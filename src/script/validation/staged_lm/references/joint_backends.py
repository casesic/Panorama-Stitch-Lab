"""Dense-joint and separate-FK references excluded from production control flow."""

from __future__ import annotations

from contextlib import contextmanager
import time
from typing import Any, Iterator

import torch

from lib_pano.human_fitting.staged_lm.config import STAGE1_PER_FRAME_DIM
from lib_pano.human_fitting.staged_lm.irls import compute_irls_point_weights
from lib_pano.human_fitting.staged_lm.linearization import (
    Linearization,
    LinearizationProvenance,
)
from lib_pano.human_fitting.staged_lm.joint.beta_derivatives import (
    joint_shape_anchor_residual_and_jacobian,
)
from lib_pano.human_fitting.staged_lm.joint import (
    Stage2JointFastCache,
    evaluate_joint_shape_state,
    joint_fast_raw_groups,
    physical_beta_from_latent,
)
import lib_pano.human_fitting.staged_lm.joint as _joint_production
from lib_pano.human_fitting.staged_lm.joint import (
    CombinedFKResult,
    accumulate_stage2_joint_arrowhead_system,
    joint_group_raw_residual_and_jacobians,
    joint_pose_anchor_residual_and_jacobian,
    validate_stage2_joint_layout,
)
from lib_pano.human_fitting.staged_lm.types import NormalEquations


def compute_joint_fk_with_pose_and_beta_jacobians_separate_reference(
    rotations: torch.Tensor,
    d_rotations_d_pose: torch.Tensor,
    joints: torch.Tensor,
    d_joints_d_beta_latent: torch.Tensor,
    parents: torch.Tensor,
) -> CombinedFKResult:
    """Propagate pose and beta FK derivatives in two independent recursions."""
    joint_count = joints.shape[0]
    pose_dim = d_rotations_d_pose.shape[-1]
    beta_dim = d_joints_d_beta_latent.shape[-1]
    relative_joints = joints.clone()
    relative_joints[1:] -= joints.index_select(0, parents[1:])
    d_relative_beta = d_joints_d_beta_latent.clone()
    d_relative_beta[1:] -= d_joints_d_beta_latent.index_select(0, parents[1:])
    local = torch.zeros(joint_count, 4, 4, dtype=joints.dtype, device=joints.device)
    local[:, :3, :3] = rotations
    local[:, :3, 3] = relative_joints
    local[:, 3, 3] = 1
    d_local_pose = torch.zeros(
        joint_count, 4, 4, pose_dim, dtype=joints.dtype, device=joints.device,
    )
    d_local_beta = torch.zeros(
        joint_count, 4, 4, beta_dim, dtype=joints.dtype, device=joints.device,
    )
    d_local_pose[:, :3, :3] = d_rotations_d_pose
    d_local_beta[:, :3, 3] = d_relative_beta
    globals_ = [local[0]]
    d_pose_list = [d_local_pose[0]]
    d_beta_list = [d_local_beta[0]]
    for joint in range(1, joint_count):
        parent = int(parents[joint])
        parent_global = globals_[parent]
        globals_.append(parent_global @ local[joint])
        d_pose_list.append(
            torch.einsum("abq,bc->acq", d_pose_list[parent], local[joint])
            + torch.einsum("ab,bcq->acq", parent_global, d_local_pose[joint])
        )
        d_beta_list.append(
            torch.einsum("abq,bc->acq", d_beta_list[parent], local[joint])
            + torch.einsum("ab,bcq->acq", parent_global, d_local_beta[joint])
        )
    global_transforms = torch.stack(globals_)
    d_global_pose = torch.stack(d_pose_list)
    d_global_beta = torch.stack(d_beta_list)
    rest = torch.eye(4, dtype=joints.dtype, device=joints.device).repeat(joint_count, 1, 1)
    rest[:, :3, 3] = -joints
    d_rest_beta = torch.zeros(
        joint_count, 4, 4, beta_dim, dtype=joints.dtype, device=joints.device,
    )
    d_rest_beta[:, :3, 3] = -d_joints_d_beta_latent
    relative = global_transforms @ rest
    return CombinedFKResult(
        global_transforms[:, :3, 3], global_transforms, relative,
        d_global_pose, torch.einsum("jabq,jbc->jacq", d_global_pose, rest),
        d_global_beta,
        torch.einsum("jabq,jbc->jacq", d_global_beta, rest)
        + torch.einsum("jab,jbcq->jacq", global_transforms, d_rest_beta),
    )


@contextmanager
def use_separate_fk_reference() -> Iterator[None]:
    """Temporarily route validation group evaluations through separate FK."""
    original = _joint_production.compute_joint_fk_with_combined_jacobian
    _joint_production.compute_joint_fk_with_combined_jacobian = (
        compute_joint_fk_with_pose_and_beta_jacobians_separate_reference
    )
    try:
        yield
    finally:
        _joint_production.compute_joint_fk_with_combined_jacobian = original


def stage2_joint_normal_equations_reference(
    cache: Stage2JointFastCache, parameters: torch.Tensor, beta_abs_limit: float,
    row_scales_by_group: list[torch.Tensor], body_pose_anchors: list[torch.Tensor],
    beta_anchor: torch.Tensor, pose_anchor_scale: float, shape_anchor_scale: float,
) -> NormalEquations:
    """Materialize the dense global joint Hessian for regression only."""
    start = time.perf_counter()
    residual, system = accumulate_stage2_joint_arrowhead_system(
        cache, parameters, beta_abs_limit, row_scales_by_group, body_pose_anchors,
        beta_anchor, pose_anchor_scale, shape_anchor_scale,
    )
    result = NormalEquations(
        residual, system.to_dense(count_call=False), system.flat_gradient(),
    )
    cache.normal_equations_ms += (time.perf_counter() - start) * 1000.0
    cache.normal_equations_calls += 1
    return result


def build_stage2_joint_reference_linearization(
    cache: Stage2JointFastCache, linearisation_point: torch.Tensor,
    beta_abs_limit: float, beta_anchor: torch.Tensor,
    body_pose_anchors: list[torch.Tensor], pose_anchor_scale: float,
    shape_anchor_scale: float, *, is_irls: bool, charbonnier_eps: float,
    aggregation_spec: dict[str, Any] | None, rep_scale: float,
    linear_solver_backend: str,
) -> Linearization:
    """Build dense or Schur analytic joint linearization for regression."""
    groups = joint_fast_raw_groups(cache, linearisation_point, beta_abs_limit)
    if is_irls:
        if aggregation_spec is None:
            raise ValueError("aggregation spec required")
        weights = compute_irls_point_weights(groups, charbonnier_eps, aggregation_spec)
        scales = [torch.sqrt(weight).repeat_interleave(2).detach() for weight in weights]
        flat = torch.cat(weights)
        stats = {
            "min": float(flat.min()), "mean": float(flat.mean()),
            "median": float(flat.median()), "max": float(flat.max()),
            "n_points": flat.numel(),
        }
    else:
        scales = [
            torch.full(
                (group.residual_2d.numel(),), rep_scale,
                dtype=linearisation_point.dtype, device=linearisation_point.device,
            )
            for group in groups
        ]
        stats = {}
    per_frame_dim, total_dim = validate_stage2_joint_layout(cache, linearisation_point)

    def anchors(parameters: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        pose = torch.cat([
            (
                parameters[
                    frame * per_frame_dim + STAGE1_PER_FRAME_DIM:
                    (frame + 1) * per_frame_dim
                ] - body_pose_anchors[frame].to(parameters)
            ) * pose_anchor_scale
            for frame in range(cache.frame_count)
        ])
        beta, _ = physical_beta_from_latent(
            parameters[-cache.model.beta_dim:], beta_abs_limit,
        )
        return pose, (beta - beta_anchor.to(parameters)) * shape_anchor_scale

    def surrogate_fn(parameters: torch.Tensor) -> torch.Tensor:
        current = joint_fast_raw_groups(cache, parameters, beta_abs_limit)
        reprojection = [
            group.residual_2d.reshape(-1) * scale.to(parameters)
            for group, scale in zip(current, scales)
        ]
        pose, shape = anchors(parameters)
        return torch.cat((*reprojection, pose, shape))

    def jacobian_fn(parameters: torch.Tensor) -> torch.Tensor:
        shape_state = evaluate_joint_shape_state(
            cache, parameters[-cache.model.beta_dim:], beta_abs_limit,
        )
        blocks = []
        for group, scale in zip(cache.groups, scales):
            result = joint_group_raw_residual_and_jacobians(
                cache, group, parameters, beta_abs_limit, shape_state,
            )
            block = torch.zeros(
                result.raw_residual.numel(), total_dim,
                dtype=parameters.dtype, device=parameters.device,
            )
            frame_slice = slice(
                group.frame_position * per_frame_dim,
                (group.frame_position + 1) * per_frame_dim,
            )
            block[:, frame_slice] = (
                result.local_pose_jacobian * scale.to(parameters).unsqueeze(1)
            )
            block[:, -cache.model.beta_dim:] = (
                result.shared_beta_jacobian * scale.to(parameters).unsqueeze(1)
            )
            blocks.append(block)
        _, pose_jacobian = joint_pose_anchor_residual_and_jacobian(
            cache, parameters, body_pose_anchors, pose_anchor_scale,
        )
        _, shape_jacobian = joint_shape_anchor_residual_and_jacobian(
            parameters[-cache.model.beta_dim:], beta_anchor, beta_abs_limit,
            shape_anchor_scale, parameter_dimension=total_dim,
        )
        return torch.cat((*blocks, pose_jacobian, shape_jacobian))

    if linear_solver_backend == "dense":
        equations_builder = stage2_joint_normal_equations_reference
        solver_label = "dense"
    elif linear_solver_backend in ("schur", "arrowhead_schur"):
        equations_builder = _joint_production.stage2_joint_arrowhead_normal_equations
        solver_label = "arrowhead_schur"
    else:
        raise ValueError(f"Unsupported linear solver backend: {linear_solver_backend}")

    def normal_equations_fn(parameters: torch.Tensor) -> NormalEquations:
        return equations_builder(
            cache, parameters, beta_abs_limit, scales, body_pose_anchors,
            beta_anchor, pose_anchor_scale, shape_anchor_scale,
        )

    return Linearization(
        surrogate=surrogate_fn,
        jacobian=jacobian_fn,
        normal_equations=normal_equations_fn,
        weight_stats=stats,
        aggregation_spec=aggregation_spec or {},
        provenance=LinearizationProvenance(
            jacobian_backend="analytic_stage2_joint",
            linearization_backend="direct_normal_equations",
            linear_solver_backend=solver_label,
        ),
    )
