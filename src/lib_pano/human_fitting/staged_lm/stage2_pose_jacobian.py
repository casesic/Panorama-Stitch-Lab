"""Explicit analytic Jacobian and direct normal equations for Stage 2 pose.

The dense global Jacobian remains debug-only; production LM accumulates
per-frame normal-equation blocks directly.
"""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any

import torch

from .analytic_stage1 import (
    _skew,
    axis_angle_to_matrix,
    project_cam2_erp_with_jacobian,
    so3_right_jacobian,
)
from .config import STAGE1_PER_FRAME_DIM
from .irls import build_primary_aggregation_spec, compute_irls_point_weights
from .parameter_layout import unpack_frame_parameters
from .stage2_pose_fast import (
    Stage2PoseFastCache,
    Stage2PoseFrameCameraCache,
    assemble_full_pose_rotations,
    fast_raw_residual,
    fast_raw_groups,
)
from .types import NormalEquations, ReprojectionGroup


ROTATION_PARAMETER_DIM = 66
POSE_PER_FRAME_DIM = 75


def validate_stage2_pose_analytic_layout(cache: Stage2PoseFastCache) -> None:
    expected_body = 63
    expected_per_frame = STAGE1_PER_FRAME_DIM + expected_body
    expected_rotation = 3 + expected_body
    model_body_joints = (cache.model.joint_count - 34) if cache.model.joint_count >= 34 else -1
    if (cache.body_pose_dim != expected_body or POSE_PER_FRAME_DIM != expected_per_frame
            or ROTATION_PARAMETER_DIM != expected_rotation or model_body_joints != 21):
        raise RuntimeError(
            "UNSUPPORTED_STAGE2_POSE_ANALYTIC_LAYOUT: "
            f"expected body_pose_dim={expected_body}, per_frame_dim={expected_per_frame}, "
            f"rotation_parameter_dim={expected_rotation}, NUM_BODY_JOINTS=21; "
            f"actual body_pose_dim={cache.body_pose_dim}, per_frame_dim="
            f"{STAGE1_PER_FRAME_DIM + cache.body_pose_dim}, rotation_parameter_dim="
            f"{3 + cache.body_pose_dim}, model_NUM_BODY_JOINTS={model_body_joints}"
        )


@dataclass
class FKJacobianResult:
    posed_joints: torch.Tensor
    global_transforms: torch.Tensor
    relative_transforms: torch.Tensor
    d_global_transforms: torch.Tensor
    d_relative_transforms: torch.Tensor


@dataclass
class SurrogateJacobianResult:
    residual: torch.Tensor
    jacobian: torch.Tensor


def rotation_matrix_jacobian_additive(axis_angle: torch.Tensor) -> torch.Tensor:
    """Return ``dR/dphi`` for direct additive axis-angle coordinates."""
    phi = axis_angle.reshape(3)
    rotation = axis_angle_to_matrix(phi)
    right_jacobian = so3_right_jacobian(phi)
    basis_skew = _skew(right_jacobian.T)
    return torch.einsum("ij,kjl->ilk", rotation, basis_skew)


def assemble_full_pose_rotation_jacobians(
    global_orient: torch.Tensor,
    body_pose: torch.Tensor,
    left_hand_pose: torch.Tensor,
    right_hand_pose: torch.Tensor,
    model_cache: Any,
) -> tuple[torch.Tensor, torch.Tensor]:
    rotations = assemble_full_pose_rotations(
        global_orient, body_pose, left_hand_pose, right_hand_pose, model_cache,
    )
    derivatives = torch.zeros(
        model_cache.joint_count, 3, 3, ROTATION_PARAMETER_DIM,
        dtype=rotations.dtype, device=rotations.device,
    )
    derivatives[0, :, :, :3] = rotation_matrix_jacobian_additive(global_orient)
    body_axis_angles = body_pose.reshape(21, 3)
    body_derivatives = torch.stack([
        rotation_matrix_jacobian_additive(body_axis_angles[joint])
        for joint in range(21)
    ])
    for joint in range(21):
        column = 3 + joint * 3
        derivatives[joint + 1, :, :, column:column + 3] = body_derivatives[joint]
    return rotations, derivatives


def compute_pose_feature_with_jacobian(
    rotations: torch.Tensor,
    rotation_jacobians: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    identity = torch.eye(3, dtype=rotations.dtype, device=rotations.device)
    feature = (rotations[1:] - identity).reshape(-1)
    derivative = rotation_jacobians[1:].reshape(-1, rotation_jacobians.shape[-1])
    return feature, derivative


def compute_selected_pose_offsets_with_jacobian(
    rotations: torch.Tensor,
    rotation_jacobians: torch.Tensor,
    selected_posedirs: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    feature, d_feature = compute_pose_feature_with_jacobian(rotations, rotation_jacobians)
    offsets = torch.einsum("p,pnc->nc", feature, selected_posedirs)
    d_offsets = torch.einsum("pq,pnc->ncq", d_feature, selected_posedirs)
    return offsets, d_offsets, feature, d_feature


def compute_fk_with_jacobian(
    rotations: torch.Tensor,
    rotation_jacobians: torch.Tensor,
    joints: torch.Tensor,
    parents: torch.Tensor,
) -> FKJacobianResult:
    """Reproduce batch_rigid_transform and propagate all parameter columns."""
    joint_count, parameter_count = rotations.shape[0], rotation_jacobians.shape[-1]
    relative_joints = joints.clone()
    relative_joints[1:] -= joints.index_select(0, parents[1:])
    local = torch.zeros(joint_count, 4, 4, dtype=rotations.dtype, device=rotations.device)
    local[:, :3, :3] = rotations
    local[:, :3, 3] = relative_joints
    local[:, 3, 3] = 1.0
    d_local = torch.zeros(
        joint_count, 4, 4, parameter_count,
        dtype=rotations.dtype, device=rotations.device,
    )
    d_local[:, :3, :3] = rotation_jacobians

    global_list = [local[0]]
    derivative_list = [d_local[0]]
    for joint in range(1, joint_count):
        parent = int(parents[joint])
        parent_global = global_list[parent]
        parent_derivative = derivative_list[parent]
        global_transform = parent_global @ local[joint]
        derivative = (
            torch.einsum("abq,bc->acq", parent_derivative, local[joint])
            + torch.einsum("ab,bcq->acq", parent_global, d_local[joint])
        )
        global_list.append(global_transform)
        derivative_list.append(derivative)
    global_transforms = torch.stack(global_list)
    d_global = torch.stack(derivative_list)
    rest_inverse = torch.eye(4, dtype=rotations.dtype, device=rotations.device).repeat(joint_count, 1, 1)
    rest_inverse[:, :3, 3] = -joints
    relative_transforms = global_transforms @ rest_inverse
    d_relative = torch.einsum("jabq,jbc->jacq", d_global, rest_inverse)
    return FKJacobianResult(
        posed_joints=global_transforms[:, :3, 3],
        global_transforms=global_transforms,
        relative_transforms=relative_transforms,
        d_global_transforms=d_global,
        d_relative_transforms=d_relative,
    )


def compute_selected_lbs_with_jacobian(
    selected_v_posed: torch.Tensor,
    d_selected_v_posed: torch.Tensor,
    selected_lbs_weights: torch.Tensor,
    relative_transforms: torch.Tensor,
    d_relative_transforms: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    joint_count = relative_transforms.shape[0]
    transforms = (
        selected_lbs_weights @ relative_transforms.reshape(joint_count, 16)
    ).reshape(-1, 4, 4)
    d_transforms = torch.einsum("nj,jabq->nabq", selected_lbs_weights, d_relative_transforms)
    homogeneous = torch.cat((
        selected_v_posed,
        torch.ones(selected_v_posed.shape[0], 1, dtype=selected_v_posed.dtype, device=selected_v_posed.device),
    ), dim=1)
    d_homogeneous = torch.cat((
        d_selected_v_posed,
        torch.zeros(
            selected_v_posed.shape[0], 1, d_selected_v_posed.shape[-1],
            dtype=selected_v_posed.dtype, device=selected_v_posed.device,
        ),
    ), dim=1)
    vertices = (transforms @ homogeneous.unsqueeze(-1))[:, :3, 0]
    derivative = (
        torch.einsum("nabq,nb->naq", d_transforms[:, :3], homogeneous)
        + torch.einsum("nab,nbq->naq", transforms[:, :3], d_homogeneous)
    )
    return vertices, derivative, transforms, d_transforms


def fast_selected_vertices_with_jacobian(
    cache: Stage2PoseFastCache,
    camera_cache: Stage2PoseFrameCameraCache,
    global_orient: torch.Tensor,
    translation: torch.Tensor,
    body_pose: torch.Tensor,
    *,
    return_intermediates: bool = False,
) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    reference = global_orient
    rotations, d_rotations = assemble_full_pose_rotation_jacobians(
        global_orient, body_pose, camera_cache.left_hand_pose.to(reference),
        camera_cache.right_hand_pose.to(reference), cache.model,
    )
    offsets, d_offsets, feature, d_feature = compute_selected_pose_offsets_with_jacobian(
        rotations, d_rotations, camera_cache.selected_posedirs.to(reference),
    )
    selected_v_posed = camera_cache.selected_v_shaped.to(reference) + offsets
    fk = compute_fk_with_jacobian(
        rotations, d_rotations, cache.model.joints.to(reference),
        cache.model.parents.to(reference.device),
    )
    skinned, d_skinned, transforms, d_transforms = compute_selected_lbs_with_jacobian(
        selected_v_posed, d_offsets, camera_cache.selected_lbs_weights.to(reference),
        fk.relative_transforms, fk.d_relative_transforms,
    )
    scale = camera_cache.scale.to(reference).reshape(1, 1)
    vertices = skinned * scale + translation.reshape(1, 3)
    d_vertices = d_skinned * scale.unsqueeze(-1)
    if not return_intermediates:
        return vertices, d_vertices
    return vertices, d_vertices, {
        "rotation_jacobians": d_rotations,
        "pose_feature_jacobian": d_feature,
        "selected_pose_offset_jacobian": d_offsets,
        "global_transform_jacobian": fk.d_global_transforms,
        "relative_transform_jacobian": fk.d_relative_transforms,
        "selected_transform_jacobian": d_transforms,
        "selected_skinned_vertex_jacobian": d_skinned,
        "pose_feature": feature,
        "global_transforms": fk.global_transforms,
        "relative_transforms": fk.relative_transforms,
        "selected_transforms": transforms,
    }


def stage2_pose_group_raw_residual_and_jacobian(
    cache: Stage2PoseFastCache,
    camera_cache: Stage2PoseFrameCameraCache,
    parameters: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    validate_stage2_pose_analytic_layout(cache)
    cache.group_analytic_calls += 1
    frame = unpack_frame_parameters(
        parameters, camera_cache.frame_position, POSE_PER_FRAME_DIM, include_body_pose=True,
    )
    if camera_cache.camera == "cam2":
        orient, translation, camera_column = frame.cam2_orient.reshape(3), frame.cam2_transl.reshape(3), 0
    else:
        orient, translation, camera_column = frame.cam4_orient.reshape(3), frame.cam4_transl.reshape(3), 6
    vertices, d_vertices = fast_selected_vertices_with_jacobian(
        cache, camera_cache, orient, translation, frame.body_pose.reshape(-1),
    )
    uv, projection_jacobian = project_cam2_erp_with_jacobian(
        vertices, cache.image_width, cache.image_height,
    )
    uv_rot_pose = torch.einsum("nab,nbq->naq", projection_jacobian, d_vertices)
    point_count = vertices.shape[0]
    local_jacobian = torch.zeros(
        point_count, 2, POSE_PER_FRAME_DIM,
        dtype=parameters.dtype, device=parameters.device,
    )
    local_jacobian[:, :, camera_column:camera_column + 3] = uv_rot_pose[:, :, :3]
    local_jacobian[:, :, camera_column + 3:camera_column + 6] = projection_jacobian
    local_jacobian[:, :, STAGE1_PER_FRAME_DIM:] = uv_rot_pose[:, :, 3:]
    residual = uv - camera_cache.targets.to(parameters)
    return residual.reshape(-1), local_jacobian.reshape(point_count * 2, POSE_PER_FRAME_DIM)


def stage2_pose_raw_residual(cache: Stage2PoseFastCache, parameters: torch.Tensor) -> torch.Tensor:
    """Residual-only selected forward; this deliberately constructs no Jacobian."""
    return fast_raw_residual(cache, parameters)


def stage2_pose_explicit_raw_jacobian(cache: Stage2PoseFastCache, parameters: torch.Tensor) -> torch.Tensor:
    """DEBUG_ONLY_GLOBAL_JACOBIAN: construct a dense multi-frame oracle J."""
    validate_stage2_pose_analytic_layout(cache)
    if cache.profile_timing and parameters.is_cuda:
        torch.cuda.synchronize(parameters.device)
    start = time.perf_counter()
    row_count = sum(group.targets.numel() for group in cache.groups)
    jacobian = torch.zeros(
        row_count, cache.frame_count * POSE_PER_FRAME_DIM,
        dtype=parameters.dtype, device=parameters.device,
    )
    row = 0
    for group in cache.groups:
        residual, local = stage2_pose_group_raw_residual_and_jacobian(cache, group, parameters)
        column = group.frame_position * POSE_PER_FRAME_DIM
        jacobian[row:row + residual.numel(), column:column + POSE_PER_FRAME_DIM] = local
        row += residual.numel()
    if cache.profile_timing and parameters.is_cuda:
        torch.cuda.synchronize(parameters.device)
    cache.explicit_jacobian_ms += (time.perf_counter() - start) * 1000.0
    cache.explicit_jacobian_calls += 1
    return jacobian


def stage2_pose_anchor_residual_and_jacobian(
    parameters: torch.Tensor,
    body_pose_anchors: list[torch.Tensor],
    frame_count: int,
    pose_anchor_scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    body_pose_dim = int(body_pose_anchors[0].numel())
    per_frame_dim = STAGE1_PER_FRAME_DIM + body_pose_dim
    residual_parts = []
    jacobian = torch.zeros(
        frame_count * body_pose_dim, frame_count * per_frame_dim,
        dtype=parameters.dtype, device=parameters.device,
    )
    identity = torch.eye(body_pose_dim, dtype=parameters.dtype, device=parameters.device) * pose_anchor_scale
    for frame in range(frame_count):
        parameter_start = frame * per_frame_dim + STAGE1_PER_FRAME_DIM
        row_start = frame * body_pose_dim
        residual_parts.append(
            (parameters[parameter_start:parameter_start + body_pose_dim] - body_pose_anchors[frame].to(parameters))
            * pose_anchor_scale
        )
        jacobian[row_start:row_start + body_pose_dim,
                 parameter_start:parameter_start + body_pose_dim] = identity
    return torch.cat(residual_parts), jacobian


def stage2_pose_surrogate_residual_and_jacobian(
    cache: Stage2PoseFastCache,
    parameters: torch.Tensor,
    body_pose_anchors: list[torch.Tensor],
    pose_anchor_scale: float,
    *,
    objective_mode: str,
    charbonnier_eps: float,
    rep_scale: float,
) -> SurrogateJacobianResult:
    validate_stage2_pose_analytic_layout(cache)
    raw_parts: list[torch.Tensor] = []
    groups: list[ReprojectionGroup] = []
    row_count = sum(group.targets.numel() for group in cache.groups)
    raw_jacobian = torch.zeros(
        row_count, cache.frame_count * POSE_PER_FRAME_DIM,
        dtype=parameters.dtype, device=parameters.device,
    )
    row = 0
    for camera_cache in cache.groups:
        residual, local = stage2_pose_group_raw_residual_and_jacobian(
            cache, camera_cache, parameters,
        )
        raw_parts.append(residual)
        groups.append(ReprojectionGroup(
            camera_cache.frame_index, camera_cache.camera, residual.reshape(-1, 2),
        ))
        column = camera_cache.frame_position * POSE_PER_FRAME_DIM
        raw_jacobian[row:row + residual.numel(), column:column + POSE_PER_FRAME_DIM] = local
        row += residual.numel()
    raw_residual = torch.cat(raw_parts)
    if objective_mode == "charbonnier_irls":
        spec = build_primary_aggregation_spec(groups)
        weights = compute_irls_point_weights(groups, charbonnier_eps, spec)
        scales = torch.cat([torch.sqrt(weight).repeat_interleave(2) for weight in weights]).to(parameters)
    elif objective_mode == "raw_l2":
        scales = torch.full(
            (raw_residual.numel(),), rep_scale,
            dtype=parameters.dtype, device=parameters.device,
        )
    else:
        raise ValueError(objective_mode)
    anchor_residual, anchor_jacobian = stage2_pose_anchor_residual_and_jacobian(
        parameters, body_pose_anchors, cache.frame_count, pose_anchor_scale,
    )
    return SurrogateJacobianResult(
        residual=torch.cat((raw_residual * scales, anchor_residual)),
        jacobian=torch.cat((raw_jacobian * scales.unsqueeze(1), anchor_jacobian), dim=0),
    )


def stage2_pose_normal_equations(
    cache: Stage2PoseFastCache,
    parameters: torch.Tensor,
    row_scales_by_group: list[torch.Tensor],
    body_pose_anchors: list[torch.Tensor],
    pose_anchor_scale: float,
) -> NormalEquations:
    """Accumulate per-frame 75x75 systems without a global dense Jacobian."""
    validate_stage2_pose_analytic_layout(cache)
    if len(row_scales_by_group) != len(cache.groups):
        raise ValueError("Stage 2 pose row-scale group count mismatch")
    if cache.profile_timing and parameters.is_cuda:
        torch.cuda.synchronize(parameters.device)
    start = time.perf_counter()
    dimension = cache.frame_count * POSE_PER_FRAME_DIM
    hessian = torch.zeros(dimension, dimension, dtype=torch.float64, device=parameters.device)
    gradient = torch.zeros(dimension, dtype=torch.float64, device=parameters.device)
    residual_parts: list[torch.Tensor] = []
    for camera_cache, row_scales in zip(cache.groups, row_scales_by_group):
        raw_residual, local_jacobian = stage2_pose_group_raw_residual_and_jacobian(
            cache, camera_cache, parameters,
        )
        scales = row_scales.to(parameters)
        if scales.numel() != raw_residual.numel():
            raise ValueError("Stage 2 pose row-scale dimension mismatch")
        scaled_residual = raw_residual * scales
        scaled_jacobian = local_jacobian * scales.unsqueeze(1)
        residual_parts.append(scaled_residual)
        local_64 = scaled_jacobian.to(torch.float64)
        residual_64 = scaled_residual.to(torch.float64)
        block = slice(
            camera_cache.frame_position * POSE_PER_FRAME_DIM,
            (camera_cache.frame_position + 1) * POSE_PER_FRAME_DIM,
        )
        hessian[block, block] += local_64.T @ local_64
        gradient[block] += local_64.T @ residual_64

    anchor_parts: list[torch.Tensor] = []
    anchor_scale_squared = float(pose_anchor_scale) ** 2
    for frame in range(cache.frame_count):
        frame_start = frame * POSE_PER_FRAME_DIM
        pose_slice = slice(frame_start + STAGE1_PER_FRAME_DIM, frame_start + POSE_PER_FRAME_DIM)
        pose_delta = parameters[pose_slice] - body_pose_anchors[frame].to(parameters)
        anchor_parts.append(pose_delta * pose_anchor_scale)
        hessian[pose_slice, pose_slice] += torch.eye(
            cache.body_pose_dim, dtype=torch.float64, device=parameters.device,
        ) * anchor_scale_squared
        gradient[pose_slice] += pose_delta.to(torch.float64) * anchor_scale_squared
    result = NormalEquations(
        residual=torch.cat((*residual_parts, *anchor_parts)),
        hessian=hessian,
        gradient=gradient,
    )
    if cache.profile_timing and parameters.is_cuda:
        torch.cuda.synchronize(parameters.device)
    cache.normal_equations_ms += (time.perf_counter() - start) * 1000.0
    cache.normal_equations_calls += 1
    return result


def build_stage2_pose_analytic_linearization(
    cache: Stage2PoseFastCache,
    linearisation_point: torch.Tensor,
    body_pose_anchors: list[torch.Tensor],
    pose_anchor_scale: float,
    *,
    is_irls: bool,
    charbonnier_eps: float,
    aggregation_spec: dict[str, Any] | None,
    rep_scale: float,
) -> dict[str, Any]:
    validate_stage2_pose_analytic_layout(cache)
    groups = fast_raw_groups(cache, linearisation_point)
    if is_irls:
        if aggregation_spec is None:
            raise ValueError("aggregation_spec is required for Stage 2 pose IRLS")
        weights = compute_irls_point_weights(groups, charbonnier_eps, aggregation_spec)
        row_scales = [torch.sqrt(weight).repeat_interleave(2).detach() for weight in weights]
        flat_weights = torch.cat(weights)
        weight_stats = {
            "min": float(flat_weights.min()), "mean": float(flat_weights.mean()),
            "median": float(flat_weights.median()), "max": float(flat_weights.max()),
            "n_points": int(flat_weights.numel()),
        }
    else:
        row_scales = [torch.full(
            (group.residual_2d.numel(),), rep_scale,
            dtype=linearisation_point.dtype, device=linearisation_point.device,
        ) for group in groups]
        weight_stats = {}

    def anchor_residual(parameters: torch.Tensor) -> torch.Tensor:
        return torch.cat([
            (parameters[frame * POSE_PER_FRAME_DIM + STAGE1_PER_FRAME_DIM:
                        (frame + 1) * POSE_PER_FRAME_DIM]
             - body_pose_anchors[frame].to(parameters)) * pose_anchor_scale
            for frame in range(cache.frame_count)
        ])

    def surrogate_fn(parameters: torch.Tensor) -> torch.Tensor:
        raw = fast_raw_groups(cache, parameters)
        scaled = [
            group.residual_2d.reshape(-1) * scales.to(parameters)
            for group, scales in zip(raw, row_scales)
        ]
        return torch.cat((*scaled, anchor_residual(parameters)))

    def jacobian_fn(parameters: torch.Tensor) -> torch.Tensor:
        row_count = sum(group.targets.numel() for group in cache.groups)
        raw_jacobian = torch.zeros(
            row_count, cache.frame_count * POSE_PER_FRAME_DIM,
            dtype=parameters.dtype, device=parameters.device,
        )
        row = 0
        for camera_cache, scales in zip(cache.groups, row_scales):
            residual, local = stage2_pose_group_raw_residual_and_jacobian(
                cache, camera_cache, parameters,
            )
            column = camera_cache.frame_position * POSE_PER_FRAME_DIM
            raw_jacobian[row:row + residual.numel(), column:column + POSE_PER_FRAME_DIM] = (
                local * scales.to(parameters).unsqueeze(1)
            )
            row += residual.numel()
        return torch.cat((raw_jacobian, stage2_pose_anchor_residual_and_jacobian(
            parameters, body_pose_anchors, cache.frame_count, pose_anchor_scale,
        )[1]), dim=0)

    def normal_equations_fn(parameters: torch.Tensor) -> NormalEquations:
        return stage2_pose_normal_equations(
            cache, parameters, row_scales, body_pose_anchors, pose_anchor_scale,
        )

    return {
        "surrogate_fn": surrogate_fn,
        "jacobian_fn": jacobian_fn,
        "normal_equations_fn": normal_equations_fn,
        "weight_stats": weight_stats,
        "spec": aggregation_spec or {},
        "jacobian_backend": "analytic_stage2_pose",
        "linearization_backend": "direct_normal_equations",
    }
