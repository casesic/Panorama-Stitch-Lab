"""Explicit physical/latent beta Jacobian for dynamic Stage 2 Joint.

Purpose: preserve the pose-fixed beta-only reference chain through dynamic
joints, FK/rest correction, selected LBS, ERP, and residuals.
Mathematical scope distinguishes physical-beta and beta-latent derivatives.
It does not merge with the combined pose/beta path or construct Arrowhead H/g.
Read :mod:`.derivatives` next for the combined analytic chain.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import torch

from ..config import STAGE1_PER_FRAME_DIM
from ..geometry.erp import project_unified_erp_with_jacobian
from ..irls import build_primary_aggregation_spec, compute_irls_point_weights
from ..parameter_layout import unpack_frame_parameters
from .cache import (
    Stage2JointFastCache,
    Stage2JointFrameCameraCache,
    validate_stage2_joint_layout,
)
from .shape import (
    Stage2JointShapeState,
    evaluate_joint_shape_state,
    physical_beta_from_latent,
)
from ..pose.forward import assemble_full_pose_rotations, compute_selected_pose_offsets
from ..pose.derivatives import compute_selected_lbs_with_jacobian
from ..types import ReprojectionGroup


@dataclass
class FKBetaJacobianResult:
    posed_joints: torch.Tensor
    global_transforms: torch.Tensor
    relative_transforms: torch.Tensor
    relative_joints: torch.Tensor
    rest_inverse: torch.Tensor
    d_posed_joints_d_beta: torch.Tensor
    d_global_transforms_d_beta: torch.Tensor
    d_relative_transforms_d_beta: torch.Tensor
    d_relative_joints_d_beta: torch.Tensor
    d_rest_inverse_d_beta: torch.Tensor


@dataclass
class JointBetaSurrogateResult:
    residual: torch.Tensor
    beta_jacobian: torch.Tensor


def compute_fk_beta_jacobian(
    rotations: torch.Tensor, joints: torch.Tensor,
    d_joints_d_beta: torch.Tensor, parents: torch.Tensor,
) -> FKBetaJacobianResult:
    joint_count, beta_dim = joints.shape[0], d_joints_d_beta.shape[-1]
    relative_joints = joints.clone()
    relative_joints[1:] -= joints.index_select(0, parents[1:])
    d_relative_joints = d_joints_d_beta.clone()
    d_relative_joints[1:] -= d_joints_d_beta.index_select(0, parents[1:])
    local = torch.zeros(joint_count, 4, 4, dtype=joints.dtype, device=joints.device)
    local[:, :3, :3] = rotations
    local[:, :3, 3] = relative_joints
    local[:, 3, 3] = 1
    d_local = torch.zeros(
        joint_count, 4, 4, beta_dim, dtype=joints.dtype, device=joints.device,
    )
    d_local[:, :3, 3] = d_relative_joints
    globals_ = [local[0]]
    d_globals = [d_local[0]]
    for joint in range(1, joint_count):
        parent = int(parents[joint])
        parent_global, parent_derivative = globals_[parent], d_globals[parent]
        globals_.append(parent_global @ local[joint])
        d_globals.append(
            torch.einsum("abq,bc->acq", parent_derivative, local[joint])
            + torch.einsum("ab,bcq->acq", parent_global, d_local[joint])
        )
    global_transforms, d_global = torch.stack(globals_), torch.stack(d_globals)
    rest_inverse = torch.eye(4, dtype=joints.dtype, device=joints.device).repeat(joint_count, 1, 1)
    rest_inverse[:, :3, 3] = -joints
    d_rest_inverse = torch.zeros(
        joint_count, 4, 4, beta_dim, dtype=joints.dtype, device=joints.device,
    )
    d_rest_inverse[:, :3, 3] = -d_joints_d_beta
    relative = global_transforms @ rest_inverse
    d_relative = (
        torch.einsum("jabq,jbc->jacq", d_global, rest_inverse)
        + torch.einsum("jab,jbcq->jacq", global_transforms, d_rest_inverse)
    )
    return FKBetaJacobianResult(
        global_transforms[:, :3, 3], global_transforms, relative, relative_joints,
        rest_inverse, d_global[:, :3, 3], d_global, d_relative,
        d_relative_joints, d_rest_inverse,
    )


def joint_fast_selected_vertices_with_beta_jacobian(
    cache: Stage2JointFastCache,
    camera_cache: Stage2JointFrameCameraCache,
    shape_state: Stage2JointShapeState,
    global_orient: torch.Tensor,
    translation: torch.Tensor,
    body_pose: torch.Tensor,
    *,
    return_intermediates: bool = False,
):
    rotations = assemble_full_pose_rotations(
        global_orient, body_pose, camera_cache.left_hand_pose.to(global_orient),
        camera_cache.right_hand_pose.to(global_orient), cache.model,
    )
    pose_offsets, pose_feature = compute_selected_pose_offsets(
        rotations, camera_cache.selected_posedirs.to(global_orient),
    )
    beta_dirs = camera_cache.selected_beta_shapedirs.to(global_orient)
    selected_v_shaped = camera_cache.selected_base_v_shaped.to(global_orient) + torch.einsum(
        "ncb,b->nc", beta_dirs, shape_state.beta,
    )
    selected_v_posed = selected_v_shaped + pose_offsets
    fk = compute_fk_beta_jacobian(
        rotations, shape_state.joints, shape_state.d_joints_d_beta,
        cache.model.parents.to(global_orient.device),
    )
    skinned, d_skinned_beta, transforms, d_transforms = compute_selected_lbs_with_jacobian(
        selected_v_posed, beta_dirs, camera_cache.selected_lbs_weights.to(global_orient),
        fk.relative_transforms, fk.d_relative_transforms_d_beta,
    )
    scale = camera_cache.scale.to(global_orient).reshape(1, 1)
    vertices = skinned * scale + translation.reshape(1, 3)
    d_vertices_beta = d_skinned_beta * scale.unsqueeze(-1)
    d_vertices_latent = d_vertices_beta * shape_state.d_beta_d_latent.reshape(1, 1, -1)
    if not return_intermediates:
        return vertices, d_vertices_beta, d_vertices_latent
    return vertices, d_vertices_beta, d_vertices_latent, {
        "physical_beta": shape_state.beta,
        "d_beta_d_latent": shape_state.d_beta_d_latent,
        "selected_v_shaped": selected_v_shaped,
        "d_selected_v_shaped_d_beta": beta_dirs,
        "d_selected_v_shaped_d_latent": beta_dirs * shape_state.d_beta_d_latent.reshape(1, 1, -1),
        "joints": shape_state.joints,
        "d_joints_d_beta": shape_state.d_joints_d_beta,
        "d_joints_d_latent": shape_state.d_joints_d_latent,
        "rotations": rotations, "pose_feature": pose_feature,
        "selected_pose_offsets": pose_offsets, "selected_v_posed": selected_v_posed,
        "relative_joints": fk.relative_joints, "global_transforms": fk.global_transforms,
        "relative_transforms": fk.relative_transforms,
        "d_global_transforms_d_beta": fk.d_global_transforms_d_beta,
        "d_rest_inverse_d_beta": fk.d_rest_inverse_d_beta,
        "d_relative_transforms_d_beta": fk.d_relative_transforms_d_beta,
        "selected_transforms": transforms, "d_selected_transforms_d_beta": d_transforms,
        "selected_skinned_vertices": skinned,
        "d_selected_skinned_vertices_d_beta": d_skinned_beta,
        "final_selected_vertices": vertices,
    }


def joint_group_raw_residual_and_beta_jacobian(
    cache: Stage2JointFastCache,
    camera_cache: Stage2JointFrameCameraCache,
    parameters: torch.Tensor,
    beta_abs_limit: float,
    shape_state: Stage2JointShapeState | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if shape_state is None:
        shape_state = evaluate_joint_shape_state(
            cache, parameters[-cache.model.beta_dim:], beta_abs_limit,
        )
    per_frame_dim, _ = validate_stage2_joint_layout(cache, parameters)
    frame = unpack_frame_parameters(parameters, camera_cache.frame_position, per_frame_dim, True)
    orient = frame.cam2_orient if camera_cache.camera == "cam2" else frame.cam4_orient
    translation = frame.cam2_transl if camera_cache.camera == "cam2" else frame.cam4_transl
    vertices, _, d_vertices_latent = joint_fast_selected_vertices_with_beta_jacobian(
        cache, camera_cache, shape_state, orient.reshape(3), translation.reshape(3),
        frame.body_pose.reshape(-1),
    )
    uv, projection_jacobian = project_unified_erp_with_jacobian(
        vertices, cache.image_width, cache.image_height,
    )
    jacobian = torch.einsum("nab,nbq->naq", projection_jacobian, d_vertices_latent)
    return (uv - camera_cache.targets.to(parameters)).reshape(-1), jacobian.reshape(-1, cache.model.beta_dim)


def joint_explicit_raw_beta_jacobian(
    cache: Stage2JointFastCache, parameters: torch.Tensor, beta_abs_limit: float,
) -> torch.Tensor:
    if cache.profile_timing and parameters.is_cuda:
        torch.cuda.synchronize(parameters.device)
    start = time.perf_counter()
    shape_state = evaluate_joint_shape_state(cache, parameters[-cache.model.beta_dim:], beta_abs_limit)
    blocks = [joint_group_raw_residual_and_beta_jacobian(
        cache, group, parameters, beta_abs_limit, shape_state,
    )[1] for group in cache.groups]
    result = torch.cat(blocks)
    if cache.profile_timing and parameters.is_cuda:
        torch.cuda.synchronize(parameters.device)
    cache.beta_jacobian_ms += (time.perf_counter() - start) * 1000.0
    cache.beta_jacobian_calls += 1
    return result


def joint_explicit_raw_global_beta_block(
    cache: Stage2JointFastCache, parameters: torch.Tensor, beta_abs_limit: float,
) -> torch.Tensor:
    """DEBUG_ONLY_GLOBAL_BETA_BLOCK: zeros in pose columns, beta J at the end."""
    beta_j = joint_explicit_raw_beta_jacobian(cache, parameters, beta_abs_limit)
    result = torch.zeros(
        beta_j.shape[0], parameters.numel(), dtype=parameters.dtype, device=parameters.device,
    )
    result[:, -cache.model.beta_dim:] = beta_j
    return result


def joint_shape_anchor_residual_and_jacobian(
    beta_latent: torch.Tensor, beta_anchor: torch.Tensor,
    beta_abs_limit: float, shape_anchor_scale: float,
    *, parameter_dimension: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    beta, derivative = physical_beta_from_latent(beta_latent, beta_abs_limit)
    residual = (beta - beta_anchor.to(beta)) * shape_anchor_scale
    local = torch.diag(derivative * shape_anchor_scale)
    if parameter_dimension is None:
        return residual, local
    result = torch.zeros(
        beta.numel(), parameter_dimension, dtype=beta.dtype, device=beta.device,
    )
    result[:, -beta.numel():] = local
    return residual, result


def joint_surrogate_residual_and_beta_jacobian(
    cache: Stage2JointFastCache,
    parameters: torch.Tensor,
    beta_abs_limit: float,
    beta_anchor: torch.Tensor,
    body_pose_anchors: list[torch.Tensor],
    pose_anchor_scale: float,
    shape_anchor_scale: float,
    *, objective_mode: str, charbonnier_eps: float, rep_scale: float,
) -> JointBetaSurrogateResult:
    shape_state = evaluate_joint_shape_state(cache, parameters[-cache.model.beta_dim:], beta_abs_limit)
    raw_parts, beta_parts, groups = [], [], []
    for camera_cache in cache.groups:
        residual, jacobian = joint_group_raw_residual_and_beta_jacobian(
            cache, camera_cache, parameters, beta_abs_limit, shape_state,
        )
        raw_parts.append(residual); beta_parts.append(jacobian)
        groups.append(ReprojectionGroup(
            camera_cache.frame_index, camera_cache.camera, residual.reshape(-1, 2),
        ))
    if objective_mode == "charbonnier_irls":
        spec = build_primary_aggregation_spec(groups)
        weights = compute_irls_point_weights(groups, charbonnier_eps, spec)
        row_scales = [torch.sqrt(weight).repeat_interleave(2) for weight in weights]
    elif objective_mode == "raw_l2":
        row_scales = [torch.full(
            (residual.numel(),), rep_scale, dtype=parameters.dtype, device=parameters.device,
        ) for residual in raw_parts]
    else:
        raise ValueError(objective_mode)
    scaled_r = [r * s.to(parameters) for r, s in zip(raw_parts, row_scales)]
    scaled_j = [j * s.to(parameters).unsqueeze(1) for j, s in zip(beta_parts, row_scales)]
    per_frame_dim, _ = validate_stage2_joint_layout(cache, parameters)
    pose_anchor = torch.cat([
        (parameters[f * per_frame_dim + STAGE1_PER_FRAME_DIM:(f + 1) * per_frame_dim]
         - body_pose_anchors[f].to(parameters)) * pose_anchor_scale
        for f in range(cache.frame_count)
    ])
    shape_residual, shape_jacobian = joint_shape_anchor_residual_and_jacobian(
        parameters[-cache.model.beta_dim:], beta_anchor, beta_abs_limit, shape_anchor_scale,
    )
    return JointBetaSurrogateResult(
        torch.cat((*scaled_r, pose_anchor, shape_residual)),
        torch.cat((
            *scaled_j,
            torch.zeros(pose_anchor.numel(), cache.model.beta_dim,
                        dtype=parameters.dtype, device=parameters.device),
            shape_jacobian,
        )),
    )
