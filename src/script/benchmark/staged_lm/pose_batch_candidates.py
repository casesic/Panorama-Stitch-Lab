"""Benchmark wrappers around the production Stage 2 Pose FULL_BATCH kernel.

The stable batch geometry/Jacobian/Hg implementation lives exclusively in
``lib_pano.human_fitting.staged_lm.stage2_pose_batched``.  This module retains
only objective candidates, profiling helpers, and compatibility wrappers used
by shadow/ablation experiments.
"""

from __future__ import annotations

import time
from typing import Any
from dataclasses import replace

import torch

from lib_pano.human_fitting.staged_lm.analytic_stage1 import (
    axis_angle_to_matrix, project_cam2_erp_with_jacobian,
)
from lib_pano.human_fitting.staged_lm.irls import compute_primary_charbonnier_tensor
from lib_pano.human_fitting.staged_lm.parameter_layout import unpack_frame_parameters
from lib_pano.human_fitting.staged_lm.stage2_pose_batched import (
    Stage2PoseBatchCache as PoseGroupBatch,
    Stage2PoseBatchParameters as PoseBatchParameters,
    build_stage2_pose_batch_cache,
    gather_pose_batch_parameters,
    stage2_pose_batched_accumulate_normal_equations,
    stage2_pose_batched_feature_offsets_from_rotations as batched_pose_feature_offsets_from_rotations,
    stage2_pose_batched_fk_by_depth as batched_pose_fk_by_depth,
    stage2_pose_batched_lbs_from_fk as batched_pose_lbs_from_fk,
    stage2_pose_batched_projection_from_vertices as batched_pose_projection_from_vertices,
    stage2_pose_batched_rotations_and_jacobians as batched_pose_rotations_and_jacobians,
)
from lib_pano.human_fitting.staged_lm.stage2_pose_fast import Stage2PoseFastCache
from lib_pano.human_fitting.staged_lm.stage2_pose_jacobian import (
    assemble_full_pose_rotation_jacobians,
    compute_fk_with_jacobian,
    compute_selected_lbs_with_jacobian,
    compute_selected_pose_offsets_with_jacobian,
)
from lib_pano.human_fitting.staged_lm.types import ReprojectionGroup


def build_pose_group_batch(cache: Stage2PoseFastCache) -> PoseGroupBatch:
    return build_stage2_pose_batch_cache(cache, pose_anchor_scale=1.0)


def gather_pose_group_parameters(
    parameters: torch.Tensor, layout,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    gathered = gather_pose_batch_parameters(parameters, layout)
    return (
        gathered.group_orientations,
        gathered.group_translations,
        gathered.group_body_poses,
    )


def batched_pose_group_analytic(
    batch: PoseGroupBatch,
    orientations: torch.Tensor,
    translations: torch.Tensor,
    body_poses: torch.Tensor,
    image_width: int,
    image_height: int,
    *,
    vectorized_scatter: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    rotations, d_rotations = batched_pose_rotations_and_jacobians(
        orientations, body_poses, batch.left_hand_pose, batch.right_hand_pose,
        derivative_mapping=batch.derivative_mapping,
        vectorized_scatter=vectorized_scatter,
    )
    _, _, _, d_offsets, v_posed = batched_pose_feature_offsets_from_rotations(
        batch, rotations, d_rotations,
    )
    relative, d_relative = batched_pose_fk_by_depth(
        rotations, d_rotations, batch.joints, batch.parents,
        batch.fk_depth_indices,
    )
    _, _, _, _, vertices, d_vertices = batched_pose_lbs_from_fk(
        batch, v_posed, d_offsets, relative, d_relative, translations,
    )
    _, _, _, residual, active = batched_pose_projection_from_vertices(
        batch, vertices, d_vertices, image_width, image_height,
    )
    return residual, active, batch.layout.valid_mask


def batched_pose_hg(
    batch: PoseGroupBatch,
    orientations: torch.Tensor,
    translations: torch.Tensor,
    body_poses: torch.Tensor,
    row_scales: torch.Tensor,
    body_pose_anchors: list[torch.Tensor],
    pose_anchor_scale: float,
    image_width: int,
    image_height: int,
    *,
    frame_body_poses: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if frame_body_poses is None:
        raise RuntimeError("MISSING_FRAME_BODY_POSES")
    residual, active, _ = batched_pose_group_analytic(
        batch, orientations, translations, body_poses, image_width, image_height,
    )
    parameters = torch.zeros(
        batch.layout.frame_count, 75, dtype=orientations.dtype,
        device=orientations.device,
    )
    parameters[:, 12:] = frame_body_poses
    gathered = PoseBatchParameters(
        orientations, translations, body_poses, frame_body_poses,
    )
    accumulation_cache = replace(
        batch,
        pose_anchor_identity_64=torch.eye(
            63, dtype=torch.float64, device=orientations.device,
        ) * float(pose_anchor_scale) ** 2,
    )
    equations = stage2_pose_batched_accumulate_normal_equations(
        accumulation_cache, parameters.reshape(-1), gathered, residual, active,
        row_scales.reshape(batch.layout.group_count, batch.layout.nmax, 2),
        body_pose_anchors, pose_anchor_scale,
    )
    return residual, active, equations.hessian, equations.gradient


def _batched_pose_values(
    batch: PoseGroupBatch,
    orientations: torch.Tensor,
    translations: torch.Tensor,
    body_poses: torch.Tensor,
) -> torch.Tensor:
    group_count = batch.layout.group_count
    zeros = torch.zeros(group_count, 3, 3, dtype=orientations.dtype, device=orientations.device)
    axis_angles = torch.cat((
        orientations[:, None], body_poses.reshape(group_count, 21, 3), zeros,
        batch.left_hand_pose.reshape(group_count, 15, 3),
        batch.right_hand_pose.reshape(group_count, 15, 3),
    ), dim=1)
    rotations = axis_angle_to_matrix(axis_angles)
    identity = torch.eye(3, dtype=rotations.dtype, device=rotations.device)
    feature = (rotations[:, 1:] - identity).reshape(group_count, -1)
    offsets = torch.einsum("gp,gpnc->gnc", feature, batch.selected_posedirs)
    v_posed = batch.selected_v_shaped + offsets
    relative, _ = batched_pose_fk_by_depth(
        rotations,
        torch.zeros(group_count, 55, 3, 3, 66, dtype=rotations.dtype, device=rotations.device),
        batch.joints, batch.parents, batch.fk_depth_indices,
    )
    transforms = torch.einsum("gnj,gjab->gnab", batch.selected_lbs_weights, relative)
    homogeneous = torch.cat((v_posed, torch.ones_like(v_posed[..., :1])), dim=-1)
    vertices = torch.einsum("gnab,gnb->gna", transforms[:, :, :3], homogeneous)
    return vertices * batch.scale.reshape(-1, 1, 1) + translations[:, None]


def batched_pose_residual_only(
    batch: PoseGroupBatch,
    orientations: torch.Tensor,
    translations: torch.Tensor,
    body_poses: torch.Tensor,
    image_width: int,
    image_height: int,
) -> torch.Tensor:
    vertices = _batched_pose_values(batch, orientations, translations, body_poses)
    group_count, nmax = vertices.shape[:2]
    uv, _ = project_cam2_erp_with_jacobian(
        vertices.reshape(-1, 3), image_width, image_height,
    )
    return (
        uv.reshape(group_count, nmax, 2) - batch.layout.targets
    ) * batch.layout.valid_mask[..., None]


def batched_pose_objective(
    batch: PoseGroupBatch,
    orientations: torch.Tensor,
    translations: torch.Tensor,
    body_poses: torch.Tensor,
    body_pose_anchors: list[torch.Tensor],
    pose_anchor_scale: float,
    image_width: int,
    image_height: int,
    *,
    objective_mode: str,
    charbonnier_eps: float,
    rep_scale: float,
    aggregation_spec: dict[str, Any] | None,
    frame_body_poses: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    if frame_body_poses is None:
        raise RuntimeError("MISSING_FRAME_BODY_POSES")
    residual = batched_pose_residual_only(
        batch, orientations, translations, body_poses, image_width, image_height,
    )
    if objective_mode == "raw_l2":
        primary = 0.5 * torch.sum((residual * rep_scale) ** 2)
    else:
        if aggregation_spec is None:
            raise ValueError("aggregation_spec is required")
        groups = []
        keys = list(aggregation_spec["terms"])
        for index, count in enumerate(batch.layout.point_counts_host):
            frame_text, camera = keys[index].split("|", 1)
            groups.append(ReprojectionGroup(
                int(frame_text), camera, residual[index, :count],
            ))
        primary = compute_primary_charbonnier_tensor(
            groups, charbonnier_eps, aggregation_spec,
        )
    pose = 0.5 * sum(torch.sum((
        (frame_body_poses[frame] - body_pose_anchors[frame].to(body_poses))
        * pose_anchor_scale
    ) ** 2) for frame in range(batch.layout.frame_count))
    return {
        "primary": primary, "pose_anchor": pose,
        "shape_anchor": primary.new_zeros(()), "total": primary + pose,
    }


def profile_pose_group_production(
    cache: Stage2PoseFastCache,
    group: Any,
    parameters: torch.Tensor,
    row_scale: torch.Tensor | None = None,
) -> dict[str, float]:
    """Section timing for the validation-only serial analytic reference."""
    sections: dict[str, float] = {}

    def timed(name, fn):
        if parameters.is_cuda:
            torch.cuda.synchronize(parameters.device)
        started = time.perf_counter()
        value = fn()
        if parameters.is_cuda:
            torch.cuda.synchronize(parameters.device)
        sections[name] = (time.perf_counter() - started) * 1000.0
        return value

    frame = timed("A_parameter_unpack", lambda: unpack_frame_parameters(
        parameters, group.frame_position, 75, True,
    ))
    orient = frame.cam2_orient.reshape(3) if group.camera == "cam2" else frame.cam4_orient.reshape(3)
    translation = frame.cam2_transl.reshape(3) if group.camera == "cam2" else frame.cam4_transl.reshape(3)
    body = frame.body_pose.reshape(-1)
    rotations, d_rotations = timed(
        "B_C_D_rotation_value_right_jacobian_scatter",
        lambda: assemble_full_pose_rotation_jacobians(
            orient, body, group.left_hand_pose, group.right_hand_pose, cache.model,
        ),
    )
    offsets, d_offsets, _, _ = timed(
        "E_F_G_H_pose_feature_offsets_jacobian",
        lambda: compute_selected_pose_offsets_with_jacobian(
            rotations, d_rotations, group.selected_posedirs,
        ),
    )
    fk = timed("I_J_FK_value_jacobian", lambda: compute_fk_with_jacobian(
        rotations, d_rotations, cache.model.joints, cache.model.parents,
    ))
    skinned, d_skinned, _, _ = timed(
        "K_L_M_N_selected_LBS_value_jacobian",
        lambda: compute_selected_lbs_with_jacobian(
            group.selected_v_shaped + offsets, d_offsets,
            group.selected_lbs_weights, fk.relative_transforms,
            fk.d_relative_transforms,
        ),
    )
    vertices = skinned * group.scale.reshape(1, 1) + translation
    timed("O_P_ERP_projection_value_jacobian", lambda: project_cam2_erp_with_jacobian(
        vertices, cache.image_width, cache.image_height,
    ))
    sections.setdefault("Q_local_reprojection_J", 0.0)
    sections.setdefault("R_Hg_GEMM", 0.0)
    sections.setdefault("S_frame_block_reduction", 0.0)
    sections.setdefault("T_anchor_contribution", 0.0)
    return sections
