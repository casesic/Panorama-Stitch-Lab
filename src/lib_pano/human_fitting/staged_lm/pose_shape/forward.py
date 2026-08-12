"""Dynamic Pose/Shape value path from shared shape through ERP residual groups.

Purpose: expose parameter unpack, dynamic shape consumption, local rotations,
pose offsets, dynamic FK, selected LBS, rigid transform, ERP projection, and
raw residual groups in mathematical order.
This value path constructs no Jacobian, objective, H/g, or Linearization.
Read :mod:`.beta_derivatives` and :mod:`.derivatives` next.
"""

from __future__ import annotations

import time

import torch
from smplx.lbs import batch_rigid_transform

from ..geometry.erp import project_unified_erp
from ..parameter_layout import unpack_frame_parameters
from ..pose.forward import (
    assemble_full_pose_rotations,
    compute_selected_lbs,
    compute_selected_pose_offsets,
)
from ..types import ReprojectionGroup
from .cache import (
    Stage2PoseShapeFastCache,
    Stage2PoseShapeFrameCameraCache,
    _sync,
    validate_stage2_pose_shape_layout,
)
from .shape import Stage2PoseShapeState, evaluate_pose_shape_state


def evaluate_pose_shape_group_state(
    cache: Stage2PoseShapeFastCache,
    camera_cache: Stage2PoseShapeFrameCameraCache,
    shape_state: Stage2PoseShapeState,
    global_orient: torch.Tensor,
    translation: torch.Tensor,
    body_pose: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Evaluate one group while preserving the historical intermediate keys."""
    rotations = assemble_full_pose_rotations(
        global_orient, body_pose, camera_cache.left_hand_pose.to(global_orient),
        camera_cache.right_hand_pose.to(global_orient), cache.model,
    )
    pose_offsets, pose_feature = compute_selected_pose_offsets(
        rotations, camera_cache.selected_posedirs.to(global_orient),
    )
    selected_v_shaped = camera_cache.selected_base_v_shaped.to(global_orient) + torch.einsum(
        "ncb,b->nc", camera_cache.selected_beta_shapedirs.to(global_orient), shape_state.beta,
    )
    selected_v_posed = selected_v_shaped + pose_offsets
    posed_joints, relative_transforms = batch_rigid_transform(
        rotations.unsqueeze(0), shape_state.joints.unsqueeze(0),
        cache.model.parents.to(global_orient.device), dtype=global_orient.dtype,
    )
    skinned, selected_transforms = compute_selected_lbs(
        selected_v_posed, camera_cache.selected_lbs_weights.to(global_orient),
        relative_transforms[0],
    )
    final = skinned * camera_cache.scale.to(global_orient).reshape(1, 1) + translation.reshape(1, 3)
    return {
        "physical_beta": shape_state.beta,
        "selected_v_shaped": selected_v_shaped, "joints": shape_state.joints,
        "rotations": rotations, "pose_feature": pose_feature,
        "selected_pose_offsets": pose_offsets, "selected_v_posed": selected_v_posed,
        "relative_transforms": relative_transforms[0], "posed_joints": posed_joints[0],
        "selected_transforms": selected_transforms,
        "selected_skinned_vertices": skinned, "final_selected_vertices": final,
    }


def pose_shape_fast_selected_vertices(
    cache: Stage2PoseShapeFastCache, camera_cache: Stage2PoseShapeFrameCameraCache,
    shape_state: Stage2PoseShapeState, global_orient: torch.Tensor,
    translation: torch.Tensor, body_pose: torch.Tensor, *, return_intermediates: bool = False,
):
    """Return selected vertices ``[N,3]`` and optionally unchanged intermediates."""
    state = evaluate_pose_shape_group_state(
        cache, camera_cache, shape_state, global_orient, translation, body_pose,
    )
    return (state["final_selected_vertices"], state) if return_intermediates else state["final_selected_vertices"]


def pose_shape_fast_raw_groups(
    cache: Stage2PoseShapeFastCache, parameters: torch.Tensor, beta_abs_limit: float,
) -> list[ReprojectionGroup]:
    """Build raw ``[N,2]`` ERP residual groups in frame/camera order."""
    per_frame_dim, _ = validate_stage2_pose_shape_layout(cache, parameters)
    shape_state = evaluate_pose_shape_state(cache, parameters[-cache.model.beta_dim:], beta_abs_limit)
    groups = []
    for camera_cache in cache.groups:
        frame = unpack_frame_parameters(parameters, camera_cache.frame_position, per_frame_dim, True)
        orient = frame.cam2_orient if camera_cache.camera == "cam2" else frame.cam4_orient
        translation = frame.cam2_transl if camera_cache.camera == "cam2" else frame.cam4_transl
        vertices = pose_shape_fast_selected_vertices(
            cache, camera_cache, shape_state, orient.reshape(3), translation.reshape(3),
            frame.body_pose.reshape(-1),
        )
        uv = project_unified_erp(vertices, cache.image_width, cache.image_height)
        groups.append(ReprojectionGroup(
            camera_cache.frame_index, camera_cache.camera,
            uv - camera_cache.targets.to(parameters),
        ))
    return groups


def pose_shape_fast_raw_residual(
    cache: Stage2PoseShapeFastCache, parameters: torch.Tensor, beta_abs_limit: float,
) -> torch.Tensor:
    """Flatten Pose/Shape reprojection groups without changing row order."""
    _sync(cache, parameters); start = time.perf_counter()
    result = torch.cat([
        group.residual_2d.reshape(-1)
        for group in pose_shape_fast_raw_groups(cache, parameters, beta_abs_limit)
    ])
    _sync(cache, parameters)
    cache.residual_ms += (time.perf_counter() - start) * 1000.0
    cache.residual_calls += 1
    return result
