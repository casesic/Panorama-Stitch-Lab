"""Stage 2 Pose value path from parameters through ERP residual groups.

Purpose
-------
Expose the serial/reference value chain in its mathematical order: local
rotations, pose blend shapes, forward kinematics, selected-vertex LBS, frame
rigid transform, ERP projection, and residual groups.

Mathematical scope
------------------
This module reproduces the installed SMPL-X Layer/LBS contract while
materializing only selected pose offsets, vertex transforms, and final
vertices. It does not construct analytic derivatives, normal equations,
objectives, metrics, or LM objects.

Read :mod:`.derivatives` next to follow the analytic chain rule. The
FULL_BATCH production equivalent is in :mod:`.batched`.
"""

from __future__ import annotations

import time

import torch
from smplx.lbs import batch_rigid_transform

from lib_pano.human_fitting import smplx_sequence as _smplx_sequence  # noqa: F401
from lib_dynamic_hvip.smpl.smpl_lib import _aa_to_rotmat

from ..config import STAGE1_PER_FRAME_DIM
from ..geometry.erp import project_unified_erp
from ..parameter_layout import unpack_frame_parameters
from ..residuals import flatten_reprojection_groups
from ..types import ReprojectionGroup
from .cache import (
    Stage2PoseFastCache,
    Stage2PoseFrameCameraCache,
    Stage2PoseModelCache,
)


def _unsupported(message: str) -> RuntimeError:
    return RuntimeError(f"UNSUPPORTED_SMPLX_FORWARD_CONTRACT: {message}")


def _sync(cache: Stage2PoseFastCache, parameters: torch.Tensor) -> None:
    if cache.profile_timing and parameters.is_cuda:
        torch.cuda.synchronize(parameters.device)


def compute_selected_pose_offsets(
    rotation_matrices: torch.Tensor,
    selected_posedirs: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute selected pose offsets from SMPL-X's non-root pose feature.

    Args:
        rotation_matrices: ``[J, 3, 3]`` local joint rotations.
        selected_posedirs: ``[(J - 1) * 9, N, 3]`` selected pose directions.

    Returns:
        offsets: ``[N, 3]`` selected-vertex pose offsets.
        pose_feature: ``[(J - 1) * 9]`` flattened non-root rotation deltas.
    """
    if rotation_matrices.ndim != 3 or rotation_matrices.shape[-2:] != (3, 3):
        raise _unsupported(f"rotation matrices have shape {tuple(rotation_matrices.shape)}")
    identity = torch.eye(3, dtype=rotation_matrices.dtype, device=rotation_matrices.device)
    pose_feature = (rotation_matrices[1:] - identity).reshape(-1)
    if selected_posedirs.shape[0] != pose_feature.numel() or selected_posedirs.shape[-1] != 3:
        raise _unsupported(
            f"selected posedirs {tuple(selected_posedirs.shape)} vs pose feature {pose_feature.numel()}"
        )
    offsets = torch.einsum("p,pnc->nc", pose_feature, selected_posedirs)
    return offsets, pose_feature


def compute_selected_lbs(
    selected_v_posed: torch.Tensor,
    selected_lbs_weights: torch.Tensor,
    relative_transforms: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Skin selected vertices without constructing all vertex transforms.

    Args:
        selected_v_posed: ``[N, 3]`` posed selected vertices.
        selected_lbs_weights: ``[N, J]`` skinning weights.
        relative_transforms: ``[J, 4, 4]`` rest-corrected FK transforms.

    Returns:
        vertices: ``[N, 3]`` skinned selected vertices.
        selected_transforms: ``[N, 4, 4]`` blended LBS transforms.
    """
    joint_count = relative_transforms.shape[0]
    if selected_lbs_weights.shape != (selected_v_posed.shape[0], joint_count):
        raise _unsupported("selected LBS weight shape mismatch")
    selected_transforms = (
        selected_lbs_weights @ relative_transforms.reshape(joint_count, 16)
    ).reshape(-1, 4, 4)
    homogeneous = torch.cat((
        selected_v_posed,
        torch.ones(
            selected_v_posed.shape[0], 1,
            dtype=selected_v_posed.dtype, device=selected_v_posed.device,
        ),
    ), dim=1)
    vertices = (selected_transforms @ homogeneous.unsqueeze(-1))[:, :3, 0]
    return vertices, selected_transforms


def assemble_full_pose_rotations(
    global_orient: torch.Tensor,
    body_pose: torch.Tensor,
    left_hand_pose: torch.Tensor,
    right_hand_pose: torch.Tensor,
    model_cache: Stage2PoseModelCache,
) -> torch.Tensor:
    """Assemble root/body/jaw/eyes/hands in installed SMPL-X order.

    Inputs are axis-angle tensors with shapes ``[3]``, ``[63]``, ``[45]``,
    and ``[45]``. The result is ``[J, 3, 3]`` where the current SMPL-X model
    uses ``J = 55``.
    """
    if global_orient.numel() != 3 or body_pose.numel() != 63:
        raise _unsupported("global/body pose dimensions")
    if left_hand_pose.numel() != 45 or right_hand_pose.numel() != 45:
        raise _unsupported("hand pose dimensions")
    zeros = torch.zeros(3, dtype=global_orient.dtype, device=global_orient.device)
    axis_angles = torch.cat((
        global_orient.reshape(1, 3), body_pose.reshape(21, 3),
        zeros.reshape(1, 3), zeros.reshape(1, 3), zeros.reshape(1, 3),
        left_hand_pose.reshape(15, 3), right_hand_pose.reshape(15, 3),
    ), dim=0)
    if axis_angles.shape[0] != model_cache.joint_count:
        raise _unsupported(f"assembled joint count {axis_angles.shape[0]}")
    return _aa_to_rotmat(axis_angles)


def fast_selected_vertices(
    cache: Stage2PoseFastCache,
    camera_cache: Stage2PoseFrameCameraCache,
    global_orient: torch.Tensor,
    translation: torch.Tensor,
    body_pose: torch.Tensor,
    *,
    return_intermediates: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute selected final vertices for one frame/camera.

    ``global_orient`` and ``translation`` are ``[3]``; ``body_pose`` is
    ``[63]``; the returned vertices are ``[N, 3]``. When requested, the
    intermediate mapping exposes the value path without changing its
    computation order.
    """
    # 1. Parameter layout and local rotations.
    reference = global_orient
    rotations = assemble_full_pose_rotations(
        global_orient.reshape(3), body_pose.reshape(-1),
        camera_cache.left_hand_pose.to(reference), camera_cache.right_hand_pose.to(reference),
        cache.model,
    )
    # 2. Pose blend shapes.
    selected_v_shaped = camera_cache.selected_v_shaped.to(reference)
    selected_posedirs = camera_cache.selected_posedirs.to(reference)
    pose_offsets, pose_feature = compute_selected_pose_offsets(rotations, selected_posedirs)
    selected_v_posed = selected_v_shaped + pose_offsets
    # 3. Forward kinematics.
    posed_joints, relative_transforms = batch_rigid_transform(
        rotations.unsqueeze(0), cache.model.joints.to(reference).unsqueeze(0),
        cache.model.parents.to(device=reference.device), dtype=reference.dtype,
    )
    # 4. Linear blend skinning.
    skinned, selected_transforms = compute_selected_lbs(
        selected_v_posed, camera_cache.selected_lbs_weights.to(reference),
        relative_transforms.squeeze(0),
    )
    # 5. Frame rigid transform (the camera-specific orient is already in FK).
    # forward_smplx_tensor_batch applies scale after model LBS, then translation.
    final_vertices = skinned * camera_cache.scale.to(reference).reshape(1, 1) + translation.reshape(1, 3)
    if final_vertices.shape[0] != camera_cache.vertex_ids.numel():
        raise _unsupported("fast path produced an unexpected vertex count")
    if not return_intermediates:
        return final_vertices
    return final_vertices, {
        "v_shaped": cache.model.v_shaped.to(reference),
        "joints": cache.model.joints.to(reference),
        "rotation_matrices": rotations,
        "pose_feature": pose_feature,
        "selected_pose_offsets": pose_offsets,
        "fk_transforms": relative_transforms.squeeze(0),
        "selected_v_posed": selected_v_posed,
        "selected_transforms": selected_transforms,
        "selected_skinned_vertices": skinned,
        "root_center": cache.model.joints[0].to(reference) * camera_cache.scale.to(reference),
        "final_selected_vertices": final_vertices,
        "posed_joints": posed_joints.squeeze(0),
    }


def fast_raw_groups(cache: Stage2PoseFastCache, parameters: torch.Tensor) -> list[ReprojectionGroup]:
    """Build raw ERP residual groups in production frame/camera order.

    Args:
        parameters: ``[T * Q_frame]`` with ``Q_frame = 12 + 63 = 75``.

    Returns:
        One :class:`ReprojectionGroup` per non-empty frame/camera group; each
        residual tensor has shape ``[N, 2]``.
    """
    per_frame_dim = STAGE1_PER_FRAME_DIM + cache.body_pose_dim
    expected = cache.frame_count * per_frame_dim
    if parameters.numel() != expected:
        raise ValueError(f"Stage 2 pose parameter dimension: expected {expected}, got {parameters.numel()}")
    groups: list[ReprojectionGroup] = []
    for camera_cache in cache.groups:
        frame = unpack_frame_parameters(
            parameters, camera_cache.frame_position, per_frame_dim, include_body_pose=True,
        )
        orient = frame.cam2_orient if camera_cache.camera == "cam2" else frame.cam4_orient
        translation = frame.cam2_transl if camera_cache.camera == "cam2" else frame.cam4_transl
        vertices = fast_selected_vertices(
            cache, camera_cache, orient.reshape(3), translation.reshape(3), frame.body_pose.reshape(-1),
        )
        # 6. ERP projection.
        projected = project_unified_erp(vertices, cache.image_width, cache.image_height)
        # 7. Residual groups.
        groups.append(ReprojectionGroup(
            frame_index=camera_cache.frame_index, camera=camera_cache.camera,
            residual_2d=projected - camera_cache.targets.to(parameters),
        ))
    return groups


def fast_raw_residual(cache: Stage2PoseFastCache, parameters: torch.Tensor) -> torch.Tensor:
    _sync(cache, parameters)
    start = time.perf_counter()
    result = flatten_reprojection_groups(fast_raw_groups(cache, parameters))
    _sync(cache, parameters)
    cache.residual_ms += (time.perf_counter() - start) * 1000.0
    cache.residual_calls += 1
    return result
