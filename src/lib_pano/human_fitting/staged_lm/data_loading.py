"""Loading and pre-processing of frame observations and SMPL-X initialisations.

This module wraps the I/O helpers from ``lib_pano`` so that ``main()`` does not
need to know about bbox JSON files, rotate rules or HMR directory layouts.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from lib_pano.fitting_io.hmr_init import (
    build_hmr_init_for_frame,
    build_segment_smplx_init,
    try_load_smplx_init,
)
from lib_pano.fitting_io.sam3d_dense_io import load_sam3d_dense_result
from lib_pano.human3d.crop_upright_coord import resolve_rotate_rule
from lib_pano.human_fitting.constraints.sam3d_dense import select_valid_sam3d_dense_points
from lib_pano.pipeline.bbox_io import load_bbox_dict as load_bbox_json

from .config import MIN_VALID_POINTS
from .types import FrozenFrameParameters, FrameValidPoints, ValidPointSet


def load_frame_observations(
    frame_index: int,
    sam3d_dense_dir: Path,
    bbox_dir: Path,
    image_width: int,
    image_height: int,
    rotate_rules: list[dict],
) -> tuple[object, object]:
    """Load SAM3D dense observations for both cameras of one frame.

    Returns a ``(cam2_obs, cam4_obs)`` tuple.  Raises ``RuntimeError`` if
    either camera's observation is missing.
    """
    cam2_bboxes = load_bbox_json(bbox_dir / "cam2_bboxes.json")
    cam4_bboxes = load_bbox_json(bbox_dir / "cam4_bboxes.json")
    rotate_rule = resolve_rotate_rule(frame_index, rotate_rules)
    cam2_obs = load_sam3d_dense_result(
        frame_index, sam3d_dense_dir, "cam2", image_width, image_height,
        crop_bbox=cam2_bboxes.get(str(frame_index)), rotate_rule=rotate_rule,
    )
    cam4_obs = load_sam3d_dense_result(
        frame_index, sam3d_dense_dir, "cam4", image_width, image_height,
        crop_bbox=cam4_bboxes.get(str(frame_index)), rotate_rule=rotate_rule,
    )
    if cam2_obs is None or cam4_obs is None:
        raise RuntimeError(f"Missing SAM3D dense for frame {frame_index}")
    return cam2_obs, cam4_obs


def build_frozen_frame_parameters(cam2_smplx_parameters, cam4_smplx_parameters, frame_index: int) -> FrozenFrameParameters:
    """Extract frozen (non-optimised) parameters for one frame.

    Hand poses and per-camera scale are detached from the computation graph so
    they act as constants during optimisation.
    """
    cam2_scale_flat = cam2_smplx_parameters.scale.detach().clone().reshape(-1)
    cam4_scale_flat = cam4_smplx_parameters.scale.detach().clone().reshape(-1)

    cam2_scale = (
        cam2_scale_flat[frame_index].reshape(1)
        if frame_index < cam2_scale_flat.shape[0]
        else cam2_scale_flat[0].reshape(1)
    )
    cam4_scale = (
        cam4_scale_flat[frame_index].reshape(1)
        if frame_index < cam4_scale_flat.shape[0]
        else cam4_scale_flat[0].reshape(1)
    )

    return FrozenFrameParameters(
        cam2_left_hand_pose=cam2_smplx_parameters.left_hand_pose.detach()[frame_index:frame_index + 1].clone(),
        cam2_right_hand_pose=cam2_smplx_parameters.right_hand_pose.detach()[frame_index:frame_index + 1].clone(),
        cam4_left_hand_pose=cam4_smplx_parameters.left_hand_pose.detach()[frame_index:frame_index + 1].clone(),
        cam4_right_hand_pose=cam4_smplx_parameters.right_hand_pose.detach()[frame_index:frame_index + 1].clone(),
        cam2_scale=cam2_scale,
        cam4_scale=cam4_scale,
    )


def precompute_valid_points(cam2_obs, cam4_obs, sample_ids) -> FrameValidPoints:
    """Select and cache valid SAM3D dense correspondences for one frame.

    Raises ``RuntimeError`` if either camera has fewer than
    :data:`MIN_VALID_POINTS` valid points.
    """

    def _select(obs, camera: str) -> ValidPointSet:
        vertex_ids, target_2d = select_valid_sam3d_dense_points(obs, sample_ids)
        if int(vertex_ids.shape[0]) < MIN_VALID_POINTS:
            raise RuntimeError(f"{camera}: {vertex_ids.shape[0]} < {MIN_VALID_POINTS}")
        return ValidPointSet(vertex_ids=vertex_ids, target_2d=target_2d)

    return FrameValidPoints(cam2=_select(cam2_obs, "cam2"), cam4=_select(cam4_obs, "cam4"))


def to_tensor_vertex_ids_and_targets(
    valid_points_by_frame: list[FrameValidPoints],
    device: torch.device = torch.device("cpu"),
    target_dtype: torch.dtype = torch.float32,
) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
    """Convert cached valid-point arrays into typed tensors.

    Returns ``(cam2_vertex_ids, cam2_targets, cam4_vertex_ids, cam4_targets)``
    where vertex ids are int64 and targets are float32, matching the dtypes
    expected by ``index_select`` and the reprojection arithmetic.
    """
    cam2_vertex_ids: list[torch.Tensor] = []
    cam2_targets: list[torch.Tensor] = []
    cam4_vertex_ids: list[torch.Tensor] = []
    cam4_targets: list[torch.Tensor] = []
    for frame_valid in valid_points_by_frame:
        cam2_vertex_ids.append(torch.as_tensor(
            frame_valid.cam2.vertex_ids, dtype=torch.long, device=device,
        ))
        cam2_targets.append(torch.as_tensor(
            frame_valid.cam2.target_2d, dtype=target_dtype, device=device,
        ))
        cam4_vertex_ids.append(torch.as_tensor(
            frame_valid.cam4.vertex_ids, dtype=torch.long, device=device,
        ))
        cam4_targets.append(torch.as_tensor(
            frame_valid.cam4.target_2d, dtype=target_dtype, device=device,
        ))
    return cam2_vertex_ids, cam2_targets, cam4_vertex_ids, cam4_targets


def load_hmr_initializations(
    frame_indices: list[int],
    hmr_dir: Path,
) -> tuple[set[int], dict[int, object], dict[int, object]]:
    """Load per-frame SMPL-X HMR initialisations for both cameras.

    Returns ``(invalid_frames, cam2_inits, cam4_inits)`` where entries may be
    ``None`` if HMR failed for that frame (see ``try_load_smplx_init``).
    """
    invalid_frames: set[int] = set()
    cam2_inits: dict[int, object] = {}
    cam4_inits: dict[int, object] = {}
    for frame_index in frame_indices:
        cam2_inits[frame_index] = try_load_smplx_init(frame_index, hmr_dir, invalid_frames, "cam2")
        cam4_inits[frame_index] = try_load_smplx_init(frame_index, hmr_dir, invalid_frames, "cam4")
    return invalid_frames, cam2_inits, cam4_inits


def build_segment_initializations(
    frame_indices: list[int],
    cam2_hmr_inits: dict[int, object],
    cam4_hmr_inits: dict[int, object],
) -> tuple[object, object]:
    """Build gap-filled SMPL-X segment initialisations for both cameras.

    Each frame that lacks a direct HMR result is filled by interpolation or a
    neighbour fallback (handled inside ``build_hmr_init_for_frame``).  The
    per-frame lists are then packed into ``SMPLX_Param_Seq`` objects by
    ``build_segment_smplx_init``.
    """
    cam2_frame_params = [
        build_hmr_init_for_frame(frame_index, frame_indices, cam2_hmr_inits)[0]
        for frame_index in frame_indices
    ]
    cam4_frame_params = [
        build_hmr_init_for_frame(frame_index, frame_indices, cam4_hmr_inits)[0]
        for frame_index in frame_indices
    ]
    return build_segment_smplx_init(cam2_frame_params, cam4_frame_params)
