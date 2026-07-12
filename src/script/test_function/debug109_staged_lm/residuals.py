"""Reprojection residual computation and summary metrics.

The residual functions map a flat optimisation vector to a list of
:class:`ReprojectionGroup` (one per frame/camera) by running the SMPL-X
forward model, projecting vertices onto the ERP image and differencing against
the observed targets.

The numerical behaviour is identical to the original ``make_raw_reproj_groups``
and ``raw_reproj_metrics``.
"""

from __future__ import annotations

from typing import Any, Callable

import torch

from lib_pano.human_fitting.smplx_sequence import forward_smplx_tensor_batch

from .parameter_layout import unpack_frame_parameters
from .types import FrozenFrameParameters, ReprojectionGroup


def raw_reproj_metrics(residual_flat: torch.Tensor) -> dict[str, float]:
    """Compute RMS / median / p95 of 2D reprojection residuals.

    Parameters
    ----------
    residual_flat:
        Flat residual vector ``[N*2]`` (x, y interleaved).
    """
    residual_2d = residual_flat.reshape(-1, 2)
    norms = torch.norm(residual_2d, dim=1).cpu()
    squared = norms ** 2
    return {
        "rms": float(torch.sqrt(squared.mean())),
        "median": float(norms.median()),
        "p95": float(norms.kthvalue(max(1, int(norms.numel() * 0.95))).values),
    }


def flatten_reprojection_groups(groups: list[ReprojectionGroup]) -> torch.Tensor:
    """Concatenate all per-group residuals into one flat vector."""
    return torch.cat([group.residual_2d.reshape(-1) for group in groups])


def make_raw_reproj_groups(
    parameters: torch.Tensor,
    frame_indices: list[int],
    cam2_vertex_ids_by_frame: list[torch.Tensor],
    cam2_targets_by_frame: list[torch.Tensor],
    cam4_vertex_ids_by_frame: list[torch.Tensor],
    cam4_targets_by_frame: list[torch.Tensor],
    frozen_params_by_frame: list[FrozenFrameParameters],
    model: Any,
    fitter: Any,
    image_width: int,
    image_height: int,
    frame_count: int,
    per_frame_dim: int,
    beta_fn: Callable[[torch.Tensor], torch.Tensor],
    body_pose_anchors: list[torch.Tensor] | None = None,
) -> list[ReprojectionGroup]:
    """Build reprojection residual groups for every (frame, camera) pair.

    Parameters
    ----------
    parameters:
        Flat optimisation vector.
    per_frame_dim:
        Stage 1 layout size when pose is frozen, or the Stage 2 per-frame
        layout size when pose is optimised.
    beta_fn:
        Maps ``parameters`` to the physical beta tensor.  Stage 1/2A pass a
        constant-frozen beta; Stage 2 joint refinement passes the active
        ``tanh`` mapping.
    body_pose_anchors:
        When provided (Stage 1), the frozen anchor is used as the body pose
        instead of reading it from ``parameters``.
    """
    device = parameters.device
    dtype = parameters.dtype
    beta = beta_fn(parameters)
    include_body_pose = body_pose_anchors is None
    groups: list[ReprojectionGroup] = []

    for frame_position in range(frame_count):
        frozen = frozen_params_by_frame[frame_position]
        pose_override = (
            body_pose_anchors[frame_position].to(device, dtype).view(1, -1)
            if body_pose_anchors is not None
            else None
        )
        frame_params = unpack_frame_parameters(
            parameters, frame_position, per_frame_dim,
            include_body_pose=include_body_pose,
            body_pose_override=pose_override,
        )

        cam2_vertices, _, _ = forward_smplx_tensor_batch(
            model,
            frame_params.cam2_orient,
            frame_params.body_pose,
            frozen.cam2_left_hand_pose.to(device, dtype),
            frozen.cam2_right_hand_pose.to(device, dtype),
            beta.unsqueeze(0).to(device, dtype),
            frame_params.cam2_transl,
            frozen.cam2_scale.expand(1).to(device, dtype),
        )
        cam4_vertices, _, _ = forward_smplx_tensor_batch(
            model,
            frame_params.cam4_orient,
            frame_params.body_pose,
            frozen.cam4_left_hand_pose.to(device, dtype),
            frozen.cam4_right_hand_pose.to(device, dtype),
            beta.unsqueeze(0).to(device, dtype),
            frame_params.cam4_transl,
            frozen.cam4_scale.expand(1).to(device, dtype),
        )

        projected_cam2 = fitter._project_cam2(cam2_vertices, image_width, image_height).squeeze(0)
        projected_cam4 = fitter._project_cam2(cam4_vertices, image_width, image_height).squeeze(0)

        cam2_vid = cam2_vertex_ids_by_frame[frame_position].to(device)
        cam4_vid = cam4_vertex_ids_by_frame[frame_position].to(device)
        cam2_tgt = cam2_targets_by_frame[frame_position].to(device, dtype)
        cam4_tgt = cam4_targets_by_frame[frame_position].to(device, dtype)

        residual_cam2 = projected_cam2.index_select(0, cam2_vid) - cam2_tgt
        residual_cam4 = projected_cam4.index_select(0, cam4_vid) - cam4_tgt

        groups.append(ReprojectionGroup(
            frame_index=int(frame_indices[frame_position]),
            camera="cam2",
            residual_2d=residual_cam2,
        ))
        groups.append(ReprojectionGroup(
            frame_index=int(frame_indices[frame_position]),
            camera="cam4",
            residual_2d=residual_cam4,
        ))

    return groups
