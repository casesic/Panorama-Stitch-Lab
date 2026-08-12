"""Parameter layout helpers that replace magic index slicing.

The optimisation vector ``x`` packs per-frame parameters (and, in Stage 2 pose/shape refinement, a
shared beta block) into a single flat tensor.  The original script accessed
fields through raw slices such as ``x[s:s+3]``, ``x[s+3:s+6]`` ... which made
the layout hard to follow and easy to break.

This module provides :class:`FrameParameterSlices` and
:func:`unpack_frame_parameters` so call-sites read::

    frame_params = unpack_frame_parameters(x, frame_index, per_frame_dim,
                                           include_body_pose=stage_uses_pose)
    forward_smplx_tensor_batch(model, frame_params.cam2_orient, ...)

The numerical layout is **identical** to the original; only the access pattern
changes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class FrameParameterSlices:
    """Slice descriptors for the four (or five) parameter blocks of one frame.

    Each attribute is a ``slice`` that can be applied directly to the flat
    optimisation vector.  ``body_pose`` is ``None`` for Stage 1 where the pose
    is frozen and not part of the vector.
    """

    cam2_orient: slice
    cam2_transl: slice
    cam4_orient: slice
    cam4_transl: slice
    body_pose: slice | None


@dataclass
class UnpackedFrameParameters:
    """Concrete parameter tensors for a single frame.

    All tensors keep a leading batch dimension of 1 to match the
    ``forward_smplx_tensor_batch`` signature.
    """

    cam2_orient: torch.Tensor
    cam2_transl: torch.Tensor
    cam4_orient: torch.Tensor
    cam4_transl: torch.Tensor
    body_pose: torch.Tensor


def frame_slices(frame_index: int, per_frame_dim: int, include_body_pose: bool) -> FrameParameterSlices:
    """Return the slice descriptor for ``frame_index`` within ``x``.

    Parameters
    ----------
    frame_index:
        Zero-based frame position within the multi-frame batch.
    per_frame_dim:
        Number of parameters consumed by one frame (12 for Stage 1, 75 for
        Stage 2 pose warmup or pose/shape refinement).
    include_body_pose:
        Whether the per-frame block contains a body-pose segment.
    """
    start = frame_index * per_frame_dim
    return FrameParameterSlices(
        cam2_orient=slice(start, start + 3),
        cam2_transl=slice(start + 3, start + 6),
        cam4_orient=slice(start + 6, start + 9),
        cam4_transl=slice(start + 9, start + 12),
        body_pose=slice(start + 12, start + per_frame_dim) if include_body_pose else None,
    )


def unpack_frame_parameters(
    parameters: torch.Tensor,
    frame_index: int,
    per_frame_dim: int,
    include_body_pose: bool,
    body_pose_override: torch.Tensor | None = None,
) -> UnpackedFrameParameters:
    """Extract per-frame parameter tensors from the flat optimisation vector.

    Parameters
    ----------
    parameters:
        Flat optimisation vector ``x``.
    frame_index, per_frame_dim, include_body_pose:
        See :func:`frame_slices`.
    body_pose_override:
        When provided (e.g. the frozen pose anchor in Stage 1), this tensor is
        used instead of slicing the body-pose block from ``parameters``.  The
        caller is responsible for ensuring shape compatibility.

    Returns
    -------
    UnpackedFrameParameters
        Each orient / transl tensor has shape ``[1, 3]`` and the body-pose
        tensor has shape ``[1, body_pose_dim]``.
    """
    slices = frame_slices(frame_index, per_frame_dim, include_body_pose or body_pose_override is not None)
    cam2_orient = parameters[slices.cam2_orient].view(1, 3)
    cam2_transl = parameters[slices.cam2_transl].view(1, 3)
    cam4_orient = parameters[slices.cam4_orient].view(1, 3)
    cam4_transl = parameters[slices.cam4_transl].view(1, 3)

    if body_pose_override is not None:
        body_pose = body_pose_override
    elif slices.body_pose is not None:
        body_pose = parameters[slices.body_pose].view(1, -1)
    else:
        raise ValueError("body_pose_override is required when include_body_pose is False")

    return UnpackedFrameParameters(
        cam2_orient=cam2_orient,
        cam2_transl=cam2_transl,
        cam4_orient=cam4_orient,
        cam4_transl=cam4_transl,
        body_pose=body_pose,
    )


def stage1_slices(frame_index: int, stage1_per_frame_dim: int = 12) -> FrameParameterSlices:
    """Convenience wrapper for Stage 1 (no body-pose block)."""
    return frame_slices(frame_index, stage1_per_frame_dim, include_body_pose=False)


def build_stage1_initial_parameters(
    cam2_smplx_parameters: Any,
    cam4_smplx_parameters: Any,
    frame_count: int,
) -> torch.Tensor:
    """Concatenate per-frame orient/transl into the Stage 1 vector.

    The layout is ``[cam2_orient | cam2_transl | cam4_orient | cam4_transl]``
    repeated for each frame, matching the original ``x0_s1`` construction.
    """
    parts: list[torch.Tensor] = []
    for t in range(frame_count):
        parts.append(cam2_smplx_parameters.orient.detach()[t].reshape(-1))
        parts.append(cam2_smplx_parameters.transl.detach()[t].reshape(-1))
        parts.append(cam4_smplx_parameters.orient.detach()[t].reshape(-1))
        parts.append(cam4_smplx_parameters.transl.detach()[t].reshape(-1))
    return torch.cat(parts).clone()
