"""Pose true objective, anchor residual, and raw reprojection metrics.

Purpose
-------
Reduce raw groups from :mod:`pose.forward` using the unchanged Charbonnier
IRLS/raw-L2 semantics and preserve objective/metrics timing counters.

Mathematical scope
------------------
Raw residual flattening, pose-anchor residuals, true objective, and metrics.
FK, LBS, SO(3), ERP derivatives, and FULL_BATCH H/g do not live here.
Read :mod:`pose.batched` next for the production normal equations.
"""

from __future__ import annotations

import time
from typing import Any

import torch

from ..config import STAGE1_PER_FRAME_DIM
from ..irls import compute_primary_charbonnier_tensor
from ..residuals import flatten_reprojection_groups, raw_reproj_metrics
from .cache import Stage2PoseFastCache
from .forward import _sync, fast_raw_groups, fast_raw_residual


def _pose_anchor_residual(
    cache: Stage2PoseFastCache,
    parameters: torch.Tensor,
    body_pose_anchors: list[torch.Tensor],
    pose_anchor_scale: float,
) -> torch.Tensor:
    per_frame_dim = STAGE1_PER_FRAME_DIM + cache.body_pose_dim
    return torch.cat([
        (parameters[frame * per_frame_dim + STAGE1_PER_FRAME_DIM:(frame + 1) * per_frame_dim]
         - body_pose_anchors[frame].to(parameters)) * pose_anchor_scale
        for frame in range(cache.frame_count)
    ])


def stage2_pose_fast_true_objective(
    cache: Stage2PoseFastCache,
    parameters: torch.Tensor,
    body_pose_anchors: list[torch.Tensor],
    pose_anchor_scale: float,
    objective_mode: str,
    charbonnier_eps: float,
    rep_scale: float,
    aggregation_spec: dict[str, Any] | None,
) -> torch.Tensor:
    """Evaluate the unchanged true Pose objective at ``parameters [T*Q]``."""
    _sync(cache, parameters)
    start = time.perf_counter()
    groups = fast_raw_groups(cache, parameters)
    if objective_mode == "charbonnier_irls":
        if aggregation_spec is None:
            raise ValueError("aggregation_spec is required for Charbonnier objective")
        primary = compute_primary_charbonnier_tensor(groups, charbonnier_eps, aggregation_spec)
    elif objective_mode == "raw_l2":
        residual = flatten_reprojection_groups(groups) * rep_scale
        primary = 0.5 * torch.sum(residual ** 2)
    else:
        raise ValueError(objective_mode)
    anchor = _pose_anchor_residual(cache, parameters, body_pose_anchors, pose_anchor_scale)
    result = primary + 0.5 * torch.sum(anchor ** 2)
    _sync(cache, parameters)
    cache.objective_ms += (time.perf_counter() - start) * 1000.0
    cache.objective_calls += 1
    return result


def stage2_pose_fast_raw_metrics(
    cache: Stage2PoseFastCache, parameters: torch.Tensor,
) -> dict[str, float]:
    """Compute raw UV metrics without changing residual/group ordering."""
    _sync(cache, parameters)
    start = time.perf_counter()
    result = raw_reproj_metrics(
        flatten_reprojection_groups(fast_raw_groups(cache, parameters)).detach(),
    )
    _sync(cache, parameters)
    cache.metrics_ms += (time.perf_counter() - start) * 1000.0
    cache.metrics_calls += 1
    return result


__all__ = [
    "fast_raw_residual",
    "stage2_pose_fast_true_objective",
    "stage2_pose_fast_raw_metrics",
]
