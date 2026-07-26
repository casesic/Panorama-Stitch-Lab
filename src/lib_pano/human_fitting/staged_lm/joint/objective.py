"""Joint objective, anchors, raw metrics, and their timing lifecycle.

Purpose: define true reprojection/anchor objectives and reporting metrics.
This module consumes :mod:`.forward` values and :mod:`.shape` beta mapping;
it does not implement shape state, FK, LBS, Jacobians, Arrowhead assembly, or
Schur solving. Read :mod:`.arrowhead` for structured H/g construction.
"""

from __future__ import annotations

import time
from typing import Any

import torch

from ..config import STAGE1_PER_FRAME_DIM
from ..irls import compute_primary_charbonnier_tensor
from ..residuals import flatten_reprojection_groups, raw_reproj_metrics
from .cache import Stage2JointFastCache, _sync, validate_stage2_joint_layout
from .forward import joint_fast_raw_groups
from .shape import physical_beta_from_latent


def _joint_anchor_residuals(
    cache: Stage2JointFastCache, parameters: torch.Tensor, beta_abs_limit: float,
    beta_anchor: torch.Tensor, body_pose_anchors: list[torch.Tensor],
    pose_anchor_scale: float, shape_anchor_scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    per_frame_dim, _ = validate_stage2_joint_layout(cache, parameters)
    pose = torch.cat([
        (parameters[f * per_frame_dim + STAGE1_PER_FRAME_DIM:(f + 1) * per_frame_dim]
         - body_pose_anchors[f].to(parameters)) * pose_anchor_scale
        for f in range(cache.frame_count)
    ])
    beta, _ = physical_beta_from_latent(parameters[-cache.model.beta_dim:], beta_abs_limit)
    shape = (beta - beta_anchor.to(parameters)) * shape_anchor_scale
    return pose, shape


def stage2_joint_fast_true_objective(
    cache: Stage2JointFastCache, parameters: torch.Tensor, beta_abs_limit: float,
    beta_anchor: torch.Tensor, body_pose_anchors: list[torch.Tensor],
    pose_anchor_scale: float, shape_anchor_scale: float,
    objective_mode: str, charbonnier_eps: float, rep_scale: float,
    aggregation_spec: dict[str, Any] | None,
) -> torch.Tensor:
    return _stage2_joint_fast_objective_terms(
        cache, parameters, beta_abs_limit, beta_anchor, body_pose_anchors,
        pose_anchor_scale, shape_anchor_scale, objective_mode,
        charbonnier_eps, rep_scale, aggregation_spec, record_timing=True,
    )["total"]


def stage2_joint_fast_objective_terms(
    cache: Stage2JointFastCache, parameters: torch.Tensor, beta_abs_limit: float,
    beta_anchor: torch.Tensor, body_pose_anchors: list[torch.Tensor],
    pose_anchor_scale: float, shape_anchor_scale: float,
    objective_mode: str, charbonnier_eps: float, rep_scale: float,
    aggregation_spec: dict[str, Any] | None,
) -> dict[str, torch.Tensor]:
    """Evaluate all joint objective terms with one dynamic selected forward."""
    return _stage2_joint_fast_objective_terms(
        cache, parameters, beta_abs_limit, beta_anchor, body_pose_anchors,
        pose_anchor_scale, shape_anchor_scale, objective_mode,
        charbonnier_eps, rep_scale, aggregation_spec, record_timing=False,
    )


def _stage2_joint_fast_objective_terms(
    cache: Stage2JointFastCache, parameters: torch.Tensor, beta_abs_limit: float,
    beta_anchor: torch.Tensor, body_pose_anchors: list[torch.Tensor],
    pose_anchor_scale: float, shape_anchor_scale: float,
    objective_mode: str, charbonnier_eps: float, rep_scale: float,
    aggregation_spec: dict[str, Any] | None,
    *, record_timing: bool,
) -> dict[str, torch.Tensor]:
    if record_timing:
        _sync(cache, parameters)
    start = time.perf_counter()
    groups = joint_fast_raw_groups(cache, parameters, beta_abs_limit)
    if objective_mode == "charbonnier_irls":
        if aggregation_spec is None: raise ValueError("aggregation spec required")
        primary = compute_primary_charbonnier_tensor(groups, charbonnier_eps, aggregation_spec)
    elif objective_mode == "raw_l2":
        raw = flatten_reprojection_groups(groups) * rep_scale
        primary = .5 * torch.sum(raw.square())
    else: raise ValueError(objective_mode)
    pose, shape = _joint_anchor_residuals(
        cache, parameters, beta_abs_limit, beta_anchor, body_pose_anchors,
        pose_anchor_scale, shape_anchor_scale,
    )
    pose_anchor = .5 * torch.sum(pose.square())
    shape_anchor = .5 * torch.sum(shape.square())
    result = {
        "primary": primary,
        "pose_anchor": pose_anchor,
        "shape_anchor": shape_anchor,
        "total": primary + pose_anchor + shape_anchor,
    }
    if record_timing:
        _sync(cache, parameters)
        cache.objective_ms += (time.perf_counter()-start)*1000
        cache.objective_calls += 1
    return result


def stage2_joint_fast_raw_metrics(
    cache: Stage2JointFastCache, parameters: torch.Tensor, beta_abs_limit: float,
) -> dict[str, float]:
    _sync(cache, parameters); start=time.perf_counter()
    result=raw_reproj_metrics(flatten_reprojection_groups(
        joint_fast_raw_groups(cache,parameters,beta_abs_limit)).detach())
    _sync(cache, parameters); cache.metrics_ms+=(time.perf_counter()-start)*1000; cache.metrics_calls+=1
    return result

