"""Benchmark-only Pose normal-equation variants for LM shadow regression."""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
import math
import time
from typing import Any

import torch

from lib_pano.human_fitting.staged_lm.irls import compute_irls_point_weights
from lib_pano.human_fitting.staged_lm.pose.forward import fast_raw_groups
from lib_pano.human_fitting.staged_lm.pose.derivatives import (
    stage2_pose_normal_equations,
    stage2_pose_group_raw_residual_and_jacobian,
)
from lib_pano.human_fitting.staged_lm.pose.batched import (
    pad_stage2_pose_row_scales,
    stage2_pose_batched_normal_equations,
)
from lib_pano.human_fitting.staged_lm.types import NormalEquations, StageDefinition
from script.benchmark.staged_lm.batch_common import numerical_error
from script.benchmark.staged_lm.pose_batch_candidates import (
    PoseBatchParameters,
    PoseGroupBatch,
    batched_pose_group_analytic,
    gather_pose_batch_parameters,
)


class PoseNEMode(str, Enum):
    SERIAL_REFERENCE = "serial_reference"
    PRODUCTION_FULL_BATCH = "production_full_batch"
    BATCH_FORWARD_SERIAL_HG = "batch_forward_serial_hg"
    SERIAL_FORWARD_BATCH_HG = "serial_forward_batch_hg"
    SERIAL_PRODUCTION = "serial_reference"
    FULL_BATCH = "production_full_batch"


@dataclass
class PoseEquationEvaluation:
    equations: NormalEquations
    residual_raw: torch.Tensor
    active_jacobian: torch.Tensor
    elapsed_ms: float


def _sync(reference: torch.Tensor) -> None:
    if reference.is_cuda:
        torch.cuda.synchronize(reference.device)


def _weight_stats(weights: torch.Tensor, valid_mask: torch.Tensor) -> dict[str, Any]:
    values = weights[valid_mask]
    return {
        "min": float(values.min()) if values.numel() else 0.0,
        "mean": float(values.mean()) if values.numel() else 0.0,
        "median": float(values.median()) if values.numel() else 0.0,
        "max": float(values.max()) if values.numel() else 0.0,
        "n_points": int(values.numel()),
    }


def compute_batched_pose_row_scales(
    residual: torch.Tensor,
    batch: PoseGroupBatch,
    aggregation_spec: dict[str, Any] | None,
    *,
    objective_mode: str,
    charbonnier_eps: float,
    rep_scale: float,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Compute production-equivalent point weights directly on padded tensors."""
    if objective_mode == "raw_l2":
        scales = torch.full_like(residual, rep_scale) * batch.layout.valid_mask[..., None]
        return scales, {}
    if aggregation_spec is None:
        raise ValueError("aggregation_spec is required")
    point_weights = residual.new_zeros(batch.layout.valid_mask.shape)
    keys = list(aggregation_spec["terms"])
    for group, key in enumerate(keys):
        count = batch.layout.point_counts_host[group]
        alpha = float(aggregation_spec["terms"][key]["alpha_per_point"])
        if count and alpha > 0:
            norms = torch.norm(residual[group, :count], dim=1)
            point_weights[group, :count] = alpha / torch.sqrt(norms.square() + charbonnier_eps ** 2)
    scales = torch.sqrt(point_weights)[..., None].expand(-1, -1, 2)
    return scales, _weight_stats(point_weights, batch.layout.valid_mask)


def serial_pose_row_scales(
    stage: StageDefinition,
    parameters: torch.Tensor,
    batch: PoseGroupBatch,
    *,
    objective_mode: str,
    charbonnier_eps: float,
    rep_scale: float,
) -> tuple[torch.Tensor, dict[str, Any]]:
    groups = fast_raw_groups(stage.analytic_cache, parameters)
    padded = parameters.new_zeros(batch.layout.group_count, batch.layout.nmax, 2)
    if objective_mode == "charbonnier_irls":
        weights = compute_irls_point_weights(groups, charbonnier_eps, stage.aggregation_spec)
        for group, weight in enumerate(weights):
            padded[group, :weight.numel()] = torch.sqrt(weight)[:, None]
        flat = torch.cat(weights)
        stats = {
            "min": float(flat.min()), "mean": float(flat.mean()),
            "median": float(flat.median()), "max": float(flat.max()),
            "n_points": int(flat.numel()),
        }
    else:
        for group, value in enumerate(groups):
            padded[group, :value.valid_point_count] = rep_scale
        stats = {}
    return padded, stats


def _active_indices(batch: PoseGroupBatch, group: int) -> torch.Tensor:
    return batch.active_indices_cam2 if batch.layout.camera_indices_host[group] == 0 else batch.active_indices_cam4


def _anchor_contribution(
    hessian: torch.Tensor,
    gradient: torch.Tensor,
    residual_parts: list[torch.Tensor],
    parameters: torch.Tensor,
    body_pose_anchors: list[torch.Tensor],
    pose_anchor_scale: float,
) -> None:
    scale2 = float(pose_anchor_scale) ** 2
    identity = torch.eye(63, dtype=torch.float64, device=parameters.device) * scale2
    for frame, anchor in enumerate(body_pose_anchors):
        start = frame * 75 + 12
        delta = parameters[start:start + 63] - anchor.to(parameters)
        residual_parts.append(delta * pose_anchor_scale)
        hessian[start:start + 63, start:start + 63] += identity
        gradient[start:start + 63] += delta.to(torch.float64) * scale2


def _assemble_from_active(
    batch: PoseGroupBatch,
    parameters: torch.Tensor,
    residual: torch.Tensor,
    active: torch.Tensor,
    row_scales: torch.Tensor,
    body_pose_anchors: list[torch.Tensor],
    pose_anchor_scale: float,
    *,
    use_bmm: bool,
    reduction_mode: str,
) -> NormalEquations:
    dimension = batch.layout.frame_count * 75
    hessian = torch.zeros(dimension, dimension, dtype=torch.float64, device=parameters.device)
    gradient = torch.zeros(dimension, dtype=torch.float64, device=parameters.device)
    residual_parts: list[torch.Tensor] = []
    if use_bmm:
        scaled_r = (residual * row_scales).reshape(batch.layout.group_count, -1).to(torch.float64)
        scaled_j = (active * row_scales[..., None]).reshape(batch.layout.group_count, -1, 69).to(torch.float64)
        group_h = torch.bmm(scaled_j.transpose(1, 2), scaled_j)
        group_g = torch.bmm(scaled_j.transpose(1, 2), scaled_r[..., None]).squeeze(-1)
    mapped_h = mapped_g = None
    if use_bmm and reduction_mode == "batched_bmm_scatter":
        mapped_h = torch.zeros(batch.layout.group_count, 75, 75, dtype=torch.float64, device=parameters.device)
        mapped_g = torch.zeros(batch.layout.group_count, 75, dtype=torch.float64, device=parameters.device)
        for group in range(batch.layout.group_count):
            indices = _active_indices(batch, group)
            mapped_h[group][indices[:, None], indices[None, :]] = group_h[group]
            mapped_g[group][indices] = group_g[group]
        frame_h = torch.zeros(batch.layout.frame_count, 75, 75, dtype=torch.float64, device=parameters.device)
        frame_g = torch.zeros(batch.layout.frame_count, 75, dtype=torch.float64, device=parameters.device)
        frame_h.index_add_(0, batch.layout.frame_positions, mapped_h)
        frame_g.index_add_(0, batch.layout.frame_positions, mapped_g)
        for frame in range(batch.layout.frame_count):
            block = slice(frame * 75, (frame + 1) * 75)
            hessian[block, block] = frame_h[frame]
            gradient[block] = frame_g[frame]
    for group in range(batch.layout.group_count):
        count = batch.layout.point_counts_host[group]
        scaled_residual = (residual[group, :count] * row_scales[group, :count]).reshape(-1)
        residual_parts.append(scaled_residual)
        indices = _active_indices(batch, group)
        frame = batch.layout.frame_positions_host[group]
        global_indices = frame * 75 + indices
        if use_bmm:
            local_h, local_g = group_h[group], group_g[group]
        else:
            local_r = scaled_residual.to(torch.float64)
            local_j = (active[group, :count] * row_scales[group, :count, :, None]).reshape(-1, 69).to(torch.float64)
            local_h, local_g = local_j.T @ local_j, local_j.T @ local_r
        # Both serial-frame and production-order deliberately preserve the
        # production frame0/cam2, frame0/cam4, ... addition order.
        if mapped_h is None:
            hessian[global_indices[:, None], global_indices[None, :]] += local_h
            gradient[global_indices] += local_g
    _anchor_contribution(hessian, gradient, residual_parts, parameters, body_pose_anchors, pose_anchor_scale)
    return NormalEquations(torch.cat(residual_parts), hessian, gradient)


def evaluate_pose_equations(
    mode: PoseNEMode,
    stage: StageDefinition,
    batch: PoseGroupBatch,
    parameters: torch.Tensor,
    row_scales: torch.Tensor,
    body_pose_anchors: list[torch.Tensor],
    pose_anchor_scale: float,
    image_width: int,
    image_height: int,
    *,
    hg_reduction_mode: str = "batched_bmm_serial_frame",
) -> PoseEquationEvaluation:
    _sync(parameters); started = time.perf_counter()
    if mode is PoseNEMode.SERIAL_REFERENCE:
        scale_list = [row_scales[g, :batch.layout.point_counts_host[g]].reshape(-1) for g in range(batch.layout.group_count)]
        equations = stage2_pose_normal_equations(
            stage.analytic_cache, parameters, scale_list,
            body_pose_anchors, pose_anchor_scale,
        )
        _sync(parameters)
        equation_elapsed_ms = (time.perf_counter() - started) * 1000.0
        raw_parts, active_parts = [], []
        for group_index, group in enumerate(stage.analytic_cache.groups):
            raw, local = stage2_pose_group_raw_residual_and_jacobian(stage.analytic_cache, group, parameters)
            count = group.targets.shape[0]
            active = local[:, _active_indices(batch, group_index)].reshape(count, 2, 69)
            raw_parts.append(raw.reshape(count, 2)); active_parts.append(active)
        raw_padded = parameters.new_zeros(batch.layout.group_count, batch.layout.nmax, 2)
        active_padded = parameters.new_zeros(batch.layout.group_count, batch.layout.nmax, 2, 69)
        for i, (raw, active) in enumerate(zip(raw_parts, active_parts)):
            raw_padded[i, :raw.shape[0]] = raw; active_padded[i, :active.shape[0]] = active
    elif mode is PoseNEMode.PRODUCTION_FULL_BATCH:
        packed = gather_pose_batch_parameters(parameters, batch.layout)
        raw_padded, active_padded, _ = batched_pose_group_analytic(
            batch, packed.group_orientations, packed.group_translations,
            packed.group_body_poses, image_width, image_height,
        )
        production_batch = replace(
            batch,
            pose_anchor_identity_64=torch.eye(
                63, dtype=torch.float64, device=parameters.device,
            ) * float(pose_anchor_scale) ** 2,
        )
        equations = stage2_pose_batched_normal_equations(
            stage.analytic_cache, production_batch, parameters, row_scales,
            body_pose_anchors, pose_anchor_scale,
        )
    else:
        packed = gather_pose_batch_parameters(parameters, batch.layout)
        if mode is PoseNEMode.SERIAL_FORWARD_BATCH_HG:
            raw_padded = parameters.new_zeros(batch.layout.group_count, batch.layout.nmax, 2)
            active_padded = parameters.new_zeros(batch.layout.group_count, batch.layout.nmax, 2, 69)
            for i, group in enumerate(stage.analytic_cache.groups):
                raw, local = stage2_pose_group_raw_residual_and_jacobian(stage.analytic_cache, group, parameters)
                count = group.targets.shape[0]
                raw_padded[i, :count] = raw.reshape(count, 2)
                active_padded[i, :count] = local[:, _active_indices(batch, i)].reshape(count, 2, 69)
        else:
            raw_padded, active_padded, _ = batched_pose_group_analytic(
                batch, packed.group_orientations, packed.group_translations,
                packed.group_body_poses, image_width, image_height,
            )
        use_bmm = mode is PoseNEMode.SERIAL_FORWARD_BATCH_HG and hg_reduction_mode not in ("serial_group", "production_order")
        equations = _assemble_from_active(
            batch, parameters, raw_padded, active_padded, row_scales,
            body_pose_anchors, pose_anchor_scale, use_bmm=use_bmm,
            reduction_mode=hg_reduction_mode,
        )
    _sync(parameters)
    elapsed = equation_elapsed_ms if mode is PoseNEMode.SERIAL_REFERENCE else (time.perf_counter() - started) * 1000.0
    return PoseEquationEvaluation(equations, raw_padded, active_padded, elapsed)


def equation_errors(candidate: PoseEquationEvaluation, reference: PoseEquationEvaluation) -> dict[str, Any]:
    return {
        "residual": numerical_error(candidate.residual_raw, reference.residual_raw),
        "jacobian": numerical_error(candidate.active_jacobian, reference.active_jacobian),
        "H": numerical_error(candidate.equations.hessian, reference.equations.hessian),
        "g": numerical_error(candidate.equations.gradient, reference.equations.gradient),
    }
