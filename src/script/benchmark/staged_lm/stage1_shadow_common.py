"""Shared normal-equation and error utilities for Stage-1 shadows."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
import time
from typing import Any, Callable

import torch

from lib_pano.human_fitting.staged_lm.analytic_stage1 import (
    Stage1AnalyticCache, _group_raw_residual_and_local_jacobian,
    stage1_normal_equations, stage1_raw_groups, stage1_raw_jacobian,
    stage1_raw_residual,
)
from lib_pano.human_fitting.staged_lm.irls import compute_irls_point_weights
from lib_pano.human_fitting.staged_lm.types import NormalEquations
from lib_pano.human_fitting.staged_lm.lm_solver import (
    compute_scaled_gradient, normal_equations_max_diagonal,
    normal_equations_quadratic_form, solve_normal_equations,
)
from script.benchmark.staged_lm.stage1_batch_candidates import (
    Stage1BatchAnalyticResult, Stage1BatchCache, pad_stage1_row_scales,
    stage1_batched_group_analytic,
)


class Stage1NEMode(str, Enum):
    SERIAL_REFERENCE = "serial_reference"
    BATCH_FORWARD_SERIAL_FRAME_GEMM = "batch_forward_serial_frame_gemm"
    SERIAL_FORWARD_BATCH_GROUP_GEMM = "serial_forward_batch_group_gemm"
    FULL_BATCH_GROUP_GEMM = "full_batch_group_gemm"


@dataclass(frozen=True)
class Stage1ShadowLinearizationCommon:
    row_scales_by_group: list[torch.Tensor]
    padded_row_scales: torch.Tensor
    flat_row_scales: torch.Tensor
    weight_stats: dict[str, Any]
    spec: dict[str, Any]
    surrogate_fn: Callable
    jacobian_fn: Callable


@dataclass
class Stage1EquationEvaluation:
    equations: NormalEquations
    raw_residual: torch.Tensor | None
    active_jacobian: torch.Tensor | None
    equation_ms: float
    full_analytic_calls: int


@dataclass(frozen=True)
class TensorError:
    max_abs: float
    mean_abs: float
    relative_frobenius: float
    reference_norm: float
    candidate_norm: float
    cosine_similarity: float
    worst_index: tuple[int, ...] | None
    worst_reference: float | None
    worst_candidate: float | None


def tensor_error(candidate: torch.Tensor, reference: torch.Tensor) -> TensorError:
    c, r = candidate.to(torch.float64), reference.to(torch.float64)
    delta = c - r; flat = delta.abs().reshape(-1)
    worst = None; rv = cv = None
    if flat.numel():
        linear = int(flat.argmax()); indices = []
        for size in reversed(delta.shape): indices.append(linear % size); linear //= size
        worst = tuple(reversed(indices)); rv = float(r[worst]); cv = float(c[worst])
    rn, cn = float(r.norm()), float(c.norm())
    cosine = float(torch.nn.functional.cosine_similarity(
        r.reshape(1, -1), c.reshape(1, -1), eps=1e-30,
    )) if r.numel() else 1.0
    return TensorError(
        float(flat.max()) if flat.numel() else 0.0,
        float(flat.mean()) if flat.numel() else 0.0,
        float(delta.norm()) / max(rn, 1e-30), rn, cn, cosine, worst, rv, cv,
    )


def build_stage1_shadow_linearization_common(
    cache: Stage1AnalyticCache, batch: Stage1BatchCache,
    linearisation_point: torch.Tensor, *, is_irls: bool,
    charbonnier_eps: float, aggregation_spec: dict[str, Any] | None,
    rep_scale: float,
) -> Stage1ShadowLinearizationCommon:
    groups = stage1_raw_groups(cache, linearisation_point)
    if is_irls:
        weights = compute_irls_point_weights(groups, charbonnier_eps, aggregation_spec)
        scales = [torch.sqrt(weight).repeat_interleave(2).detach() for weight in weights]
        flat_weights = torch.cat(weights)
        stats = {"min":float(flat_weights.min()), "mean":float(flat_weights.mean()),
                 "median":float(flat_weights.median()), "max":float(flat_weights.max()),
                 "n_points":int(flat_weights.numel())}
    else:
        scales = [torch.full((group.residual_2d.numel(),), rep_scale,
                             dtype=linearisation_point.dtype,
                             device=linearisation_point.device) for group in groups]
        stats = {}
    flat = torch.cat(scales)
    padded = pad_stage1_row_scales(batch, scales)
    return Stage1ShadowLinearizationCommon(
        scales, padded, flat, stats, aggregation_spec or {},
        lambda x: stage1_raw_residual(cache, x) * flat.to(x),
        lambda x: stage1_raw_jacobian(cache, x) * flat.to(x).unsqueeze(1),
    )


def _serial_forward_padded(cache, batch, parameters) -> Stage1BatchAnalyticResult:
    residual = torch.zeros(batch.layout.group_count, batch.layout.nmax, 2,
                           dtype=parameters.dtype, device=parameters.device)
    active = torch.zeros(batch.layout.group_count, batch.layout.nmax, 2, 6,
                         dtype=parameters.dtype, device=parameters.device)
    for index, (group, count, camera) in enumerate(zip(
        cache.groups, batch.layout.point_counts_host, batch.layout.camera_indices_host,
    )):
        rr, jj = _group_raw_residual_and_local_jacobian(cache, parameters, group)
        residual[index, :count] = rr.reshape(count, 2)
        active[index, :count] = jj.reshape(count, 2, 12)[..., camera*6:camera*6+6]
    return Stage1BatchAnalyticResult(residual, active)


def _full_group_gemm(batch, result, scales):
    g, n = result.residual.shape[:2]
    r = (result.residual * scales).reshape(g, 2*n).to(torch.float64)
    j = (result.active_jacobian * scales[..., None]).reshape(g, 2*n, 6).to(torch.float64)
    return r, torch.bmm(j.transpose(1, 2), j), torch.bmm(j.transpose(1, 2), r[..., None]).squeeze(-1)


def _group_system(batch, result, scales, parameters):
    r, hg, gg = _full_group_gemm(batch, result, scales)
    dimension = batch.layout.frame_count * 12
    h = torch.zeros(dimension, dimension, dtype=torch.float64, device=parameters.device)
    grad = torch.zeros(dimension, dtype=torch.float64, device=parameters.device)
    residuals = []
    for index, (frame, camera, count) in enumerate(zip(
        batch.layout.frame_positions_host, batch.layout.camera_indices_host,
        batch.layout.point_counts_host,
    )):
        block = slice(frame*12+camera*6, frame*12+camera*6+6)
        h[block, block] += hg[index]; grad[block] += gg[index]
        residuals.append(r[index, :2*count].to(parameters.dtype))
    return NormalEquations(torch.cat(residuals), h, grad)


def _frame_system(batch, result, scales, parameters):
    dimension = batch.layout.frame_count * 12
    h = torch.zeros(dimension, dimension, dtype=torch.float64, device=parameters.device)
    grad = torch.zeros(dimension, dtype=torch.float64, device=parameters.device)
    residuals = []; frame_r = [[] for _ in range(batch.layout.frame_count)]
    frame_j = [[] for _ in range(batch.layout.frame_count)]
    for index, (frame, camera, count) in enumerate(zip(
        batch.layout.frame_positions_host, batch.layout.camera_indices_host,
        batch.layout.point_counts_host,
    )):
        r = (result.residual[index, :count] * scales[index, :count]).reshape(-1)
        active = (result.active_jacobian[index, :count] * scales[index, :count, :, None]).reshape(-1, 6)
        local = torch.zeros(active.shape[0], 12, dtype=active.dtype, device=active.device)
        local[:, camera*6:camera*6+6] = active
        residuals.append(r); frame_r[frame].append(r); frame_j[frame].append(local)
    for frame in range(batch.layout.frame_count):
        rr = torch.cat(frame_r[frame]).to(torch.float64)
        jj = torch.cat(frame_j[frame]).to(torch.float64)
        block = slice(frame*12, (frame+1)*12)
        h[block, block] = jj.T @ jj; grad[block] = jj.T @ rr
    return NormalEquations(torch.cat(residuals), h, grad)


def evaluate_stage1_equations(
    mode: Stage1NEMode, cache: Stage1AnalyticCache, batch: Stage1BatchCache,
    parameters: torch.Tensor, padded_row_scales: torch.Tensor,
    flat_row_scales: torch.Tensor,
) -> Stage1EquationEvaluation:
    before = batch.full_analytic_calls
    if parameters.is_cuda: torch.cuda.synchronize(parameters.device)
    started = time.perf_counter()
    raw = active = None
    if mode is Stage1NEMode.SERIAL_REFERENCE:
        equations = stage1_normal_equations(cache, parameters, flat_row_scales)
    else:
        result = (_serial_forward_padded(cache, batch, parameters)
                  if mode is Stage1NEMode.SERIAL_FORWARD_BATCH_GROUP_GEMM
                  else stage1_batched_group_analytic(
                      batch, parameters, cache.image_width, cache.image_height))
        raw, active = result.residual, result.active_jacobian
        equations = (_frame_system(batch, result, padded_row_scales, parameters)
                     if mode is Stage1NEMode.BATCH_FORWARD_SERIAL_FRAME_GEMM
                     else _group_system(batch, result, padded_row_scales, parameters))
    if parameters.is_cuda: torch.cuda.synchronize(parameters.device)
    elapsed = (time.perf_counter() - started) * 1000.0
    return Stage1EquationEvaluation(
        equations, raw, active, elapsed, batch.full_analytic_calls - before,
    )


def equation_errors(candidate, reference) -> dict[str, Any]:
    result = {
        "scaled_residual": asdict(tensor_error(candidate.equations.residual, reference.equations.residual)),
        "H": asdict(tensor_error(candidate.equations.hessian, reference.equations.hessian)),
        "g": asdict(tensor_error(candidate.equations.gradient, reference.equations.gradient)),
    }
    return result


def analytic_errors(cache, batch, parameters, candidate) -> dict[str, Any]:
    serial = _serial_forward_padded(cache, batch, parameters)
    if candidate.raw_residual is None or candidate.active_jacobian is None:
        return {}
    return {
        "residual": asdict(tensor_error(candidate.raw_residual, serial.residual)),
        "active_jacobian": asdict(tensor_error(candidate.active_jacobian, serial.active_jacobian)),
    }


def step_errors(candidate, reference, parameter_dimension: int) -> dict[str, Any]:
    damping=normal_equations_max_diagonal(reference.equations)*1e-3
    rd=solve_normal_equations(reference.equations,damping).delta
    cd=solve_normal_equations(candidate.equations,damping).delta
    output={"max_diagonal_abs":abs(normal_equations_max_diagonal(candidate.equations)-normal_equations_max_diagonal(reference.equations)),
            "scaled_gradient_abs":abs(compute_scaled_gradient(candidate.equations)-compute_scaled_gradient(reference.equations))}
    if rd is None or cd is None:return {**output,"solve_success_match":rd is None and cd is None}
    predicted_reference=float(-reference.equations.gradient.dot(rd)-.5*normal_equations_quadratic_form(reference.equations,rd))
    predicted_candidate=float(-candidate.equations.gradient.dot(cd)-.5*normal_equations_quadratic_form(candidate.equations,cd))
    output.update({"solve_success_match":True,"delta":asdict(tensor_error(cd,rd)),
                   "predicted_reference":predicted_reference,"predicted_candidate":predicted_candidate,
                   "predicted_relative":abs(predicted_candidate-predicted_reference)/max(abs(predicted_reference),1e-30)})
    return output
