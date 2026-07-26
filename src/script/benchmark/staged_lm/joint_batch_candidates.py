"""Benchmark-only mode selection around the production Joint FULL_BATCH kernel."""

from __future__ import annotations

from enum import Enum

import torch

from lib_pano.human_fitting.staged_lm.joint.batched import (
    Stage2JointBatchAnalyticResult as JointBatchAnalyticResult,
    Stage2JointBatchCache as JointBatchCache,
    Stage2JointBatchLayout as JointBatchLayout,
    Stage2JointBatchParameters as JointBatchParameters,
    build_stage2_joint_batch_cache,
    gather_stage2_joint_batch_parameters,
    pad_stage2_joint_row_scales,
    stage2_joint_batched_accumulate_arrowhead,
    stage2_joint_batched_dynamic_shape,
    stage2_joint_batched_group_analytic,
    stage2_joint_batched_normal_equations,
)
from lib_pano.human_fitting.staged_lm.joint import (
    Stage2JointFastCache,
    evaluate_joint_shape_state,
    physical_beta_from_latent,
)
from lib_pano.human_fitting.staged_lm.arrowhead_schur import ArrowheadNormalEquationSystem
from lib_pano.human_fitting.staged_lm.joint import (
    joint_group_raw_residual_and_jacobians,
    stage2_joint_arrowhead_normal_equations,
)
from lib_pano.human_fitting.staged_lm.types import NormalEquations


class JointNEMode(str, Enum):
    SERIAL_REFERENCE = "serial_reference"
    BATCH_FORWARD_SERIAL_ACCUMULATION = "batch_forward_serial_accumulation"
    SERIAL_FORWARD_BATCH_GEMM = "serial_forward_batch_gemm"
    FULL_BATCH_PRODUCTION_REDUCTION = "full_batch_production_reduction"
    FULL_BATCH_TENSOR_BETA_REDUCTION = "full_batch_tensor_beta_reduction"


# Compatibility names are aliases, not copies of the production mathematics.
def build_joint_batch_cache(
    cache: Stage2JointFastCache, pose_anchor_scale: float,
) -> JointBatchCache:
    return build_stage2_joint_batch_cache(
        cache, pose_anchor_scale=pose_anchor_scale,
    )
gather_joint_batch_parameters = gather_stage2_joint_batch_parameters
joint_batched_dynamic_shape = stage2_joint_batched_dynamic_shape
joint_batched_group_analytic = stage2_joint_batched_group_analytic
pad_joint_row_scales = pad_stage2_joint_row_scales


def _serial_forward_padded(
    cache: Stage2JointFastCache,
    batch: JointBatchCache,
    parameters: torch.Tensor,
    beta_abs_limit: float,
) -> JointBatchAnalyticResult:
    shape = evaluate_joint_shape_state(
        cache, parameters[-batch.layout.beta_dim :], beta_abs_limit,
    )
    residual = torch.zeros(
        batch.layout.group_count, batch.layout.nmax, 2,
        dtype=parameters.dtype, device=parameters.device,
    )
    active = torch.zeros(
        batch.layout.group_count, batch.layout.nmax, 2, 69,
        dtype=parameters.dtype, device=parameters.device,
    )
    beta_j = torch.zeros(
        batch.layout.group_count, batch.layout.nmax, 2, batch.layout.beta_dim,
        dtype=parameters.dtype, device=parameters.device,
    )
    for index, (group, count) in enumerate(
        zip(cache.groups, batch.layout.point_counts_host)
    ):
        value = joint_group_raw_residual_and_jacobians(
            cache, group, parameters, beta_abs_limit, shape,
        )
        indices = (
            batch.active_indices_cam2
            if batch.layout.camera_indices_host[index] == 0
            else batch.active_indices_cam4
        )
        residual[index, :count] = value.raw_residual.reshape(count, 2)
        active[index, :count] = value.local_pose_jacobian.reshape(
            count, 2, 75,
        ).index_select(-1, indices)
        beta_j[index, :count] = value.shared_beta_jacobian.reshape(count, 2, -1)
    return JointBatchAnalyticResult(residual, active, beta_j)


def _diagnostic_accumulate(
    cache, batch, parameters, result, scales, body_pose_anchors, beta_anchor,
    beta_abs_limit, pose_anchor_scale, shape_anchor_scale, *, serial_gemm,
    tensor_beta_reduction,
):
    """Benchmark-only alternate GEMM/reduction order around production tensors."""
    g, n, b = batch.layout.group_count, batch.layout.nmax, batch.layout.beta_dim
    contributions = []
    for index, count in enumerate(batch.layout.point_counts_host):
        rows = count if serial_gemm else n
        r = (result.residual[index, :rows] * scales[index, :rows]).reshape(-1).to(torch.float64)
        ja = (result.active_pose_jacobian[index, :rows] * scales[index, :rows, :, None]).reshape(-1, 69).to(torch.float64)
        jb = (result.beta_jacobian[index, :rows] * scales[index, :rows, :, None]).reshape(-1, b).to(torch.float64)
        contributions.append((r[:2*count], ja.T @ ja, ja.T @ jb, jb.T @ jb, ja.T @ r, jb.T @ r))
    if not serial_gemm:
        r = (result.residual * scales).reshape(g, 2*n).to(torch.float64)
        ja = (result.active_pose_jacobian * scales[..., None]).reshape(g, 2*n, 69).to(torch.float64)
        jb = (result.beta_jacobian * scales[..., None]).reshape(g, 2*n, b).to(torch.float64)
        contributions = [
            (r[i, :2*batch.layout.point_counts_host[i]], ha, hb, hc, ga, gb)
            for i, (ha, hb, hc, ga, gb) in enumerate(zip(
                torch.bmm(ja.transpose(1, 2), ja), torch.bmm(ja.transpose(1, 2), jb),
                torch.bmm(jb.transpose(1, 2), jb),
                torch.bmm(ja.transpose(1, 2), r[..., None]).squeeze(-1),
                torch.bmm(jb.transpose(1, 2), r[..., None]).squeeze(-1),
            ))
        ]
    a = torch.zeros(cache.frame_count, 75, 75, dtype=torch.float64, device=parameters.device)
    cross = torch.zeros(cache.frame_count, 75, b, dtype=torch.float64, device=parameters.device)
    c = torch.zeros(b, b, dtype=torch.float64, device=parameters.device)
    gf = torch.zeros(cache.frame_count, 75, dtype=torch.float64, device=parameters.device)
    gbeta = torch.zeros(b, dtype=torch.float64, device=parameters.device)
    residuals = []
    for index, (r, ha, hb, hc, ga, gb) in enumerate(contributions):
        frame = batch.layout.frame_positions_host[index]
        indices = batch.active_indices_cam2 if batch.layout.camera_indices_host[index] == 0 else batch.active_indices_cam4
        a[frame][indices[:, None], indices[None, :]] += ha
        cross[frame][indices] += hb; gf[frame][indices] += ga
        if not tensor_beta_reduction: c += hc; gbeta += gb
        residuals.append(r.to(parameters.dtype))
    if tensor_beta_reduction:
        c = torch.stack([item[3] for item in contributions]).sum(0)
        gbeta = torch.stack([item[5] for item in contributions]).sum(0)
    packed = gather_stage2_joint_batch_parameters(parameters, batch.layout, b)
    pose_residuals = []
    for frame in range(cache.frame_count):
        delta = packed.frame_body_poses[frame] - body_pose_anchors[frame].to(parameters)
        pose_residuals.append(delta * pose_anchor_scale)
        a[frame, 12:, 12:] += batch.pose_anchor_identity_64
        gf[frame, 12:] += delta.to(torch.float64) * float(pose_anchor_scale) ** 2
    beta, dbdz = physical_beta_from_latent(packed.beta_latent, beta_abs_limit)
    shape_residual = (beta - beta_anchor.to(parameters)) * shape_anchor_scale
    diagonal = (dbdz * shape_anchor_scale).to(torch.float64)
    c += torch.diag(diagonal.square()); gbeta += diagonal * shape_residual.to(torch.float64)
    system = ArrowheadNormalEquationSystem(a, cross, c, gf, gbeta, timing_sink=cache)
    return NormalEquations(torch.cat((*residuals, *pose_residuals, shape_residual)), None,
                           system.flat_gradient(), structured_system=system)


def evaluate_joint_normal_equations(
    mode: JointNEMode,
    cache: Stage2JointFastCache,
    batch: JointBatchCache,
    parameters: torch.Tensor,
    beta_abs_limit: float,
    row_scales_by_group: list[torch.Tensor],
    body_pose_anchors: list[torch.Tensor],
    beta_anchor: torch.Tensor,
    pose_anchor_scale: float,
    shape_anchor_scale: float,
) -> NormalEquations:
    if mode is JointNEMode.SERIAL_REFERENCE:
        return stage2_joint_arrowhead_normal_equations(
            cache, parameters, beta_abs_limit, row_scales_by_group,
            body_pose_anchors, beta_anchor, pose_anchor_scale,
            shape_anchor_scale,
        )
    padded = pad_stage2_joint_row_scales(batch, row_scales_by_group)
    if mode is JointNEMode.FULL_BATCH_PRODUCTION_REDUCTION:
        return stage2_joint_batched_normal_equations(
            cache, batch, parameters, beta_abs_limit, padded,
            body_pose_anchors, beta_anchor, pose_anchor_scale,
            shape_anchor_scale,
        )
    result = (
        _serial_forward_padded(cache, batch, parameters, beta_abs_limit)
        if mode is JointNEMode.SERIAL_FORWARD_BATCH_GEMM
        else stage2_joint_batched_group_analytic(
            cache, batch, parameters, beta_abs_limit,
        )
    )
    if mode is JointNEMode.SERIAL_FORWARD_BATCH_GEMM:
        return stage2_joint_batched_accumulate_arrowhead(
            cache, batch, parameters, result, padded, body_pose_anchors,
            beta_anchor, beta_abs_limit, pose_anchor_scale, shape_anchor_scale,
        )
    return _diagnostic_accumulate(
        cache, batch, parameters, result, padded, body_pose_anchors,
        beta_anchor, beta_abs_limit, pose_anchor_scale, shape_anchor_scale,
        tensor_beta_reduction=(
            mode is JointNEMode.FULL_BATCH_TENSOR_BETA_REDUCTION
        ),
        serial_gemm=(
            mode is JointNEMode.BATCH_FORWARD_SERIAL_ACCUMULATION
        ),
    )
