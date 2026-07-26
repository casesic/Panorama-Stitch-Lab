"""Benchmark-only Stage-1 FULL_BATCH geometry/Jacobian candidates."""
from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any

import torch

from lib_pano.human_fitting.staged_lm.analytic_stage1 import (
    Stage1AnalyticCache,
)
from lib_pano.human_fitting.staged_lm.geometry.erp import (
    project_unified_erp_with_jacobian,
)
from lib_pano.human_fitting.staged_lm.geometry.so3 import (
    axis_angle_to_matrix, skew, so3_right_jacobian,
)
from lib_pano.human_fitting.staged_lm.irls import compute_primary_charbonnier_tensor
from lib_pano.human_fitting.staged_lm.types import ReprojectionGroup


def _bytes(tensor: torch.Tensor) -> int:
    return tensor.numel() * tensor.element_size()


@dataclass(frozen=True)
class Stage1BatchLayout:
    frame_positions: torch.Tensor
    camera_indices: torch.Tensor
    point_counts: torch.Tensor
    valid_mask: torch.Tensor
    targets: torch.Tensor
    rigid_parameter_indices: torch.Tensor
    frame_positions_host: tuple[int, ...]
    camera_indices_host: tuple[int, ...]
    point_counts_host: tuple[int, ...]
    group_count: int
    frame_count: int
    nmax: int
    padding_ratio: float
    padded_bytes: int


@dataclass
class Stage1BatchCache:
    layout: Stage1BatchLayout
    local_vertices: torch.Tensor
    rotation_centers: torch.Tensor
    setup_ms: float
    layout_setup_ms: float
    vertex_padding_ms: float
    static_indices_ms: float
    persistent_bytes: int
    full_analytic_calls: int = 0


@dataclass(frozen=True)
class Stage1BatchParameters:
    group_orientations: torch.Tensor
    group_translations: torch.Tensor


@dataclass
class Stage1BatchAnalyticResult:
    residual: torch.Tensor
    active_jacobian: torch.Tensor


def build_stage1_batch_cache(cache: Stage1AnalyticCache) -> Stage1BatchCache:
    if not cache.groups:
        raise ValueError("EMPTY_STAGE1_BATCH")
    started = time.perf_counter()
    groups = cache.groups
    device, dtype = groups[0].local_vertices.device, groups[0].local_vertices.dtype
    counts = tuple(int(group.local_vertices.shape[0]) for group in groups)
    frames = tuple(int(group.frame_position) for group in groups)
    cameras = tuple(0 if group.camera == "cam2" else 1 for group in groups)
    g, nmax = len(groups), max(counts)
    section = time.perf_counter()
    frame_t = torch.tensor(frames, dtype=torch.long, device=device)
    camera_t = torch.tensor(cameras, dtype=torch.long, device=device)
    count_t = torch.tensor(counts, dtype=torch.long, device=device)
    valid = torch.zeros(g, nmax, dtype=torch.bool, device=device)
    targets = torch.zeros(g, nmax, 2, dtype=dtype, device=device)
    for index, (group, count) in enumerate(zip(groups, counts)):
        valid[index, :count] = True
        targets[index, :count] = group.targets
    layout_ms = (time.perf_counter() - section) * 1000.0
    section = time.perf_counter()
    vertices = torch.zeros(g, nmax, 3, dtype=dtype, device=device)
    for index, (group, count) in enumerate(zip(groups, counts)):
        vertices[index, :count] = group.local_vertices
    vertex_ms = (time.perf_counter() - section) * 1000.0
    section = time.perf_counter()
    centers = torch.stack([group.rotation_center for group in groups])
    base = torch.arange(6, dtype=torch.long, device=device)
    rigid_indices = base[None] + camera_t[:, None] * 6
    static_ms = (time.perf_counter() - section) * 1000.0
    layout_tensors = (frame_t, camera_t, count_t, valid, targets, rigid_indices)
    layout = Stage1BatchLayout(
        frame_t, camera_t, count_t, valid, targets, rigid_indices,
        frames, cameras, counts, g, cache.frame_count, nmax,
        1.0 - sum(counts) / max(g * nmax, 1),
        sum(_bytes(x) for x in layout_tensors),
    )
    persistent = sum(_bytes(x) for x in (*layout_tensors, vertices, centers))
    return Stage1BatchCache(
        layout, vertices, centers, (time.perf_counter() - started) * 1000.0,
        layout_ms, vertex_ms, static_ms, persistent,
    )


def gather_stage1_batch_parameters(
    parameters: torch.Tensor, layout: Stage1BatchLayout,
) -> Stage1BatchParameters:
    framed = parameters.reshape(layout.frame_count, 12)
    grouped = framed.index_select(0, layout.frame_positions)
    rigid = grouped.gather(1, layout.rigid_parameter_indices)
    return Stage1BatchParameters(rigid[:, :3], rigid[:, 3:])


def stage1_batched_group_analytic(
    batch: Stage1BatchCache, parameters: torch.Tensor,
    image_width: int, image_height: int,
) -> Stage1BatchAnalyticResult:
    batch.full_analytic_calls += 1
    packed = gather_stage1_batch_parameters(parameters, batch.layout)
    rotations = axis_angle_to_matrix(packed.group_orientations)
    points = torch.einsum("gnc,gdc->gnd", batch.local_vertices, rotations)
    points = points + batch.rotation_centers[:, None] + packed.group_translations[:, None]
    right = so3_right_jacobian(packed.group_orientations)
    rotated = -torch.einsum(
        "gij,gnjk,gkl->gnil", rotations, skew(batch.local_vertices), right,
    )
    g, n = points.shape[:2]
    uv, projection = project_unified_erp_with_jacobian(
        points.reshape(g * n, 3), image_width, image_height,
    )
    uv = uv.reshape(g, n, 2)
    projection = projection.reshape(g, n, 2, 3)
    orient_j = torch.einsum("gnij,gnjk->gnik", projection, rotated)
    mask = batch.layout.valid_mask
    residual = (uv - batch.layout.targets) * mask[..., None]
    jacobian = torch.cat((orient_j, projection), -1) * mask[..., None, None]
    return Stage1BatchAnalyticResult(residual, jacobian)


def pad_stage1_row_scales(
    batch: Stage1BatchCache, row_scales_by_group: list[torch.Tensor],
) -> torch.Tensor:
    output = torch.zeros(
        batch.layout.group_count, batch.layout.nmax, 2,
        dtype=row_scales_by_group[0].dtype, device=row_scales_by_group[0].device,
    )
    for index, (count, scales) in enumerate(
        zip(batch.layout.point_counts_host, row_scales_by_group)
    ):
        output[index, :count] = scales.reshape(count, 2)
    return output


# Compatibility wrappers used by the earlier bake-off/tests.
Stage1GroupBatch = Stage1BatchCache
build_stage1_group_batch = build_stage1_batch_cache


def gather_stage1_group_parameters(parameters, layout):
    packed = gather_stage1_batch_parameters(parameters, layout)
    return packed.group_orientations, packed.group_translations


def batched_stage1_residual_and_jacobian(
    batch, orientations, translations, image_width, image_height,
):
    framed = torch.zeros(batch.layout.frame_count, 12, dtype=orientations.dtype, device=orientations.device)
    for i, (frame, camera) in enumerate(zip(batch.layout.frame_positions_host, batch.layout.camera_indices_host)):
        framed[frame, camera*6:camera*6+3] = orientations[i]
        framed[frame, camera*6+3:camera*6+6] = translations[i]
    result = stage1_batched_group_analytic(batch, framed.reshape(-1), image_width, image_height)
    return result.residual, result.active_jacobian


def batched_stage1_residual(batch, orientations, translations, image_width, image_height):
    return batched_stage1_residual_and_jacobian(
        batch, orientations, translations, image_width, image_height,
    )[0]


def batched_stage1_hg(
    batch, orientations, translations, row_scales, image_width, image_height,
):
    residual, jacobian = batched_stage1_residual_and_jacobian(
        batch, orientations, translations, image_width, image_height,
    )
    g, n = residual.shape[:2]
    r = (residual * row_scales).reshape(g, 2*n).to(torch.float64)
    j = (jacobian * row_scales[..., None]).reshape(g, 2*n, 6).to(torch.float64)
    hg = torch.bmm(j.transpose(1, 2), j)
    gg = torch.bmm(j.transpose(1, 2), r[..., None]).squeeze(-1)
    h = torch.zeros(batch.layout.frame_count*12, batch.layout.frame_count*12,
                    dtype=torch.float64, device=r.device)
    grad = torch.zeros(batch.layout.frame_count, 12, dtype=torch.float64, device=r.device)
    for i, (frame, camera) in enumerate(zip(batch.layout.frame_positions_host, batch.layout.camera_indices_host)):
        block = slice(frame*12+camera*6, frame*12+camera*6+6)
        h[block, block] += hg[i]; grad[frame, camera*6:camera*6+6] += gg[i]
    return residual, jacobian, hg, h, grad.reshape(-1)


def batched_stage1_objective(
    batch, orientations, translations, image_width, image_height, *,
    objective_mode, charbonnier_eps, rep_scale, aggregation_spec,
):
    residual = batched_stage1_residual(batch, orientations, translations, image_width, image_height)
    if objective_mode == "raw_l2":
        return 0.5 * torch.sum((residual * rep_scale) ** 2)
    groups = [
        ReprojectionGroup(int(term.split("|", 1)[0]), term.split("|", 1)[1], residual[i, :count])
        for i, (term, count) in enumerate(zip(aggregation_spec["terms"], batch.layout.point_counts_host))
    ]
    return compute_primary_charbonnier_tensor(groups, charbonnier_eps, aggregation_spec)


def profile_stage1_group_production(cache, group, parameters):
    """Retained section-profiler entry point for the unified bake-off."""
    sections={}
    def timed(name,fn):
        start=time.perf_counter();value=fn();sections[name]=(time.perf_counter()-start)*1000;return value
    offset=group.frame_position*12+(0 if group.camera=="cam2" else 6)
    orient,translation=timed("A_parameter_unpack",lambda:(parameters[offset:offset+3],parameters[offset+3:offset+6]))
    rotation=timed("B_rotation_value",lambda:axis_angle_to_matrix(orient))
    right=timed("C_right_jacobian",lambda:so3_right_jacobian(orient))
    rotated=timed("D_rotated_point_jacobian",lambda:-torch.einsum("ij,njk,kl->nil",rotation,skew(group.local_vertices),right))
    points=timed("E_vertex_transform",lambda:group.local_vertices@rotation.T+group.rotation_center+translation)
    uv,projection=timed("F_G_erp_projection_value_jacobian",lambda:project_unified_erp_with_jacobian(points,cache.image_width,cache.image_height))
    local=timed("H_local_jacobian_assembly",lambda:torch.cat((torch.einsum("nij,njk->nik",projection,rotated),projection),-1))
    residual=(uv-group.targets).reshape(-1);j=local.reshape(-1,6)
    timed("J_group_hessian_gradient",lambda:(j.to(torch.float64).T@j.to(torch.float64),j.to(torch.float64).T@residual.to(torch.float64)))
    sections.setdefault("I_row_scale",0.0);sections.setdefault("K_frame_reduction",0.0)
    return sections
