"""Production FULL_BATCH normal equations for Stage 2 Joint pose+beta.

Serial IRLS weights, true objective, metrics, LM and Arrowhead Schur remain
on their existing production paths.
"""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any

import torch

from .analytic_stage1 import project_cam2_erp_with_jacobian
from .arrowhead_schur import ArrowheadNormalEquationSystem
from .stage2_joint_fast import (
    Stage2JointFastCache, Stage2JointShapeState, evaluate_joint_shape_state,
    physical_beta_from_latent,
)
from .stage2_pose_batched import stage2_pose_batched_rotations_and_jacobians
from .types import NormalEquations

@dataclass(frozen=True)
class Stage2JointBatchLayout:
    frame_positions: torch.Tensor
    camera_indices: torch.Tensor
    point_counts: torch.Tensor
    valid_mask: torch.Tensor
    targets: torch.Tensor
    frame_positions_host: tuple[int, ...]
    camera_indices_host: tuple[int, ...]
    point_counts_host: tuple[int, ...]
    group_count: int
    frame_count: int
    nmax: int
    beta_dim: int
    padding_ratio: float
    padded_bytes: int


@dataclass(frozen=True)
class Stage2JointBatchParameters:
    group_orientations: torch.Tensor
    group_translations: torch.Tensor
    group_body_poses: torch.Tensor
    frame_body_poses: torch.Tensor
    beta_latent: torch.Tensor


@dataclass
class Stage2JointBatchCache:
    layout: Stage2JointBatchLayout
    selected_base_v_shaped: torch.Tensor
    selected_beta_shapedirs: torch.Tensor
    selected_posedirs: torch.Tensor
    selected_lbs_weights: torch.Tensor
    left_hand_pose: torch.Tensor
    right_hand_pose: torch.Tensor
    scale: torch.Tensor
    parents: torch.Tensor
    derivative_mapping: torch.Tensor
    fk_depth_indices: tuple[torch.Tensor, ...]
    active_indices_cam2: torch.Tensor
    active_indices_cam4: torch.Tensor
    pose_anchor_identity_64: torch.Tensor
    setup_ms: float
    padding_layout_ms: float
    padding_base_vertices_ms: float
    padding_beta_dirs_ms: float
    padding_posedirs_ms: float
    padding_weights_ms: float
    stack_hands_scale_ms: float
    static_layout_ms: float
    persistent_bytes: int


@dataclass
class Stage2JointBatchAnalyticResult:
    residual: torch.Tensor
    active_pose_jacobian: torch.Tensor
    beta_jacobian: torch.Tensor
    rotations: torch.Tensor | None = None
    d_rotations: torch.Tensor | None = None
    relative_transforms: torch.Tensor | None = None
    d_relative_transforms: torch.Tensor | None = None
    vertices: torch.Tensor | None = None
    d_vertices_pose: torch.Tensor | None = None
    d_vertices_beta: torch.Tensor | None = None


def _bytes(tensor: torch.Tensor) -> int:
    return tensor.numel() * tensor.element_size()


def _depth_indices(parents: torch.Tensor) -> tuple[torch.Tensor, ...]:
    parents_host = tuple(int(value) for value in parents.detach().cpu().tolist())
    depths = [0] * len(parents_host)
    for joint in range(1, len(parents_host)):
        depths[joint] = depths[parents_host[joint]] + 1
    return tuple(torch.tensor(
        [joint for joint, value in enumerate(depths) if value == depth],
        dtype=torch.long, device=parents.device,
    ) for depth in range(max(depths) + 1))


def build_stage2_joint_batch_cache(
    cache: Stage2JointFastCache, *, pose_anchor_scale: float,
) -> Stage2JointBatchCache:
    if not cache.groups:
        raise ValueError("EMPTY_JOINT_GROUP_BATCH")
    started = time.perf_counter()
    groups = cache.groups
    device = groups[0].selected_base_v_shaped.device
    dtype = groups[0].selected_base_v_shaped.dtype
    counts = tuple(int(group.vertex_ids.numel()) for group in groups)
    frames = tuple(int(group.frame_position) for group in groups)
    cameras = tuple(0 if group.camera == "cam2" else 1 for group in groups)
    group_count, nmax = len(groups), max(counts)
    beta_dim = cache.model.beta_dim

    section = time.perf_counter()
    valid = torch.zeros(group_count, nmax, dtype=torch.bool, device=device)
    targets = torch.zeros(group_count, nmax, 2, dtype=dtype, device=device)
    for index, (group, count) in enumerate(zip(groups, counts)):
        valid[index, :count] = True
        targets[index, :count] = group.targets
    frame_tensor = torch.tensor(frames, dtype=torch.long, device=device)
    camera_tensor = torch.tensor(cameras, dtype=torch.long, device=device)
    count_tensor = torch.tensor(counts, dtype=torch.long, device=device)
    layout_ms = (time.perf_counter() - section) * 1000

    section = time.perf_counter()
    base = torch.zeros(group_count, nmax, 3, dtype=dtype, device=device)
    for i, (group, count) in enumerate(zip(groups, counts)): base[i, :count] = group.selected_base_v_shaped
    base_ms = (time.perf_counter() - section) * 1000
    section = time.perf_counter()
    beta_dirs = torch.zeros(group_count, nmax, 3, beta_dim, dtype=dtype, device=device)
    for i, (group, count) in enumerate(zip(groups, counts)): beta_dirs[i, :count] = group.selected_beta_shapedirs
    beta_ms = (time.perf_counter() - section) * 1000
    pose_dim = groups[0].selected_posedirs.shape[0]
    section = time.perf_counter()
    posedirs = torch.zeros(group_count, pose_dim, nmax, 3, dtype=dtype, device=device)
    for i, (group, count) in enumerate(zip(groups, counts)): posedirs[i, :, :count] = group.selected_posedirs
    posedirs_ms = (time.perf_counter() - section) * 1000
    joint_count = groups[0].selected_lbs_weights.shape[-1]
    section = time.perf_counter()
    weights = torch.zeros(group_count, nmax, joint_count, dtype=dtype, device=device)
    for i, (group, count) in enumerate(zip(groups, counts)): weights[i, :count] = group.selected_lbs_weights
    weights_ms = (time.perf_counter() - section) * 1000
    section = time.perf_counter()
    left = torch.stack([group.left_hand_pose.reshape(-1) for group in groups])
    right = torch.stack([group.right_hand_pose.reshape(-1) for group in groups])
    scale = torch.stack([group.scale.reshape(-1) for group in groups])
    stack_ms = (time.perf_counter() - section) * 1000

    section = time.perf_counter()
    mapping = torch.zeros(22, 66, 3, dtype=dtype, device=device)
    joints = torch.arange(22, device=device)
    components = torch.arange(3, device=device)
    mapping[joints[:, None], joints[:, None] * 3 + components[None], components[None]] = 1
    depth = _depth_indices(cache.model.parents)
    cam2 = torch.cat((torch.arange(0, 6, device=device), torch.arange(12, 75, device=device)))
    cam4 = torch.cat((torch.arange(6, 12, device=device), torch.arange(12, 75, device=device)))
    anchor_eye = torch.eye(63, dtype=torch.float64, device=device) * float(pose_anchor_scale) ** 2
    static_ms = (time.perf_counter() - section) * 1000
    layout_tensors = (frame_tensor, camera_tensor, count_tensor, valid, targets)
    persistent = (*layout_tensors, base, beta_dirs, posedirs, weights, left, right, scale,
                  mapping, cam2, cam4, anchor_eye, *depth)
    padded_points = group_count * nmax
    layout = Stage2JointBatchLayout(
        frame_tensor, camera_tensor, count_tensor, valid, targets,
        frames, cameras, counts, group_count, cache.frame_count, nmax, beta_dim,
        1.0 - sum(counts) / max(padded_points, 1), sum(_bytes(x) for x in layout_tensors),
    )
    return Stage2JointBatchCache(
        layout, base, beta_dirs, posedirs, weights, left, right, scale,
        cache.model.parents, mapping, depth, cam2, cam4, anchor_eye,
        (time.perf_counter() - started) * 1000, layout_ms, base_ms, beta_ms,
        posedirs_ms, weights_ms, stack_ms, static_ms,
        sum(_bytes(x) for x in persistent),
    )


def gather_stage2_joint_batch_parameters(
    parameters: torch.Tensor, layout: Stage2JointBatchLayout, beta_dim: int,
) -> Stage2JointBatchParameters:
    framed = parameters[:-beta_dim].reshape(layout.frame_count, 75)
    grouped = framed.index_select(0, layout.frame_positions)
    offsets = layout.camera_indices * 6
    xyz = torch.arange(3, device=parameters.device)
    frame_body = framed[:, 12:75]
    return Stage2JointBatchParameters(
        grouped.gather(1, offsets[:, None] + xyz[None]),
        grouped.gather(1, offsets[:, None] + 3 + xyz[None]),
        frame_body.index_select(0, layout.frame_positions), frame_body,
        parameters[-beta_dim:],
    )


def stage2_joint_batched_dynamic_shape(
    batch: Stage2JointBatchCache, shape: Stage2JointShapeState,
) -> tuple[torch.Tensor, torch.Tensor]:
    value = batch.selected_base_v_shaped + torch.einsum(
        "gncb,b->gnc", batch.selected_beta_shapedirs, shape.beta,
    )
    derivative = batch.selected_beta_shapedirs * shape.d_beta_d_latent[None, None, None, :]
    return value, derivative


def stage2_joint_batched_combined_fk_by_depth(
    rotations: torch.Tensor, d_rotations: torch.Tensor, joints: torch.Tensor,
    d_joints: torch.Tensor, parents: torch.Tensor,
    depth_indices: tuple[torch.Tensor, ...],
) -> tuple[torch.Tensor, torch.Tensor]:
    group_count, joint_count = rotations.shape[:2]
    pose_dim, beta_dim = d_rotations.shape[-1], d_joints.shape[-1]
    joints_g = joints[None].expand(group_count, -1, -1)
    dj_g = d_joints[None].expand(group_count, -1, -1, -1)
    relative = joints_g.clone(); relative[:, 1:] -= joints_g.index_select(1, parents[1:])
    d_relative = dj_g.clone(); d_relative[:, 1:] -= dj_g.index_select(1, parents[1:])
    local = torch.zeros(group_count, joint_count, 4, 4, dtype=rotations.dtype, device=rotations.device)
    local[:, :, :3, :3] = rotations; local[:, :, :3, 3] = relative; local[:, :, 3, 3] = 1
    d_local = torch.zeros(group_count, joint_count, 4, 4, pose_dim + beta_dim,
                          dtype=rotations.dtype, device=rotations.device)
    d_local[:, :, :3, :3, :pose_dim] = d_rotations
    d_local[:, :, :3, 3, pose_dim:] = d_relative
    global_ = torch.zeros_like(local); d_global = torch.zeros_like(d_local)
    global_[:, 0] = local[:, 0]; d_global[:, 0] = d_local[:, 0]
    for indices in depth_indices[1:]:
        pi = parents.index_select(0, indices)
        pg, pd = global_.index_select(1, pi), d_global.index_select(1, pi)
        child, d_child = local.index_select(1, indices), d_local.index_select(1, indices)
        global_[:, indices] = pg @ child
        d_global[:, indices] = (torch.einsum("gjabq,gjbc->gjacq", pd, child)
                                + torch.einsum("gjab,gjbcq->gjacq", pg, d_child))
    rest = torch.eye(4, dtype=rotations.dtype, device=rotations.device)[None, None].repeat(group_count, joint_count, 1, 1)
    rest[:, :, :3, 3] = -joints_g
    d_rest = torch.zeros_like(d_global); d_rest[:, :, :3, 3, pose_dim:] = -dj_g
    return global_ @ rest, (torch.einsum("gjabq,gjbc->gjacq", d_global, rest)
                            + torch.einsum("gjab,gjbcq->gjacq", global_, d_rest))


def stage2_joint_batched_group_analytic(
    cache: Stage2JointFastCache, batch: Stage2JointBatchCache,
    parameters: torch.Tensor, beta_abs_limit: float, *,
    return_intermediates: bool = False,
) -> Stage2JointBatchAnalyticResult:
    packed = gather_stage2_joint_batch_parameters(parameters, batch.layout, batch.layout.beta_dim)
    cache.batched_shape_state_calls += 1
    cache.batched_full_analytic_calls += 1
    shape = evaluate_joint_shape_state(cache, packed.beta_latent, beta_abs_limit)
    rotations, d_rotations = stage2_pose_batched_rotations_and_jacobians(
        packed.group_orientations, packed.group_body_poses, batch.left_hand_pose,
        batch.right_hand_pose, batch.derivative_mapping,
    )
    identity = torch.eye(3, dtype=rotations.dtype, device=rotations.device)
    feature = (rotations[:, 1:] - identity).reshape(batch.layout.group_count, -1)
    d_feature = d_rotations[:, 1:].reshape(batch.layout.group_count, -1, 66)
    offsets = torch.einsum("gp,gpnc->gnc", feature, batch.selected_posedirs)
    d_offsets = torch.einsum("gpq,gpnc->gncq", d_feature, batch.selected_posedirs)
    shaped, d_beta_shape = stage2_joint_batched_dynamic_shape(batch, shape)
    relative, d_relative = stage2_joint_batched_combined_fk_by_depth(
        rotations, d_rotations, shape.joints, shape.d_joints_d_latent,
        batch.parents, batch.fk_depth_indices,
    )
    posed = shaped + offsets
    d_posed = torch.cat((d_offsets, d_beta_shape), -1)
    homogeneous = torch.cat((posed, torch.ones_like(posed[..., :1])), -1)
    d_homogeneous = torch.cat((d_posed, torch.zeros_like(d_posed[..., :1, :])), -2)
    transforms = torch.einsum("gnj,gjab->gnab", batch.selected_lbs_weights, relative[:, :, :3])
    vertices = torch.einsum("gnab,gnb->gna", transforms, homogeneous)
    d_vertices = (torch.einsum("gnj,gjabq,gnb->gnaq", batch.selected_lbs_weights,
                               d_relative[:, :, :3], homogeneous)
                  + torch.einsum("gnab,gnbq->gnaq", transforms, d_homogeneous))
    scale = batch.scale.reshape(-1, 1, 1)
    vertices = vertices * scale + packed.group_translations[:, None]
    d_vertices = d_vertices * scale[..., None]
    g, n = vertices.shape[:2]
    uv, projection = project_cam2_erp_with_jacobian(vertices.reshape(g*n, 3), cache.image_width, cache.image_height)
    uv = uv.reshape(g, n, 2); projection = projection.reshape(g, n, 2, 3)
    uv_all = torch.einsum("gnab,gnbq->gnaq", projection, d_vertices)
    active = torch.cat((uv_all[..., :3], projection, uv_all[..., 3:66]), -1)
    beta_j = uv_all[..., 66:]
    mask = batch.layout.valid_mask[..., None]
    residual = (uv - batch.layout.targets) * mask
    active = active * mask[..., None]; beta_j = beta_j * mask[..., None]
    return Stage2JointBatchAnalyticResult(
        residual, active, beta_j,
        rotations if return_intermediates else None,
        d_rotations if return_intermediates else None,
        relative if return_intermediates else None,
        d_relative if return_intermediates else None,
        vertices if return_intermediates else None,
        d_vertices[..., :66] if return_intermediates else None,
        d_vertices[..., 66:] if return_intermediates else None,
    )


def pad_stage2_joint_row_scales(batch: Stage2JointBatchCache, scales: list[torch.Tensor]) -> torch.Tensor:
    output = torch.zeros(batch.layout.group_count, batch.layout.nmax, 2,
                         dtype=scales[0].dtype, device=scales[0].device)
    for index, (count, scale) in enumerate(zip(batch.layout.point_counts_host, scales)):
        output[index, :count] = scale.reshape(count, 2)
    return output


def _batched_gemm(result: Stage2JointBatchAnalyticResult, scales: torch.Tensor):
    g, n = result.residual.shape[:2]
    r = (result.residual * scales).reshape(g, 2*n).to(torch.float64)
    ja = (result.active_pose_jacobian * scales[..., None]).reshape(g, 2*n, 69).to(torch.float64)
    jb = (result.beta_jacobian * scales[..., None]).reshape(g, 2*n, result.beta_jacobian.shape[-1]).to(torch.float64)
    return (r, ja, jb, torch.bmm(ja.transpose(1, 2), ja),
            torch.bmm(ja.transpose(1, 2), jb), torch.bmm(jb.transpose(1, 2), jb),
            torch.bmm(ja.transpose(1, 2), r[..., None]).squeeze(-1),
            torch.bmm(jb.transpose(1, 2), r[..., None]).squeeze(-1))


def stage2_joint_batched_accumulate_arrowhead(
    cache: Stage2JointFastCache, batch: Stage2JointBatchCache, parameters: torch.Tensor,
    result: Stage2JointBatchAnalyticResult, scales: torch.Tensor,
    body_pose_anchors: list[torch.Tensor], beta_anchor: torch.Tensor,
    beta_abs_limit: float, pose_anchor_scale: float, shape_anchor_scale: float,
) -> NormalEquations:
    g, n, b = batch.layout.group_count, batch.layout.nmax, batch.layout.beta_dim
    r, _, _, ha, hb, hc, ga, gb = _batched_gemm(result, scales)
    contributions = [
        (r[i, :2 * batch.layout.point_counts_host[i]], ha[i], hb[i], hc[i], ga[i], gb[i])
        for i in range(g)
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
        c += hc; gbeta += gb
        residuals.append(r.to(parameters.dtype))
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
    c += torch.diag(diagonal.square())
    gbeta += diagonal * shape_residual.to(torch.float64)
    system = ArrowheadNormalEquationSystem(
        a, cross, c, gf, gbeta, timing_sink=cache,
    )
    residual = torch.cat((*residuals, *pose_residuals, shape_residual))
    return NormalEquations(
        residual, None, system.flat_gradient(), structured_system=system,
    )


def stage2_joint_batched_normal_equations(
    fast_cache: Stage2JointFastCache, batch_cache: Stage2JointBatchCache,
    parameters: torch.Tensor, beta_abs_limit: float, padded_row_scales: torch.Tensor,
    body_pose_anchors: list[torch.Tensor], beta_anchor: torch.Tensor,
    pose_anchor_scale: float, shape_anchor_scale: float,
) -> NormalEquations:
    if fast_cache.profile_timing and parameters.is_cuda:
        torch.cuda.synchronize(parameters.device)
    started = time.perf_counter()
    analytic = stage2_joint_batched_group_analytic(
        fast_cache, batch_cache, parameters, beta_abs_limit,
    )
    result = stage2_joint_batched_accumulate_arrowhead(
        fast_cache, batch_cache, parameters, analytic, padded_row_scales,
        body_pose_anchors, beta_anchor, beta_abs_limit, pose_anchor_scale,
        shape_anchor_scale,
    )
    if fast_cache.profile_timing and parameters.is_cuda:
        torch.cuda.synchronize(parameters.device)
    elapsed = (time.perf_counter() - started) * 1000.0
    fast_cache.normal_equations_ms += elapsed
    fast_cache.normal_equations_calls += 1
    fast_cache.batched_normal_equations_ms += elapsed
    fast_cache.batched_normal_equations_calls += 1
    fast_cache.arrowhead_accumulation_ms += elapsed
    fast_cache.arrowhead_accumulation_calls += 1
    return result


def build_stage2_joint_batched_linearization(
    fast_cache: Stage2JointFastCache, batch_cache: Stage2JointBatchCache,
    linearisation_point: torch.Tensor, beta_abs_limit: float,
    beta_anchor: torch.Tensor, body_pose_anchors: list[torch.Tensor],
    pose_anchor_scale: float, shape_anchor_scale: float, *, is_irls: bool,
    charbonnier_eps: float, aggregation_spec: dict[str, Any] | None,
    rep_scale: float,
) -> dict[str, Any]:
    from .stage2_joint_jacobian import build_stage2_joint_linearization_common
    common = build_stage2_joint_linearization_common(
        fast_cache, linearisation_point, beta_abs_limit, beta_anchor,
        body_pose_anchors, pose_anchor_scale, shape_anchor_scale,
        is_irls=is_irls, charbonnier_eps=charbonnier_eps,
        aggregation_spec=aggregation_spec, rep_scale=rep_scale,
    )
    padded = pad_stage2_joint_row_scales(batch_cache, common.row_scales_by_group)
    return {
        "surrogate_fn": common.surrogate_fn,
        "jacobian_fn": common.jacobian_fn,
        "normal_equations_fn": lambda parameters: stage2_joint_batched_normal_equations(
            fast_cache, batch_cache, parameters, beta_abs_limit, padded,
            body_pose_anchors, beta_anchor, pose_anchor_scale, shape_anchor_scale,
        ),
        "weight_stats": common.weight_stats, "spec": common.spec,
        "jacobian_backend": "analytic_stage2_joint_batched",
        "linearization_backend": "direct_arrowhead_normal_equations_batched_full_group",
        "linear_solver_backend": "arrowhead_schur",
    }
