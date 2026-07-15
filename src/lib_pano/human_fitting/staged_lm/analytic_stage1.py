"""Analytic residual and Jacobian backend for Stage 1 rigid parameters only."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any

import torch

from lib_pano.human_fitting.projection import ROTATION_MATRIX
from lib_pano.human_fitting.smplx_sequence import forward_smplx_tensor_batch

from .config import STAGE1_PER_FRAME_DIM
from .irls import compute_irls_point_weights, compute_primary_charbonnier_tensor
from .residuals import flatten_reprojection_groups, raw_reproj_metrics
from .types import NormalEquations, ReprojectionGroup, StageContext


def _skew(vectors: torch.Tensor) -> torch.Tensor:
    zeros = torch.zeros_like(vectors[..., 0])
    x, y, z = vectors.unbind(-1)
    return torch.stack((
        zeros, -z, y,
        z, zeros, -x,
        -y, x, zeros,
    ), dim=-1).reshape(vectors.shape[:-1] + (3, 3))


def axis_angle_to_matrix(axis_angle: torch.Tensor) -> torch.Tensor:
    """Stable Rodrigues map supporting exactly zero axis-angle vectors."""
    theta2 = torch.sum(axis_angle * axis_angle, dim=-1, keepdim=True)
    theta = torch.sqrt(theta2)
    small = theta2 < 1e-8
    safe_theta = torch.where(small, torch.ones_like(theta), theta)
    a_regular = torch.sin(theta) / safe_theta
    b_regular = (1.0 - torch.cos(theta)) / torch.where(small, torch.ones_like(theta2), theta2)
    a = torch.where(small, 1.0 - theta2 / 6.0 + theta2 * theta2 / 120.0, a_regular)
    b = torch.where(small, 0.5 - theta2 / 24.0 + theta2 * theta2 / 720.0, b_regular)
    phi_hat = _skew(axis_angle)
    identity = torch.eye(3, dtype=axis_angle.dtype, device=axis_angle.device)
    return identity + a[..., None] * phi_hat + b[..., None] * (phi_hat @ phi_hat)


def so3_right_jacobian(axis_angle: torch.Tensor) -> torch.Tensor:
    """Right Jacobian for additive axis-angle coordinates with stable Taylor terms."""
    theta2 = torch.sum(axis_angle * axis_angle, dim=-1, keepdim=True)
    theta = torch.sqrt(theta2)
    small = theta2 < 1e-8
    safe_theta2 = torch.where(small, torch.ones_like(theta2), theta2)
    a_regular = (1.0 - torch.cos(theta)) / safe_theta2
    b_regular = (theta - torch.sin(theta)) / torch.where(
        small, torch.ones_like(theta2), safe_theta2 * torch.where(small, torch.ones_like(theta), theta)
    )
    a = torch.where(small, 0.5 - theta2 / 24.0 + theta2 * theta2 / 720.0, a_regular)
    b = torch.where(small, 1.0 / 6.0 - theta2 / 120.0 + theta2 * theta2 / 5040.0, b_regular)
    phi_hat = _skew(axis_angle)
    identity = torch.eye(3, dtype=axis_angle.dtype, device=axis_angle.device)
    return identity - a[..., None] * phi_hat + b[..., None] * (phi_hat @ phi_hat)


def rotated_point_jacobian(axis_angle: torch.Tensor, local_points: torch.Tensor) -> torch.Tensor:
    """Return ``d(R(phi)p)/d(phi)`` for direct additive axis-angle coordinates."""
    rotation = axis_angle_to_matrix(axis_angle.reshape(3))
    right_jacobian = so3_right_jacobian(axis_angle.reshape(3))
    return -torch.einsum("ij,njk,kl->nil", rotation, _skew(local_points), right_jacobian)


def _erp_projection_values(
    points: torch.Tensor,
    image_width: int,
    image_height: int,
) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
    """Shared ERP values used by both the UV-only and derivative paths."""
    x, y, z = points.unbind(-1)
    horizontal2 = x * x + z * z
    horizontal = torch.sqrt(horizontal2.clamp_min(1e-12))
    radius2 = horizontal2 + y * y
    longitude = torch.atan2(x, z)
    # Match production ``project_erp`` operation order for FP32 parity.
    radius = torch.norm(points, dim=-1).clamp_min(1e-6)
    latitude_argument = (-y / radius).clamp(-1.0 + 1e-6, 1.0 - 1e-6)
    latitude = torch.asin(latitude_argument)
    u = (longitude / (2.0 * math.pi) + 0.5) * image_width
    v = (0.5 - latitude / math.pi) * image_height
    return torch.stack((u, v), dim=-1), (x, y, z, horizontal2, horizontal, radius2, radius)


def erp_projection(points: torch.Tensor, image_width: int, image_height: int) -> torch.Tensor:
    """ERP UV-only path; it never constructs a projection Jacobian."""
    uv, _ = _erp_projection_values(points, image_width, image_height)
    return uv


def erp_projection_with_jacobian(
    points: torch.Tensor,
    image_width: int,
    image_height: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Project ERP camera-space points and return batched ``d(uv)/d(xyz)``."""
    uv, values = _erp_projection_values(points, image_width, image_height)
    x, y, z, horizontal2, horizontal, radius2, radius = values
    safe_radius2 = radius2.clamp_min(1e-12)

    safe_horizontal2 = horizontal2.clamp_min(1e-12)
    du_scale = image_width / (2.0 * math.pi)
    dv_scale = image_height / math.pi
    du = torch.stack((
        du_scale * z / safe_horizontal2,
        torch.zeros_like(x),
        -du_scale * x / safe_horizontal2,
    ), dim=-1)
    dv = torch.stack((
        -dv_scale * y * x / (horizontal * safe_radius2),
        dv_scale * horizontal / safe_radius2,
        -dv_scale * y * z / (horizontal * safe_radius2),
    ), dim=-1)
    latitude_active = (
        (radius2 > 1e-12)
        & ((-y / radius) > -1.0 + 1e-6)
        & ((-y / radius) < 1.0 - 1e-6)
    )
    dv = torch.where(latitude_active.unsqueeze(-1), dv, torch.zeros_like(dv))
    return uv, torch.stack((du, dv), dim=-2)


def project_cam2_erp(points: torch.Tensor, image_width: int, image_height: int) -> torch.Tensor:
    """Apply the fixed cam2 rotation and compute UV without derivative work."""
    fixed_rotation = ROTATION_MATRIX.to(device=points.device, dtype=points.dtype)
    return erp_projection(points @ fixed_rotation.T, image_width, image_height)


def project_cam2_erp_with_jacobian(
    points: torch.Tensor, image_width: int, image_height: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Match the current Stage 1 ``fitter._project_cam2`` path exactly."""
    fixed_rotation = ROTATION_MATRIX.to(device=points.device, dtype=points.dtype)
    camera_points = points @ fixed_rotation.T
    uv, jacobian_camera = erp_projection_with_jacobian(camera_points, image_width, image_height)
    return uv, torch.einsum("nij,jk->nik", jacobian_camera, fixed_rotation)


# Explicit compatibility alias for existing diagnostics.
project_cam2_with_jacobian = project_cam2_erp_with_jacobian


def metric_close_with_absolute_or_relative(
    actual: float, expected: float, *, relative_tolerance: float = 1e-5,
    absolute_tolerance: float = 1e-3,
) -> bool:
    absolute_difference = abs(actual - expected)
    relative_difference = absolute_difference / max(abs(expected), 1e-16)
    return absolute_difference <= absolute_tolerance or relative_difference <= relative_tolerance


@dataclass
class Stage1AnalyticGroupCache:
    frame_position: int
    frame_index: int
    camera: str
    local_vertices: torch.Tensor
    rotation_center: torch.Tensor
    targets: torch.Tensor


@dataclass
class Stage1AnalyticCache:
    groups: list[Stage1AnalyticGroupCache]
    frame_count: int
    image_width: int
    image_height: int
    setup_ms: float
    profile_timing: bool = False
    residual_ms: float = 0.0
    jacobian_ms: float = 0.0
    normal_equations_ms: float = 0.0
    objective_ms: float = 0.0
    metrics_ms: float = 0.0
    residual_calls: int = 0
    jacobian_calls: int = 0
    normal_equations_calls: int = 0
    objective_calls: int = 0
    metrics_calls: int = 0

    def reset_timing(self) -> None:
        self.residual_ms = 0.0
        self.jacobian_ms = 0.0
        self.normal_equations_ms = 0.0
        self.objective_ms = 0.0
        self.metrics_ms = 0.0
        self.residual_calls = 0
        self.jacobian_calls = 0
        self.normal_equations_calls = 0
        self.objective_calls = 0
        self.metrics_calls = 0

    @staticmethod
    def _per_call(total_ms: float, calls: int) -> float:
        return total_ms / calls if calls else 0.0

    def timing_dict(self) -> dict[str, float]:
        return {
            "analytic_cache_setup_ms": self.setup_ms,
            "analytic_residual_ms": self.residual_ms,
            "analytic_residual_calls": self.residual_calls,
            "analytic_residual_ms_per_call": self._per_call(self.residual_ms, self.residual_calls),
            "analytic_jacobian_ms": self.jacobian_ms,
            "analytic_jacobian_calls": self.jacobian_calls,
            "analytic_jacobian_ms_per_call": self._per_call(self.jacobian_ms, self.jacobian_calls),
            "analytic_normal_equations_ms": self.normal_equations_ms,
            "analytic_normal_equations_calls": self.normal_equations_calls,
            "analytic_normal_equations_ms_per_call": self._per_call(
                self.normal_equations_ms, self.normal_equations_calls,
            ),
            "analytic_objective_ms": self.objective_ms,
            "analytic_objective_calls": self.objective_calls,
            "analytic_objective_ms_per_call": self._per_call(self.objective_ms, self.objective_calls),
            "analytic_metrics_ms": self.metrics_ms,
            "analytic_metrics_calls": self.metrics_calls,
            "analytic_metrics_ms_per_call": self._per_call(self.metrics_ms, self.metrics_calls),
        }

    def raw_residual(self, parameters: torch.Tensor) -> torch.Tensor:
        return stage1_raw_residual(self, parameters)

    def raw_groups(self, parameters: torch.Tensor) -> list[ReprojectionGroup]:
        return stage1_raw_groups(self, parameters)

    def explicit_raw_jacobian(self, parameters: torch.Tensor) -> torch.Tensor:
        return stage1_raw_jacobian(self, parameters)

    def raw_normal_equations(
        self, parameters: torch.Tensor, row_scales: torch.Tensor,
    ) -> NormalEquations:
        return stage1_normal_equations(self, parameters, row_scales)

    def true_objective(
        self,
        parameters: torch.Tensor,
        objective_mode: str,
        charbonnier_eps: float,
        rep_scale: float,
        aggregation_spec: dict[str, Any] | None,
    ) -> torch.Tensor:
        _sync(parameters, self.profile_timing)
        start = time.perf_counter()
        groups = stage1_raw_groups(self, parameters)
        if objective_mode == "charbonnier_irls":
            if aggregation_spec is None:
                raise ValueError("aggregation_spec is required for Charbonnier objective")
            objective = compute_primary_charbonnier_tensor(
                groups, charbonnier_eps, aggregation_spec,
            )
        elif objective_mode == "raw_l2":
            residual = flatten_reprojection_groups(groups) * rep_scale
            objective = 0.5 * torch.sum(residual ** 2)
        else:
            raise ValueError(f"Unsupported objective mode: {objective_mode}")
        _sync(parameters, self.profile_timing)
        self.objective_ms += (time.perf_counter() - start) * 1000.0
        self.objective_calls += 1
        return objective

    def raw_metrics(self, parameters: torch.Tensor) -> dict[str, float]:
        _sync(parameters, self.profile_timing)
        start = time.perf_counter()
        residual = flatten_reprojection_groups(stage1_raw_groups(self, parameters)).detach()
        metrics = raw_reproj_metrics(residual)
        _sync(parameters, self.profile_timing)
        self.metrics_ms += (time.perf_counter() - start) * 1000.0
        self.metrics_calls += 1
        return metrics


def _sync(tensor: torch.Tensor, enabled: bool) -> None:
    if enabled and tensor.is_cuda:
        torch.cuda.synchronize(tensor.device)


def build_stage1_analytic_cache(context: StageContext) -> Stage1AnalyticCache:
    """Cache selected, scaled vertices before global orient and translation."""
    reference = context.beta_anchor
    _sync(reference, context.args.profile_timing)
    start = time.perf_counter()
    groups: list[Stage1AnalyticGroupCache] = []
    zero_orient = torch.zeros(1, 3, device=reference.device, dtype=reference.dtype)
    zero_transl = torch.zeros_like(zero_orient)
    with torch.no_grad():
        for position, frame_index in enumerate(context.frame_indices):
            frozen = context.frozen_params_by_frame[position]
            body_pose = context.body_pose_anchors[position].to(reference).reshape(1, -1)
            camera_inputs = (
                ("cam2", frozen.cam2_left_hand_pose, frozen.cam2_right_hand_pose,
                 frozen.cam2_scale, context.cam2_vertex_ids_by_frame[position], context.cam2_targets_by_frame[position]),
                ("cam4", frozen.cam4_left_hand_pose, frozen.cam4_right_hand_pose,
                 frozen.cam4_scale, context.cam4_vertex_ids_by_frame[position], context.cam4_targets_by_frame[position]),
            )
            for camera, left_hand, right_hand, scale, vertex_ids, targets in camera_inputs:
                vertices, _, joints_body = forward_smplx_tensor_batch(
                    context.model, zero_orient, body_pose,
                    left_hand.to(reference), right_hand.to(reference),
                    context.beta_anchor.reshape(1, -1), zero_transl,
                    scale.expand(1).to(reference),
                )
                ids = vertex_ids.to(device=vertices.device)
                rotation_center = joints_body[0, 0].detach().clone()
                local_vertices = (
                    vertices.squeeze(0).index_select(0, ids) - rotation_center
                ).detach().clone()
                groups.append(Stage1AnalyticGroupCache(
                    frame_position=position,
                    frame_index=int(frame_index),
                    camera=camera,
                    local_vertices=local_vertices,
                    rotation_center=rotation_center,
                    targets=targets.to(reference).detach().clone(),
                ))
    _sync(reference, context.args.profile_timing)
    return Stage1AnalyticCache(
        groups=groups, frame_count=context.frame_count,
        image_width=context.image_width, image_height=context.image_height,
        setup_ms=(time.perf_counter() - start) * 1000.0,
        profile_timing=context.args.profile_timing,
    )


def _group_transform(parameters: torch.Tensor, group: Stage1AnalyticGroupCache) -> tuple[torch.Tensor, torch.Tensor]:
    offset = group.frame_position * STAGE1_PER_FRAME_DIM + (0 if group.camera == "cam2" else 6)
    orient = parameters[offset:offset + 3]
    translation = parameters[offset + 3:offset + 6]
    local = group.local_vertices.to(parameters)
    rotation = axis_angle_to_matrix(orient)
    points = local @ rotation.T + group.rotation_center.to(parameters) + translation
    return points, orient


def stage1_raw_groups(cache: Stage1AnalyticCache, parameters: torch.Tensor) -> list[ReprojectionGroup]:
    groups = []
    for cached in cache.groups:
        points, _ = _group_transform(parameters, cached)
        uv = project_cam2_erp(points, cache.image_width, cache.image_height)
        groups.append(ReprojectionGroup(
            frame_index=cached.frame_index, camera=cached.camera,
            residual_2d=uv - cached.targets.to(parameters),
        ))
    return groups


def stage1_raw_residual(cache: Stage1AnalyticCache, parameters: torch.Tensor) -> torch.Tensor:
    _sync(parameters, cache.profile_timing)
    start = time.perf_counter()
    residual = torch.cat([group.residual_2d.reshape(-1) for group in stage1_raw_groups(cache, parameters)])
    _sync(parameters, cache.profile_timing)
    cache.residual_ms += (time.perf_counter() - start) * 1000.0
    cache.residual_calls += 1
    return residual


def stage1_raw_jacobian(cache: Stage1AnalyticCache, parameters: torch.Tensor) -> torch.Tensor:
    _sync(parameters, cache.profile_timing)
    start = time.perf_counter()
    row_count = sum(group.targets.shape[0] * 2 for group in cache.groups)
    jacobian = torch.zeros(
        row_count, cache.frame_count * STAGE1_PER_FRAME_DIM,
        device=parameters.device, dtype=parameters.dtype,
    )
    row = 0
    for cached in cache.groups:
        points, orient = _group_transform(parameters, cached)
        _, projection_jacobian = project_cam2_erp_with_jacobian(
            points, cache.image_width, cache.image_height,
        )
        local = cached.local_vertices.to(parameters)
        point_orient_jacobian = rotated_point_jacobian(orient, local)
        orient_jacobian = torch.einsum("nij,njk->nik", projection_jacobian, point_orient_jacobian)
        point_count = local.shape[0]
        block = torch.cat((orient_jacobian, projection_jacobian), dim=-1).reshape(point_count * 2, 6)
        column = cached.frame_position * STAGE1_PER_FRAME_DIM + (0 if cached.camera == "cam2" else 6)
        jacobian[row:row + point_count * 2, column:column + 6] = block
        row += point_count * 2
    _sync(parameters, cache.profile_timing)
    cache.jacobian_ms += (time.perf_counter() - start) * 1000.0
    cache.jacobian_calls += 1
    return jacobian


def _group_raw_residual_and_local_jacobian(
    cache: Stage1AnalyticCache,
    parameters: torch.Tensor,
    cached: Stage1AnalyticGroupCache,
) -> tuple[torch.Tensor, torch.Tensor]:
    points, orient = _group_transform(parameters, cached)
    uv, projection_jacobian = project_cam2_erp_with_jacobian(
        points, cache.image_width, cache.image_height,
    )
    local = cached.local_vertices.to(parameters)
    point_orient_jacobian = rotated_point_jacobian(orient, local)
    orient_jacobian = torch.einsum("nij,njk->nik", projection_jacobian, point_orient_jacobian)
    residual = (uv - cached.targets.to(parameters)).reshape(-1)
    camera_jacobian = torch.cat((orient_jacobian, projection_jacobian), dim=-1).reshape(-1, 6)
    local_jacobian = torch.zeros(
        residual.numel(), STAGE1_PER_FRAME_DIM,
        dtype=parameters.dtype, device=parameters.device,
    )
    column = 0 if cached.camera == "cam2" else 6
    local_jacobian[:, column:column + 6] = camera_jacobian
    return residual, local_jacobian


def stage1_normal_equations(
    cache: Stage1AnalyticCache,
    parameters: torch.Tensor,
    row_scales: torch.Tensor,
) -> NormalEquations:
    """Accumulate per-frame 12x12 systems without constructing the global J."""
    _sync(parameters, cache.profile_timing)
    start = time.perf_counter()
    parameter_count = cache.frame_count * STAGE1_PER_FRAME_DIM
    hessian = torch.zeros(
        parameter_count, parameter_count, dtype=torch.float64, device=parameters.device,
    )
    gradient = torch.zeros(parameter_count, dtype=torch.float64, device=parameters.device)
    residual_parts: list[torch.Tensor] = []
    frame_residuals: list[list[torch.Tensor]] = [[] for _ in range(cache.frame_count)]
    frame_jacobians: list[list[torch.Tensor]] = [[] for _ in range(cache.frame_count)]
    row = 0
    for cached in cache.groups:
        raw_residual, local_jacobian = _group_raw_residual_and_local_jacobian(cache, parameters, cached)
        rows = raw_residual.numel()
        scales = row_scales[row:row + rows].to(parameters)
        residual = raw_residual * scales
        local_jacobian = local_jacobian * scales.unsqueeze(1)
        frame_residuals[cached.frame_position].append(residual)
        frame_jacobians[cached.frame_position].append(local_jacobian)
        residual_parts.append(residual)
        row += rows
    for frame_position in range(cache.frame_count):
        frame_residual_64 = torch.cat(frame_residuals[frame_position]).to(torch.float64)
        frame_jacobian_64 = torch.cat(frame_jacobians[frame_position]).to(torch.float64)
        block = slice(
            frame_position * STAGE1_PER_FRAME_DIM,
            (frame_position + 1) * STAGE1_PER_FRAME_DIM,
        )
        hessian[block, block] = frame_jacobian_64.T @ frame_jacobian_64
        gradient[block] = frame_jacobian_64.T @ frame_residual_64
    result = NormalEquations(torch.cat(residual_parts), hessian, gradient)
    _sync(parameters, cache.profile_timing)
    cache.normal_equations_ms += (time.perf_counter() - start) * 1000.0
    cache.normal_equations_calls += 1
    return result


def build_stage1_analytic_linearization(
    cache: Stage1AnalyticCache,
    linearisation_point: torch.Tensor,
    *,
    is_irls: bool,
    charbonnier_eps: float,
    aggregation_spec: dict[str, Any] | None,
    rep_scale: float,
) -> dict[str, Any]:
    """Build frozen-weight analytic Stage 1 surrogate residual and Jacobian."""
    if is_irls:
        groups = stage1_raw_groups(cache, linearisation_point)
        weights = compute_irls_point_weights(groups, charbonnier_eps, aggregation_spec)
        point_scales = torch.cat([torch.sqrt(weight).repeat_interleave(2) for weight in weights]).detach()
        flat_weights = torch.cat(weights)
        weight_stats = {
            "min": float(flat_weights.min()), "mean": float(flat_weights.mean()),
            "median": float(flat_weights.median()), "max": float(flat_weights.max()),
            "n_points": int(flat_weights.numel()),
        }
    else:
        row_count = sum(group.targets.shape[0] * 2 for group in cache.groups)
        point_scales = torch.full(
            (row_count,), rep_scale, dtype=linearisation_point.dtype,
            device=linearisation_point.device,
        )
        weight_stats = {}

    def surrogate_fn(parameters: torch.Tensor) -> torch.Tensor:
        return stage1_raw_residual(cache, parameters) * point_scales.to(parameters)

    def jacobian_fn(parameters: torch.Tensor) -> torch.Tensor:
        return stage1_raw_jacobian(cache, parameters) * point_scales.to(parameters).unsqueeze(1)

    def normal_equations_fn(parameters: torch.Tensor) -> NormalEquations:
        return stage1_normal_equations(cache, parameters, point_scales)

    return {
        "surrogate_fn": surrogate_fn,
        "jacobian_fn": jacobian_fn,
        "normal_equations_fn": normal_equations_fn,
        "weight_stats": weight_stats,
        "spec": aggregation_spec or {},
        "jacobian_backend": "analytic_stage1",
        "linearization_backend": "direct_normal_equations",
    }
