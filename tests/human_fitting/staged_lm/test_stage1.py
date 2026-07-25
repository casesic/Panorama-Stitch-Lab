"""Pure mathematical tests for the Stage 1 analytic Jacobian backend."""

from __future__ import annotations

import sys
import math
from pathlib import Path

import torch
from torch.func import jacfwd

try:
    from ._bootstrap import ensure_test_paths
except ImportError:
    from _bootstrap import ensure_test_paths
ensure_test_paths(__file__)

from lib_pano.human_fitting.staged_lm.analytic_stage1 import (
    Stage1AnalyticCache,
    Stage1AnalyticGroupCache,
    axis_angle_to_matrix,
    build_stage1_analytic_linearization,
    erp_projection_with_jacobian,
    project_cam2_erp,
    project_cam2_erp_with_jacobian,
    metric_close_with_absolute_or_relative,
    rotated_point_jacobian,
    stage1_raw_jacobian,
    stage1_normal_equations,
)
from lib_pano.human_fitting.staged_lm.irls import build_primary_aggregation_spec
from lib_pano.human_fitting.staged_lm.types import ReprojectionGroup


def _relative_frobenius(actual: torch.Tensor, expected: torch.Tensor) -> float:
    return float(torch.linalg.norm(actual - expected) / torch.linalg.norm(expected).clamp_min(1e-16))


def test_erp_projection_jacobian() -> None:
    torch.manual_seed(1)
    points = torch.randn(20, 3, dtype=torch.float64)
    points[:, 2] += 2.0
    _, analytic = erp_projection_with_jacobian(points, 5760, 2880)
    oracle = jacfwd(lambda value: erp_projection_with_jacobian(value, 5760, 2880)[0])(points)
    oracle = torch.stack([oracle[i, :, i, :] for i in range(points.shape[0])])
    assert _relative_frobenius(analytic, oracle) <= 1e-5
    uv_fast = project_cam2_erp(points, 5760, 2880)
    uv_with_jacobian, _ = project_cam2_erp_with_jacobian(points, 5760, 2880)
    torch.testing.assert_close(uv_fast, uv_with_jacobian, rtol=1e-6, atol=1e-5)
    near_pole = torch.tensor([[1e-7, 1.0, -2e-7], [2e-6, -1.0, 1e-6]], dtype=torch.float64)
    uv, jacobian = erp_projection_with_jacobian(near_pole, 5760, 2880)
    assert torch.isfinite(uv).all() and torch.isfinite(jacobian).all()
    print("test_erp_projection_jacobian: PASS")


def test_axis_angle_point_jacobian_and_small_angles() -> None:
    torch.manual_seed(2)
    points = torch.randn(9, 3, dtype=torch.float64)
    for axis_angle in (
        torch.tensor([0.3, -0.2, 0.1], dtype=torch.float64),
        torch.zeros(3, dtype=torch.float64),
        torch.tensor([1e-8, -1e-8, 0.0], dtype=torch.float64),
        torch.tensor([1e-5, 0.0, -1e-5], dtype=torch.float64),
    ):
        analytic = rotated_point_jacobian(axis_angle, points)
        oracle_raw = jacfwd(lambda phi: points @ axis_angle_to_matrix(phi).T)(axis_angle)
        assert torch.isfinite(analytic).all()
        assert _relative_frobenius(analytic, oracle_raw) <= 1e-6
    print("test_axis_angle_point_jacobian_and_small_angles: PASS")


def test_axis_angle_near_pi() -> None:
    torch.manual_seed(3)
    points = torch.randn(11, 3, dtype=torch.float64)
    axes = [
        torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64),
        torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64),
        torch.tensor([-2.0, 1.0, 0.5], dtype=torch.float64),
    ]
    for axis in axes:
        axis = axis / axis.norm()
        for offset in (-1e-4, -1e-6, 1e-6, 1e-4):
            phi = axis * (math.pi + offset)
            analytic = rotated_point_jacobian(phi, points)
            oracle = jacfwd(lambda value: points @ axis_angle_to_matrix(value).T)(phi)
            error = analytic - oracle
            rel = _relative_frobenius(analytic, oracle)
            assert torch.isfinite(analytic).all()
            assert rel <= 5e-5
            best_cd = float("inf")
            for epsilon in (1e-4, 3e-4, 1e-3, 3e-3):
                columns = []
                for column in range(3):
                    delta = torch.zeros_like(phi)
                    delta[column] = epsilon
                    columns.append(
                        ((points @ axis_angle_to_matrix(phi + delta).T)
                         - (points @ axis_angle_to_matrix(phi - delta).T)) / (2 * epsilon)
                    )
                central = torch.stack(columns, dim=-1)
                best_cd = min(best_cd, _relative_frobenius(analytic, central))
            print(
                f"near_pi norm={float(phi.norm()):.8f} rel={rel:.3e} "
                f"max_abs={float(error.abs().max()):.3e} best_cd={best_cd:.3e}"
            )
            assert best_cd <= 5e-5
    print("test_axis_angle_near_pi: PASS")


def _synthetic_cache(frame_count: int) -> Stage1AnalyticCache:
    groups = []
    for frame in range(frame_count):
        for camera_index, camera in enumerate(("cam2", "cam4")):
            local = torch.tensor([[0.2, -0.1, 2.0], [-0.3, 0.4, 1.7]], dtype=torch.float64)
            local = local + frame * 0.1 + camera_index * 0.05
            groups.append(Stage1AnalyticGroupCache(
                frame_position=frame, frame_index=10 + frame, camera=camera,
                local_vertices=local, rotation_center=torch.zeros(3, dtype=torch.float64),
                targets=torch.zeros(2, 2, dtype=torch.float64),
            ))
    return Stage1AnalyticCache(groups, frame_count, 5760, 2880, 0.0)


def test_translation_and_parameter_blocks() -> None:
    for frame_count in (1, 2):
        cache = _synthetic_cache(frame_count)
        parameters = torch.zeros(frame_count * 12, dtype=torch.float64)
        parameters[2::12] = 0.05
        jacobian = stage1_raw_jacobian(cache, parameters)
        oracle = jacfwd(lambda value: torch.cat([
            group.residual_2d.reshape(-1)
            for group in __import__(
                "lib_pano.human_fitting.staged_lm.analytic_stage1", fromlist=["stage1_raw_groups"]
            ).stage1_raw_groups(cache, value)
        ]))(parameters)
        assert _relative_frobenius(jacobian, oracle) <= 1e-6
        rows_per_group = 4
        for group_index, group in enumerate(cache.groups):
            expected_start = group.frame_position * 12 + (0 if group.camera == "cam2" else 6)
            block_rows = jacobian[group_index * rows_per_group:(group_index + 1) * rows_per_group]
            nonzero_columns = torch.nonzero(block_rows.abs().sum(0) > 0).reshape(-1).tolist()
            assert nonzero_columns == list(range(expected_start, expected_start + 6))
    print("test_translation_and_parameter_blocks: PASS")


def test_irls_and_raw_l2_scaling() -> None:
    cache = _synthetic_cache(1)
    parameters = torch.tensor([0.1, -0.05, 0.03, 0.2, -0.1, 0.0,
                               -0.02, 0.04, 0.1, -0.1, 0.05, 0.2], dtype=torch.float64)
    groups = []
    from lib_pano.human_fitting.staged_lm.analytic_stage1 import stage1_raw_groups, stage1_raw_residual
    groups = stage1_raw_groups(cache, parameters)
    spec = build_primary_aggregation_spec(groups)
    irls = build_stage1_analytic_linearization(
        cache, parameters, is_irls=True, charbonnier_eps=1.0,
        aggregation_spec=spec, rep_scale=0.2,
    )
    raw = stage1_raw_residual(cache, parameters)
    raw_jacobian = stage1_raw_jacobian(cache, parameters)
    irls_residual = irls["surrogate_fn"](parameters)
    ratios = irls_residual / raw
    assert torch.allclose(ratios[0::2], ratios[1::2])
    assert torch.allclose(irls["jacobian_fn"](parameters), raw_jacobian * ratios.unsqueeze(1))
    equations = irls["normal_equations_fn"](parameters)
    weighted_jacobian = irls["jacobian_fn"](parameters).to(torch.float64)
    weighted_residual = irls_residual.to(torch.float64)
    torch.testing.assert_close(
        equations.hessian, weighted_jacobian.T @ weighted_jacobian, rtol=1e-6, atol=1e-6,
    )
    torch.testing.assert_close(
        equations.gradient, weighted_jacobian.T @ weighted_residual, rtol=1e-6, atol=1e-6,
    )

    l2 = build_stage1_analytic_linearization(
        cache, parameters, is_irls=False, charbonnier_eps=1.0,
        aggregation_spec=None, rep_scale=0.2,
    )
    assert torch.allclose(l2["surrogate_fn"](parameters), raw * 0.2)
    assert torch.allclose(l2["jacobian_fn"](parameters), raw_jacobian * 0.2)
    l2_equations = l2["normal_equations_fn"](parameters)
    l2_jacobian = (raw_jacobian * 0.2).to(torch.float64)
    l2_residual = (raw * 0.2).to(torch.float64)
    torch.testing.assert_close(l2_equations.hessian, l2_jacobian.T @ l2_jacobian, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(l2_equations.gradient, l2_jacobian.T @ l2_residual, rtol=1e-6, atol=1e-6)
    print("test_irls_and_raw_l2_scaling: PASS")


def test_multiframe_direct_normal_equations_blocks() -> None:
    cache = _synthetic_cache(2)
    parameters = torch.linspace(-0.1, 0.2, 24, dtype=torch.float64)
    scales = torch.linspace(0.2, 0.8, 16, dtype=torch.float64)
    explicit_jacobian = stage1_raw_jacobian(cache, parameters) * scales.unsqueeze(1)
    from lib_pano.human_fitting.staged_lm.analytic_stage1 import stage1_raw_residual
    explicit_residual = stage1_raw_residual(cache, parameters) * scales
    equations = stage1_normal_equations(cache, parameters, scales)
    torch.testing.assert_close(equations.residual, explicit_residual, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(
        equations.hessian, explicit_jacobian.T @ explicit_jacobian, rtol=1e-6, atol=1e-6,
    )
    torch.testing.assert_close(
        equations.gradient, explicit_jacobian.T @ explicit_residual, rtol=1e-6, atol=1e-6,
    )
    assert torch.count_nonzero(equations.hessian[:12, 12:]) == 0
    assert torch.count_nonzero(equations.hessian[12:, :12]) == 0
    for frame in range(2):
        block = slice(frame * 12, (frame + 1) * 12)
        rows = slice(frame * 8, (frame + 1) * 8)
        torch.testing.assert_close(
            equations.hessian[block, block],
            explicit_jacobian[rows, block].T @ explicit_jacobian[rows, block],
        )
        torch.testing.assert_close(
            equations.gradient[block],
            explicit_jacobian[rows, block].T @ explicit_residual[rows],
        )
    print("test_multiframe_direct_normal_equations_blocks: PASS")


def test_timing_reset_and_cuda_solve() -> None:
    cache = _synthetic_cache(1)
    cache.setup_ms = 7.0
    cache.residual_ms, cache.jacobian_ms, cache.normal_equations_ms = 1.0, 2.0, 3.0
    cache.residual_calls, cache.jacobian_calls, cache.normal_equations_calls = 4, 5, 6
    cache.reset_timing()
    summary = cache.timing_dict()
    assert cache.setup_ms == 7.0
    assert all(summary[key] == 0.0 for key in summary if key != "analytic_cache_setup_ms")
    parameters = torch.zeros(12, dtype=torch.float64)
    scales = torch.ones(8, dtype=torch.float64)
    stage1_normal_equations(cache, parameters, scales)
    stage1_normal_equations(cache, parameters, scales)
    assert cache.timing_dict()["analytic_normal_equations_calls"] == 2
    cache.reset_timing()
    stage1_normal_equations(cache, parameters, scales)
    second_run = cache.timing_dict()
    assert second_run["analytic_cache_setup_ms"] == 7.0
    assert second_run["analytic_normal_equations_calls"] == 1
    if torch.cuda.is_available():
        from lib_pano.human_fitting.staged_lm.lm_solver import solve_damped_normal_equations
        hessian = torch.eye(3, dtype=torch.float64, device="cuda")
        gradient = torch.ones(3, dtype=torch.float64, device="cuda")
        result = solve_damped_normal_equations(hessian, gradient, 1e-3)
        assert (
            result.backend == "cholesky"
            and result.delta is not None
            and result.delta.device == hessian.device
        )
    print("test_timing_reset_and_cuda_solve: PASS")


def test_fast_objectives_metrics_and_timing() -> None:
    from lib_pano.human_fitting.staged_lm.analytic_stage1 import stage1_raw_groups
    from lib_pano.human_fitting.staged_lm.residuals import flatten_reprojection_groups, raw_reproj_metrics

    cache = _synthetic_cache(2)
    # Give the four groups different residuals while retaining two frames/camera.
    for index, group in enumerate(cache.groups):
        group.targets = group.targets + (index + 1) * 0.15
    parameters = torch.linspace(-0.1, 0.2, 24, dtype=torch.float64)
    groups = stage1_raw_groups(cache, parameters)
    spec = build_primary_aggregation_spec(groups)
    epsilon = 0.7
    analytic_charbonnier = cache.true_objective(
        parameters, "charbonnier_irls", epsilon, 0.2, spec,
    )
    manual_charbonnier = torch.zeros((), dtype=parameters.dtype)
    for group in groups:
        alpha = spec["terms"][f"{group.frame_index}|{group.camera}"]["alpha_per_point"]
        norms = torch.norm(group.residual_2d, dim=1)
        manual_charbonnier += alpha * torch.sum(torch.sqrt(norms ** 2 + epsilon ** 2) - epsilon)
    torch.testing.assert_close(analytic_charbonnier, manual_charbonnier)

    rep_scale = 0.2
    raw = flatten_reprojection_groups(groups)
    analytic_l2 = cache.true_objective(parameters, "raw_l2", epsilon, rep_scale, None)
    torch.testing.assert_close(analytic_l2, 0.5 * torch.sum((raw * rep_scale) ** 2))
    assert cache.raw_metrics(parameters) == raw_reproj_metrics(raw)
    assert cache.objective_calls == 2 and cache.metrics_calls == 1
    setup_ms = cache.setup_ms
    cache.reset_timing()
    assert cache.setup_ms == setup_ms
    assert cache.objective_calls == cache.metrics_calls == 0
    assert cache.objective_ms == cache.metrics_ms == 0.0

    assert metric_close_with_absolute_or_relative(10.00011, 10.0)
    assert not metric_close_with_absolute_or_relative(10.01, 10.0)
    print("test_fast_objectives_metrics_and_timing: PASS")


if __name__ == "__main__":
    test_erp_projection_jacobian()
    test_axis_angle_point_jacobian_and_small_angles()
    test_axis_angle_near_pi()
    test_translation_and_parameter_blocks()
    test_irls_and_raw_l2_scaling()
    test_multiframe_direct_normal_equations_blocks()
    test_timing_reset_and_cuda_solve()
    test_fast_objectives_metrics_and_timing()
    print("\nAll analytic Stage 1 tests PASSED")
