"""Validate and benchmark the staged-LM Stage 1 analytic Jacobian."""

from __future__ import annotations

import argparse
import math
import time
from dataclasses import replace
import torch
from torch.func import jacfwd

from script.human_fitting.staged_lm.app import build_context
from lib_pano.human_fitting.staged_lm.analytic_stage1 import (
    build_stage1_analytic_cache,
    build_stage1_analytic_linearization,
    Stage1AnalyticCache,
    metric_close_with_absolute_or_relative,
    stage1_raw_jacobian,
    stage1_raw_residual,
    stage1_normal_equations,
)
from lib_pano.human_fitting.staged_lm.config import STAGE1_PER_FRAME_DIM
from lib_pano.human_fitting.staged_lm.linearization import Linearization
from lib_pano.human_fitting.staged_lm.irls import build_primary_aggregation_spec, make_irls_linearization
from lib_pano.human_fitting.staged_lm.residuals import flatten_reprojection_groups
from script.validation.staged_lm.references.stage_backends import build_stage1_definition, run_stage


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", required=True)
    parser.add_argument("--hmr_dir", required=True)
    parser.add_argument("--sam3d_dense_dir", required=True)
    parser.add_argument("--bbox_dir", required=True)
    parser.add_argument("--image_width", type=int, default=5760)
    parser.add_argument("--image_height", type=int, default=2880)
    return parser.parse_args()


def _context_args(cli: argparse.Namespace, backend: str, objective: str = "charbonnier_irls") -> argparse.Namespace:
    return argparse.Namespace(
        **vars(cli), output_dir="/tmp/debug111", reproj_objective=objective,
        stop_after_stage="stage1", run_lbfgs_oracle=False,
        jacobian_backend=backend, validate_irls_first_order=False,
        stage1_eval_backend="full_forward", validate_stage1_fast_eval=False,
        profile_timing=True,
    )


def _error(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, float]:
    difference = actual - expected
    return {
        "max_abs": float(difference.abs().max()),
        "mean_abs": float(difference.abs().mean()),
        "rel": float(difference.norm() / expected.norm().clamp_min(1e-16)),
    }


def _print_group_errors(label: str, actual: torch.Tensor, expected: torch.Tensor, context) -> None:
    row = 0
    print(f"[{label}] overall={_error(actual, expected)}")
    for position, frame_index in enumerate(context.frame_indices):
        for camera, targets in (
            ("cam2", context.cam2_targets_by_frame[position]),
            ("cam4", context.cam4_targets_by_frame[position]),
        ):
            rows = targets.shape[0] * 2
            print(f"  frame={frame_index} camera={camera} {_error(actual[row:row + rows], expected[row:row + rows])}")
            row += rows


def _parameter_points(initial: torch.Tensor) -> list[tuple[str, torch.Tensor]]:
    generator = torch.Generator(device=initial.device).manual_seed(109)
    points = [("initial", initial.clone())]
    for index in range(3):
        perturbation = torch.randn(initial.shape, generator=generator, device=initial.device) * 1e-3
        points.append((f"random_{index}", initial + perturbation))
    rotation = initial.clone()
    rotation[0:3] += torch.tensor([1e-4, -2e-4, 3e-4], device=initial.device)
    points.append(("small_rotation", rotation))
    translation = initial.clone()
    translation[3:6] += torch.tensor([0.01, -0.02, 0.03], device=initial.device)
    points.append(("translation", translation))
    return points


def check_residual_and_jacobian(context, definition, cache) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    full_residual = lambda value: flatten_reprojection_groups(definition.raw_groups_fn(value))
    initial = definition.initial_lm_parameters
    for label, point in _parameter_points(initial):
        analytic = stage1_raw_residual(cache, point)
        oracle = full_residual(point)
        _print_group_errors(f"residual {label}", analytic, oracle, context)
        error = _error(analytic, oracle)
        if error["max_abs"] > 1e-3 or error["rel"] > 1e-6:
            raise RuntimeError(f"ANALYTIC_STAGE1_RESIDUAL_MISMATCH at {label}: {error}")

    analytic_jacobian = stage1_raw_jacobian(cache, initial)
    autograd_jacobian = jacfwd(full_residual)(initial.detach().clone().requires_grad_(True)).detach()
    overall = _error(analytic_jacobian, autograd_jacobian)
    print(f"[jacobian] overall={overall}")
    if overall["rel"] > 5e-4:
        raise RuntimeError(f"ANALYTIC_STAGE1_JACOBIAN_MISMATCH: {overall}")

    row = 0
    for position, frame_index in enumerate(context.frame_indices):
        for camera, targets, column_offset in (
            ("cam2", context.cam2_targets_by_frame[position], 0),
            ("cam4", context.cam4_targets_by_frame[position], 6),
        ):
            rows = targets.shape[0] * 2
            columns = position * STAGE1_PER_FRAME_DIM + column_offset
            analytic_block = analytic_jacobian[row:row + rows, columns:columns + 6]
            oracle_block = autograd_jacobian[row:row + rows, columns:columns + 6]
            print(
                f"  frame={frame_index} camera={camera} rel_frob={_error(analytic_block, oracle_block)['rel']:.3e} "
                f"orient={_error(analytic_block[:, :3], oracle_block[:, :3])} "
                f"translation={_error(analytic_block[:, 3:], oracle_block[:, 3:])}"
            )
            row += rows

    best = None
    for epsilon in (3e-4, 1e-3, 3e-3, 1e-2):
        columns = []
        for column in range(initial.numel()):
            delta = torch.zeros_like(initial)
            delta[column] = epsilon
            columns.append((full_residual(initial + delta) - full_residual(initial - delta)) / (2.0 * epsilon))
        finite_difference = torch.stack(columns, dim=1)
        error = _error(analytic_jacobian, finite_difference)
        print(f"[central_difference] eps={epsilon:.0e} {error}")
        if best is None or error["rel"] < best[1]["rel"]:
            best = (epsilon, error)
    print(f"[central_difference] best_eps={best[0]:.0e} best={best[1]}")
    return initial, analytic_jacobian, autograd_jacobian


def check_surrogates(context, definition, cache, initial) -> None:
    spec = build_primary_aggregation_spec(definition.raw_groups_fn(initial))
    for objective in ("charbonnier_irls", "raw_l2"):
        is_irls = objective == "charbonnier_irls"
        if is_irls:
            oracle = make_irls_linearization(
                definition.raw_groups_fn, None, initial, context.charbonnier_eps, spec,
            )
        else:
            scale = context.scaling.rep_scale
            def residual(value):
                return flatten_reprojection_groups(definition.raw_groups_fn(value)) * scale
            oracle = Linearization(surrogate=residual)
        analytic = build_stage1_analytic_linearization(
            cache, initial, is_irls=is_irls, charbonnier_eps=context.charbonnier_eps,
            aggregation_spec=spec if is_irls else None, rep_scale=context.scaling.rep_scale,
        )
        residual_error = _error(analytic.surrogate(initial), oracle.surrogate(initial))
        jacobian_error = _error(
            analytic.jacobian(initial), jacfwd(oracle.surrogate)(initial),
        )
        explicit_residual = analytic.surrogate(initial)
        explicit_jacobian = analytic.jacobian(initial).to(torch.float64)
        equations = analytic.normal_equations(initial)
        hessian_error = _error(equations.hessian, explicit_jacobian.T @ explicit_jacobian)
        gradient_error = _error(
            equations.gradient, explicit_jacobian.T @ explicit_residual.to(torch.float64),
        )
        print(
            f"[surrogate {objective}] residual={residual_error} jacobian={jacobian_error} "
            f"H={hessian_error} g={gradient_error}"
        )
        if residual_error["rel"] > 1e-6 or jacobian_error["rel"] > 5e-4:
            raise RuntimeError(f"ANALYTIC_STAGE1_SURROGATE_MISMATCH: {objective}")
        if hessian_error["rel"] > 1e-6 or gradient_error["rel"] > 1e-6:
            raise RuntimeError(f"ANALYTIC_STAGE1_NORMAL_EQUATIONS_MISMATCH: {objective}")


def check_lm_regression(context, autograd_definition, analytic_definition) -> bool:
    context.args.jacobian_backend = "autograd"
    autograd_outcome = run_stage(autograd_definition, context)
    context.args.jacobian_backend = "analytic_stage1"
    analytic_outcome = run_stage(analytic_definition, context)
    auto = autograd_outcome.lm_result
    analytic = analytic_outcome.lm_result
    comparisons = {
        "objective_relative": abs(analytic.objective - auto.objective) / max(abs(auto.objective), 1e-16),
        "rms_relative": abs(analytic.rms_metrics["rms"] - auto.rms_metrics["rms"]) / max(abs(auto.rms_metrics["rms"]), 1e-16),
        "median_relative": abs(analytic.rms_metrics["median"] - auto.rms_metrics["median"]) / max(abs(auto.rms_metrics["median"]), 1e-16),
        "p95_relative": abs(analytic.rms_metrics["p95"] - auto.rms_metrics["p95"]) / max(abs(auto.rms_metrics["p95"]), 1e-16),
        "parameter_max_abs": float((analytic.parameters - auto.parameters).abs().max()),
    }
    print(f"[lm_regression] {comparisons}")
    print(
        f"  autograd acc/rej={auto.counters.accepted_steps}/{auto.counters.rejected_trials}; "
        f"analytic acc/rej={analytic.counters.accepted_steps}/{analytic.counters.rejected_trials}"
    )
    print(
        f"  autograd status={auto.status} lambda={auto.final_lambda:.6e}; "
        f"analytic status={analytic.status} lambda={analytic.final_lambda:.6e} "
        f"linearization_count={analytic.counters.linearization_count}"
    )
    metrics_ok = comparisons["objective_relative"] <= 1e-5 and all(
        metric_close_with_absolute_or_relative(
            analytic.rms_metrics[name], auto.rms_metrics[name],
        )
        for name in ("rms", "median", "p95")
    )
    parameters_ok = torch.allclose(analytic.parameters, auto.parameters, rtol=1e-4, atol=1e-4)
    # Validate the fast residual against the full oracle at the same final
    # analytic parameter point; parameter agreement is checked separately.
    auto_raw = flatten_reprojection_groups(
        autograd_definition.raw_groups_fn(analytic.parameters)
    )
    analytic_raw = analytic_definition.analytic_cache.raw_residual(analytic.parameters)
    try:
        torch.testing.assert_close(analytic_raw, auto_raw, rtol=1e-6, atol=1e-3)
        residual_ok = True
    except AssertionError:
        residual_ok = False
    status_ok = analytic.status == auto.status
    counts_ok = (
        analytic.counters.accepted_steps == auto.counters.accepted_steps
        and analytic.counters.rejected_trials == auto.counters.rejected_trials
    )
    lambda_ok = math.isclose(
        analytic.final_lambda, auto.final_lambda, rel_tol=1e-6, abs_tol=1e-12,
    )
    print(f"  final_raw_error={_error(analytic_raw, auto_raw)} residual_ok={residual_ok}")
    return metrics_ok and parameters_ok and residual_ok and status_ok and counts_ok and lambda_ok


def benchmark(context, definition, cache, initial) -> None:
    full_residual = lambda value: flatten_reprojection_groups(definition.raw_groups_fn(value))
    def synchronize():
        if initial.is_cuda:
            torch.cuda.synchronize(initial.device)
    for _ in range(3):
        jacfwd(full_residual)(initial)
        stage1_raw_jacobian(cache, initial)
        stage1_normal_equations(
            cache, initial, torch.ones(sum(group.targets.numel() for group in cache.groups), device=initial.device),
        )
    repeats = 20
    synchronize(); start = time.perf_counter()
    for _ in range(repeats):
        jacfwd(full_residual)(initial)
    synchronize(); autograd_ms = (time.perf_counter() - start) * 1000.0 / repeats
    synchronize(); start = time.perf_counter()
    for _ in range(repeats):
        stage1_raw_jacobian(cache, initial)
    synchronize(); analytic_ms = (time.perf_counter() - start) * 1000.0 / repeats
    synchronize(); start = time.perf_counter()
    for _ in range(repeats):
        stage1_raw_residual(cache, initial)
    synchronize(); residual_ms = (time.perf_counter() - start) * 1000.0 / repeats
    row_scales = torch.ones(
        sum(group.targets.numel() for group in cache.groups),
        dtype=initial.dtype, device=initial.device,
    )
    synchronize(); start = time.perf_counter()
    for _ in range(repeats):
        stage1_normal_equations(cache, initial, row_scales)
    synchronize(); normal_ms = (time.perf_counter() - start) * 1000.0 / repeats
    residual_rows = row_scales.numel()
    explicit_elements = residual_rows * initial.numel()
    print(
        f"[speed] cache_setup_ms={cache.setup_ms:.3f} residual_ms/call={residual_ms:.3f} "
        f"autograd_J_ms/call={autograd_ms:.3f} analytic_J_ms/call={analytic_ms:.3f} "
        f"direct_normal_ms/call={normal_ms:.3f} explicit_J_elements={explicit_elements} "
        f"explicit_J_bytes={explicit_elements * initial.element_size()} "
        f"direct_constructs_global_J=False speedup_vs_autograd={autograd_ms / normal_ms:.2f}x"
    )

    base_groups = cache.groups[:2]
    base_parameters = initial[:STAGE1_PER_FRAME_DIM]
    for frame_count in (1, 2, 4):
        groups = []
        for frame in range(frame_count):
            groups.extend([
                replace(group, frame_position=frame, frame_index=frame)
                for group in base_groups
            ])
        scaled_cache = Stage1AnalyticCache(
            groups, frame_count, cache.image_width, cache.image_height, 0.0,
        )
        parameters = base_parameters.repeat(frame_count)
        scales = torch.ones(
            sum(group.targets.numel() for group in groups),
            dtype=parameters.dtype, device=parameters.device,
        )
        for _ in range(3):
            stage1_normal_equations(scaled_cache, parameters, scales)
        synchronize(); start = time.perf_counter()
        for _ in range(repeats):
            stage1_normal_equations(scaled_cache, parameters, scales)
        synchronize(); elapsed = (time.perf_counter() - start) * 1000.0 / repeats
        global_j_elements = scales.numel() * parameters.numel()
        max_frame_rows = max(
            sum(group.targets.numel() for group in groups if group.frame_position == frame)
            for frame in range(frame_count)
        )
        direct_temp_elements = max_frame_rows * STAGE1_PER_FRAME_DIM + parameters.numel() ** 2
        print(
            f"[scaling T={frame_count}] global_J_elements={global_j_elements} "
            f"direct_temp_elements~={direct_temp_elements} dense_H_elements={parameters.numel() ** 2} "
            f"direct_normal_ms/call={elapsed:.3f}"
        )


def run_all_stage1_validations(cli=None) -> int:
    cli = cli or parse_args()
    context = build_context(_context_args(cli, "autograd"))
    context.args.jacobian_backend = "autograd"
    autograd_definition = build_stage1_definition(context)
    cache = build_stage1_analytic_cache(context)
    initial, _, _ = check_residual_and_jacobian(context, autograd_definition, cache)
    check_surrogates(context, autograd_definition, cache, initial)
    context.args.jacobian_backend = "analytic_stage1"
    analytic_definition = build_stage1_definition(context)
    lm_ok = check_lm_regression(context, autograd_definition, analytic_definition)
    benchmark(context, autograd_definition, cache, initial)
    if not lm_ok:
        raise RuntimeError("ANALYTIC_STAGE1_LM_REGRESSION_MISMATCH")
    print("[PASS] Stage 1 analytic residual/J, surrogates, and LM regression")
    return 0


run_stage1_jacobian_validation = run_all_stage1_validations
run_stage1_surrogate_validation = run_all_stage1_validations
run_stage1_solver_validation = run_all_stage1_validations
run_stage1_benchmark = run_all_stage1_validations


def run_stage1_batch_shadow(cli=None) -> int:
    """Concise serial-reference versus FULL_BATCH Stage-1 solver gate."""
    import contextlib, io
    from lib_pano.human_fitting.staged_lm.lm.runner import run_lm_optimization
    from lib_pano.human_fitting.staged_lm.stage_definitions.rigid import build_stage1_definition as build_production_stage1
    from script.benchmark.staged_lm.stage1_batch_candidates import build_stage1_batch_cache
    from script.benchmark.staged_lm.stage1_shadow_common import Stage1NEMode
    from script.benchmark.staged_lm.stage1_shadow_lm import (
        compare_stage1_lm_results, run_stage1_lm_independent_batch,
        run_stage1_lm_lockstep,
    )
    cli=cli or parse_args();context=build_context(_context_args(cli,"analytic_stage1"))
    stage=build_production_stage1(context);batch=build_stage1_batch_cache(stage.analytic_cache)
    with contextlib.redirect_stdout(io.StringIO()):
        serial=run_lm_optimization(
            "STAGE1_SERIAL_REFERENCE",lambda x:stage.linearization_builder(x,stage.aggregation_spec),
            stage.raw_metrics_fn,stage.initial_lm_parameters,stage.parameter_dimension,
            stage.lm_max_accepted_steps,context.solver_config.lm_max_trials,
            context.solver_config,True,stage.lm_true_objective_fn)
        candidate=run_stage1_lm_independent_batch(stage,batch,context,
            mode=Stage1NEMode.FULL_BATCH_GROUP_GEMM)
        lock=run_stage1_lm_lockstep(stage,batch,context,
            candidate_mode=Stage1NEMode.FULL_BATCH_GROUP_GEMM)
    comparison=compare_stage1_lm_results(serial,candidate)
    threshold=5e-5 if context.device.type=="cuda" else 1e-5
    ok=(comparison["first_divergent_trial"] is None and comparison["status_match"]
        and comparison["accepted_match"] and comparison["rejected_match"]
        and comparison["trial_count_match"] and comparison["solve_path_match"]
        and comparison["final_parameter_max_abs"]<=threshold
        and comparison["objective_relative"]<=1e-6
        and lock.acceptance_mismatch_count==0 and lock.solve_path_mismatch_count==0)
    print(f"[STAGE1_BATCH_SHADOW] acceptance_mismatch={lock.acceptance_mismatch_count} "
          f"first_divergent={comparison['first_divergent_trial']} "
          f"param_max={comparison['final_parameter_max_abs']:.3e} "
          f"obj_rel={comparison['objective_relative']:.3e} "
          f"metrics_max={comparison['metrics_max_abs']:.3e} "
          f"serial_ms={serial.wall_time_ms:.3f} batch_ms={candidate.wall_time_ms:.3f} "
          f"speedup={serial.wall_time_ms/candidate.wall_time_ms:.3f}x")
    if not ok:raise RuntimeError("STAGE1_FULL_BATCH_SHADOW_GATE_FAILED")
    print("[PASS] Stage1 serial-reference vs FULL_BATCH shadow");return 0
