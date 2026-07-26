"""Levenberg-Marquardt main control-flow runner.

Read equations, trial, and convergence before this module. The runner keeps
the production local-variable state machine and timing boundaries unchanged.
"""

from __future__ import annotations

import time
from typing import Any, Callable

import torch

from ..config import SolverConfig
from ..linearization import Linearization
from ..types import LMResult, LMSolverCounters
from .convergence import (
    check_convergence,
    classify_no_accepted_trial,
    increase_damping_after_rejection,
    initialise_damping_from_max_diagonal,
    update_damping_after_acceptance,
)
from .equations import (
    _evaluate_linearization, compute_jacobian, compute_scaled_gradient,
    compute_scaled_gradient_from_normal_equations, normal_equations_max_diagonal,
    solve_normal_equations,
)
from .trial import (
    build_evaluated_trial_record, build_failed_trial_record,
    evaluate_trial_reduction,
)


def _elapsed_ms(start: float, ref_tensor: torch.Tensor | None, do_sync: bool) -> float:
    """Wall-clock milliseconds since ``start``, optionally syncing CUDA."""
    if ref_tensor is not None and do_sync and ref_tensor.is_cuda:
        torch.cuda.synchronize(ref_tensor.device)
    return (time.perf_counter() - start) * 1000.0


def run_lm_optimization(
    label: str,
    make_linearization: Callable[[torch.Tensor], Linearization],
    raw_metrics_fn: Callable[[torch.Tensor], dict[str, float]],
    initial_parameters: torch.Tensor,
    param_dim: int,
    max_accepted_steps: int,
    max_trials: int,
    solver_config: SolverConfig,
    profile_timing: bool,
    true_objective_fn: Callable[[torch.Tensor], torch.Tensor],
) -> LMResult:
    """Run the Levenberg-Marquardt optimiser and return structured results.

    Parameters
    ----------
    make_linearization:
        Returns the typed frozen-weight local least-squares model.
    true_objective_fn:
        The true scalar objective used for acceptance and convergence checks.
    """

    profile = profile_timing
    do_sync = profile and initial_parameters.is_cuda

    def sync() -> None:
        if do_sync:
            torch.cuda.synchronize(initial_parameters.device)

    timing: dict[str, float] = {
        "initial_linearization_ms": 0.0,
        "solver_jacobian_ms": 0.0,
        "objective_eval_ms": 0.0,
        "linear_solve_ms": 0.0,
        "final_metrics_ms": 0.0,
    }

    # ---- Initial linearisation ----
    sync()
    t_init = time.perf_counter()
    t0 = time.perf_counter()
    x = initial_parameters.detach().clone()
    linearization = make_linearization(x)

    t_j = time.perf_counter()
    equations, uses_direct_normal_equations = _evaluate_linearization(linearization, x)
    if profile:
        timing["solver_jacobian_ms"] += (time.perf_counter() - t_j) * 1000.0
    if profile:
        timing["initial_linearization_ms"] = (time.perf_counter() - t_init) * 1000.0

    # Damping initial value: 1e-3 * max diagonal of J^T J (Marquardt's heuristic).
    initial_max_diagonal = normal_equations_max_diagonal(equations)
    damping = initialise_damping_from_max_diagonal(initial_max_diagonal)

    counters = LMSolverCounters(
        solver_jacobian_count=1,
        linearization_count=1,
        normal_equation_count=1 if uses_direct_normal_equations else 0,
    )
    status = "RUNNING"
    weight_history: list[dict[str, Any]] = [
        {"outer": -1, **linearization.weight_stats}
    ]
    spec = dict(linearization.aggregation_spec)
    linear_solver_backend = linearization.provenance.linear_solver_backend
    trial_history: list[dict[str, Any]] = []

    sync()
    t_o = time.perf_counter()
    current_true_objective = float(
        true_objective_fn(x).detach().item()
    )
    if profile:
        timing["objective_eval_ms"] += (
            time.perf_counter() - t_o
        ) * 1000.0

    # ---- Outer accepted-step loop ----
    for outer in range(max_accepted_steps):
        gradient_64 = equations.gradient

        obj_before = current_true_objective

        accepted_this = False
        # ---- Inner trial loop: find an acceptable step by adjusting damping ----
        for _trial in range(max_trials):
            sync()
            t_ls = time.perf_counter()
            trial_damping = damping
            solve_result = solve_normal_equations(equations, damping)
            counters.record_linear_solve(solve_result)
            delta = solve_result.delta
            if not solve_result.success:
                # Both solvers failed: increase damping and retry.
                damping = increase_damping_after_rejection(damping)
                counters.rejected_trials += 1
                trial_history.append(build_failed_trial_record(
                    outer, _trial, trial_damping, linear_solver_backend,
                    solve_result.schur_failed,
                ))
                if profile:
                    timing["linear_solve_ms"] += (time.perf_counter() - t_ls) * 1000.0
                continue
            if profile:
                timing["linear_solve_ms"] += (time.perf_counter() - t_ls) * 1000.0

            candidate = (x + delta.to(dtype=torch.float32)).detach()
            sync()
            t_o = time.perf_counter()
            obj_trial = float(true_objective_fn(candidate).detach().item())
            if profile:
                timing["objective_eval_ms"] += (time.perf_counter() - t_o) * 1000.0

            delta_64 = delta.to(torch.float64)
            predicted_reduction, actual_reduction, rho, accepted_trial = (
                evaluate_trial_reduction(
                    equations, gradient_64, delta_64, obj_before, obj_trial,
                )
            )
            trial_history.append(build_evaluated_trial_record(
                outer, _trial, trial_damping, linear_solver_backend, delta,
                predicted_reduction, actual_reduction, rho, accepted_trial,
                solve_result.schur_failed,
            ))
            if accepted_trial:
                x = candidate
                current_true_objective = obj_trial
                counters.accepted_steps += 1
                accepted_this = True
                damping = update_damping_after_acceptance(damping, rho)
                break
            else:
                damping = increase_damping_after_rejection(damping)
                counters.rejected_trials += 1

        # ---- No acceptable step found this outer iteration ----
        if not accepted_this:
            if uses_direct_normal_equations:
                scaled_grad = compute_scaled_gradient(equations)
            else:
                residual_now = linearization.surrogate(x).detach()
                t_j = time.perf_counter()
                jacobian_now = compute_jacobian(linearization, x)
                if profile:
                    timing["solver_jacobian_ms"] += (time.perf_counter() - t_j) * 1000.0
                counters.solver_jacobian_count += 1
                jacobian_now_64 = jacobian_now.to(torch.float64)
                residual_now_64 = residual_now.to(torch.float64)
                scaled_grad = compute_scaled_gradient_from_normal_equations(
                    jacobian_now_64.T @ jacobian_now_64,
                    jacobian_now_64.T @ residual_now_64,
                )
            status = classify_no_accepted_trial(
                counters.accepted_steps, scaled_grad,
            )
            break

        # ---- Re-linearise at the accepted point ----
        next_linearization = make_linearization(x)
        t_j = time.perf_counter()
        next_equations, next_uses_direct = _evaluate_linearization(next_linearization, x)
        if profile:
            timing["solver_jacobian_ms"] += (time.perf_counter() - t_j) * 1000.0
        counters.solver_jacobian_count += 1
        counters.linearization_count += 1
        counters.normal_equation_count += int(next_uses_direct)
        weight_history.append(
            {"outer": outer, **next_linearization.weight_stats}
        )

        scaled_grad_final = compute_scaled_gradient(next_equations)
        rel = (
            obj_before - current_true_objective
        ) / max(abs(obj_before), 1e-16)

        convergence = check_convergence(scaled_grad_final, float(torch.norm(delta)), rel, solver_config)
        if convergence is not None:
            status = convergence
            break
        linearization = next_linearization
        equations = next_equations
        uses_direct_normal_equations = next_uses_direct

    if status == "RUNNING":
        status = "MAX_ACCEPTED_STEPS"

    # ---- Final metrics ----
    sync()
    t_fm = time.perf_counter()
    final_parameters = x.detach().clone()
    rms_metrics = raw_metrics_fn(final_parameters)
    obj_final = current_true_objective
    if profile:
        timing["final_metrics_ms"] = (time.perf_counter() - t_fm) * 1000.0

    wall_ms = _elapsed_ms(t0, x, do_sync)

    print(f"[{label}] {status} acc={counters.accepted_steps} rej={counters.rejected_trials} "
          f"jac={counters.solver_jacobian_count} lins={counters.linearization_count} "
          f"slv={counters.solve_attempt_count} "
          f"chol_ok={counters.cholesky_success_count} fb={counters.solve_fallback_count} "
          f"obj={obj_final:.6e} rms={rms_metrics['rms']:.1f}")

    return LMResult(
        label=label,
        parameters=final_parameters,
        objective=obj_final,
        status=status,
        counters=counters,
        final_lambda=damping,
        wall_time_ms=wall_ms,
        rms_metrics=rms_metrics,
        weight_history=weight_history,
        first_order_checks={},
        spec=spec,
        jacobian_backend=linearization.provenance.jacobian_backend,
        linearization_backend=linearization.provenance.linearization_backend,
        linear_solver_backend=linear_solver_backend,
        trial_history=trial_history,
        profile_timing=timing if profile else None,
    )
