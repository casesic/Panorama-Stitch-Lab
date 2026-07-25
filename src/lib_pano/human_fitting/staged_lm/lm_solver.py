"""Levenberg-Marquardt optimiser for staged fitting.
"""

from __future__ import annotations

import math
import time
from typing import Any, Callable

import torch
from torch.func import jacfwd

from .config import SolverConfig
from .types import LinearSolveResult, LMResult, LMSolverCounters, NormalEquations

# Damping bounds.  ``lam_min`` prevents the damping from collapsing to zero
# (which would make the normal equations ill-conditioned), while ``lam_max``
# caps the growth so the solver can still recover.
_LAMBDA_MIN = 1e-10
_LAMBDA_MAX = 1e12


def _elapsed_ms(start: float, ref_tensor: torch.Tensor | None, do_sync: bool) -> float:
    """Wall-clock milliseconds since ``start``, optionally syncing CUDA."""
    if ref_tensor is not None and do_sync and ref_tensor.is_cuda:
        torch.cuda.synchronize(ref_tensor.device)
    return (time.perf_counter() - start) * 1000.0


def compute_jacobian(linearization: dict[str, Any], x: torch.Tensor) -> torch.Tensor:
    """Use an optional analytic Jacobian, otherwise forward-mode AD."""
    jacobian_fn = linearization.get("jacobian_fn")
    if jacobian_fn is not None:
        return jacobian_fn(x).detach()
    return jacfwd(linearization["surrogate_fn"])(
        x.detach().clone().requires_grad_(True)
    ).detach()


def solve_damped_normal_equations(
    hessian: torch.Tensor,
    gradient: torch.Tensor,
    damping: float,
) -> LinearSolveResult:
    """Solve one dense damped system with Cholesky then ``linalg.solve``."""
    parameter_dimension = int(hessian.shape[0])
    identity = torch.eye(
        parameter_dimension, dtype=hessian.dtype, device=hessian.device,
    )
    matrix = hessian + damping * identity
    try:
        lower, info = torch.linalg.cholesky_ex(matrix)
        if info.item():
            raise RuntimeError
        delta = torch.cholesky_solve(-gradient.unsqueeze(1), lower).squeeze(1)
        return LinearSolveResult(delta, "cholesky")
    except Exception:
        try:
            delta = torch.linalg.solve(matrix, -gradient.unsqueeze(1)).squeeze(1)
            return LinearSolveResult(delta, "dense_solve")
        except Exception:
            return LinearSolveResult(None, "dense_solve")


def compute_scaled_gradient_from_normal_equations(
    hessian: torch.Tensor, gradient: torch.Tensor,
) -> float:
    """Scaled gradient ``||g|| / max(1, ||H||)`` for either linearization path.

    Normalising by the Hessian norm removes problem-scale dependence so the
    same tolerance works across stages with different parameter dimensions.
    """
    return float(gradient.norm() / max(1, hessian.norm()))


def _validate_normal_equations(
    equations: NormalEquations,
    parameters: torch.Tensor,
) -> None:
    """Validate representation, device and dtype once after construction."""
    dense = equations.hessian is not None
    structured = equations.structured_system is not None
    if dense == structured:
        raise RuntimeError("INVALID_NORMAL_EQUATION_REPRESENTATION")
    if structured:
        flat = equations.structured_system.flat_gradient()
        if equations.gradient.shape != flat.shape or not torch.equal(equations.gradient, flat):
            raise RuntimeError("INVALID_NORMAL_EQUATION_REPRESENTATION")

    expected_device = parameters.device
    tensors = [("residual", equations.residual), ("gradient", equations.gradient)]
    if dense:
        tensors.append(("hessian", equations.hessian))
    else:
        system = equations.structured_system
        tensors.extend((
            ("frame_hessians", system.frame_hessians),
            ("frame_beta_cross", system.frame_beta_cross),
            ("beta_hessian", system.beta_hessian),
            ("frame_gradients", system.frame_gradients),
            ("beta_gradient", system.beta_gradient),
        ))
    for name, tensor in tensors:
        if tensor.device != expected_device:
            error = (
                "UNEXPECTED_CPU_FALLBACK_IN_CUDA_PIPELINE"
                if parameters.is_cuda else "DEVICE_CONTRACT_VIOLATION"
            )
            raise RuntimeError(
                f"{error}: {name} expected={expected_device} actual={tensor.device}"
            )

    double_tensors = [("gradient", equations.gradient)]
    if dense:
        double_tensors.append(("hessian", equations.hessian))
    else:
        double_tensors.extend(
            (f"structured.{name}", tensor) for name, tensor in tensors[2:]
        )
    for name, tensor in double_tensors:
        if tensor.dtype != torch.float64:
            raise RuntimeError(
                "DEVICE_CONTRACT_VIOLATION: "
                f"{name} expected_dtype=torch.float64 actual_dtype={tensor.dtype}"
            )


def normal_equations_max_diagonal(equations: NormalEquations) -> float:
    value = (torch.diagonal(equations.hessian).max() if equations.hessian is not None
             else equations.structured_system.max_diagonal())
    return float(value)


def normal_equations_hessian_norm(equations: NormalEquations) -> torch.Tensor:
    return (equations.hessian.norm() if equations.hessian is not None
            else equations.structured_system.hessian_frobenius_norm())


def normal_equations_quadratic_form(
    equations: NormalEquations, delta: torch.Tensor,
) -> torch.Tensor:
    if equations.hessian is not None:
        return delta @ equations.hessian @ delta
    return equations.structured_system.quadratic_form(delta)


def compute_scaled_gradient(equations: NormalEquations) -> float:
    return float(equations.gradient.norm() / max(1, normal_equations_hessian_norm(equations)))


def _solve_structured_normal_equations(
    equations: NormalEquations, damping: float,
) -> LinearSolveResult:
    """Solve Arrowhead equations, materialising dense H only after failure."""
    system = equations.structured_system
    delta = system.solve_damped(damping)
    if delta is not None:
        return LinearSolveResult(delta, "schur")
    dense = system.to_dense()
    if system.timing_sink is not None:
        system.timing_sink.schur_dense_fallback_calls += 1
    dense_result = solve_damped_normal_equations(
        dense, equations.gradient, damping,
    )
    return LinearSolveResult(
        dense_result.delta, dense_result.backend, schur_failed=True,
    )


def solve_normal_equations(
    equations: NormalEquations, damping: float,
) -> LinearSolveResult:
    """Dispatch a dense or Arrowhead normal-equation solve."""
    if equations.hessian is not None:
        return solve_damped_normal_equations(
            equations.hessian, equations.gradient, damping,
        )
    return _solve_structured_normal_equations(equations, damping)


def _evaluate_linearization(
    linearization: dict[str, Any], x: torch.Tensor,
) -> tuple[NormalEquations, bool]:
    normal_equations_fn = linearization.get("normal_equations_fn")
    if normal_equations_fn is not None:
        equations = normal_equations_fn(x)
        if not isinstance(equations, NormalEquations):
            raise TypeError(f"normal_equations_fn returned {type(equations)}")
        _validate_normal_equations(equations, x)
        return equations, True
    residual = linearization["surrogate_fn"](x).detach()
    jacobian = compute_jacobian(linearization, x)
    jacobian_64 = jacobian.to(torch.float64)
    residual_64 = residual.to(torch.float64)
    equations = NormalEquations(
        residual=residual,
        hessian=jacobian_64.T @ jacobian_64,
        gradient=jacobian_64.T @ residual_64,
    )
    _validate_normal_equations(equations, x)
    return equations, False


def check_convergence(
    scaled_gradient: float,
    step_norm: float,
    rel_objective_change: float,
    solver_config: SolverConfig,
) -> str | None:
    """Return a convergence status string, or ``None`` to continue iterating."""
    if scaled_gradient < solver_config.scaled_grad_tol:
        return "CONVERGED_SCALED_GRADIENT"
    if step_norm < solver_config.step_tol:
        return "CONVERGED_STEP"
    if abs(rel_objective_change) < solver_config.rel_obj_tol:
        return "CONVERGED_REL_OBJECTIVE"
    return None


def run_lm_optimization(
    label: str,
    make_linearization: Callable[[torch.Tensor], dict[str, Any]],
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
        Returns a dict with ``surrogate_fn``, ``weight_stats`` and ``spec``.
        For IRLS mode this builds the weighted surrogate; for raw-L2 mode it
        returns the plain residual.
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
    if not math.isfinite(initial_max_diagonal) or initial_max_diagonal <= 0:
        raise RuntimeError("INVALID_INITIAL_LM_DAMPING")
    damping = initial_max_diagonal * 1e-3

    counters = LMSolverCounters(
        solver_jacobian_count=1,
        linearization_count=1,
        normal_equation_count=1 if uses_direct_normal_equations else 0,
    )
    status = "RUNNING"
    weight_history: list[dict[str, Any]] = [{"outer": -1, **linearization.get("weight_stats", {})}]
    spec = linearization.get("spec", {})
    linear_solver_backend = linearization.get("linear_solver_backend", "dense")
    trial_history: list[dict[str, Any]] = []

    # ---- Outer accepted-step loop ----
    for outer in range(max_accepted_steps):
        gradient_64 = equations.gradient

        sync()
        t_o = time.perf_counter()
        obj_before = float(true_objective_fn(x).detach().item())
        if profile:
            timing["objective_eval_ms"] += (time.perf_counter() - t_o) * 1000.0

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
                damping = min(damping * 10, _LAMBDA_MAX)
                counters.rejected_trials += 1
                trial_history.append({
                    "outer": outer, "trial": _trial, "damping": trial_damping,
                    "linear_solver_backend": linear_solver_backend,
                    "solve_success": False, "delta_norm": None,
                    "predicted_reduction": None, "actual_reduction": None,
                    "rho": None, "accepted": False,
                    "used_dense_fallback": solve_result.schur_failed,
                })
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
            predicted_reduction = float(
                -gradient_64.dot(delta_64)
                - 0.5 * normal_equations_quadratic_form(equations, delta_64)
            )
            actual_reduction = obj_before - obj_trial
            rho = actual_reduction / max(predicted_reduction, 1e-12) if predicted_reduction > 0 else 0.0

            accepted_trial = predicted_reduction > 0 and actual_reduction > 0 and rho > 0
            trial_history.append({
                "outer": outer, "trial": _trial, "damping": trial_damping,
                "linear_solver_backend": linear_solver_backend,
                "solve_success": True, "delta_norm": float(delta.norm()),
                "predicted_reduction": predicted_reduction,
                "actual_reduction": actual_reduction, "rho": rho,
                "accepted": accepted_trial,
                "used_dense_fallback": solve_result.schur_failed,
            })
            if accepted_trial:
                x = candidate
                counters.accepted_steps += 1
                accepted_this = True
                if rho > 0.75:
                    damping = max(damping / 3, _LAMBDA_MIN)
                elif rho < 0.25:
                    damping = min(damping * 3, _LAMBDA_MAX)
                break
            else:
                damping = min(damping * 10, _LAMBDA_MAX)
                counters.rejected_trials += 1

        # ---- No acceptable step found this outer iteration ----
        if not accepted_this:
            if uses_direct_normal_equations:
                scaled_grad = compute_scaled_gradient(equations)
            else:
                residual_now = linearization["surrogate_fn"](x).detach()
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
            status = "CONVERGED_SCALED_GRADIENT" if (counters.accepted_steps > 4 and scaled_grad < 1e-4) else "LM_NO_ACCEPTED_TRIAL"
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
        weight_history.append({"outer": outer, **next_linearization.get("weight_stats", {})})

        scaled_grad_final = compute_scaled_gradient(next_equations)
        sync()
        t_o = time.perf_counter()
        rel = (obj_before - float(true_objective_fn(x).detach().item())) / max(abs(obj_before), 1e-16)
        if profile:
            timing["objective_eval_ms"] += (time.perf_counter() - t_o) * 1000.0

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
    obj_final = float(true_objective_fn(final_parameters).detach().item())
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
        jacobian_backend=linearization.get("jacobian_backend", "autograd"),
        linearization_backend=linearization.get("linearization_backend", "jacobian"),
        linear_solver_backend=linear_solver_backend,
        trial_history=trial_history,
        profile_timing=timing if profile else None,
    )
