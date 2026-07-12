"""Levenberg-Marquardt optimiser and L-BFGS oracle.

The LM solver (``run_lm_optimization``) is the primary optimiser under test.
It implements a classic damped Gauss-Newton iteration with:

* Cholesky-based normal-equation solve with a ``torch.linalg.solve`` fallback.
* Levenberg-Marquardz damping update (``rho > 0.75`` -> shrink, ``rho < 0.25``
  -> grow).
* Three convergence criteria: scaled gradient, step norm, relative objective
  change.
* Optional IRLS first-order validation after each accepted step.

The L-BFGS oracle (``run_lbfgs_oracle``) solves the same objective with
PyTorch's built-in L-BFGS so the LM result can be compared against a
well-known reference.

The numerical behaviour — damping initialisation, lambda update rule,
predicted/actual reduction, Cholesky-then-fallback order, convergence
thresholds and float32/float64 usage — is identical to the original
``identity_lm`` and ``run_lbfgs`` functions.
"""

from __future__ import annotations

import time
from typing import Any, Callable

import torch
from torch.func import jacfwd

from .irls import validate_first_order
from .config import SolverConfig
from .types import LBFGSResult, LMResult, LMSolverCounters

# Damping bounds.  ``lam_min`` prevents the damping from collapsing to zero
# (which would make the normal equations ill-conditioned), while ``lam_max``
# caps the growth so the solver can still recover.
_LAMBDA_MIN = 1e-10
_LAMBDA_MAX = 1e12


def _elapsed_ms(start: float, ref_tensor: torch.Tensor | None, do_sync: bool) -> float:
    """Wall-clock milliseconds since ``start``, optionally syncing CUDA."""
    if ref_tensor is not None and do_sync and ref_tensor.is_cuda:
        torch.cuda.synchronize()
    return (time.perf_counter() - start) * 1000.0


def compute_jacobian(surrogate_fn: Callable[[torch.Tensor], torch.Tensor], x: torch.Tensor) -> torch.Tensor:
    """Compute the Jacobian of ``surrogate_fn`` at ``x`` via forward-mode AD."""
    return jacfwd(surrogate_fn)(x.detach().clone().requires_grad_(True)).detach()


def solve_damped_normal_equations(
    hessian: torch.Tensor,
    gradient: torch.Tensor,
    damping: float,
    param_dim: int,
) -> tuple[torch.Tensor | None, bool, bool]:
    """Solve ``(H + lambda*I) delta = -g`` with Cholesky then dense fallback.

    Returns
    -------
    (delta, cholesky_ok, fallback_ok):
        ``delta`` is ``None`` if both solvers fail.  The two booleans indicate
        which path succeeded so the caller can update counters.
    """
    matrix = hessian + damping * torch.eye(param_dim, dtype=torch.float64)
    try:
        lower, info = torch.linalg.cholesky_ex(matrix)
        if info.item():
            raise RuntimeError
        delta = torch.cholesky_solve(-gradient.unsqueeze(1), lower).squeeze(1)
        return delta, True, False
    except Exception:
        try:
            delta = torch.linalg.solve(matrix, -gradient.unsqueeze(1)).squeeze(1)
            return delta, False, True
        except Exception:
            return None, False, False


def compute_scaled_gradient(jacobian: torch.Tensor, residual: torch.Tensor) -> float:
    """Scaled gradient norm ``||J^T r|| / max(1, ||J^T J||)``.

    Normalising by the Hessian norm removes problem-scale dependence so the
    same tolerance works across stages with different parameter dimensions.
    """
    j64 = jacobian.to(torch.float64)
    r64 = residual.to(torch.float64)
    return float((j64.T @ r64).norm() / max(1, (j64.T @ j64).norm()))


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
    progress_every_accepted: int,
    true_objective_fn: Callable[[torch.Tensor], torch.Tensor],
    validate_irls: bool = False,
    init_irls_check: dict[str, Any] | None = None,
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
    init_irls_check:
        Optional pre-computed initial first-order validation (IRLS mode only).
    """
    if true_objective_fn is None:
        raise RuntimeError("identity_lm requires explicit true_obj_fn")

    profile = profile_timing
    progress_every = progress_every_accepted
    do_sync = profile and torch.cuda.is_available()

    def sync() -> None:
        if do_sync:
            torch.cuda.synchronize()

    timing: dict[str, float] = {
        "initial_linearization_ms": 0.0,
        "solver_jacobian_ms": 0.0,
        "validation_ms": 0.0,
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
    jacobian = compute_jacobian(linearization["surrogate_fn"], x)
    if profile:
        timing["solver_jacobian_ms"] += (time.perf_counter() - t_j) * 1000.0
    if profile:
        timing["initial_linearization_ms"] = (time.perf_counter() - t_init) * 1000.0

    # Damping initial value: 1e-3 * max diagonal of J^T J (Marquardt's heuristic).
    damping = float(torch.diag((jacobian.to(torch.float64).T @ jacobian.to(torch.float64))).max()) * 1e-3

    counters = LMSolverCounters(solver_jacobian_count=1, linearization_count=1)
    status = "RUNNING"
    weight_history: list[dict[str, Any]] = [{"outer": -1, **linearization.get("weight_stats", {})}]
    irls_checks: list[dict[str, Any]] = []
    if init_irls_check is not None:
        irls_checks.append(init_irls_check)
    spec = linearization.get("spec", {})

    # ---- Outer accepted-step loop ----
    for outer in range(max_accepted_steps):
        residual_64 = linearization["surrogate_fn"](x).detach().to(torch.float64)
        jacobian_64 = jacobian.to(torch.float64)
        hessian_64 = jacobian_64.T @ jacobian_64
        gradient_64 = jacobian_64.T @ residual_64

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
            counters.solve_attempt_count += 1
            delta, chol_ok, fb_ok = solve_damped_normal_equations(
                hessian_64, gradient_64, damping, param_dim,
            )
            if chol_ok:
                counters.cholesky_success_count += 1
            elif fb_ok:
                counters.solve_fallback_count += 1
            else:
                # Both solvers failed: increase damping and retry.
                damping = min(damping * 10, _LAMBDA_MAX)
                counters.rejected_trials += 1
                if profile:
                    timing["linear_solve_ms"] += (time.perf_counter() - t_ls) * 1000.0
                continue
            if profile:
                timing["linear_solve_ms"] += (time.perf_counter() - t_ls) * 1000.0

            candidate = (x + delta.to(torch.float32)).detach()
            sync()
            t_o = time.perf_counter()
            obj_trial = float(true_objective_fn(candidate).detach().item())
            if profile:
                timing["objective_eval_ms"] += (time.perf_counter() - t_o) * 1000.0

            delta_64 = delta.to(torch.float64)
            predicted_reduction = float(-gradient_64.dot(delta_64) - 0.5 * (delta_64 @ hessian_64 @ delta_64))
            actual_reduction = obj_before - obj_trial
            rho = actual_reduction / max(predicted_reduction, 1e-12) if predicted_reduction > 0 else 0.0

            if predicted_reduction > 0 and actual_reduction > 0 and rho > 0:
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
            residual_now = linearization["surrogate_fn"](x).detach()
            t_j = time.perf_counter()
            jacobian_now = compute_jacobian(linearization["surrogate_fn"], x)
            if profile:
                timing["solver_jacobian_ms"] += (time.perf_counter() - t_j) * 1000.0
            counters.validation_jacobian_count += 1
            scaled_grad = compute_scaled_gradient(jacobian_now, residual_now)
            status = "CONVERGED_SCALED_GRADIENT" if (counters.accepted_steps > 4 and scaled_grad < 1e-4) else "LM_NO_ACCEPTED_TRIAL"
            break

        # ---- Re-linearise at the accepted point ----
        next_linearization = make_linearization(x)
        t_j = time.perf_counter()
        next_jacobian = compute_jacobian(next_linearization["surrogate_fn"], x)
        if profile:
            timing["solver_jacobian_ms"] += (time.perf_counter() - t_j) * 1000.0
        counters.solver_jacobian_count += 1
        counters.linearization_count += 1
        weight_history.append({"outer": outer, **next_linearization.get("weight_stats", {})})

        if validate_irls:
            sync()
            t_v = time.perf_counter()
            check = validate_first_order(
                next_linearization["surrogate_fn"], true_objective_fn, x,
                f"{label}_acc{counters.accepted_steps}", solver_config.irls_gradient_rel_tol,
            )
            if profile:
                timing["validation_ms"] += (time.perf_counter() - t_v) * 1000.0
            irls_checks.append({"label": f"acc_{counters.accepted_steps}", **check})
            if not check["pass"]:
                raise RuntimeError(f"IRLS_FIRST_ORDER_MISMATCH at {label} outer={outer}")

        residual_final = next_linearization["surrogate_fn"](x).detach()
        jacobian_final = next_jacobian.to(torch.float64)
        scaled_grad_final = compute_scaled_gradient(jacobian_final, residual_final)
        sync()
        t_o = time.perf_counter()
        rel = (obj_before - float(true_objective_fn(x).detach().item())) / max(abs(obj_before), 1e-16)
        if profile:
            timing["objective_eval_ms"] += (time.perf_counter() - t_o) * 1000.0

        if progress_every > 0 and counters.accepted_steps % progress_every == 0:
            print(f"[{label}] prog acc={counters.accepted_steps} rej={counters.rejected_trials} "
                  f"obj={float(true_objective_fn(x).detach().item()):.6e} "
                  f"lam={damping:.2e} wall={(time.perf_counter()-t0)*1000:.0f}ms")

        convergence = check_convergence(scaled_grad_final, float(torch.norm(delta)), rel, solver_config)
        if convergence is not None:
            status = convergence
            break
        linearization, jacobian = next_linearization, next_jacobian

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

    first_order_checks: dict[str, Any] = {}
    if irls_checks:
        first_order_checks = {
            "initial": irls_checks[0],
            "accepted_endpoints": irls_checks[1:] if len(irls_checks) > 1 else [],
            "worst_abs_max": max(c["abs_max"] for c in irls_checks),
            "worst_rel_l2": max(c["rel_l2"] for c in irls_checks),
            "all_pass": all(c["pass"] for c in irls_checks),
        }

    print(f"[{label}] {status} acc={counters.accepted_steps} rej={counters.rejected_trials} "
          f"solver_jac={counters.solver_jacobian_count} val_jac={counters.validation_jacobian_count} "
          f"total_jac={counters.jacobian_count_total} lins={counters.linearization_count} "
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
        first_order_checks=first_order_checks,
        spec=spec,
        profile_timing=timing if profile else None,
    )


def run_lbfgs_oracle(
    label: str,
    true_objective_fn: Callable[[torch.Tensor], torch.Tensor],
    raw_metrics_fn: Callable[[torch.Tensor], dict[str, float]],
    initial_parameters: torch.Tensor,
    outer_steps: int,
    solver_config: SolverConfig,
) -> LBFGSResult:
    """Run L-BFGS on the same objective as a reference oracle."""
    x = initial_parameters.detach().clone().requires_grad_(True)
    optimizer = torch.optim.LBFGS(
        [x], line_search_fn="strong_wolfe", lr=solver_config.lbfgs_lr,
        max_iter=solver_config.lbfgs_max_iter,
    )
    closure_calls = [0]
    t0 = time.perf_counter()

    def closure():
        closure_calls[0] += 1
        optimizer.zero_grad()
        obj = true_objective_fn(x)
        if not isinstance(obj, torch.Tensor):
            raise TypeError(f"Type: {type(obj)}")
        if obj.ndim != 0:
            raise RuntimeError(f"Ndims: {obj.shape}")
        obj.backward()
        return obj

    for _ in range(outer_steps):
        optimizer.step(closure)

    final_parameters = x.detach()
    wall_ms = _elapsed_ms(t0, final_parameters, False)
    obj_final = float(true_objective_fn(final_parameters).detach().item())
    print(f"[{label}] cls={closure_calls[0]} obj={obj_final:.6e} rms={raw_metrics_fn(final_parameters)['rms']:.1f}")
    return LBFGSResult(
        label=label,
        parameters=final_parameters,
        objective=obj_final,
        closure_calls=closure_calls[0],
        wall_time_ms=wall_ms,
    )
