"""Levenberg-Marquardt optimiser for staged fitting.

The LM solver (``run_lm_optimization``) is the primary optimiser under test.
It implements a classic damped Gauss-Newton iteration with:

* Cholesky-based normal-equation solve with a ``torch.linalg.solve`` fallback.
* Levenberg-Marquardz damping update (``rho > 0.75`` -> shrink, ``rho < 0.25``
  -> grow).
* Three convergence criteria: scaled gradient, step norm, relative objective
  change.
* Optional IRLS first-order validation after each accepted step.

The numerical behaviour — damping initialisation, lambda update rule,
predicted/actual reduction, Cholesky-then-fallback order, convergence
thresholds and float32/float64 usage — is identical to the original
``identity_lm`` and ``run_lbfgs`` functions.
"""

from __future__ import annotations

import math
import time
from typing import Any, Callable

import torch
from torch.func import jacfwd

from .irls import validate_first_order
from .config import SolverConfig
from .types import LMResult, LMSolverCounters, NormalEquations

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
    param_dim: int,
) -> tuple[torch.Tensor | None, bool, bool]:
    """Solve ``(H + lambda*I) delta = -g`` with Cholesky then dense fallback.

    Returns
    -------
    (delta, cholesky_ok, fallback_ok):
        ``delta`` is ``None`` if both solvers fail.  The two booleans indicate
        which path succeeded so the caller can update counters.
    """
    identity = torch.eye(
        param_dim, dtype=hessian.dtype, device=hessian.device,
    )
    matrix = hessian + damping * identity
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


def compute_scaled_gradient_from_normal_equations(
    hessian: torch.Tensor, gradient: torch.Tensor,
) -> float:
    """Scaled gradient ``||g|| / max(1, ||H||)`` for either linearization path.

    Normalising by the Hessian norm removes problem-scale dependence so the
    same tolerance works across stages with different parameter dimensions.
    """
    return float(gradient.norm() / max(1, hessian.norm()))


def _validate_normal_equation_representation(equations: NormalEquations) -> None:
    dense = equations.hessian is not None
    structured = equations.structured_system is not None
    if dense == structured:
        raise RuntimeError("INVALID_NORMAL_EQUATION_REPRESENTATION")
    if structured:
        flat = equations.structured_system.flat_gradient()
        if equations.gradient.shape != flat.shape or not torch.equal(equations.gradient, flat):
            raise RuntimeError("INVALID_NORMAL_EQUATION_REPRESENTATION")


def normal_equations_max_diagonal(equations: NormalEquations) -> float:
    _validate_normal_equation_representation(equations)
    value = (torch.diagonal(equations.hessian).max() if equations.hessian is not None
             else equations.structured_system.max_diagonal())
    return float(value)


def normal_equations_hessian_norm(equations: NormalEquations) -> torch.Tensor:
    _validate_normal_equation_representation(equations)
    return (equations.hessian.norm() if equations.hessian is not None
            else equations.structured_system.hessian_frobenius_norm())


def normal_equations_quadratic_form(
    equations: NormalEquations, delta: torch.Tensor,
) -> torch.Tensor:
    _validate_normal_equation_representation(equations)
    if equations.hessian is not None:
        return delta @ equations.hessian @ delta
    return equations.structured_system.quadratic_form(delta)


def compute_scaled_gradient(equations: NormalEquations) -> float:
    return float(equations.gradient.norm() / max(1, normal_equations_hessian_norm(equations)))


def solve_normal_equations(
    equations: NormalEquations, damping: float, param_dim: int,
) -> tuple[torch.Tensor | None, bool, bool, bool, bool, bool]:
    """Solve dense or structured equations.

    Returns ``delta, cholesky_ok, fallback_ok, schur_ok,
    schur_dense_fallback_ok, structured_to_dense``.
    """
    _validate_normal_equation_representation(equations)
    if equations.hessian is not None:
        delta, chol, fallback = solve_damped_normal_equations(
            equations.hessian, equations.gradient, damping, param_dim,
        )
        return delta, chol, fallback, False, False, False
    system = equations.structured_system
    delta = system.solve_damped(damping)
    if delta is not None:
        return delta, True, False, True, False, False
    dense = system.to_dense()
    if system.timing_sink is not None:
        system.timing_sink.schur_dense_fallback_calls += 1
    delta, chol, fallback = solve_damped_normal_equations(
        dense, equations.gradient, damping, param_dim,
    )
    dense_ok = delta is not None
    # A successful dense fallback is represented by the legacy fallback
    # counter regardless of whether dense Cholesky or linalg.solve succeeded.
    return delta, False, dense_ok, False, dense_ok, True


def _evaluate_linearization(
    linearization: dict[str, Any], x: torch.Tensor,
) -> tuple[NormalEquations, bool]:
    normal_equations_fn = linearization.get("normal_equations_fn")
    if normal_equations_fn is not None:
        equations = normal_equations_fn(x)
        if not isinstance(equations, NormalEquations):
            raise TypeError(f"normal_equations_fn returned {type(equations)}")
        _validate_normal_equation_representation(equations)
        return equations, True
    residual = linearization["surrogate_fn"](x).detach()
    jacobian = compute_jacobian(linearization, x)
    jacobian_64 = jacobian.to(torch.float64)
    residual_64 = residual.to(torch.float64)
    return NormalEquations(
        residual=residual,
        hessian=jacobian_64.T @ jacobian_64,
        gradient=jacobian_64.T @ residual_64,
    ), False


def _validate_equations_device(equations: NormalEquations, parameters: torch.Tensor) -> None:
    """Reject any CPU fallback or cross-device normal-equation representation."""
    expected = parameters.device
    tensors = [("residual", equations.residual), ("gradient", equations.gradient)]
    if equations.hessian is not None:
        tensors.append(("hessian", equations.hessian))
    elif equations.structured_system is not None:
        system = equations.structured_system
        tensors.extend((
            ("frame_hessians", system.frame_hessians),
            ("frame_beta_cross", system.frame_beta_cross),
            ("beta_hessian", system.beta_hessian),
            ("frame_gradients", system.frame_gradients),
            ("beta_gradient", system.beta_gradient),
        ))
    for name, tensor in tensors:
        if tensor.device != expected:
            error = "UNEXPECTED_CPU_FALLBACK_IN_CUDA_PIPELINE" if parameters.is_cuda else "DEVICE_CONTRACT_VIOLATION"
            raise RuntimeError(f"{error}: {name} expected={expected} actual={tensor.device}")
    if equations.gradient.dtype != torch.float64:
        raise RuntimeError(
            "DEVICE_CONTRACT_VIOLATION: gradient "
            f"expected_dtype=torch.float64 actual_dtype={equations.gradient.dtype}"
        )
    if equations.hessian is not None and equations.hessian.dtype != torch.float64:
        raise RuntimeError(
            "DEVICE_CONTRACT_VIOLATION: hessian "
            f"expected_dtype=torch.float64 actual_dtype={equations.hessian.dtype}"
        )
    if equations.structured_system is not None:
        for name in (
            "frame_hessians", "frame_beta_cross", "beta_hessian",
            "frame_gradients", "beta_gradient",
        ):
            tensor = getattr(equations.structured_system, name)
            if tensor.dtype != torch.float64:
                raise RuntimeError(
                    "DEVICE_CONTRACT_VIOLATION: "
                    f"structured.{name} expected_dtype=torch.float64 "
                    f"actual_dtype={tensor.dtype}"
                )


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
    do_sync = profile and initial_parameters.is_cuda

    def sync() -> None:
        if do_sync:
            torch.cuda.synchronize(initial_parameters.device)

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
    equations, uses_direct_normal_equations = _evaluate_linearization(linearization, x)
    _validate_equations_device(equations, x)
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
    irls_checks: list[dict[str, Any]] = []
    if init_irls_check is not None:
        irls_checks.append(init_irls_check)
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
            counters.solve_attempt_count += 1
            trial_damping = damping
            delta, chol_ok, fb_ok, schur_ok, schur_fb_ok, materialized = solve_normal_equations(
                equations, damping, param_dim,
            )
            counters.schur_success_count += int(schur_ok)
            counters.schur_dense_fallback_count += int(schur_fb_ok)
            counters.schur_failure_count += int(materialized)
            counters.structured_to_dense_count += int(materialized)
            if chol_ok:
                counters.cholesky_success_count += 1
            elif fb_ok:
                counters.solve_fallback_count += 1
            else:
                # Both solvers failed: increase damping and retry.
                damping = min(damping * 10, _LAMBDA_MAX)
                counters.rejected_trials += 1
                trial_history.append({
                    "outer": outer, "trial": _trial, "damping": trial_damping,
                    "linear_solver_backend": linear_solver_backend,
                    "solve_success": False, "delta_norm": None,
                    "predicted_reduction": None, "actual_reduction": None,
                    "rho": None, "accepted": False,
                    "used_dense_fallback": materialized,
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
                "accepted": accepted_trial, "used_dense_fallback": materialized,
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
                counters.validation_jacobian_count += 1
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
        _validate_equations_device(next_equations, x)
        if profile:
            timing["solver_jacobian_ms"] += (time.perf_counter() - t_j) * 1000.0
        counters.solver_jacobian_count += 1
        counters.linearization_count += 1
        counters.normal_equation_count += int(next_uses_direct)
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

        scaled_grad_final = compute_scaled_gradient(next_equations)
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
        jacobian_backend=linearization.get("jacobian_backend", "autograd"),
        linearization_backend=linearization.get("linearization_backend", "jacobian"),
        linear_solver_backend=linear_solver_backend,
        trial_history=trial_history,
        profile_timing=timing if profile else None,
    )

