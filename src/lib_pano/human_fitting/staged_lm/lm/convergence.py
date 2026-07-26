"""LM damping updates and convergence/status policy.

All comparisons and legacy thresholds intentionally preserve production
behavior. This module contains no tensor solve and does not run the solver.
"""

from __future__ import annotations

import math

from ..config import SolverConfig

_LAMBDA_MIN = 1e-10
_LAMBDA_MAX = 1e12


def initialise_damping_from_max_diagonal(initial_max_diagonal: float) -> float:
    if not math.isfinite(initial_max_diagonal) or initial_max_diagonal <= 0:
        raise RuntimeError("INVALID_INITIAL_LM_DAMPING")
    return initial_max_diagonal * 1e-3


def increase_damping_after_rejection(damping: float) -> float:
    return min(damping * 10, _LAMBDA_MAX)


def update_damping_after_acceptance(damping: float, rho: float) -> float:
    if rho > 0.75:
        return max(damping / 3, _LAMBDA_MIN)
    elif rho < 0.25:
        return min(damping * 3, _LAMBDA_MAX)
    return damping


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


def classify_no_accepted_trial(
    accepted_steps: int,
    scaled_gradient: float,
) -> str:
    """Legacy behavior intentionally preserved; it is not config-consistent."""
    return (
        "CONVERGED_SCALED_GRADIENT"
        if accepted_steps > 4 and scaled_gradient < 1e-4
        else "LM_NO_ACCEPTED_TRIAL"
    )

