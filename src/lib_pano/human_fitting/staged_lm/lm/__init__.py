"""Canonical staged-LM solver API."""

from .equations import (
    compute_jacobian,
    compute_scaled_gradient,
    compute_scaled_gradient_from_normal_equations,
    normal_equations_hessian_norm,
    normal_equations_max_diagonal,
    normal_equations_quadratic_form,
    solve_damped_normal_equations,
    solve_normal_equations,
)
from .convergence import check_convergence
from .runner import run_lm_optimization

__all__ = [
    "compute_jacobian",
    "solve_damped_normal_equations",
    "compute_scaled_gradient_from_normal_equations",
    "normal_equations_max_diagonal",
    "normal_equations_hessian_norm",
    "normal_equations_quadratic_form",
    "compute_scaled_gradient",
    "solve_normal_equations",
    "check_convergence",
    "run_lm_optimization",
]
