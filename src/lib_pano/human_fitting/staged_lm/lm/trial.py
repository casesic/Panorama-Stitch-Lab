"""Scalar LM trial reduction, acceptance, and history-record construction."""

from __future__ import annotations

from typing import Any

import torch

from ..types import NormalEquations
from .equations import normal_equations_quadratic_form


def evaluate_trial_reduction(
    equations: NormalEquations,
    gradient_64: torch.Tensor,
    delta_64: torch.Tensor,
    objective_before: float,
    objective_candidate: float,
) -> tuple[float, float, float, bool]:
    predicted_reduction = float(
        -gradient_64.dot(delta_64)
        - 0.5 * normal_equations_quadratic_form(
            equations,
            delta_64,
        )
    )
    actual_reduction = objective_before - objective_candidate
    rho = (
        actual_reduction
        / max(predicted_reduction, 1e-12)
        if predicted_reduction > 0
        else 0.0
    )
    accepted = (
        predicted_reduction > 0
        and actual_reduction > 0
        and rho > 0
    )
    return predicted_reduction, actual_reduction, rho, accepted


def build_failed_trial_record(
    outer: int,
    trial: int,
    damping: float,
    linear_solver_backend: str,
    used_dense_fallback: bool,
) -> dict[str, Any]:
    return {
        "outer": outer, "trial": trial, "damping": damping,
        "linear_solver_backend": linear_solver_backend,
        "solve_success": False, "delta_norm": None,
        "predicted_reduction": None, "actual_reduction": None,
        "rho": None, "accepted": False,
        "used_dense_fallback": used_dense_fallback,
    }


def build_evaluated_trial_record(
    outer: int,
    trial: int,
    damping: float,
    linear_solver_backend: str,
    delta: torch.Tensor,
    predicted_reduction: float,
    actual_reduction: float,
    rho: float,
    accepted: bool,
    used_dense_fallback: bool,
) -> dict[str, Any]:
    return {
        "outer": outer, "trial": trial, "damping": damping,
        "linear_solver_backend": linear_solver_backend,
        "solve_success": True, "delta_norm": float(delta.norm()),
        "predicted_reduction": predicted_reduction,
        "actual_reduction": actual_reduction, "rho": rho,
        "accepted": accepted,
        "used_dense_fallback": used_dense_fallback,
    }

