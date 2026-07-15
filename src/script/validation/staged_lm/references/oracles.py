"""Autograd and full-forward objective oracles for staged-LM validation."""

from __future__ import annotations

from typing import Any, Callable

import torch

from lib_pano.human_fitting.staged_lm.irls import (
    build_primary_aggregation_spec,
    compute_primary_charbonnier_tensor,
    make_irls_linearization,
    validate_first_order,
)
from lib_pano.human_fitting.staged_lm.residuals import flatten_reprojection_groups


def build_full_true_objective(
    raw_groups_fn: Callable[[torch.Tensor], list[Any]],
    anchor_residual_fn: Callable[[torch.Tensor], torch.Tensor] | None,
    initial_parameters: torch.Tensor,
    *,
    is_irls: bool,
    charbonnier_eps: float,
    rep_scale: float,
) -> Callable[[torch.Tensor], torch.Tensor]:
    """Build the differentiable full-forward objective used by oracles."""
    if is_irls:
        spec = build_primary_aggregation_spec(raw_groups_fn(initial_parameters))

        def objective(parameters: torch.Tensor) -> torch.Tensor:
            value = compute_primary_charbonnier_tensor(
                raw_groups_fn(parameters), charbonnier_eps, spec,
            )
            if anchor_residual_fn is not None:
                value = value + 0.5 * torch.sum(anchor_residual_fn(parameters) ** 2)
            return value
    else:
        def objective(parameters: torch.Tensor) -> torch.Tensor:
            residual = flatten_reprojection_groups(raw_groups_fn(parameters)) * rep_scale
            if anchor_residual_fn is not None:
                residual = torch.cat((residual, anchor_residual_fn(parameters)))
            return 0.5 * torch.sum(residual ** 2)
    return objective


__all__ = [
    "build_full_true_objective",
    "make_irls_linearization",
    "validate_first_order",
]
