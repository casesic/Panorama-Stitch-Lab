"""Typed contract between staged fitting problems and the LM solver."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

import torch

from .types import NormalEquations

TensorFunction = Callable[[torch.Tensor], torch.Tensor]
NormalEquationsFunction = Callable[[torch.Tensor], NormalEquations]
WeightStatistics = Mapping[str, Any]
AggregationSpec = Mapping[str, Any]


@dataclass(frozen=True)
class LinearizationProvenance:
    """Names written unchanged to LM result and profiling output."""

    jacobian_backend: str = "autograd"
    linearization_backend: str = "jacobian"
    linear_solver_backend: str = "dense"


@dataclass(frozen=True)
class Linearization:
    """One frozen-weight local least-squares model at an LM iterate."""

    surrogate: TensorFunction
    jacobian: TensorFunction | None = None
    normal_equations: NormalEquationsFunction | None = None
    weight_stats: WeightStatistics = field(default_factory=dict)
    aggregation_spec: AggregationSpec = field(default_factory=dict)
    provenance: LinearizationProvenance = field(
        default_factory=LinearizationProvenance,
    )

    def __post_init__(self) -> None:
        if not callable(self.surrogate):
            raise TypeError("Linearization.surrogate must be callable")
        if self.jacobian is not None and not callable(self.jacobian):
            raise TypeError("Linearization.jacobian must be callable or None")
        if self.normal_equations is not None and not callable(self.normal_equations):
            raise TypeError(
                "Linearization.normal_equations must be callable or None"
            )

    @property
    def uses_direct_normal_equations(self) -> bool:
        return self.normal_equations is not None

    @property
    def has_analytic_jacobian(self) -> bool:
        return self.jacobian is not None
