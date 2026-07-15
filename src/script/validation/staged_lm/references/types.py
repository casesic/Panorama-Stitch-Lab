"""Validation-only extensions of production staged-LM result structures."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch

from lib_pano.human_fitting.staged_lm.types import StageDefinition


@dataclass
class ReferenceStageDefinition(StageDefinition):
    """Stage definition fields needed only by oracle/regression runners."""

    initial_lbfgs_parameters: torch.Tensor | None = None
    lbfgs_outer_steps: int = 0
    analytic_linearization: bool = False
    lbfgs_true_objective_fn: Callable[[torch.Tensor], torch.Tensor] | None = None
    fast_eval_validator: Callable[[str, torch.Tensor], None] | None = None
    full_forward_oracle_validation_ms: float = 0.0
