"""Validation-only extensions of production staged-LM result structures."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch

from lib_pano.human_fitting.staged_lm.types import StageDefinition


LBFGS_MAX_ITER = 10
LBFGS_LR = 0.3
STAGE1_LBFGS_OUTER_STEPS = 15
POSE_LBFGS_OUTER_STEPS = 20
JOINT_LBFGS_OUTER_STEPS = 20


@dataclass
class LBFGSResult:
    """Validation-only result of one L-BFGS oracle run."""

    label: str
    parameters: torch.Tensor
    objective: float
    closure_calls: int
    wall_time_ms: float


@dataclass
class ReferenceStageOutcome:
    """Validation result containing optional L-BFGS oracle output."""

    lm_result: object
    lbfgs_result: LBFGSResult | None
    verdict: str | None
    true_objective_fn: Callable[[torch.Tensor], torch.Tensor]
    definition: StageDefinition | None = None


@dataclass
class ReferenceStageDefinition(StageDefinition):
    """Stage definition fields needed only by oracle/regression runners."""

    initial_lbfgs_parameters: torch.Tensor | None = None
    lbfgs_outer_steps: int = 0
    analytic_linearization: bool = False
    lbfgs_true_objective_fn: Callable[[torch.Tensor], torch.Tensor] | None = None
    fast_eval_validator: Callable[[str, torch.Tensor], None] | None = None
    full_forward_oracle_validation_ms: float = 0.0
