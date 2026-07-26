"""Generic execution of one fully constructed staged-LM stage."""

from __future__ import annotations

from typing import Callable

import torch

from .device import validate_stage_initial_parameters
from .linearization import Linearization
from .lm.runner import run_lm_optimization
from .types import StageContext, StageDefinition, StageOutcome


def _build_linearization_and_objective(
    stage: StageDefinition,
) -> tuple[
    Callable[[torch.Tensor], Linearization],
    Callable[[torch.Tensor], torch.Tensor],
]:
    """Bind the production analytic linearization and fast true objective."""
    if stage.linearization_builder is None or stage.lm_true_objective_fn is None:
        raise RuntimeError("INCOMPLETE_PRODUCTION_STAGE_DEFINITION")
    def make_linearization(parameters: torch.Tensor) -> Linearization:
        return stage.linearization_builder(parameters, stage.aggregation_spec)

    return make_linearization, stage.lm_true_objective_fn


def run_stage(stage: StageDefinition, context: StageContext) -> StageOutcome:
    """Run one production LM stage without validation/oracle side paths."""
    if stage.analytic_cache is not None:
        stage.analytic_cache.reset_timing()
    validate_stage_initial_parameters(
        stage, context,
        include_cache=getattr(context.args, "validate_device_contract", False),
    )
    make_linearization, true_objective = _build_linearization_and_objective(stage)
    summary = stage.input_summary
    if summary is None:
        raise RuntimeError("MISSING_PRODUCTION_STAGE_INPUT_SUMMARY")
    print(
        f"\n{'=' * 60}\n[{stage.name}] T={context.frame_count} "
        f"param_dim={stage.parameter_dimension} groups={summary.group_count} "
        f"valid_pts={summary.total_valid_points} "
        f"surr_dim={summary.surrogate_residual_dimension}\n{'=' * 60}"
    )

    result = run_lm_optimization(
        label=f"{stage.name}_LM",
        make_linearization=make_linearization,
        raw_metrics_fn=stage.raw_metrics_fn,
        initial_parameters=stage.initial_lm_parameters,
        param_dim=stage.parameter_dimension,
        max_accepted_steps=stage.lm_max_accepted_steps,
        max_trials=context.solver_config.lm_max_trials,
        solver_config=context.solver_config,
        profile_timing=context.args.profile_timing,
        true_objective_fn=true_objective,
    )
    result.eval_backend = stage.eval_backend
    if context.args.profile_timing and stage.analytic_cache is not None:
        if result.profile_timing is not None:
            result.profile_timing.update(stage.analytic_cache.timing_dict())
    if result.profile_timing is not None:
        result.profile_timing["eval_backend"] = stage.eval_backend
        result.profile_timing["linearization_backend"] = stage.linearization_backend
        result.profile_timing["linear_solver_backend"] = stage.linear_solver_backend

    return StageOutcome(
        lm_result=result,
        lbfgs_result=None,
        verdict=None,
        true_objective_fn=true_objective,
        definition=stage,
    )
