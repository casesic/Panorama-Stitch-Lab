"""Construction of the production Stage 1 rigid-alignment problem."""

from __future__ import annotations

import time
from typing import Any

import torch

from ..analytic_stage1 import (
    build_stage1_analytic_cache,
    build_stage1_analytic_linearization,
)
from ..config import (
    PRODUCTION_STAGE1_EVALUATION,
    PRODUCTION_STAGE1_LINEARIZATION,
    STAGE1_PER_FRAME_DIM,
)
from ..linearization import Linearization
from ..parameter_layout import build_stage1_initial_parameters
from ..residuals import make_raw_reproj_groups
from ..stage_runtime import run_stage
from ..types import (
    StageContext,
    StageDefinition,
    StageOutcome,
    StagePreparationTiming,
)
from .common import (
    build_input_summary,
    build_static_aggregation_spec,
    synchronize_context_device,
)


def _build_stage1_initial(context: StageContext) -> torch.Tensor:
    return build_stage1_initial_parameters(
        context.cam2_smplx_parameters,
        context.cam4_smplx_parameters,
        context.frame_count,
    )


def build_stage1_definition(context: StageContext) -> StageDefinition:
    """Build fixed analytic/fast Stage 1."""
    synchronize_context_device(context)
    cache_start = time.perf_counter()
    cache = build_stage1_analytic_cache(context)
    synchronize_context_device(context)
    cache_setup_ms = (time.perf_counter() - cache_start) * 1000.0

    synchronize_context_device(context)
    bind_start = time.perf_counter()
    initial_parameters = _build_stage1_initial(context)
    parameter_dimension = context.frame_count * STAGE1_PER_FRAME_DIM

    def beta_frozen(_parameters: torch.Tensor) -> torch.Tensor:
        return context.beta_anchor

    def raw_groups_fn(parameters: torch.Tensor):
        return make_raw_reproj_groups(
            parameters, context.frame_indices,
            context.cam2_vertex_ids_by_frame, context.cam2_targets_by_frame,
            context.cam4_vertex_ids_by_frame, context.cam4_targets_by_frame,
            context.frozen_params_by_frame, context.model, context.fitter,
            context.image_width, context.image_height, context.frame_count,
            STAGE1_PER_FRAME_DIM, beta_frozen, context.body_pose_anchors,
        )

    aggregation_spec = build_static_aggregation_spec(context)
    input_summary = build_input_summary(context, parameter_dimension, 0)

    def linearization_builder(
        parameters: torch.Tensor, spec: dict[str, Any] | None,
    ) -> Linearization:
        return build_stage1_analytic_linearization(
            cache, parameters, is_irls=context.is_irls,
            charbonnier_eps=context.charbonnier_eps,
            aggregation_spec=spec, rep_scale=context.scaling.rep_scale,
        )

    def true_objective(parameters: torch.Tensor) -> torch.Tensor:
        return cache.true_objective(
            parameters, context.args.reproj_objective,
            context.charbonnier_eps, context.scaling.rep_scale,
            aggregation_spec,
        )

    stage = StageDefinition(
        name="S1",
        parameter_dimension=parameter_dimension,
        initial_lm_parameters=initial_parameters,
        raw_groups_fn=raw_groups_fn,
        anchor_residual_fn=None,
        raw_metrics_fn=cache.raw_metrics,
        lm_max_accepted_steps=context.solver_config.stage1_lm_max_acc,
        analytic_cache=cache,
        linearization_builder=linearization_builder,
        lm_true_objective_fn=true_objective,
        eval_backend=PRODUCTION_STAGE1_EVALUATION,
        linearization_backend=PRODUCTION_STAGE1_LINEARIZATION,
        linear_solver_backend="dense",
        aggregation_spec=aggregation_spec,
        input_summary=input_summary,
    )
    synchronize_context_device(context)
    stage.preparation_timing = StagePreparationTiming(
        handoff_ms=0.0,
        cache_setup_ms=cache_setup_ms,
        definition_bind_ms=(time.perf_counter() - bind_start) * 1000.0,
    )
    return stage


def run_stage1(context: StageContext) -> StageOutcome:
    """Build and run fixed production Stage 1."""
    return run_stage(build_stage1_definition(context), context)
