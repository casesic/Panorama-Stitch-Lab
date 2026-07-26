"""Construction of the production Stage 2 pose-warmup problem."""

from __future__ import annotations

import time
from typing import Any

import torch

from ..config import (
    PRODUCTION_POSE_EVALUATION,
    PRODUCTION_POSE_LINEARIZATION,
    STAGE1_PER_FRAME_DIM,
)
from ..linearization import Linearization
from ..residuals import make_raw_reproj_groups
from ..pose.batched import (
    build_stage2_pose_batch_cache,
    build_stage2_pose_batched_linearization,
)
from ..pose.cache import (
    build_stage2_pose_fast_cache,
)
from ..stage2_shared_cache import Stage2SharedGeometryCache
from ..stage_handoff import check_s1_to_pose, handoff_s1_to_pose
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


def build_pose_stage_definition(
    context: StageContext,
    stage1_outcome: StageOutcome,
    shared_cache: Stage2SharedGeometryCache,
) -> tuple[StageDefinition, dict[str, Any]]:
    """Build fixed analytic/fast Stage 2 pose warmup."""
    per_frame_dim = STAGE1_PER_FRAME_DIM + context.body_pose_dim
    parameter_dimension = context.frame_count * per_frame_dim
    pose_anchor_scale = context.scaling.pose_anchor_scale
    synchronize_context_device(context)
    handoff_start = time.perf_counter()
    initial_parameters = handoff_s1_to_pose(
        stage1_outcome.lm_result.parameters, context.body_pose_anchors,
        context.frame_count, context.body_pose_dim,
    )
    handoff = check_s1_to_pose(
        stage1_outcome.lm_result.parameters, initial_parameters,
        context.body_pose_anchors, context.frame_count, context.body_pose_dim,
    )
    synchronize_context_device(context)
    handoff_ms = (time.perf_counter() - handoff_start) * 1000.0

    synchronize_context_device(context)
    cache_start = time.perf_counter()
    cache = build_stage2_pose_fast_cache(context, shared_cache)
    batch_cache = build_stage2_pose_batch_cache(
        cache, pose_anchor_scale=pose_anchor_scale,
    )
    cache.batch_cache = batch_cache
    cache.batch_cache_setup_ms = batch_cache.setup_ms
    cache.batch_padding_ratio = batch_cache.layout.padding_ratio
    cache.batch_persistent_bytes = batch_cache.persistent_bytes
    cache.batch_group_count = batch_cache.layout.group_count
    cache.batch_nmax = batch_cache.layout.nmax
    synchronize_context_device(context)
    cache_setup_ms = (time.perf_counter() - cache_start) * 1000.0

    synchronize_context_device(context)
    bind_start = time.perf_counter()

    def beta_frozen(_parameters: torch.Tensor) -> torch.Tensor:
        return context.beta_anchor

    def raw_groups_fn(parameters: torch.Tensor):
        return make_raw_reproj_groups(
            parameters, context.frame_indices,
            context.cam2_vertex_ids_by_frame, context.cam2_targets_by_frame,
            context.cam4_vertex_ids_by_frame, context.cam4_targets_by_frame,
            context.frozen_params_by_frame, context.model, context.fitter,
            context.image_width, context.image_height, context.frame_count,
            per_frame_dim, beta_frozen,
        )

    def pose_anchor_residual(parameters: torch.Tensor) -> torch.Tensor:
        return torch.cat([
            (
                parameters[
                    frame * per_frame_dim + STAGE1_PER_FRAME_DIM:
                    (frame + 1) * per_frame_dim
                ] - context.body_pose_anchors[frame]
            ) * pose_anchor_scale
            for frame in range(context.frame_count)
        ])

    aggregation_spec = build_static_aggregation_spec(context)
    input_summary = build_input_summary(
        context, parameter_dimension,
        context.frame_count * context.body_pose_dim,
    )

    def linearization_builder(
        parameters: torch.Tensor, spec: dict[str, Any] | None,
    ) -> Linearization:
        return build_stage2_pose_batched_linearization(
            cache, batch_cache, parameters,
            context.body_pose_anchors, pose_anchor_scale,
            is_irls=context.is_irls, charbonnier_eps=context.charbonnier_eps,
            aggregation_spec=spec, rep_scale=context.scaling.rep_scale,
        )

    def true_objective(parameters: torch.Tensor) -> torch.Tensor:
        return cache.true_objective(
            parameters, context.body_pose_anchors, pose_anchor_scale,
            context.args.reproj_objective, context.charbonnier_eps,
            context.scaling.rep_scale, aggregation_spec,
        )

    stage = StageDefinition(
        name="STAGE2_POSE",
        parameter_dimension=parameter_dimension,
        initial_lm_parameters=initial_parameters,
        raw_groups_fn=raw_groups_fn,
        anchor_residual_fn=pose_anchor_residual,
        raw_metrics_fn=cache.raw_metrics,
        lm_max_accepted_steps=context.solver_config.stage2_pose_warmup_lm_max_acc,
        analytic_cache=cache,
        linearization_builder=linearization_builder,
        lm_true_objective_fn=true_objective,
        eval_backend=PRODUCTION_POSE_EVALUATION,
        linearization_backend=PRODUCTION_POSE_LINEARIZATION,
        linear_solver_backend="dense",
        aggregation_spec=aggregation_spec,
        input_summary=input_summary,
    )
    synchronize_context_device(context)
    stage.preparation_timing = StagePreparationTiming(
        handoff_ms=handoff_ms,
        cache_setup_ms=cache_setup_ms,
        definition_bind_ms=(time.perf_counter() - bind_start) * 1000.0,
    )
    return stage, {"lm": handoff, "lbfgs": None}


def run_pose_stage(
    context: StageContext,
    stage1_outcome: StageOutcome,
    shared_cache: Stage2SharedGeometryCache,
) -> tuple[StageOutcome, StageDefinition, dict[str, Any]]:
    """Build and run fixed production Stage 2 pose warmup."""
    definition, handoff = build_pose_stage_definition(
        context, stage1_outcome, shared_cache,
    )
    return run_stage(definition, context), definition, handoff
