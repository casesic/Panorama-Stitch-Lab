"""Construction of the production Stage 2 pose/beta joint problem."""

from __future__ import annotations

import time
from typing import Any

import torch

from ..config import (
    PRODUCTION_JOINT_EVALUATION,
    PRODUCTION_JOINT_LINEARIZATION,
    PRODUCTION_JOINT_LINEAR_SOLVER,
    STAGE1_PER_FRAME_DIM,
)
from ..linearization import Linearization
from ..residuals import make_raw_reproj_groups
from ..joint.batched import (
    build_stage2_joint_batch_cache,
    build_stage2_joint_batched_linearization,
)
from ..joint.cache import build_stage2_joint_fast_cache
from ..joint.objective import stage2_joint_fast_objective_terms
from ..stage2_shared_cache import Stage2SharedGeometryCache
from ..stage_handoff import check_pose_to_joint_state
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


def build_joint_stage_definition(
    context: StageContext,
    pose_outcome: StageOutcome,
    shared_cache: Stage2SharedGeometryCache,
) -> tuple[StageDefinition, dict[str, Any]]:
    """Build fixed analytic/fast/Schur/combined-FK joint refinement."""
    per_frame_dim = STAGE1_PER_FRAME_DIM + context.body_pose_dim
    pose_dimension = context.frame_count * per_frame_dim
    parameter_dimension = pose_dimension + context.beta_dim
    beta_abs_limit = context.objective_config.beta_abs_limit
    pose_anchor_scale = context.scaling.pose_anchor_scale
    shape_anchor_scale = context.scaling.shape_anchor_scale
    synchronize_context_device(context)
    handoff_start = time.perf_counter()
    beta_latent = torch.atanh(context.beta_anchor / beta_abs_limit)
    initial_parameters = torch.cat((
        pose_outcome.lm_result.parameters.clone(), beta_latent.clone(),
    )).clone()
    handoff = check_pose_to_joint_state(
        pose_outcome.lm_result.parameters, initial_parameters,
        context.beta_anchor, beta_abs_limit, shape_anchor_scale,
    )
    synchronize_context_device(context)
    handoff_ms = (time.perf_counter() - handoff_start) * 1000.0

    synchronize_context_device(context)
    cache_start = time.perf_counter()
    cache = build_stage2_joint_fast_cache(context, shared_cache)
    batch_cache = build_stage2_joint_batch_cache(
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

    def beta_active(parameters: torch.Tensor) -> torch.Tensor:
        return beta_abs_limit * torch.tanh(parameters[-context.beta_dim:])

    def raw_groups_fn(parameters: torch.Tensor):
        return make_raw_reproj_groups(
            parameters, context.frame_indices,
            context.cam2_vertex_ids_by_frame, context.cam2_targets_by_frame,
            context.cam4_vertex_ids_by_frame, context.cam4_targets_by_frame,
            context.frozen_params_by_frame, context.model, context.fitter,
            context.image_width, context.image_height, context.frame_count,
            per_frame_dim, beta_active,
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

    def shape_anchor_residual(parameters: torch.Tensor) -> torch.Tensor:
        return (
            beta_abs_limit * torch.tanh(parameters[-context.beta_dim:])
            - context.beta_anchor
        ) * shape_anchor_scale

    def all_anchors_residual(parameters: torch.Tensor) -> torch.Tensor:
        return torch.cat((
            pose_anchor_residual(parameters), shape_anchor_residual(parameters),
        ))

    aggregation_spec = build_static_aggregation_spec(context)
    input_summary = build_input_summary(
        context, parameter_dimension,
        context.frame_count * context.body_pose_dim + context.beta_dim,
    )

    def linearization_builder(
        parameters: torch.Tensor, spec: dict[str, Any] | None,
    ) -> Linearization:
        return build_stage2_joint_batched_linearization(
            cache, batch_cache, parameters, beta_abs_limit, context.beta_anchor,
            context.body_pose_anchors, pose_anchor_scale, shape_anchor_scale,
            is_irls=context.is_irls, charbonnier_eps=context.charbonnier_eps,
            aggregation_spec=spec, rep_scale=context.scaling.rep_scale,
        )

    def true_objective(parameters: torch.Tensor) -> torch.Tensor:
        return cache.true_objective(
            parameters, beta_abs_limit, context.beta_anchor,
            context.body_pose_anchors, pose_anchor_scale, shape_anchor_scale,
            context.args.reproj_objective, context.charbonnier_eps,
            context.scaling.rep_scale, aggregation_spec,
        )

    def objective_terms(
        parameters: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        return stage2_joint_fast_objective_terms(
            cache, parameters, beta_abs_limit, context.beta_anchor,
            context.body_pose_anchors, pose_anchor_scale, shape_anchor_scale,
            context.args.reproj_objective, context.charbonnier_eps,
            context.scaling.rep_scale, aggregation_spec,
        )

    stage = StageDefinition(
        name="STAGE2_JOINT",
        parameter_dimension=parameter_dimension,
        initial_lm_parameters=initial_parameters,
        raw_groups_fn=raw_groups_fn,
        anchor_residual_fn=all_anchors_residual,
        raw_metrics_fn=lambda parameters: cache.raw_metrics(parameters, beta_abs_limit),
        lm_max_accepted_steps=context.solver_config.stage2_joint_lm_max_acc,
        analytic_cache=cache,
        linearization_builder=linearization_builder,
        lm_true_objective_fn=true_objective,
        eval_backend=PRODUCTION_JOINT_EVALUATION,
        linearization_backend=PRODUCTION_JOINT_LINEARIZATION,
        linear_solver_backend=PRODUCTION_JOINT_LINEAR_SOLVER,
        aggregation_spec=aggregation_spec,
        input_summary=input_summary,
        objective_terms_fn=objective_terms,
    )
    synchronize_context_device(context)
    stage.preparation_timing = StagePreparationTiming(
        handoff_ms=handoff_ms,
        cache_setup_ms=cache_setup_ms,
        definition_bind_ms=(time.perf_counter() - bind_start) * 1000.0,
    )
    return stage, {"lm": handoff, "lbfgs": None}


def run_joint_stage(
    context: StageContext,
    pose_outcome: StageOutcome,
    shared_cache: Stage2SharedGeometryCache,
) -> tuple[StageOutcome, StageDefinition, dict[str, Any]]:
    """Build and run fixed production Stage 2 joint refinement."""
    definition, handoff = build_joint_stage_definition(
        context, pose_outcome, shared_cache,
    )
    return run_stage(definition, context), definition, handoff
