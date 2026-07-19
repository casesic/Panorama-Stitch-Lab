"""Single-path production orchestration for staged SMPL-X fitting.

The production path is fixed to analytic Stage 1, analytic/fast Stage 2 pose,
and analytic/fast/Schur/combined-FK Stage 2 joint refinement. Alternative
autograd, full-forward, dense-joint, separate-FK, and L-BFGS implementations
live in ``src/script/validation/staged_lm/references``.
"""

from __future__ import annotations

import contextlib
import time
from typing import Any, Callable

import torch

from .analytic_stage1 import (
    build_stage1_analytic_cache,
    build_stage1_analytic_linearization,
)
from .config import (
    PRODUCTION_JOINT_EVALUATION,
    PRODUCTION_JOINT_LINEARIZATION,
    PRODUCTION_JOINT_LINEAR_SOLVER,
    PRODUCTION_POSE_EVALUATION,
    PRODUCTION_POSE_LINEARIZATION,
    PRODUCTION_STAGE1_EVALUATION,
    PRODUCTION_STAGE1_LINEARIZATION,
    STAGE1_PER_FRAME_DIM,
)
from .device import validate_stage_initial_parameters
from .irls import build_primary_aggregation_spec_from_observations
from .lm_solver import run_lm_optimization
from .parameter_layout import build_stage1_initial_parameters
from .residuals import make_raw_reproj_groups
from .stage2_joint_fast import (
    build_stage2_joint_fast_cache,
    stage2_joint_fast_objective_terms,
)
from .stage2_joint_batched import (
    build_stage2_joint_batch_cache,
    build_stage2_joint_batched_linearization,
)
from .stage2_pose_fast import build_stage2_pose_fast_cache
from .stage2_pose_batched import (
    build_stage2_pose_batch_cache,
    build_stage2_pose_batched_linearization,
)
from .stage2_shared_cache import (
    Stage2SharedGeometryCache,
    build_stage2_shared_geometry_cache,
)
from .types import (
    Stage2Outcome,
    StageContext,
    StageDefinition,
    StageInputSummary,
    StageOutcome,
    StagePreparationTiming,
)


def _build_linearization_and_objective(
    stage: StageDefinition,
) -> tuple[
    Callable[[torch.Tensor], dict[str, Any]],
    Callable[[torch.Tensor], torch.Tensor],
]:
    """Bind the production analytic linearization and fast true objective."""
    if stage.linearization_builder is None or stage.lm_true_objective_fn is None:
        raise RuntimeError("INCOMPLETE_PRODUCTION_STAGE_DEFINITION")
    def make_linearization(parameters: torch.Tensor) -> dict[str, Any]:
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
        progress_every_accepted=context.args.progress_every_accepted,
        true_objective_fn=true_objective,
        validate_irls=False,
        init_irls_check=None,
    )
    result.eval_backend = stage.eval_backend
    if stage.name == "S1":
        result.stage1_eval_backend = stage.eval_backend
    if context.args.profile_timing and stage.analytic_cache is not None:
        if result.profile_timing is not None:
            result.profile_timing.update(stage.analytic_cache.timing_dict())
    if result.profile_timing is not None:
        result.profile_timing["eval_backend"] = stage.eval_backend
        result.profile_timing["linearization_backend"] = stage.linearization_backend
        result.profile_timing["linear_solver_backend"] = stage.linear_solver_backend
        if stage.name == "S1":
            result.profile_timing["stage1_eval_backend"] = stage.eval_backend
        result.profile_timing["full_forward_oracle_validation_ms"] = 0.0

    return StageOutcome(
        lm_result=result,
        lbfgs_result=None,
        verdict=None,
        true_objective_fn=true_objective,
        definition=stage,
    )


def _build_stage1_initial(context: StageContext) -> torch.Tensor:
    return build_stage1_initial_parameters(
        context.cam2_smplx_parameters,
        context.cam4_smplx_parameters,
        context.frame_count,
    )


def _build_input_summary(
    context: StageContext,
    parameter_dimension: int,
    anchor_residual_dimension: int,
) -> StageInputSummary:
    total_valid_points = sum(
        int(ids.numel())
        for ids in (
            *context.cam2_vertex_ids_by_frame,
            *context.cam4_vertex_ids_by_frame,
        )
    )
    reprojection_dimension = 2 * total_valid_points
    return StageInputSummary(
        frame_count=context.frame_count,
        parameter_dimension=parameter_dimension,
        group_count=2 * context.frame_count,
        total_valid_points=total_valid_points,
        reprojection_residual_dimension=reprojection_dimension,
        anchor_residual_dimension=anchor_residual_dimension,
        surrogate_residual_dimension=(
            reprojection_dimension + anchor_residual_dimension
        ),
    )


def _build_static_aggregation_spec(
    context: StageContext,
) -> dict[str, Any] | None:
    if not context.is_irls:
        return None
    return build_primary_aggregation_spec_from_observations(
        context.frame_indices,
        context.cam2_vertex_ids_by_frame,
        context.cam4_vertex_ids_by_frame,
    )


def build_stage1_definition(context: StageContext) -> StageDefinition:
    """Build fixed analytic/fast Stage 1."""
    _pipeline_sync(context)
    cache_start = time.perf_counter()
    cache = build_stage1_analytic_cache(context)
    _pipeline_sync(context)
    cache_setup_ms = (time.perf_counter() - cache_start) * 1000.0

    _pipeline_sync(context)
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

    aggregation_spec = _build_static_aggregation_spec(context)
    input_summary = _build_input_summary(context, parameter_dimension, 0)

    def linearization_builder(
        parameters: torch.Tensor, spec: dict[str, Any] | None,
    ) -> dict[str, Any]:
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
    _pipeline_sync(context)
    stage.preparation_timing = StagePreparationTiming(
        handoff_ms=0.0,
        cache_setup_ms=cache_setup_ms,
        definition_bind_ms=(time.perf_counter() - bind_start) * 1000.0,
    )
    return stage


def run_stage1(context: StageContext) -> StageOutcome:
    """Build and run fixed production Stage 1."""
    return run_stage(build_stage1_definition(context), context)


def build_pose_stage_definition(
    context: StageContext,
    stage1_outcome: StageOutcome,
    shared_cache: Stage2SharedGeometryCache,
) -> tuple[StageDefinition, dict[str, Any]]:
    """Build fixed analytic/fast Stage 2 pose warmup."""
    per_frame_dim = STAGE1_PER_FRAME_DIM + context.body_pose_dim
    parameter_dimension = context.frame_count * per_frame_dim
    pose_anchor_scale = context.scaling.pose_anchor_scale
    _pipeline_sync(context)
    handoff_start = time.perf_counter()
    initial_parameters = handoff_s1_to_pose(
        stage1_outcome.lm_result.parameters, context.body_pose_anchors,
        context.frame_count, context.body_pose_dim,
    )
    handoff = check_s1_to_pose(
        stage1_outcome.lm_result.parameters, initial_parameters,
        context.body_pose_anchors, context.frame_count, context.body_pose_dim,
    )
    _pipeline_sync(context)
    handoff_ms = (time.perf_counter() - handoff_start) * 1000.0

    _pipeline_sync(context)
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
    _pipeline_sync(context)
    cache_setup_ms = (time.perf_counter() - cache_start) * 1000.0

    _pipeline_sync(context)
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

    aggregation_spec = _build_static_aggregation_spec(context)
    input_summary = _build_input_summary(
        context, parameter_dimension,
        context.frame_count * context.body_pose_dim,
    )

    def linearization_builder(
        parameters: torch.Tensor, spec: dict[str, Any] | None,
    ) -> dict[str, Any]:
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
    _pipeline_sync(context)
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
    _pipeline_sync(context)
    handoff_start = time.perf_counter()
    beta_latent = torch.atanh(context.beta_anchor / beta_abs_limit)
    initial_parameters = torch.cat((
        pose_outcome.lm_result.parameters.clone(), beta_latent.clone(),
    )).clone()
    handoff = check_pose_to_joint_state(
        pose_outcome.lm_result.parameters, initial_parameters,
        context.beta_anchor, beta_abs_limit, shape_anchor_scale,
    )
    _pipeline_sync(context)
    handoff_ms = (time.perf_counter() - handoff_start) * 1000.0

    _pipeline_sync(context)
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
    _pipeline_sync(context)
    cache_setup_ms = (time.perf_counter() - cache_start) * 1000.0

    _pipeline_sync(context)
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

    aggregation_spec = _build_static_aggregation_spec(context)
    input_summary = _build_input_summary(
        context, parameter_dimension,
        context.frame_count * context.body_pose_dim + context.beta_dim,
    )

    def linearization_builder(
        parameters: torch.Tensor, spec: dict[str, Any] | None,
    ) -> dict[str, Any]:
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
    _pipeline_sync(context)
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


def _print_handoff(handoff: dict[str, Any] | None, tag: str, kind: str) -> None:
    if handoff is None:
        return
    if kind == "pose":
        print(
            f"[HANDOFF_{tag}_S1_STAGE2_POSE] ot={handoff['ot_max']:.2e} "
            f"bp={handoff['bp_max']:.2e} ok={handoff['ok']}"
        )
    else:
        raw_value = handoff["raw_reprojection_max_diff"]
        relative_value = handoff["relative_difference_joint_init_to_pose"]
        raw_text = "NOT_RUN" if raw_value is None else f"{raw_value:.2e}"
        relative_text = (
            "NOT_RUN" if relative_value is None else f"{relative_value:.2e}"
        )
        print(
            f"[HANDOFF_{tag}_STAGE2_POSE_JOINT] "
            f"state={handoff['state_max_diff']:.2e} "
            f"bp={handoff['physical_beta_init_max_diff_to_anchor']:.2e} "
            f"raw={raw_text} "
            f"sa={handoff['shape_anchor_objective_at_init']:.2e} "
            f"rel={relative_text} "
            f"scope={handoff.get('validation_scope', 'full')} "
            f"ok={handoff['ok']}"
        )


def _pipeline_sync(context: StageContext) -> None:
    if context.device.type == "cuda":
        torch.cuda.synchronize(context.device)


def _record_preparation_timing(
    pipeline_timing: dict[str, float] | None,
    prefix: str,
    stage: StageDefinition,
) -> None:
    if pipeline_timing is None:
        return
    preparation = stage.preparation_timing
    pipeline_timing[f"{prefix}_prepare_total_ms"] = preparation.total_ms
    pipeline_timing[f"{prefix}_handoff_ms"] = preparation.handoff_ms
    pipeline_timing[f"{prefix}_cache_setup_ms"] = preparation.cache_setup_ms
    pipeline_timing[f"{prefix}_definition_bind_ms"] = preparation.definition_bind_ms
    if prefix in ("stage2_pose", "stage2_joint"):
        pipeline_timing[f"{prefix}_derived_cache_setup_ms"] = stage.analytic_cache.setup_ms
    if prefix == "stage2_pose":
        cache = stage.analytic_cache
        pipeline_timing["stage2_pose_fast_cache_setup_ms"] = cache.setup_ms
        pipeline_timing["stage2_pose_batch_cache_setup_ms"] = cache.batch_cache_setup_ms
        pipeline_timing["stage2_pose_batch_padding_ratio"] = cache.batch_padding_ratio
        pipeline_timing["stage2_pose_batch_persistent_bytes"] = cache.batch_persistent_bytes
        pipeline_timing["stage2_pose_batch_group_count"] = cache.batch_group_count
        pipeline_timing["stage2_pose_batch_nmax"] = cache.batch_nmax
    if prefix == "stage2_joint":
        cache = stage.analytic_cache
        pipeline_timing["stage2_joint_fast_cache_setup_ms"] = cache.setup_ms
        pipeline_timing["stage2_joint_batch_cache_setup_ms"] = cache.batch_cache_setup_ms
        pipeline_timing["stage2_joint_batch_padding_ratio"] = cache.batch_padding_ratio
        pipeline_timing["stage2_joint_batch_persistent_bytes"] = cache.batch_persistent_bytes
        pipeline_timing["stage2_joint_batch_group_count"] = cache.batch_group_count
        pipeline_timing["stage2_joint_batch_nmax"] = cache.batch_nmax


def _record_stage2_cache_totals(
    pipeline_timing: dict[str, float] | None,
    shared_cache: Stage2SharedGeometryCache,
    pose_definition: StageDefinition,
    joint_definition: StageDefinition | None = None,
) -> None:
    if pipeline_timing is None:
        return
    pipeline_timing.update(shared_cache.timing_dict())
    pose_preparation = pose_definition.preparation_timing
    joint_preparation = (
        joint_definition.preparation_timing
        if joint_definition is not None else StagePreparationTiming()
    )
    pipeline_timing["stage2_total_cache_setup_ms"] = (
        shared_cache.setup_ms
        + pose_preparation.cache_setup_ms
        + joint_preparation.cache_setup_ms
    )
    pipeline_timing["stage2_prepare_total_ms"] = (
        shared_cache.setup_ms
        + pose_preparation.total_ms
        + joint_preparation.total_ms
    )


def run_stage2(
    context: StageContext,
    stage1_outcome: StageOutcome,
    pipeline_timing: dict[str, float] | None = None,
    profiler_ranges: bool = False,
) -> Stage2Outcome:
    """Run fixed production pose warmup followed by joint refinement."""
    range_fn = (
        torch.profiler.record_function
        if profiler_ranges else lambda _name: contextlib.nullcontext()
    )
    with range_fn("stage2_shared_cache"):
        shared_cache = build_stage2_shared_geometry_cache(context)
    if pipeline_timing is not None:
        pipeline_timing.update(shared_cache.timing_dict())
    with range_fn("stage2_pose_definition"):
        pose_definition, pose_handoff = build_pose_stage_definition(
            context, stage1_outcome, shared_cache,
        )
    _print_handoff(pose_handoff["lm"], "LM", "pose")
    if not pose_handoff["lm"]["ok"]:
        raise RuntimeError("LM handoff Stage 1->Stage 2 pose failed")
    _record_preparation_timing(
        pipeline_timing, "stage2_pose", pose_definition,
    )

    _pipeline_sync(context)
    start = time.perf_counter()
    with range_fn("stage2_pose_solver"):
        pose_outcome = run_stage(pose_definition, context)
    _pipeline_sync(context)
    if pipeline_timing is not None:
        run_ms = (time.perf_counter() - start) * 1000.0
        pipeline_timing["stage2_pose_run_ms"] = run_ms
        pipeline_timing["stage2_pose_run_overhead_ms"] = max(
            0.0, run_ms - pose_outcome.lm_result.wall_time_ms,
        )
    handoffs = {
        "stage1_to_stage2_pose_lm": pose_handoff["lm"],
        "stage1_to_stage2_pose_lbfgs": None,
    }
    if context.args.stop_after_stage == "stage2_pose":
        _record_stage2_cache_totals(
            pipeline_timing, shared_cache, pose_definition,
        )
        return Stage2Outcome(
            pose_warmup=pose_outcome, joint_refinement=None, handoffs=handoffs,
        )

    with range_fn("stage2_joint_definition"):
        joint_definition, joint_handoff = build_joint_stage_definition(
            context, pose_outcome, shared_cache,
        )
    _print_handoff(joint_handoff["lm"], "LM", "joint")
    if not joint_handoff["lm"]["ok"]:
        raise RuntimeError("LM handoff Stage 2 pose->joint failed")
    _record_preparation_timing(
        pipeline_timing, "stage2_joint", joint_definition,
    )
    _record_stage2_cache_totals(
        pipeline_timing, shared_cache, pose_definition, joint_definition,
    )
    handoffs.update({
        "stage2_pose_to_joint_lm": joint_handoff["lm"],
        "stage2_pose_to_joint_lbfgs": None,
    })

    _pipeline_sync(context)
    start = time.perf_counter()
    with range_fn("stage2_joint_solver"):
        joint_outcome = run_stage(joint_definition, context)
    _pipeline_sync(context)
    if pipeline_timing is not None:
        run_ms = (time.perf_counter() - start) * 1000.0
        pipeline_timing["stage2_joint_run_ms"] = run_ms
        pipeline_timing["stage2_joint_run_overhead_ms"] = max(
            0.0, run_ms - joint_outcome.lm_result.wall_time_ms,
        )
    return Stage2Outcome(
        pose_warmup=pose_outcome,
        joint_refinement=joint_outcome,
        handoffs=handoffs,
    )


def check_s1_to_pose(
    stage1_parameters: torch.Tensor,
    stage2_pose_parameters: torch.Tensor,
    body_pose_anchors: list[torch.Tensor],
    frame_count: int,
    body_pose_dim: int,
) -> dict[str, Any]:
    """Verify the Stage 1 to pose handoff."""
    per_frame_dim = STAGE1_PER_FRAME_DIM + body_pose_dim
    orient_transl_max_diff = 0.0
    body_pose_max_diff = 0.0
    for frame in range(frame_count):
        pose_start = frame * per_frame_dim
        stage1_start = frame * STAGE1_PER_FRAME_DIM
        orient_transl_max_diff = max(
            orient_transl_max_diff,
            float((
                stage2_pose_parameters[
                    pose_start:pose_start + STAGE1_PER_FRAME_DIM
                ] - stage1_parameters[
                    stage1_start:stage1_start + STAGE1_PER_FRAME_DIM
                ]
            ).abs().max()),
        )
        body_pose_max_diff = max(
            body_pose_max_diff,
            float((
                stage2_pose_parameters[
                    pose_start + STAGE1_PER_FRAME_DIM:(frame + 1) * per_frame_dim
                ] - body_pose_anchors[frame]
            ).abs().max()),
        )
    return {
        "ot_max": orient_transl_max_diff,
        "bp_max": body_pose_max_diff,
        "ok": orient_transl_max_diff < 1e-6 and body_pose_max_diff < 1e-6,
    }


def handoff_s1_to_pose(
    stage1_parameters: torch.Tensor,
    body_pose_anchors: list[torch.Tensor],
    frame_count: int,
    body_pose_dim: int,
) -> torch.Tensor:
    """Construct the pose-stage initial vector from Stage 1 and anchors."""
    parts: list[torch.Tensor] = []
    for frame in range(frame_count):
        if body_pose_anchors[frame].numel() != body_pose_dim:
            raise ValueError("body pose anchor dimension mismatch")
        parts.append(stage1_parameters[
            frame * STAGE1_PER_FRAME_DIM:(frame + 1) * STAGE1_PER_FRAME_DIM
        ])
        parts.append(body_pose_anchors[frame].clone())
    return torch.cat(parts).clone()


def check_pose_to_joint_state(
    stage2_pose_parameters: torch.Tensor,
    stage2_joint_parameters: torch.Tensor,
    beta_anchor: torch.Tensor,
    beta_abs_limit: float,
    shape_anchor_scale: float,
) -> dict[str, Any]:
    """Check pose state and bounded-beta initialization without model forward."""
    pose_dimension = stage2_pose_parameters.shape[0]
    beta_dimension = stage2_joint_parameters.shape[0] - pose_dimension
    state_max_diff = float((
        stage2_joint_parameters[:pose_dimension] - stage2_pose_parameters
    ).abs().max())
    beta_initial = beta_abs_limit * torch.tanh(
        stage2_joint_parameters[-beta_dimension:]
    )
    beta_max_diff = float((beta_initial - beta_anchor).abs().max())
    shape_anchor_objective = float(
        0.5 * torch.sum(
            (beta_initial - beta_anchor) ** 2 * shape_anchor_scale ** 2
        )
    )
    ok = (
        state_max_diff <= 1e-7
        and beta_max_diff <= 1e-6
        and shape_anchor_objective <= 1e-10
    )
    return {
        "state_max_diff": state_max_diff,
        "physical_beta_init_max_diff_to_anchor": beta_max_diff,
        "raw_reprojection_max_diff": None,
        "shape_anchor_objective_at_init": shape_anchor_objective,
        "relative_difference_joint_init_to_pose": None,
        "expensive_validation_performed": False,
        "validation_scope": "state_only",
        "ok": ok,
    }
