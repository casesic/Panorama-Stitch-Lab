"""Stage orchestration for rigid alignment and pose/shape fitting.

Stage 1 performs dual-camera orient/transl rigid alignment. Stage 2 performs
pose warmup followed by pose/shape joint refinement.

Each stage is described by a :class:`StageDefinition` (the optimisation
problem) and executed by :func:`run_stage` (which wires up the IRLS / raw-L2
linearisation, runs the LM solver and the optional L-BFGS oracle, and checks
the LM-vs-LBFGS verdict).

Stage variable layouts
----------------------
* **Stage 1** — per-frame camera orient/transl parameters. Body pose is frozen to
  the HMR anchor; beta is frozen to the shared initial value.
* **Stage 2 pose warmup** — Stage 1 parameters plus the runtime body-pose vector.
  Beta still frozen; body pose is optimised with a pose-anchor regulariser.
* **Stage 2 joint refinement** — ``[pose vector | beta_latent(10)]``. Beta is optimised
  through a bounded ``tanh`` parameterisation with a shape-anchor regulariser.
"""

from __future__ import annotations

import contextlib

import time
from typing import Any, Callable

import torch

from lib_pano.human_fitting.staged_lm.irls import (
    build_primary_aggregation_spec,
    compute_primary_charbonnier_tensor,
    make_irls_linearization,
    validate_first_order,
)
from lib_pano.human_fitting.staged_lm.analytic_stage1 import (
    build_stage1_analytic_cache,
    build_stage1_analytic_linearization,
    metric_close_with_absolute_or_relative,
)
from lib_pano.human_fitting.staged_lm.lm.runner import run_lm_optimization
from lib_pano.human_fitting.staged_lm.linearization import Linearization
from .lbfgs import run_lbfgs_oracle
from lib_pano.human_fitting.staged_lm.pose.cache import build_stage2_pose_fast_cache
from lib_pano.human_fitting.staged_lm.pose.forward import fast_raw_groups
from lib_pano.human_fitting.staged_lm.pose.derivatives import build_stage2_pose_analytic_linearization
from lib_pano.human_fitting.staged_lm.joint import build_stage2_joint_fast_cache, joint_fast_raw_groups
from lib_pano.human_fitting.staged_lm.stage2_shared_cache import build_stage2_shared_geometry_cache
from .joint_backends import (
    build_stage2_joint_reference_linearization as build_stage2_joint_analytic_linearization,
)
from lib_pano.human_fitting.staged_lm.config import STAGE1_PER_FRAME_DIM
from lib_pano.human_fitting.staged_lm.parameter_layout import build_stage1_initial_parameters
from lib_pano.human_fitting.staged_lm.residuals import flatten_reprojection_groups, make_raw_reproj_groups, raw_reproj_metrics
from lib_pano.human_fitting.staged_lm.types import (
    LBFGSResult,
    LMResult,
    StageContext,
    StageOutcome,
    Stage2Outcome,
)
from .types import ReferenceStageDefinition as StageDefinition
from lib_pano.human_fitting.staged_lm.device import validate_stage_initial_parameters


def stage_verdict(
    lm_result: LMResult,
    lbfgs_result: LBFGSResult | None,
    raw_metrics_fn: Callable[[torch.Tensor], dict[str, float]],
) -> str | None:
    """Compare LM against the L-BFGS oracle.

    Returns ``"LM_MATCHES_LBFGS"`` if LM is within 2 % on objective and 5 % on
    RMS / median reprojection, and the LM status is not a failure.
    """
    if lbfgs_result is None:
        return None
    lbfgs_rms = raw_metrics_fn(lbfgs_result.parameters)
    ok = (
        lm_result.objective <= lbfgs_result.objective * 1.02
        and lm_result.rms_metrics["rms"] <= lbfgs_rms["rms"] * 1.05
        and lm_result.rms_metrics["median"] <= lbfgs_rms["median"] * 1.05
        and lm_result.status not in ("LM_NO_ACCEPTED_TRIAL", "NUMERICAL_FAILURE")
    )
    return "LM_MATCHES_LBFGS" if ok else "LM_LBFGS_GAP"


def _fast_eval_error(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, float]:
    difference = actual - expected
    return {
        "max_abs": float(difference.abs().max()),
        "mean_abs": float(difference.abs().mean()),
        "rel_l2": float(difference.norm() / expected.norm().clamp_min(1e-16)),
    }


def _build_linearization_and_objective(
    stage: StageDefinition,
    context: StageContext,
) -> tuple[
    Callable[[torch.Tensor], Linearization],
    Callable[[torch.Tensor], torch.Tensor],
    Callable[[torch.Tensor], torch.Tensor],
    dict[str, Any] | None,
    float,
]:
    """Create the linearisation factory and true objective for a stage.

    Returns ``(make_linearization, true_objective_fn, init_check, init_validation_ms)``.
    For IRLS mode the surrogate is the weighted least-squares majoriser; for
    raw-L2 mode it is the plain scaled residual.
    """
    init_validation_ms = 0.0
    if context.is_irls:
        static_spec = build_primary_aggregation_spec(stage.raw_groups_fn(stage.initial_lm_parameters))

        if stage.linearization_builder is not None:
            def make_linearization(x: torch.Tensor) -> Linearization:
                return stage.linearization_builder(x, static_spec)
        elif stage.analytic_linearization:
            def make_linearization(x: torch.Tensor) -> Linearization:
                return build_stage1_analytic_linearization(
                    stage.analytic_cache, x, is_irls=True,
                    charbonnier_eps=context.charbonnier_eps,
                    aggregation_spec=static_spec, rep_scale=context.scaling.rep_scale,
                )
        else:
            def make_linearization(x: torch.Tensor) -> Linearization:
                return make_irls_linearization(
                    stage.raw_groups_fn, stage.anchor_residual_fn, x,
                    context.charbonnier_eps, static_spec,
                )

        def full_true_objective(x: torch.Tensor) -> torch.Tensor:
            groups = stage.raw_groups_fn(x)
            objective = compute_primary_charbonnier_tensor(groups, context.charbonnier_eps, static_spec)
            if stage.anchor_residual_fn is not None:
                objective = objective + 0.5 * torch.sum(stage.anchor_residual_fn(x) ** 2)
            return objective

        init_check = None
        do_initial_validation = context.args.validate_irls_first_order
        if do_initial_validation:
            if context.args.profile_timing and stage.initial_lm_parameters.is_cuda:
                torch.cuda.synchronize(stage.initial_lm_parameters.device)
            t_iv = time.perf_counter() if context.args.profile_timing else 0
            init_linearization = make_linearization(stage.initial_lm_parameters)
            init_check = validate_first_order(
                init_linearization.surrogate,
                stage.lm_true_objective_fn or full_true_objective,
                stage.initial_lm_parameters, f"{stage.name}_init",
                context.solver_config.irls_gradient_rel_tol,
            )
            if context.args.profile_timing:
                if stage.initial_lm_parameters.is_cuda:
                    torch.cuda.synchronize(stage.initial_lm_parameters.device)
                init_validation_ms = (time.perf_counter() - t_iv) * 1000.0
            if not init_check["pass"]:
                raise RuntimeError(f"IRLS_FIRST_ORDER_MISMATCH at {stage.name}")
    else:
        rep_scale = context.scaling.rep_scale

        def total_residual(x: torch.Tensor) -> torch.Tensor:
            residual = flatten_reprojection_groups(stage.raw_groups_fn(x)) * rep_scale
            if stage.anchor_residual_fn is None:
                return residual
            return torch.cat([residual, stage.anchor_residual_fn(x)])

        if stage.linearization_builder is not None:
            def make_linearization(x: torch.Tensor) -> Linearization:
                return stage.linearization_builder(x, None)
        elif stage.analytic_linearization:
            def make_linearization(x: torch.Tensor) -> Linearization:
                return build_stage1_analytic_linearization(
                    stage.analytic_cache, x, is_irls=False,
                    charbonnier_eps=context.charbonnier_eps,
                    aggregation_spec=None, rep_scale=context.scaling.rep_scale,
                )
        else:
            def make_linearization(x: torch.Tensor) -> Linearization:
                return Linearization(surrogate=total_residual)

        def full_true_objective(x: torch.Tensor) -> torch.Tensor:
            return 0.5 * torch.sum(total_residual(x) ** 2)

        init_check = None

    lm_true_objective = stage.lm_true_objective_fn or full_true_objective
    lbfgs_true_objective = stage.lbfgs_true_objective_fn or full_true_objective
    return make_linearization, lm_true_objective, lbfgs_true_objective, init_check, init_validation_ms


def run_stage(stage: StageDefinition, context: StageContext) -> StageOutcome:
    """Execute one optimisation stage (LM + optional L-BFGS + verdict).

    Prints the stage header, runs the LM solver, optionally runs the L-BFGS
    oracle, and computes the LM-vs-LBFGS verdict.
    """
    if stage.analytic_cache is not None:
        stage.analytic_cache.reset_timing()
    validate_stage_initial_parameters(
        stage, context,
        include_cache=getattr(context.args, "validate_device_contract", False),
    )
    stage.full_forward_oracle_validation_ms = 0.0
    make_linearization, lm_true_objective, lbfgs_true_objective, init_check, init_validation_ms = (
        _build_linearization_and_objective(stage, context)
    )
    if ((stage.linearization_backend == "autograd" and stage.eval_backend not in ("full_forward",))
            or (stage.linearization_backend != "autograd" and stage.eval_backend == "full_forward")):
        print(
            f"[{stage.name}_MIXED_BACKEND] linearization={stage.linearization_backend} "
            f"evaluation={stage.eval_backend}"
        )
    if stage.fast_eval_validator is not None:
        start = time.perf_counter()
        stage.fast_eval_validator("INIT", stage.initial_lm_parameters)
        stage.full_forward_oracle_validation_ms += (time.perf_counter() - start) * 1000.0

    # ---- Print stage header (same format as original) ----
    sample_groups = stage.raw_groups_fn(stage.initial_lm_parameters)
    n_groups = len(sample_groups)
    total_valid = sum(group.valid_point_count for group in sample_groups)
    flat_sample = flatten_reprojection_groups(sample_groups)
    surrogate_dim = int(flat_sample.numel())
    if stage.anchor_residual_fn is not None:
        surrogate_dim += int(stage.anchor_residual_fn(stage.initial_lm_parameters).numel())
    print(f"\n{'='*60}\n[{stage.name}] T={context.frame_count} param_dim={stage.parameter_dimension} "
          f"groups={n_groups} valid_pts={total_valid} surr_dim={surrogate_dim}\n{'='*60}")

    lm_result = run_lm_optimization(
        label=f"{stage.name}_LM",
        make_linearization=make_linearization,
        raw_metrics_fn=stage.raw_metrics_fn,
        initial_parameters=stage.initial_lm_parameters,
        param_dim=stage.parameter_dimension,
        max_accepted_steps=stage.lm_max_accepted_steps,
        max_trials=context.solver_config.lm_max_trials,
        solver_config=context.solver_config,
        profile_timing=context.args.profile_timing,
        true_objective_fn=lm_true_objective,
    )

    if init_check is not None:
        lm_result.first_order_checks = {
            "initial": init_check,
            "worst_abs_max": init_check["abs_max"],
            "worst_rel_l2": init_check["rel_l2"],
            "all_pass": init_check["pass"],
        }

    if context.args.profile_timing and init_check is not None and lm_result.profile_timing is not None:
        lm_result.profile_timing["initial_irls_validation_ms"] = init_validation_ms
    if stage.fast_eval_validator is not None:
        start = time.perf_counter()
        stage.fast_eval_validator("LM_FINAL", lm_result.parameters)
        stage.full_forward_oracle_validation_ms += (time.perf_counter() - start) * 1000.0

    lbfgs_result = None
    if context.args.run_lbfgs_oracle and stage.initial_lbfgs_parameters is not None:
        lbfgs_result = run_lbfgs_oracle(
            label=f"{stage.name}_LBFGS",
            true_objective_fn=lbfgs_true_objective,
            raw_metrics_fn=stage.raw_metrics_fn,
            initial_parameters=stage.initial_lbfgs_parameters,
            outer_steps=stage.lbfgs_outer_steps,
            solver_config=context.solver_config,
        )
        if stage.fast_eval_validator is not None:
            start = time.perf_counter()
            stage.fast_eval_validator("LBFGS_FINAL", lbfgs_result.parameters)
            stage.full_forward_oracle_validation_ms += (time.perf_counter() - start) * 1000.0

    verdict = stage_verdict(lm_result, lbfgs_result, stage.raw_metrics_fn)
    if verdict:
        print(f"[V_{stage.name}] {verdict}")
    lm_result.eval_backend = stage.eval_backend
    if context.args.profile_timing and stage.analytic_cache is not None and lm_result.profile_timing is not None:
        lm_result.profile_timing.update(stage.analytic_cache.timing_dict())
    if lm_result.profile_timing is not None:
        lm_result.profile_timing["eval_backend"] = stage.eval_backend
        lm_result.profile_timing["linearization_backend"] = stage.linearization_backend
        lm_result.profile_timing["linear_solver_backend"] = stage.linear_solver_backend
        lm_result.profile_timing["full_forward_oracle_validation_ms"] = (
            stage.full_forward_oracle_validation_ms
        )

    return StageOutcome(
        lm_result=lm_result,
        lbfgs_result=lbfgs_result,
        verdict=verdict,
        true_objective_fn=lm_true_objective,
        definition=stage,
    )


# ------------------------------------------------------------------
# Stage builders
# ------------------------------------------------------------------

def build_stage1_definition(context: StageContext) -> StageDefinition:
    """Build the Stage 1 definition.

    Optimised variables: per-frame cam2/cam4 orient and transl.
    Frozen variables: body pose (anchor), beta (anchor), hand poses, scale.
    No anchor residual.
    """
    initial_parameters = _build_stage1_initial(context)
    parameter_dimension = context.frame_count * STAGE1_PER_FRAME_DIM

    def beta_frozen(_x: torch.Tensor) -> torch.Tensor:
        return context.beta_anchor

    def raw_groups_fn(x: torch.Tensor):
        return make_raw_reproj_groups(
            x, context.frame_indices,
            context.cam2_vertex_ids_by_frame, context.cam2_targets_by_frame,
            context.cam4_vertex_ids_by_frame, context.cam4_targets_by_frame,
            context.frozen_params_by_frame, context.model, context.fitter,
            context.image_width, context.image_height, context.frame_count,
            STAGE1_PER_FRAME_DIM, beta_frozen, context.body_pose_anchors,
        )

    def raw_metrics_fn(x: torch.Tensor) -> dict[str, float]:
        return raw_reproj_metrics(flatten_reprojection_groups(raw_groups_fn(x)).detach())

    needs_cache = (
        context.args.jacobian_backend == "analytic_stage1"
        or context.args.stage1_eval_backend == "analytic"
    )
    analytic_cache = build_stage1_analytic_cache(context) if needs_cache else None
    aggregation_spec = (
        build_primary_aggregation_spec(raw_groups_fn(initial_parameters))
        if context.args.stage1_eval_backend == "analytic" else None
    )

    lm_true_objective_fn = None
    selected_metrics_fn = raw_metrics_fn
    fast_eval_validator = None
    if context.args.stage1_eval_backend == "analytic":
        def analytic_objective(x: torch.Tensor) -> torch.Tensor:
            return analytic_cache.true_objective(
                x, context.args.reproj_objective, context.charbonnier_eps,
                context.scaling.rep_scale,
                aggregation_spec if context.is_irls else None,
            )

        lm_true_objective_fn = analytic_objective
        selected_metrics_fn = analytic_cache.raw_metrics

        if context.args.validate_stage1_fast_eval:
            def validate_fast_eval(label: str, x: torch.Tensor) -> None:
                analytic_groups = analytic_cache.raw_groups(x)
                full_groups = raw_groups_fn(x)
                analytic_raw = flatten_reprojection_groups(analytic_groups)
                full_raw = flatten_reprojection_groups(full_groups)
                error = _fast_eval_error(analytic_raw, full_raw)
                row = 0
                block_errors = []
                for analytic_group, full_group in zip(analytic_groups, full_groups):
                    rows = analytic_group.residual_2d.numel()
                    block_error = _fast_eval_error(
                        analytic_raw[row:row + rows], full_raw[row:row + rows],
                    )
                    block_errors.append(
                        f"{analytic_group.frame_index}/{analytic_group.camera}="
                        f"{block_error['max_abs']:.3e}"
                    )
                    row += rows
                try:
                    torch.testing.assert_close(analytic_raw, full_raw, rtol=1e-6, atol=1e-3)
                except AssertionError as exc:
                    raise RuntimeError("STAGE1_FAST_EVAL_MISMATCH") from exc

                if context.is_irls:
                    analytic_objective_value = compute_primary_charbonnier_tensor(
                        analytic_groups, context.charbonnier_eps, aggregation_spec,
                    )
                    full_objective_value = compute_primary_charbonnier_tensor(
                        full_groups, context.charbonnier_eps, aggregation_spec,
                    )
                else:
                    analytic_scaled = analytic_raw * context.scaling.rep_scale
                    full_scaled = full_raw * context.scaling.rep_scale
                    analytic_objective_value = 0.5 * torch.sum(analytic_scaled ** 2)
                    full_objective_value = 0.5 * torch.sum(full_scaled ** 2)
                objective_absolute = abs(float(analytic_objective_value - full_objective_value))
                objective_relative = objective_absolute / max(abs(float(full_objective_value)), 1e-16)
                analytic_metrics = raw_reproj_metrics(analytic_raw.detach())
                full_metrics = raw_reproj_metrics(full_raw.detach())
                metrics_ok = all(
                    metric_close_with_absolute_or_relative(
                        analytic_metrics[name], full_metrics[name],
                    )
                    for name in ("rms", "median", "p95")
                )
                print(
                    f"[STAGE1_FAST_EVAL_{label}] raw={error} blocks={' '.join(block_errors)} "
                    f"obj_abs={objective_absolute:.3e} obj_rel={objective_relative:.3e} "
                    f"analytic_metrics={analytic_metrics} full_metrics={full_metrics}"
                )
                if objective_relative > 1e-6 or not metrics_ok:
                    raise RuntimeError("STAGE1_FAST_EVAL_MISMATCH")

            fast_eval_validator = validate_fast_eval

    stage1_linearization_builder = None
    if context.args.jacobian_backend == "analytic_stage1":
        def stage1_linearization_builder(x: torch.Tensor, spec: dict[str, Any] | None) -> dict[str, Any]:
            return build_stage1_analytic_linearization(
                analytic_cache, x, is_irls=context.is_irls,
                charbonnier_eps=context.charbonnier_eps,
                aggregation_spec=spec, rep_scale=context.scaling.rep_scale,
            )

    stage = StageDefinition(
        name="S1",
        parameter_dimension=parameter_dimension,
        initial_lm_parameters=initial_parameters,
        initial_lbfgs_parameters=initial_parameters.clone(),
        raw_groups_fn=raw_groups_fn,
        anchor_residual_fn=None,
        raw_metrics_fn=selected_metrics_fn,
        lm_max_accepted_steps=context.solver_config.stage1_lm_max_acc,
        lbfgs_outer_steps=context.solver_config.stage1_bf_outer,
        analytic_cache=analytic_cache,
        linearization_builder=stage1_linearization_builder,
        lm_true_objective_fn=lm_true_objective_fn,
        eval_backend=context.args.stage1_eval_backend,
        linearization_backend=(
            "analytic_stage1" if context.args.jacobian_backend == "analytic_stage1" else "autograd"
        ),
        fast_eval_validator=fast_eval_validator,
    )
    return stage


def _build_stage1_initial(context: StageContext) -> torch.Tensor:
    """Concatenate per-frame orient/transl into the Stage 1 vector.

    Delegates to :func:`build_stage1_initial_parameters`.
    """
    return build_stage1_initial_parameters(
        context.cam2_smplx_parameters, context.cam4_smplx_parameters, context.frame_count,
    )


def _build_pose_warmup_definition(
    context: StageContext,
    stage1_outcome: StageOutcome,
) -> tuple[StageDefinition, dict[str, Any]]:
    """Build the Stage 2 pose warmup definition and S1->STAGE2_POSE handoff check.

    Optimised variables: per-frame orient/transl plus runtime-sized body pose.
    Frozen variables: beta (anchor), hand poses, scale.
    Anchor: pose-anchor residual penalising deviation from the HMR pose.
    """
    per_frame_dim = STAGE1_PER_FRAME_DIM + context.body_pose_dim
    parameter_dimension = context.frame_count * per_frame_dim
    pose_anchor_scale = context.scaling.pose_anchor_scale

    initial_lm = handoff_s1_to_pose(
        stage1_outcome.lm_result.parameters, context.body_pose_anchors,
        context.frame_count, context.body_pose_dim,
    )
    initial_lbfgs = None
    if stage1_outcome.lbfgs_result is not None:
        initial_lbfgs = handoff_s1_to_pose(
            stage1_outcome.lbfgs_result.parameters, context.body_pose_anchors,
            context.frame_count, context.body_pose_dim,
        )

    handoff_lm = check_s1_to_pose(
        stage1_outcome.lm_result.parameters, initial_lm,
        context.body_pose_anchors, context.frame_count, context.body_pose_dim,
    )
    handoff_lbfgs = None
    if stage1_outcome.lbfgs_result is not None:
        handoff_lbfgs = check_s1_to_pose(
            stage1_outcome.lbfgs_result.parameters, initial_lbfgs,
            context.body_pose_anchors, context.frame_count, context.body_pose_dim,
        )

    def beta_frozen(_x: torch.Tensor) -> torch.Tensor:
        return context.beta_anchor

    def raw_groups_fn(x: torch.Tensor):
        return make_raw_reproj_groups(
            x, context.frame_indices,
            context.cam2_vertex_ids_by_frame, context.cam2_targets_by_frame,
            context.cam4_vertex_ids_by_frame, context.cam4_targets_by_frame,
            context.frozen_params_by_frame, context.model, context.fitter,
            context.image_width, context.image_height, context.frame_count,
            per_frame_dim, beta_frozen,
        )

    def pose_anchor_residual(x: torch.Tensor) -> torch.Tensor:
        parts: list[torch.Tensor] = []
        for t in range(context.frame_count):
            start = t * per_frame_dim + STAGE1_PER_FRAME_DIM
            end = (t + 1) * per_frame_dim
            parts.append((x[start:end] - context.body_pose_anchors[t]) * pose_anchor_scale)
        return torch.cat(parts)

    def raw_metrics_fn(x: torch.Tensor) -> dict[str, float]:
        return raw_reproj_metrics(flatten_reprojection_groups(raw_groups_fn(x)).detach())

    linearization_backend = getattr(
        context.args, "stage2_pose_linearization_backend", "autograd",
    )
    eval_backend = getattr(context.args, "stage2_pose_eval_backend", "full_forward")
    validate_fast = getattr(context.args, "validate_stage2_pose_fast_eval", False)
    needs_fast_cache = linearization_backend == "analytic" or eval_backend == "fast" or validate_fast
    fast_cache = (
        build_stage2_pose_fast_cache(
            context, build_stage2_shared_geometry_cache(context),
        )
        if needs_fast_cache else None
    )
    fast_spec = (
        build_primary_aggregation_spec(raw_groups_fn(initial_lm)) if context.is_irls and needs_fast_cache
        else None
    )

    linearization_builder = None
    if linearization_backend == "analytic":
        def linearization_builder(x: torch.Tensor, spec: dict[str, Any] | None) -> dict[str, Any]:
            return build_stage2_pose_analytic_linearization(
                fast_cache, x, context.body_pose_anchors, pose_anchor_scale,
                is_irls=context.is_irls, charbonnier_eps=context.charbonnier_eps,
                aggregation_spec=spec, rep_scale=context.scaling.rep_scale,
            )

    lm_true_objective_fn = None
    selected_metrics_fn = raw_metrics_fn
    if eval_backend == "fast":
        def lm_true_objective_fn(x: torch.Tensor) -> torch.Tensor:
            return fast_cache.true_objective(
                x, context.body_pose_anchors, pose_anchor_scale,
                context.args.reproj_objective, context.charbonnier_eps,
                context.scaling.rep_scale, fast_spec,
            )

        selected_metrics_fn = fast_cache.raw_metrics

    fast_eval_validator = None
    if validate_fast:
        def fast_eval_validator(label: str, x: torch.Tensor) -> None:
            fast_groups = fast_raw_groups(fast_cache, x)
            full_groups = raw_groups_fn(x)
            fast_raw = flatten_reprojection_groups(fast_groups)
            full_raw = flatten_reprojection_groups(full_groups)
            error = _fast_eval_error(fast_raw, full_raw)
            try:
                torch.testing.assert_close(fast_raw, full_raw, rtol=1e-6, atol=1e-3)
            except AssertionError as exc:
                raise RuntimeError("STAGE2_POSE_FAST_EVAL_MISMATCH") from exc
            if context.is_irls:
                fast_primary = compute_primary_charbonnier_tensor(
                    fast_groups, context.charbonnier_eps, fast_spec,
                )
                full_primary = compute_primary_charbonnier_tensor(
                    full_groups, context.charbonnier_eps, fast_spec,
                )
            else:
                fast_scaled = fast_raw * context.scaling.rep_scale
                full_scaled = full_raw * context.scaling.rep_scale
                fast_primary = 0.5 * torch.sum(fast_scaled ** 2)
                full_primary = 0.5 * torch.sum(full_scaled ** 2)
            anchor_objective = 0.5 * torch.sum(pose_anchor_residual(x) ** 2)
            fast_objective = fast_primary + anchor_objective
            full_objective = full_primary + anchor_objective
            objective_absolute = abs(float(fast_objective - full_objective))
            objective_relative = objective_absolute / max(abs(float(full_objective)), 1e-16)
            fast_metrics = raw_reproj_metrics(fast_raw.detach())
            full_metrics = raw_reproj_metrics(full_raw.detach())
            metrics_ok = all(metric_close_with_absolute_or_relative(
                fast_metrics[name], full_metrics[name],
            ) for name in ("rms", "median", "p95"))
            print(
                f"[STAGE2_POSE_FAST_EVAL_{label}] raw={error} "
                f"obj_abs={objective_absolute:.3e} obj_rel={objective_relative:.3e} "
                f"fast_metrics={fast_metrics} full_metrics={full_metrics}"
            )
            if objective_relative > 1e-6 or not metrics_ok:
                raise RuntimeError("STAGE2_POSE_FAST_EVAL_MISMATCH")

    stage = StageDefinition(
        name="STAGE2_POSE",
        parameter_dimension=parameter_dimension,
        initial_lm_parameters=initial_lm,
        initial_lbfgs_parameters=initial_lbfgs,
        raw_groups_fn=raw_groups_fn,
        anchor_residual_fn=pose_anchor_residual,
        raw_metrics_fn=selected_metrics_fn,
        lm_max_accepted_steps=context.solver_config.stage2_pose_warmup_lm_max_acc,
        lbfgs_outer_steps=context.solver_config.stage2_pose_warmup_bf_outer,
        analytic_cache=fast_cache,
        linearization_builder=linearization_builder,
        lm_true_objective_fn=lm_true_objective_fn,
        # L-BFGS deliberately falls back to the generic differentiable full objective.
        lbfgs_true_objective_fn=None,
        eval_backend=eval_backend,
        linearization_backend=linearization_backend,
        fast_eval_validator=fast_eval_validator,
    )
    handoff_info = {"lm": handoff_lm, "lbfgs": handoff_lbfgs}
    return stage, handoff_info


def _build_joint_refinement_definition(
    context: StageContext,
    stage2_pose_outcome: StageOutcome,
    stage2_pose_true_objective: Callable[[torch.Tensor], torch.Tensor],
    stage2_pose_raw_groups_fn: Callable[[torch.Tensor], Any],
) -> tuple[StageDefinition, dict[str, Any]]:
    """Build the Stage 2 joint refinement definition and STAGE2_POSE->STAGE2_JOINT handoff check.

    Optimised variables: Stage 2 pose warmup vector + beta_latent (10).
    Beta is parameterised as ``beta_abs_limit * tanh(beta_latent)`` to keep it
    bounded. Anchors: pose-anchor plus shape-anchor (penalises
    deviation from the beta anchor).
    """
    per_frame_dim = STAGE1_PER_FRAME_DIM + context.body_pose_dim
    stage2_pose_dim = context.frame_count * per_frame_dim
    parameter_dimension = stage2_pose_dim + context.beta_dim
    beta_abs_limit = context.objective_config.beta_abs_limit
    pose_anchor_scale = context.scaling.pose_anchor_scale
    shape_anchor_scale = context.scaling.shape_anchor_scale

    beta_latent = torch.atanh(context.beta_anchor / beta_abs_limit)
    initial_lm = torch.cat([stage2_pose_outcome.lm_result.parameters.clone(), beta_latent.clone()]).clone()
    initial_lbfgs = None
    if stage2_pose_outcome.lbfgs_result is not None:
        initial_lbfgs = torch.cat([stage2_pose_outcome.lbfgs_result.parameters.clone(), beta_latent.clone()]).clone()

    def beta_active(x: torch.Tensor) -> torch.Tensor:
        return beta_abs_limit * torch.tanh(x[-context.beta_dim:])

    def raw_groups_fn(x: torch.Tensor):
        return make_raw_reproj_groups(
            x, context.frame_indices,
            context.cam2_vertex_ids_by_frame, context.cam2_targets_by_frame,
            context.cam4_vertex_ids_by_frame, context.cam4_targets_by_frame,
            context.frozen_params_by_frame, context.model, context.fitter,
            context.image_width, context.image_height, context.frame_count,
            per_frame_dim, beta_active,
        )

    def pose_anchor_residual(x: torch.Tensor) -> torch.Tensor:
        parts: list[torch.Tensor] = []
        for t in range(context.frame_count):
            start = t * per_frame_dim + STAGE1_PER_FRAME_DIM
            end = (t + 1) * per_frame_dim
            parts.append((x[start:end] - context.body_pose_anchors[t]) * pose_anchor_scale)
        return torch.cat(parts)

    def shape_anchor_residual(x: torch.Tensor) -> torch.Tensor:
        return (beta_abs_limit * torch.tanh(x[-context.beta_dim:]) - context.beta_anchor) * shape_anchor_scale

    def all_anchors_residual(x: torch.Tensor) -> torch.Tensor:
        return torch.cat([pose_anchor_residual(x), shape_anchor_residual(x)])

    def raw_metrics_fn(x: torch.Tensor) -> dict[str, float]:
        return raw_reproj_metrics(flatten_reprojection_groups(raw_groups_fn(x)).detach())

    linearization_backend=getattr(context.args,"stage2_joint_linearization_backend","autograd")
    requested_linear_solver=getattr(context.args,"stage2_joint_linear_solver","dense")
    if requested_linear_solver=="schur" and linearization_backend!="analytic":
        raise RuntimeError("SCHUR_REQUIRES_ANALYTIC_STAGE2_JOINT")
    linear_solver_backend="arrowhead_schur" if requested_linear_solver=="schur" else "dense"
    eval_backend=getattr(context.args,"stage2_joint_eval_backend","full_forward")
    validate_fast=getattr(context.args,"validate_stage2_joint_fast_eval",False)
    needs_cache=linearization_backend=="analytic" or eval_backend=="fast" or validate_fast
    joint_fast_cache=(
        build_stage2_joint_fast_cache(
            context, build_stage2_shared_geometry_cache(context),
        ) if needs_cache else None
    )
    aggregation_spec=(build_primary_aggregation_spec(raw_groups_fn(initial_lm)) if context.is_irls and needs_cache else None)
    linearization_builder=None
    if linearization_backend=="analytic":
        def linearization_builder(x:torch.Tensor,spec:dict[str,Any]|None)->dict[str,Any]:
            return build_stage2_joint_analytic_linearization(
                joint_fast_cache,x,beta_abs_limit,context.beta_anchor,context.body_pose_anchors,
                pose_anchor_scale,shape_anchor_scale,is_irls=context.is_irls,
                charbonnier_eps=context.charbonnier_eps,aggregation_spec=spec,rep_scale=context.scaling.rep_scale,
                linear_solver_backend=linear_solver_backend)
    selected_metrics_fn=raw_metrics_fn;lm_true_objective_fn=None
    if eval_backend=="fast":
        def lm_true_objective_fn(x:torch.Tensor)->torch.Tensor:
            return joint_fast_cache.true_objective(
                x,beta_abs_limit,context.beta_anchor,context.body_pose_anchors,pose_anchor_scale,
                shape_anchor_scale,context.args.reproj_objective,context.charbonnier_eps,
                context.scaling.rep_scale,aggregation_spec)
        def selected_metrics_fn(x:torch.Tensor)->dict[str,float]:
            return joint_fast_cache.raw_metrics(x,beta_abs_limit)

    full_joint_objective=_make_stage2_joint_true_objective_from_functions(context,raw_groups_fn,all_anchors_residual,initial_lm)
    if context.is_irls:
        pose_spec=build_primary_aggregation_spec(stage2_pose_raw_groups_fn(stage2_pose_outcome.lm_result.parameters))
        def full_pose_objective(x):
            pose_anchor=pose_anchor_residual(x)
            return compute_primary_charbonnier_tensor(stage2_pose_raw_groups_fn(x),context.charbonnier_eps,pose_spec)+.5*torch.sum(pose_anchor**2)
    else:
        def full_pose_objective(x):
            raw=flatten_reprojection_groups(stage2_pose_raw_groups_fn(x))*context.scaling.rep_scale
            return .5*torch.sum(torch.cat((raw,pose_anchor_residual(x)))**2)
    fast_eval_validator=None
    if validate_fast:
        def fast_eval_validator(label:str,x:torch.Tensor)->None:
            fast_groups=joint_fast_raw_groups(joint_fast_cache,x,beta_abs_limit);full_groups=raw_groups_fn(x)
            fast_raw=flatten_reprojection_groups(fast_groups);full_raw=flatten_reprojection_groups(full_groups)
            try:torch.testing.assert_close(fast_raw,full_raw,rtol=1e-6,atol=1e-3)
            except AssertionError as exc:raise RuntimeError("STAGE2_JOINT_FAST_EVAL_MISMATCH") from exc
            fast_obj=joint_fast_cache.true_objective(x,beta_abs_limit,context.beta_anchor,context.body_pose_anchors,
                pose_anchor_scale,shape_anchor_scale,context.args.reproj_objective,context.charbonnier_eps,
                context.scaling.rep_scale,aggregation_spec);full_obj=full_joint_objective(x)
            obj_abs=abs(float(fast_obj-full_obj));obj_rel=obj_abs/max(abs(float(full_obj)),1e-16)
            fm=raw_reproj_metrics(fast_raw.detach());om=raw_reproj_metrics(full_raw.detach())
            metrics_ok=all(metric_close_with_absolute_or_relative(fm[n],om[n]) for n in ("rms","median","p95"))
            beta=beta_abs_limit*torch.tanh(x[-context.beta_dim:]);pose_anchor=pose_anchor_residual(x);shape_anchor=shape_anchor_residual(x)
            print(f"[STAGE2_JOINT_FAST_EVAL_{label}] raw={_fast_eval_error(fast_raw,full_raw)} obj_abs={obj_abs:.3e} obj_rel={obj_rel:.3e} fast_metrics={fm} full_metrics={om} beta_finite={bool(torch.isfinite(beta).all())} pose_anchor_norm={float(pose_anchor.norm()):.3e} shape_anchor_norm={float(shape_anchor.norm()):.3e}")
            if obj_rel>1e-6 or not metrics_ok:raise RuntimeError("STAGE2_JOINT_FAST_EVAL_MISMATCH")

    stage = StageDefinition(
        name="STAGE2_JOINT",
        parameter_dimension=parameter_dimension,
        initial_lm_parameters=initial_lm,
        initial_lbfgs_parameters=initial_lbfgs,
        raw_groups_fn=raw_groups_fn,
        anchor_residual_fn=all_anchors_residual,
        raw_metrics_fn=selected_metrics_fn,
        lm_max_accepted_steps=context.solver_config.stage2_joint_lm_max_acc,
        lbfgs_outer_steps=context.solver_config.stage2_joint_bf_outer,
        analytic_cache=joint_fast_cache,linearization_builder=linearization_builder,
        lm_true_objective_fn=lm_true_objective_fn,lbfgs_true_objective_fn=full_joint_objective,
        eval_backend=eval_backend,linearization_backend=linearization_backend,
        linear_solver_backend=linear_solver_backend,
        fast_eval_validator=fast_eval_validator,
    )

    # STAGE2_POSE->STAGE2_JOINT handoff checks
    handoff_lm = check_pose_to_joint(
        stage2_pose_outcome.lm_result.parameters, initial_lm,
        context.beta_anchor, beta_abs_limit, shape_anchor_scale,
        stage2_pose_raw_groups_fn, raw_groups_fn,
        full_pose_objective, full_joint_objective,
    )
    handoff_lbfgs = None
    if stage2_pose_outcome.lbfgs_result is not None and initial_lbfgs is not None:
        handoff_lbfgs = check_pose_to_joint(
            stage2_pose_outcome.lbfgs_result.parameters, initial_lbfgs,
            context.beta_anchor, beta_abs_limit, shape_anchor_scale,
            stage2_pose_raw_groups_fn, raw_groups_fn,
            full_pose_objective, full_joint_objective,
        )

    handoff_info = {"lm": handoff_lm, "lbfgs": handoff_lbfgs}
    return stage, handoff_info


def _make_stage2_joint_true_objective(
    stage: StageDefinition,
    context: StageContext,
    raw_groups_fn: Callable[[torch.Tensor], Any],
    all_anchors_residual: Callable[[torch.Tensor], torch.Tensor],
) -> Callable[[torch.Tensor], torch.Tensor]:
    """Build the true objective for Stage 2 joint refinement (needed for handoff check)."""
    if context.is_irls:
        static_spec = build_primary_aggregation_spec(raw_groups_fn(stage.initial_lm_parameters))

        def true_objective(x: torch.Tensor) -> torch.Tensor:
            groups = raw_groups_fn(x)
            objective = compute_primary_charbonnier_tensor(groups, context.charbonnier_eps, static_spec)
            objective = objective + 0.5 * torch.sum(all_anchors_residual(x) ** 2)
            return objective
    else:
        rep_scale = context.scaling.rep_scale

        def true_objective(x: torch.Tensor) -> torch.Tensor:
            residual = flatten_reprojection_groups(raw_groups_fn(x)) * rep_scale
            return 0.5 * torch.sum(torch.cat([residual, all_anchors_residual(x)]) ** 2)

    return true_objective


def _make_stage2_joint_true_objective_from_functions(context,raw_groups_fn,anchors_fn,initial):
    if context.is_irls:
        spec=build_primary_aggregation_spec(raw_groups_fn(initial))
        def objective(x):return compute_primary_charbonnier_tensor(raw_groups_fn(x),context.charbonnier_eps,spec)+.5*torch.sum(anchors_fn(x)**2)
    else:
        def objective(x):return .5*torch.sum(torch.cat((flatten_reprojection_groups(raw_groups_fn(x))*context.scaling.rep_scale,anchors_fn(x)))**2)
    return objective


def _print_handoff(handoff: dict[str, Any] | None, tag: str, kind: str) -> None:
    if handoff is None:
        return
    if kind == "pose":
        print(f"[HANDOFF_{tag}_S1_STAGE2_POSE] ot={handoff['ot_max']:.2e} "
              f"bp={handoff['bp_max']:.2e} ok={handoff['ok']}")
    else:
        print(f"[HANDOFF_{tag}_STAGE2_POSE_JOINT] state={handoff['state_max_diff']:.2e} "
              f"bp={handoff['physical_beta_init_max_diff_to_anchor']:.2e} "
              f"raw={handoff['raw_reprojection_max_diff']:.2e} "
              f"sa={handoff['shape_anchor_objective_at_init']:.2e} "
              f"rel={handoff['relative_difference_joint_init_to_pose']:.2e} "
              f"ok={handoff['ok']}")


def _pipeline_sync(context: StageContext) -> None:
    if context.device.type == "cuda":
        torch.cuda.synchronize(context.device)


def run_stage2(
    context: StageContext,
    stage1_outcome: StageOutcome,
    pipeline_timing: dict[str, float] | None = None,
    profiler_ranges: bool = False,
) -> Stage2Outcome:
    """Run Stage 2 as pose warmup followed by joint pose/shape refinement."""
    range_fn = (
        torch.profiler.record_function
        if profiler_ranges else lambda _name: contextlib.nullcontext()
    )
    _pipeline_sync(context)
    start = time.perf_counter()
    with range_fn("stage2_pose_definition"):
        pose_definition, s1_to_pose = _build_pose_warmup_definition(context, stage1_outcome)
    for label, tag in (("lm", "LM"), ("lbfgs", "BF")):
        _print_handoff(s1_to_pose[label], tag, "pose")
        if s1_to_pose[label] is not None and not s1_to_pose[label]["ok"]:
            raise RuntimeError(f"{tag} handoff Stage 1->Stage 2 pose failed")
    _pipeline_sync(context)
    if pipeline_timing is not None:
        pipeline_timing["stage2_pose_definition_and_handoff_ms"] = (
            time.perf_counter() - start
        ) * 1000.0
    _pipeline_sync(context)
    start = time.perf_counter()
    with range_fn("stage2_pose_solver"):
        pose_outcome = run_stage(pose_definition, context)
    _pipeline_sync(context)
    if pipeline_timing is not None:
        pipeline_timing["stage2_pose_run_ms"] = (time.perf_counter() - start) * 1000.0
    handoffs = {
        "stage1_to_stage2_pose_lm": s1_to_pose["lm"],
        "stage1_to_stage2_pose_lbfgs": s1_to_pose["lbfgs"],
    }
    if context.args.stop_after_stage == "stage2_pose":
        return Stage2Outcome(pose_warmup=pose_outcome, joint_refinement=None, handoffs=handoffs)

    _pipeline_sync(context)
    start = time.perf_counter()
    with range_fn("stage2_joint_definition"):
        joint_definition, pose_to_joint = _build_joint_refinement_definition(
            context, pose_outcome, pose_outcome.true_objective_fn, pose_definition.raw_groups_fn,
        )
    for label, tag in (("lm", "LM"), ("lbfgs", "BF")):
        _print_handoff(pose_to_joint[label], tag, "joint")
        if pose_to_joint[label] is not None and not pose_to_joint[label]["ok"]:
            raise RuntimeError(f"{tag} handoff Stage 2 pose->joint failed")
    _pipeline_sync(context)
    if pipeline_timing is not None:
        pipeline_timing["stage2_joint_definition_and_handoff_ms"] = (
            time.perf_counter() - start
        ) * 1000.0
    handoffs.update({
        "stage2_pose_to_joint_lm": pose_to_joint["lm"],
        "stage2_pose_to_joint_lbfgs": pose_to_joint["lbfgs"],
    })
    _pipeline_sync(context)
    start = time.perf_counter()
    with range_fn("stage2_joint_solver"):
        joint_outcome = run_stage(joint_definition, context)
    _pipeline_sync(context)
    if pipeline_timing is not None:
        pipeline_timing["stage2_joint_run_ms"] = (time.perf_counter() - start) * 1000.0
    return Stage2Outcome(
        pose_warmup=pose_outcome,
        joint_refinement=joint_outcome,
        handoffs=handoffs,
    )


# ------------------------------------------------------------------
# Handoff helpers
# ------------------------------------------------------------------

def check_s1_to_pose(
    stage1_parameters: torch.Tensor,
    stage2_pose_parameters: torch.Tensor,
    body_pose_anchors: list[torch.Tensor],
    frame_count: int,
    body_pose_dim: int,
) -> dict[str, Any]:
    """Verify the S1->STAGE2_POSE handoff preserves orient/transl and seeds pose from anchor."""
    per_frame_dim = STAGE1_PER_FRAME_DIM + body_pose_dim
    orient_transl_max_diff = 0.0
    body_pose_max_diff = 0.0
    for t in range(frame_count):
        ot_diff = float((stage2_pose_parameters[t * per_frame_dim:t * per_frame_dim + STAGE1_PER_FRAME_DIM]
                         - stage1_parameters[t * STAGE1_PER_FRAME_DIM:(t + 1) * STAGE1_PER_FRAME_DIM]).abs().max())
        orient_transl_max_diff = max(orient_transl_max_diff, ot_diff)
        bp_diff = float((stage2_pose_parameters[t * per_frame_dim + STAGE1_PER_FRAME_DIM:(t + 1) * per_frame_dim]
                         - body_pose_anchors[t]).abs().max())
        body_pose_max_diff = max(body_pose_max_diff, bp_diff)
    return {"ot_max": orient_transl_max_diff, "bp_max": body_pose_max_diff,
            "ok": orient_transl_max_diff < 1e-6 and body_pose_max_diff < 1e-6}


def handoff_s1_to_pose(
    stage1_parameters: torch.Tensor,
    body_pose_anchors: list[torch.Tensor],
    frame_count: int,
    body_pose_dim: int,
) -> torch.Tensor:
    """Construct the Stage 2 pose warmup initial vector from Stage 1 + pose anchors."""
    parts: list[torch.Tensor] = []
    for t in range(frame_count):
        if body_pose_anchors[t].numel() != body_pose_dim:
            raise ValueError("body pose anchor dimension mismatch")
        parts.append(stage1_parameters[t * STAGE1_PER_FRAME_DIM:(t + 1) * STAGE1_PER_FRAME_DIM])
        parts.append(body_pose_anchors[t].clone())
    return torch.cat(parts).clone()


def check_pose_to_joint(
    stage2_pose_parameters: torch.Tensor,
    stage2_joint_parameters: torch.Tensor,
    beta_anchor: torch.Tensor,
    beta_abs_limit: float,
    shape_anchor_scale: float,
    raw_groups_fn_pose: Callable[[torch.Tensor], Any],
    raw_groups_fn_joint: Callable[[torch.Tensor], Any],
    true_objective_pose: Callable[[torch.Tensor], torch.Tensor],
    true_objective_joint: Callable[[torch.Tensor], torch.Tensor],
) -> dict[str, Any]:
    """Verify the STAGE2_POSE->STAGE2_JOINT handoff preserves state and correctly inits beta."""
    stage2_pose_dim = stage2_pose_parameters.shape[0]
    beta_dim = stage2_joint_parameters.shape[0] - stage2_pose_dim
    state_max_diff = float((stage2_joint_parameters[:stage2_pose_dim] - stage2_pose_parameters).abs().max())
    beta_init = beta_abs_limit * torch.tanh(stage2_joint_parameters[-beta_dim:])
    beta_max_diff = float((beta_init - beta_anchor).abs().max())
    raw_diff = float((flatten_reprojection_groups(raw_groups_fn_joint(stage2_joint_parameters)).detach()
                      - flatten_reprojection_groups(raw_groups_fn_pose(stage2_pose_parameters)).detach()).abs().max())
    shape_anchor_objective_init = float(0.5 * torch.sum((beta_init - beta_anchor) ** 2 * shape_anchor_scale ** 2))
    objective_pose = float(true_objective_pose(stage2_pose_parameters).detach().item())
    objective_joint_init = float(true_objective_joint(stage2_joint_parameters).detach().item())
    relative_objective = abs(objective_joint_init - objective_pose) / max(abs(objective_pose), 1e-16)
    ok = (state_max_diff <= 1e-7 and beta_max_diff <= 1e-6 and raw_diff <= 1e-3
          and shape_anchor_objective_init <= 1e-10 and relative_objective <= 1e-6)
    return {
        "state_max_diff": state_max_diff,
        "physical_beta_init_max_diff_to_anchor": beta_max_diff,
        "raw_reprojection_max_diff": raw_diff,
        "shape_anchor_objective_at_init": shape_anchor_objective_init,
        "relative_difference_joint_init_to_pose": relative_objective,
        "ok": ok,
    }
