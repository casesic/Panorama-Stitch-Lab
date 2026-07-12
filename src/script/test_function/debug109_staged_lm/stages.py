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

import time
from typing import Any, Callable

import torch

from .irls import (
    build_primary_aggregation_spec,
    compute_primary_charbonnier_tensor,
    make_irls_linearization,
    validate_first_order,
)
from .lm_solver import run_lbfgs_oracle, run_lm_optimization
from .config import STAGE1_PER_FRAME_DIM
from .parameter_layout import build_stage1_initial_parameters
from .residuals import flatten_reprojection_groups, make_raw_reproj_groups, raw_reproj_metrics
from .types import (
    LBFGSResult,
    LMResult,
    StageContext,
    StageDefinition,
    StageOutcome,
    Stage2Outcome,
)


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


def _build_linearization_and_objective(
    stage: StageDefinition,
    context: StageContext,
) -> tuple[Callable[[torch.Tensor], dict[str, Any]], Callable[[torch.Tensor], torch.Tensor], dict[str, Any] | None, float]:
    """Create the linearisation factory and true objective for a stage.

    Returns ``(make_linearization, true_objective_fn, init_check, init_validation_ms)``.
    For IRLS mode the surrogate is the weighted least-squares majoriser; for
    raw-L2 mode it is the plain scaled residual.
    """
    init_validation_ms = 0.0
    if context.is_irls:
        static_spec = build_primary_aggregation_spec(stage.raw_groups_fn(stage.initial_lm_parameters))

        def make_linearization(x: torch.Tensor) -> dict[str, Any]:
            return make_irls_linearization(
                stage.raw_groups_fn, stage.anchor_residual_fn, x,
                context.charbonnier_eps, static_spec,
            )

        def true_objective(x: torch.Tensor) -> torch.Tensor:
            groups = stage.raw_groups_fn(x)
            objective = compute_primary_charbonnier_tensor(groups, context.charbonnier_eps, static_spec)
            if stage.anchor_residual_fn is not None:
                objective = objective + 0.5 * torch.sum(stage.anchor_residual_fn(x) ** 2)
            return objective

        init_check = None
        do_initial_validation = context.args.validate_irls_first_order
        if do_initial_validation:
            if context.args.profile_timing and torch.cuda.is_available():
                torch.cuda.synchronize()
            t_iv = time.perf_counter() if context.args.profile_timing else 0
            init_linearization = make_linearization(stage.initial_lm_parameters)
            init_check = validate_first_order(
                init_linearization["surrogate_fn"], true_objective,
                stage.initial_lm_parameters, f"{stage.name}_init",
                context.solver_config.irls_gradient_rel_tol,
            )
            if context.args.profile_timing:
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

        def make_linearization(x: torch.Tensor) -> dict[str, Any]:
            return {"surrogate_fn": total_residual, "weight_stats": {}}

        def true_objective(x: torch.Tensor) -> torch.Tensor:
            return 0.5 * torch.sum(total_residual(x) ** 2)

        init_check = None

    return make_linearization, true_objective, init_check, init_validation_ms


def run_stage(stage: StageDefinition, context: StageContext) -> StageOutcome:
    """Execute one optimisation stage (LM + optional L-BFGS + verdict).

    Prints the stage header, runs the LM solver, optionally runs the L-BFGS
    oracle, and computes the LM-vs-LBFGS verdict.
    """
    make_linearization, true_objective, init_check, init_validation_ms = _build_linearization_and_objective(stage, context)

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

    validate_each_accepted = context.args.validate_irls_each_accepted and context.is_irls

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
        progress_every_accepted=context.args.progress_every_accepted,
        true_objective_fn=true_objective,
        validate_irls=validate_each_accepted,
        init_irls_check=init_check,
    )

    # Fold the initial-validation timing (measured outside the solver) into
    # the solver's validation_ms counter, matching the original behaviour.
    if context.args.profile_timing and init_check is not None and lm_result.profile_timing is not None:
        lm_result.profile_timing["validation_ms"] += init_validation_ms

    lbfgs_result = None
    if context.args.run_lbfgs_oracle and stage.initial_lbfgs_parameters is not None:
        lbfgs_result = run_lbfgs_oracle(
            label=f"{stage.name}_LBFGS",
            true_objective_fn=true_objective,
            raw_metrics_fn=stage.raw_metrics_fn,
            initial_parameters=stage.initial_lbfgs_parameters,
            outer_steps=stage.lbfgs_outer_steps,
            solver_config=context.solver_config,
        )

    verdict = stage_verdict(lm_result, lbfgs_result, stage.raw_metrics_fn)
    if verdict:
        print(f"[V_{stage.name}] {verdict}")

    return StageOutcome(
        lm_result=lm_result,
        lbfgs_result=lbfgs_result,
        verdict=verdict,
        true_objective_fn=true_objective,
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

    return StageDefinition(
        name="S1",
        parameter_dimension=parameter_dimension,
        initial_lm_parameters=initial_parameters,
        initial_lbfgs_parameters=initial_parameters.clone(),
        raw_groups_fn=raw_groups_fn,
        anchor_residual_fn=None,
        raw_metrics_fn=raw_metrics_fn,
        lm_max_accepted_steps=context.solver_config.stage1_lm_max_acc,
        lbfgs_outer_steps=context.solver_config.stage1_bf_outer,
    )


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

    stage = StageDefinition(
        name="STAGE2_POSE",
        parameter_dimension=parameter_dimension,
        initial_lm_parameters=initial_lm,
        initial_lbfgs_parameters=initial_lbfgs,
        raw_groups_fn=raw_groups_fn,
        anchor_residual_fn=pose_anchor_residual,
        raw_metrics_fn=raw_metrics_fn,
        lm_max_accepted_steps=context.solver_config.stage2_pose_warmup_lm_max_acc,
        lbfgs_outer_steps=context.solver_config.stage2_pose_warmup_bf_outer,
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

    stage = StageDefinition(
        name="STAGE2_JOINT",
        parameter_dimension=parameter_dimension,
        initial_lm_parameters=initial_lm,
        initial_lbfgs_parameters=initial_lbfgs,
        raw_groups_fn=raw_groups_fn,
        anchor_residual_fn=all_anchors_residual,
        raw_metrics_fn=raw_metrics_fn,
        lm_max_accepted_steps=context.solver_config.stage2_joint_lm_max_acc,
        lbfgs_outer_steps=context.solver_config.stage2_joint_bf_outer,
    )

    # STAGE2_POSE->STAGE2_JOINT handoff checks
    handoff_lm = check_pose_to_joint(
        stage2_pose_outcome.lm_result.parameters, initial_lm,
        context.beta_anchor, beta_abs_limit, shape_anchor_scale,
        stage2_pose_raw_groups_fn, raw_groups_fn,
        stage2_pose_true_objective, _make_stage2_joint_true_objective(stage, context, raw_groups_fn, all_anchors_residual),
    )
    handoff_lbfgs = None
    if stage2_pose_outcome.lbfgs_result is not None and initial_lbfgs is not None:
        handoff_lbfgs = check_pose_to_joint(
            stage2_pose_outcome.lbfgs_result.parameters, initial_lbfgs,
            context.beta_anchor, beta_abs_limit, shape_anchor_scale,
            stage2_pose_raw_groups_fn, raw_groups_fn,
            stage2_pose_true_objective, _make_stage2_joint_true_objective(stage, context, raw_groups_fn, all_anchors_residual),
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


def run_stage2(context: StageContext, stage1_outcome: StageOutcome) -> Stage2Outcome:
    """Run Stage 2 as pose warmup followed by joint pose/shape refinement."""
    pose_definition, s1_to_pose = _build_pose_warmup_definition(context, stage1_outcome)
    for label, tag in (("lm", "LM"), ("lbfgs", "BF")):
        _print_handoff(s1_to_pose[label], tag, "pose")
        if s1_to_pose[label] is not None and not s1_to_pose[label]["ok"]:
            raise RuntimeError(f"{tag} handoff Stage 1->Stage 2 pose failed")
    pose_outcome = run_stage(pose_definition, context)
    handoffs = {
        "stage1_to_stage2_pose_lm": s1_to_pose["lm"],
        "stage1_to_stage2_pose_lbfgs": s1_to_pose["lbfgs"],
    }
    if context.args.stop_after_stage == "stage2_pose":
        return Stage2Outcome(pose_warmup=pose_outcome, joint_refinement=None, handoffs=handoffs)

    joint_definition, pose_to_joint = _build_joint_refinement_definition(
        context, pose_outcome, pose_outcome.true_objective_fn, pose_definition.raw_groups_fn,
    )
    for label, tag in (("lm", "LM"), ("lbfgs", "BF")):
        _print_handoff(pose_to_joint[label], tag, "joint")
        if pose_to_joint[label] is not None and not pose_to_joint[label]["ok"]:
            raise RuntimeError(f"{tag} handoff Stage 2 pose->joint failed")
    handoffs.update({
        "stage2_pose_to_joint_lm": pose_to_joint["lm"],
        "stage2_pose_to_joint_lbfgs": pose_to_joint["lbfgs"],
    })
    return Stage2Outcome(
        pose_warmup=pose_outcome,
        joint_refinement=run_stage(joint_definition, context),
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
