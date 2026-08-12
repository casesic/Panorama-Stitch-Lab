"""Serial-driven lockstep and independent Pose LM shadow experiments."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any

import torch

from lib_pano.human_fitting.staged_lm.lm.convergence import check_convergence
from lib_pano.human_fitting.staged_lm.lm.equations import (
    compute_scaled_gradient, normal_equations_max_diagonal,
    normal_equations_quadratic_form, solve_normal_equations,
)
from lib_pano.human_fitting.staged_lm.lm.runner import run_lm_optimization
from lib_pano.human_fitting.staged_lm.linearization import (
    Linearization,
    LinearizationProvenance,
)
from lib_pano.human_fitting.staged_lm.types import LMResult, StageDefinition
from script.benchmark.staged_lm.pose_batch_candidates import (
    PoseGroupBatch, batched_pose_residual_only, gather_pose_batch_parameters,
)
from script.benchmark.staged_lm.pose_shadow_common import (
    PoseNEMode, compute_batched_pose_row_scales, equation_errors,
    evaluate_pose_equations, serial_pose_row_scales,
)


@dataclass
class PoseShadowTrialRecord:
    outer: int
    trial: int
    damping: float
    serial_delta_norm: float | None
    batch_delta_norm: float | None
    delta_max_abs: float | None
    delta_relative: float | None
    serial_predicted_reduction: float | None
    batch_predicted_reduction: float | None
    predicted_relative: float | None
    serial_actual_reduction: float | None
    batch_actual_reduction: float | None
    serial_rho: float | None
    batch_rho: float | None
    rho_abs_diff: float | None
    serial_accepted: bool
    batch_accepted: bool
    acceptance_match: bool
    serial_used_fallback: bool
    batch_used_fallback: bool
    h_relative: float
    h_max_abs: float
    g_relative: float
    g_max_abs: float
    g_norm_serial: float
    g_norm_batch: float
    residual_relative: float
    residual_max_abs: float
    jacobian_relative: float
    jacobian_max_abs: float
    serial_equation_ms: float
    batch_equation_ms: float
    worst_parameter_index: int | None
    worst_h_index: tuple[int, int] | None
    worst_g_index: int | None
    g_abs_l2: float
    g_scaled_error: float
    g_cosine_similarity: float
    g_sign_mismatch_count: int
    near_acceptance_boundary: bool
    predicted_margin: float | None
    actual_margin: float | None
    rho_margin: float | None


@dataclass
class PoseLockstepResult:
    status: str
    final_parameters: torch.Tensor
    final_objective: float
    trial_records: list[PoseShadowTrialRecord]
    acceptance_mismatch_count: int
    first_acceptance_mismatch: dict[str, Any] | None
    accepted_steps: int
    rejected_trials: int
    final_lambda: float

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "status": self.status, "final_objective": self.final_objective,
            "accepted_steps": self.accepted_steps, "rejected_trials": self.rejected_trials,
            "final_lambda": self.final_lambda,
            "acceptance_mismatch_count": self.acceptance_mismatch_count,
            "first_acceptance_mismatch": self.first_acceptance_mismatch,
            "trial_records": [asdict(record) for record in self.trial_records],
        }


def _candidate_scales(
    stage: StageDefinition, batch: PoseGroupBatch, parameters: torch.Tensor,
    context: Any, weight_source: str,
) -> tuple[torch.Tensor, dict[str, Any]]:
    if weight_source == "serial":
        return serial_pose_row_scales(
            stage, parameters, batch, objective_mode=context.args.reproj_objective,
            charbonnier_eps=context.charbonnier_eps, rep_scale=context.scaling.rep_scale,
        )
    packed = gather_pose_batch_parameters(parameters, batch.layout)
    residual = batched_pose_residual_only(
        batch, packed.group_orientations, packed.group_translations,
        packed.group_body_poses, context.image_width, context.image_height,
    )
    return compute_batched_pose_row_scales(
        residual, batch, stage.aggregation_spec,
        objective_mode=context.args.reproj_objective,
        charbonnier_eps=context.charbonnier_eps, rep_scale=context.scaling.rep_scale,
    )


def make_batched_pose_linearization(
    stage: StageDefinition, batch: PoseGroupBatch, context: Any,
    linearization_point: torch.Tensor, *, mode: PoseNEMode,
    weight_source: str, hg_reduction_mode: str,
) -> Linearization:
    scales, stats = _candidate_scales(stage, batch, linearization_point, context, weight_source)
    def equations(parameters: torch.Tensor):
        return evaluate_pose_equations(
            mode, stage, batch, parameters, scales, context.body_pose_anchors,
            context.scaling.pose_anchor_scale, context.image_width, context.image_height,
            hg_reduction_mode=hg_reduction_mode,
        ).equations

    return Linearization(
        normal_equations=equations,
        # Direct normal equations are the active path.  Keep a lightweight
        # placeholder for the generic LM interface without rebuilding serial
        # weights a second time.
        surrogate=lambda x: x.new_empty(0),
        weight_stats=stats,
        aggregation_spec=stage.aggregation_spec or {},
        provenance=LinearizationProvenance(
            jacobian_backend="batched_pose_shadow",
            linearization_backend="direct_normal_equations_batch_shadow",
            linear_solver_backend="dense",
        ),
    )


def run_pose_lm_independent_batch(
    stage: StageDefinition, batch: PoseGroupBatch, context: Any, *,
    candidate_mode: PoseNEMode, weight_source: str, hg_reduction_mode: str,
    initial_parameters: torch.Tensor | None = None, profile_timing: bool = True,
) -> LMResult:
    initial = stage.initial_lm_parameters if initial_parameters is None else initial_parameters
    return run_lm_optimization(
        label="STAGE2_POSE_BATCH_SHADOW",
        make_linearization=lambda x: make_batched_pose_linearization(
            stage, batch, context, x, mode=candidate_mode,
            weight_source=weight_source, hg_reduction_mode=hg_reduction_mode,
        ),
        raw_metrics_fn=stage.raw_metrics_fn,
        initial_parameters=initial,
        param_dim=stage.parameter_dimension,
        max_accepted_steps=stage.lm_max_accepted_steps,
        max_trials=context.solver_config.lm_max_trials,
        solver_config=context.solver_config,
        profile_timing=profile_timing,
        true_objective_fn=stage.lm_true_objective_fn,
    )


def _worst_index(value: torch.Tensor) -> int | None:
    return int(value.abs().argmax()) if value.numel() else None


def run_pose_lm_lockstep(
    serial_stage_definition: StageDefinition, pose_batch: PoseGroupBatch,
    context: Any, *, candidate_mode: PoseNEMode, weight_source: str,
    hg_reduction_mode: str, initial_parameters: torch.Tensor | None = None,
) -> PoseLockstepResult:
    stage = serial_stage_definition
    x = (stage.initial_lm_parameters if initial_parameters is None else initial_parameters).detach().clone()
    records: list[PoseShadowTrialRecord] = []
    damping: float | None = None
    accepted_steps = rejected_trials = 0
    status = "RUNNING"
    for outer in range(stage.lm_max_accepted_steps):
        serial_scales, _ = serial_pose_row_scales(
            stage, x, pose_batch, objective_mode=context.args.reproj_objective,
            charbonnier_eps=context.charbonnier_eps, rep_scale=context.scaling.rep_scale,
        )
        candidate_scales, _ = _candidate_scales(stage, pose_batch, x, context, weight_source)
        serial_eval = evaluate_pose_equations(
            PoseNEMode.SERIAL_PRODUCTION, stage, pose_batch, x, serial_scales,
            context.body_pose_anchors, context.scaling.pose_anchor_scale,
            context.image_width, context.image_height,
        )
        candidate_eval = evaluate_pose_equations(
            candidate_mode, stage, pose_batch, x, candidate_scales,
            context.body_pose_anchors, context.scaling.pose_anchor_scale,
            context.image_width, context.image_height,
            hg_reduction_mode=hg_reduction_mode,
        )
        if damping is None:
            damping = normal_equations_max_diagonal(serial_eval.equations) * 1e-3
        obj_before = float(stage.lm_true_objective_fn(x).detach())
        errors = equation_errors(candidate_eval, serial_eval)
        accepted_this = False
        for trial in range(context.solver_config.lm_max_trials):
            trial_damping = damping
            serial_solve = solve_normal_equations(serial_eval.equations, damping)
            batch_solve = solve_normal_equations(candidate_eval.equations, damping)
            sd, bd = serial_solve.delta, batch_solve.delta
            if sd is None:
                damping = min(damping * 10, 1e12); rejected_trials += 1
                continue
            serial_x = (x + sd.to(torch.float32)).detach()
            serial_trial_obj = float(stage.lm_true_objective_fn(serial_x).detach())
            sp = float(-serial_eval.equations.gradient.dot(sd) - .5 * normal_equations_quadratic_form(serial_eval.equations, sd))
            sa = obj_before - serial_trial_obj; sr = sa / max(sp, 1e-12) if sp > 0 else 0.0
            sacc = sp > 0 and sa > 0 and sr > 0
            if bd is None:
                bp = ba = br = None; bacc = False; batch_obj = None
            else:
                batch_x = (x + bd.to(torch.float32)).detach()
                batch_obj = float(stage.lm_true_objective_fn(batch_x).detach())
                bp = float(-candidate_eval.equations.gradient.dot(bd) - .5 * normal_equations_quadratic_form(candidate_eval.equations, bd))
                ba = obj_before - batch_obj; br = ba / max(bp, 1e-12) if bp > 0 else 0.0
                bacc = bp > 0 and ba > 0 and br > 0
            delta_diff = None if bd is None else bd - sd
            gdiff = candidate_eval.equations.gradient - serial_eval.equations.gradient
            gref, gcand = serial_eval.equations.gradient, candidate_eval.equations.gradient
            sign_mask = gref.abs() > 1e-6
            hdiff = candidate_eval.equations.hessian - serial_eval.equations.hessian
            worst_h_flat = _worst_index(hdiff)
            record = PoseShadowTrialRecord(
                outer, trial, trial_damping, float(sd.norm()), None if bd is None else float(bd.norm()),
                None if delta_diff is None else float(delta_diff.abs().max()),
                None if delta_diff is None else float(delta_diff.norm()) / max(float(sd.norm()), 1e-30),
                sp, bp, None if bp is None else abs(bp-sp)/max(abs(sp),1e-30), sa, ba, sr, br,
                None if br is None else abs(br-sr), sacc, bacc, sacc == bacc,
                bool(serial_solve.backend == "dense_solve" or serial_solve.schur_failed),
                bool(batch_solve.backend == "dense_solve" or batch_solve.schur_failed),
                errors["H"]["relative_frobenius"], errors["H"]["max_abs"],
                errors["g"]["relative_frobenius"], errors["g"]["max_abs"],
                float(gref.norm()), float(gcand.norm()), errors["residual"]["relative_frobenius"],
                errors["residual"]["max_abs"], errors["jacobian"]["relative_frobenius"],
                errors["jacobian"]["max_abs"], serial_eval.elapsed_ms, candidate_eval.elapsed_ms,
                _worst_index(delta_diff) if delta_diff is not None else None,
                None if worst_h_flat is None else (worst_h_flat // hdiff.shape[1], worst_h_flat % hdiff.shape[1]),
                _worst_index(gdiff), float(gdiff.norm()),
                float(gdiff.norm()) / max(float(gref.norm()), math.sqrt(gref.numel()) * 1e-8),
                float(torch.nn.functional.cosine_similarity(gref[None], gcand[None])),
                int(((torch.sign(gref) != torch.sign(gcand)) & sign_mask).sum()),
                abs(sp) <= 1e-6 or abs(sa) <= 1e-6 or abs(sr) <= 1e-5,
                sp, sa, sr,
            )
            records.append(record)
            if sacc:
                x = serial_x; accepted_steps += 1; accepted_this = True
                if sr > .75: damping = max(damping / 3, 1e-10)
                elif sr < .25: damping = min(damping * 3, 1e12)
                break
            damping = min(damping * 10, 1e12); rejected_trials += 1
        if not accepted_this:
            status = "LM_NO_ACCEPTED_TRIAL"; break
        next_linear = stage.linearization_builder(x, stage.aggregation_spec)
        next_eq = next_linear.normal_equations(x)
        scaled = compute_scaled_gradient(next_eq)
        rel = (obj_before - float(stage.lm_true_objective_fn(x).detach())) / max(abs(obj_before), 1e-16)
        convergence = check_convergence(scaled, float(sd.norm()), rel, context.solver_config)
        if convergence is not None:
            status = convergence; break
    if status == "RUNNING": status = "MAX_ACCEPTED_STEPS"
    mismatches = [record for record in records if not record.acceptance_match]
    return PoseLockstepResult(
        status, x, float(stage.lm_true_objective_fn(x).detach()), records,
        len(mismatches), None if not mismatches else {"outer": mismatches[0].outer, "trial": mismatches[0].trial},
        accepted_steps, rejected_trials, float(damping),
    )


def compare_lm_results(serial: LMResult, batch: LMResult) -> dict[str, Any]:
    difference = batch.parameters - serial.parameters
    first = None
    for index, (left, right) in enumerate(zip(serial.trial_history, batch.trial_history)):
        for key in ("outer", "trial", "solve_success", "accepted", "schur_primary_failed"):
            if left.get(key) != right.get(key):
                first = {"index": index, "difference_type": key, "serial": left.get(key), "batch": right.get(key)}; break
        if first: break
    if first is None and len(serial.trial_history) != len(batch.trial_history):
        first = {"difference_type": "trial_history_length", "serial": len(serial.trial_history), "batch": len(batch.trial_history)}
    return {
        "first_divergent_trial": first,
        "final_parameter_max_abs": float(difference.abs().max()),
        "final_parameter_relative": float(difference.norm()) / max(float(serial.parameters.norm()), 1e-30),
        "final_objective_relative": abs(batch.objective-serial.objective)/max(abs(serial.objective),1e-30),
        "status_match": serial.status == batch.status,
        "accepted_match": serial.counters.accepted_steps == batch.counters.accepted_steps,
        "rejected_match": serial.counters.rejected_trials == batch.counters.rejected_trials,
        "trial_count_match": len(serial.trial_history) == len(batch.trial_history),
        "final_lambda_relative": abs(batch.final_lambda-serial.final_lambda)/max(abs(serial.final_lambda),1e-30),
        "metrics_max_abs": max(abs(batch.rms_metrics[k]-serial.rms_metrics[k]) for k in ("rms","median","p95")),
        "weight_history_match": serial.weight_history == batch.weight_history,
    }
