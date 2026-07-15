"""Stage 2 pose/joint backend, Schur, and FK solver regressions.

Mechanically consolidated from the historical per-check scripts.
"""

from __future__ import annotations

import argparse, statistics
import torch
from script.human_fitting.staged_lm.app import build_context
from lib_pano.human_fitting.staged_lm.residuals import flatten_reprojection_groups
from script.validation.staged_lm.references.stage_backends import _build_joint_refinement_definition, _build_pose_warmup_definition, build_stage1_definition, run_stage
import argparse
import statistics
from contextlib import nullcontext
from script.validation.staged_lm.references.joint_backends import use_separate_fk_reference
from script.validation.staged_lm.references.gate_c import evaluate_gate_c


def backend_args(c):return argparse.Namespace(frames=c.frames,hmr_dir=c.hmr_dir,sam3d_dense_dir=c.sam3d_dense_dir,bbox_dir=c.bbox_dir,
    image_width=c.image_width,image_height=c.image_height,output_dir="/tmp/debug117",reproj_objective="charbonnier_irls",stop_after_stage="stage2",
    run_lbfgs_oracle=False,jacobian_backend="analytic_stage1",stage1_eval_backend="analytic",validate_stage1_fast_eval=False,
    stage2_pose_linearization_backend="analytic",stage2_pose_eval_backend="fast",validate_stage2_pose_fast_eval=False,
    stage2_joint_linearization_backend="autograd",stage2_joint_eval_backend="full_forward",validate_stage2_joint_fast_eval=False,
    validate_irls_first_order=False,validate_irls_each_accepted=False,profile_timing=c.profile,progress_every_accepted=0)


def backend_metric_close(a,b):return abs(a-b)<=1e-3 or abs(a-b)/max(abs(b),1e-16)<=1e-5


def backend_run_mode(context,pose,label,linearization,evaluation):
    context.args.stage2_joint_linearization_backend=linearization;context.args.stage2_joint_eval_backend=evaluation
    context.args.validate_stage2_joint_fast_eval=evaluation=="fast"
    definition,_=_build_joint_refinement_definition(context,pose,pose.true_objective_fn,pose.definition.raw_groups_fn)
    outcome=run_stage(definition,context);raw=flatten_reprojection_groups(definition.raw_groups_fn(outcome.lm_result.parameters)).detach()
    beta=context.objective_config.beta_abs_limit*torch.tanh(outcome.lm_result.parameters[-context.beta_dim:])
    return label,outcome,raw,beta


def backend_compare(reference,candidate):
    rl,ro,rr,rb=reference;cl,co,cr,cb=candidate;r,g=ro.lm_result,co.lm_result
    obj=abs(g.objective-r.objective)/max(abs(r.objective),1e-16)
    if obj>1e-5:raise RuntimeError(f"JOINT_OBJECTIVE_MISMATCH {cl}")
    torch.testing.assert_close(cr,rr,rtol=1e-6,atol=1e-3);torch.testing.assert_close(g.parameters,r.parameters,rtol=1e-4,atol=1e-4)
    torch.testing.assert_close(cb,rb,rtol=1e-5,atol=1e-5)
    if not all(backend_metric_close(g.rms_metrics[n],r.rms_metrics[n]) for n in ("rms","median","p95")):raise RuntimeError("JOINT_METRIC_MISMATCH")
    print(f"[REGRESSION {rl}->{cl}] obj_rel={obj:.3e} raw_max={float((cr-rr).abs().max()):.3e} param_max={float((g.parameters-r.parameters).abs().max()):.3e} beta_max={float((cb-rb).abs().max()):.3e} acc/rej={g.counters.accepted_steps}/{g.counters.rejected_trials} status={g.status} lambda={g.final_lambda:.6e}")


def run_joint_backend_regression(cli=None):
    cli=cli or parse_args();context=build_context(backend_args(cli));stage1=run_stage(build_stage1_definition(context),context)
    pose_def,_=_build_pose_warmup_definition(context,stage1);pose=run_stage(pose_def,context)
    modes=(("A_AUTOGRAD_FULL","autograd","full_forward"),("B_ANALYTIC_FULL","analytic","full_forward"),("C_ANALYTIC_FAST","analytic","fast"))
    repeats=3 if cli.profile else 1;runs={m[0]:[] for m in modes}
    for _ in range(repeats):
        for mode in modes:runs[mode[0]].append(backend_run_mode(context,pose,*mode))
    reps=[runs[m[0]][0] for m in modes];backend_compare(reps[0],reps[1]);backend_compare(reps[0],reps[2])
    for label,values in runs.items():
        def med(key):return statistics.median((v[1].lm_result.profile_timing or {}).get(key,0.) for v in values)
        lm=values[0][1].lm_result
        print(f"[TIMING {label}] wall={statistics.median(v[1].lm_result.wall_time_ms for v in values):.3f} jac={med('solver_jacobian_ms'):.3f} obj={med('objective_eval_ms'):.3f} metrics={med('final_metrics_ms'):.3f} solve={med('linear_solve_ms'):.3f} normal_eq={lm.counters.normal_equation_count} acc/rej={lm.counters.accepted_steps}/{lm.counters.rejected_trials} shape={med('stage2_joint_shape_state_ms'):.3f} direct={med('stage2_joint_normal_equations_ms'):.3f} fast_obj={med('stage2_joint_objective_ms'):.3f} fast_metrics={med('stage2_joint_metrics_ms'):.3f}")
    print("[PASS] Stage 2 joint A/B/C solver regression");return 0


def schur_make_args(cli: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(
        frames=cli.frames, hmr_dir=cli.hmr_dir,
        sam3d_dense_dir=cli.sam3d_dense_dir, bbox_dir=cli.bbox_dir,
        image_width=cli.image_width, image_height=cli.image_height,
        output_dir="/tmp/debug118", reproj_objective="charbonnier_irls",
        stop_after_stage="stage2", run_lbfgs_oracle=False,
        jacobian_backend="analytic_stage1", stage1_eval_backend="analytic",
        validate_stage1_fast_eval=False,
        stage2_pose_linearization_backend="analytic", stage2_pose_eval_backend="fast",
        validate_stage2_pose_fast_eval=False,
        stage2_joint_linearization_backend="autograd",
        stage2_joint_eval_backend="full_forward", stage2_joint_linear_solver="dense",
        validate_stage2_joint_fast_eval=False, validate_irls_first_order=False,
        validate_irls_each_accepted=False, profile_timing=cli.profile,
        progress_every_accepted=0,
    )


def schur_run_mode(context, pose, label, linearization, evaluation, solver):
    context.args.stage2_joint_linearization_backend = linearization
    context.args.stage2_joint_eval_backend = evaluation
    context.args.stage2_joint_linear_solver = solver
    context.args.validate_stage2_joint_fast_eval = evaluation == "fast"
    definition, _ = _build_joint_refinement_definition(
        context, pose, pose.true_objective_fn, pose.definition.raw_groups_fn,
    )
    outcome = run_stage(definition, context)
    raw = flatten_reprojection_groups(
        definition.raw_groups_fn(outcome.lm_result.parameters)
    ).detach()
    beta = context.objective_config.beta_abs_limit * torch.tanh(
        outcome.lm_result.parameters[-context.beta_dim:]
    )
    return label, outcome, raw, beta


def schur_metric_close(a: float, b: float) -> bool:
    return abs(a-b) <= 1e-3 or abs(a-b)/max(abs(b), 1e-16) <= 1e-5


def compare_dense_schur(dense, schur) -> None:
    _, dense_outcome, dense_raw, dense_beta = dense
    _, schur_outcome, schur_raw, schur_beta = schur
    reference, candidate = dense_outcome.lm_result, schur_outcome.lm_result
    objective_rel = abs(candidate.objective-reference.objective)/max(abs(reference.objective), 1e-16)
    if objective_rel > 1e-7:
        raise RuntimeError("SCHUR_FINAL_OBJECTIVE_MISMATCH")
    torch.testing.assert_close(schur_raw, dense_raw, rtol=1e-6, atol=1e-3)
    torch.testing.assert_close(candidate.parameters, reference.parameters, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(schur_beta, dense_beta, rtol=1e-6, atol=1e-6)
    if not all(schur_metric_close(candidate.rms_metrics[name], reference.rms_metrics[name])
               for name in ("rms", "median", "p95")):
        raise RuntimeError("SCHUR_FINAL_METRIC_MISMATCH")
    exact_fields = (
        candidate.counters.accepted_steps == reference.counters.accepted_steps,
        candidate.counters.rejected_trials == reference.counters.rejected_trials,
        candidate.status == reference.status,
        candidate.counters.linearization_count == reference.counters.linearization_count,
        candidate.final_lambda == reference.final_lambda,
        len(candidate.trial_history) == len(reference.trial_history),
    )
    if not all(exact_fields):
        raise RuntimeError("SCHUR_LM_TRAJECTORY_MISMATCH")
    for dense_trial, schur_trial in zip(reference.trial_history, candidate.trial_history):
        for key in ("outer", "trial", "damping", "accepted"):
            if dense_trial[key] != schur_trial[key]:
                raise RuntimeError(f"SCHUR_TRIAL_FIELD_MISMATCH {key}")
        for key, tolerance in (("delta_norm", 1e-10), ("predicted_reduction", 1e-10), ("rho", 1e-7)):
            left, right = dense_trial[key], schur_trial[key]
            if left is not None and abs(left-right)/max(abs(left), 1e-16) > tolerance:
                raise RuntimeError(f"SCHUR_TRIAL_NUMERIC_MISMATCH {key}")
    print(
        f"[DENSE_SCHUR] objective_rel={objective_rel:.3e} "
        f"raw_max={float((schur_raw-dense_raw).abs().max()):.3e} "
        f"parameter_max={float((candidate.parameters-reference.parameters).abs().max()):.3e} "
        f"beta_max={float((schur_beta-dense_beta).abs().max()):.3e} "
        f"trials={len(candidate.trial_history)} schur_success={candidate.counters.schur_success_count} "
        f"fallback={candidate.counters.schur_dense_fallback_count}"
    )


def run_dense_schur_regression(cli=None) -> int:
    cli = cli or parse_args()
    context = build_context(schur_make_args(cli))
    stage1 = run_stage(build_stage1_definition(context), context)
    pose_definition, _ = _build_pose_warmup_definition(context, stage1)
    pose = run_stage(pose_definition, context)
    all_modes = (
        ("A_AUTOGRAD_FULL_DENSE", "autograd", "full_forward", "dense"),
        ("B_ANALYTIC_FULL_DENSE", "analytic", "full_forward", "dense"),
        ("C_ANALYTIC_FAST_DENSE", "analytic", "fast", "dense"),
        ("D_ANALYTIC_FAST_SCHUR", "analytic", "fast", "schur"),
    )
    modes = all_modes[2:] if cli.scaling_only else all_modes
    repeats = 3 if cli.profile else 1
    runs = {mode[0]: [] for mode in modes}
    for _ in range(repeats):
        for mode in modes:
            runs[mode[0]].append(schur_run_mode(context, pose, *mode))
    compare_dense_schur(
        runs["C_ANALYTIC_FAST_DENSE"][0], runs["D_ANALYTIC_FAST_SCHUR"][0],
    )
    frame_count = context.frame_count
    p, b = 12 + context.body_pose_dim, context.beta_dim
    dense_elements = (frame_count*p+b)**2
    arrow_elements = frame_count*p*p + frame_count*p*b + b*b + frame_count*p + b
    print(
        f"[STORAGE] T={frame_count} P={p} B={b} param_dim={frame_count*p+b} "
        f"dense_bytes={dense_elements*8} arrowhead_bytes={arrow_elements*8}"
    )
    for label, values in runs.items():
        def median(key):
            return statistics.median((v[1].lm_result.profile_timing or {}).get(key, 0.0) for v in values)
        lm = values[0][1].lm_result
        print(
            f"[TIMING {label}] wall={statistics.median(v[1].lm_result.wall_time_ms for v in values):.3f} "
            f"normal_eq={median('stage2_joint_normal_equations_ms'):.3f} "
            f"solve={median('linear_solve_ms'):.3f} schur={median('stage2_joint_schur_solve_ms'):.3f} "
            f"local={median('stage2_joint_schur_local_factor_ms'):.3f} "
            f"reduced_build={median('stage2_joint_schur_reduced_build_ms'):.3f} "
            f"reduced_solve={median('stage2_joint_schur_reduced_solve_ms'):.3f} "
            f"backsub={median('stage2_joint_schur_back_substitution_ms'):.3f} "
            f"acc/rej={lm.counters.accepted_steps}/{lm.counters.rejected_trials} obj={lm.objective:.8e}"
        )
    print("[PASS] Stage 2 joint dense/Schur regression")
    return 0


def fk_make_args(cli: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(
        frames=cli.frames, hmr_dir=cli.hmr_dir,
        sam3d_dense_dir=cli.sam3d_dense_dir, bbox_dir=cli.bbox_dir,
        image_width=cli.image_width, image_height=cli.image_height,
        output_dir="/tmp/staged_lm_joint_fk", reproj_objective="charbonnier_irls",
        stop_after_stage="stage2",
        run_lbfgs_oracle=getattr(cli, "run_lbfgs_oracle", False),
        jacobian_backend="analytic_stage1", stage1_eval_backend="analytic",
        validate_stage1_fast_eval=False,
        stage2_pose_linearization_backend="analytic", stage2_pose_eval_backend="fast",
        validate_stage2_pose_fast_eval=False,
        stage2_joint_linearization_backend="analytic",
        stage2_joint_eval_backend="fast", stage2_joint_linear_solver="schur",
        stage2_joint_fk_jacobian_backend="combined",
        validate_stage2_joint_fast_eval=False, validate_irls_first_order=False,
        validate_irls_each_accepted=False, profile_timing=cli.profile,
        progress_every_accepted=0,
    )


def run_backend(context, pose, backend: str, *, run_lbfgs: bool = False):
    context.args.run_lbfgs_oracle = run_lbfgs
    manager = use_separate_fk_reference() if backend == "separate" else nullcontext()
    with manager:
        definition, handoff = _build_joint_refinement_definition(
            context, pose, pose.true_objective_fn, pose.definition.raw_groups_fn,
        )
        outcome = run_stage(definition, context)
    raw = flatten_reprojection_groups(
        definition.raw_groups_fn(outcome.lm_result.parameters)
    ).detach()
    beta = context.objective_config.beta_abs_limit * torch.tanh(
        outcome.lm_result.parameters[-context.beta_dim:]
    )
    return outcome, raw, beta, handoff


def _metric_close(left: float, right: float) -> bool:
    return abs(left-right) <= 1e-4 or abs(left-right)/max(abs(right), 1e-16) <= 1e-6


def _relative(left: float, right: float) -> float:
    return abs(left-right)/max(abs(right), 1e-16)


def fk_compare(reference_data, candidate_data) -> None:
    reference_outcome, reference_raw, reference_beta = reference_data[:3]
    candidate_outcome, candidate_raw, candidate_beta = candidate_data[:3]
    reference, candidate = reference_outcome.lm_result, candidate_outcome.lm_result
    objective_rel = _relative(candidate.objective, reference.objective)
    if objective_rel > 1e-7:
        raise RuntimeError("COMBINED_FK_FINAL_OBJECTIVE_MISMATCH")
    torch.testing.assert_close(candidate_raw, reference_raw, rtol=1e-6, atol=1e-3)
    torch.testing.assert_close(candidate.parameters, reference.parameters, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(candidate_beta, reference_beta, rtol=1e-7, atol=1e-7)
    if not all(_metric_close(candidate.rms_metrics[name], reference.rms_metrics[name])
               for name in ("rms", "median", "p95")):
        raise RuntimeError("COMBINED_FK_FINAL_METRIC_MISMATCH")
    exact = (
        candidate.counters.accepted_steps == reference.counters.accepted_steps,
        candidate.counters.rejected_trials == reference.counters.rejected_trials,
        candidate.status == reference.status,
        candidate.final_lambda == reference.final_lambda,
        candidate.counters.linearization_count == reference.counters.linearization_count,
        candidate.counters.normal_equation_count == reference.counters.normal_equation_count,
        candidate.counters.schur_success_count == reference.counters.schur_success_count,
        candidate.counters.schur_dense_fallback_count == reference.counters.schur_dense_fallback_count,
        len(candidate.trial_history) == len(reference.trial_history),
    )
    if not all(exact):
        raise RuntimeError("COMBINED_FK_LM_TRAJECTORY_MISMATCH")
    for separate, combined in zip(reference.trial_history, candidate.trial_history):
        for key in ("outer", "trial", "damping", "solve_success", "accepted", "used_dense_fallback"):
            if separate.get(key) != combined.get(key):
                raise RuntimeError(f"COMBINED_FK_TRIAL_FIELD_MISMATCH {key}")
        for key, tolerance in (("delta_norm", 1e-9), ("predicted_reduction", 1e-9),
                               ("rho", 1e-7)):
            left, right = separate.get(key), combined.get(key)
            if left is not None and _relative(float(left), float(right)) > tolerance:
                raise RuntimeError(f"COMBINED_FK_TRIAL_NUMERIC_MISMATCH {key}")
        left, right = separate.get("actual_reduction"), combined.get("actual_reduction")
        if left is not None and abs(float(left)-float(right)) > 1e-6:
            raise RuntimeError("COMBINED_FK_TRIAL_NUMERIC_MISMATCH actual_reduction")
    first_lambda = candidate.trial_history[0]["damping"] if candidate.trial_history else None
    print(
        f"[SEPARATE_COMBINED] objective_rel={objective_rel:.3e} "
        f"raw_max={float((candidate_raw-reference_raw).abs().max()):.3e} "
        f"parameter_max={float((candidate.parameters-reference.parameters).abs().max()):.3e} "
        f"beta_max={float((candidate_beta-reference_beta).abs().max()):.3e} "
        f"initial_lambda={first_lambda} final_lambda={candidate.final_lambda} "
        f"trials={len(candidate.trial_history)} acc/rej="
        f"{candidate.counters.accepted_steps}/{candidate.counters.rejected_trials}"
    )


def run_combined_fk_regression(cli=None) -> int:
    cli = cli or parse_args()
    context = build_context(fk_make_args(cli))
    stage1 = run_stage(build_stage1_definition(context), context)
    pose_definition, pose_handoff = _build_pose_warmup_definition(context, stage1)
    pose = run_stage(pose_definition, context)
    repeats = 1 if cli.run_lbfgs_oracle else (3 if cli.profile else 1)
    runs = {"separate": [], "combined": []}
    for _ in range(repeats):
        for backend in ("separate", "combined"):
            runs[backend].append(run_backend(
                context, pose, backend,
                run_lbfgs=cli.run_lbfgs_oracle and backend == "combined",
            ))
    fk_compare(runs["separate"][0], runs["combined"][0])
    for backend, values in runs.items():
        def median_profile(key: str) -> float:
            return statistics.median(
                (value[0].lm_result.profile_timing or {}).get(key, 0.0) for value in values
            )
        result = values[0][0].lm_result
        print(
            f"[TIMING {backend.upper()}] wall={statistics.median(v[0].lm_result.wall_time_ms for v in values):.3f} "
            f"solver_jacobian={median_profile('solver_jacobian_ms'):.3f} "
            f"normal_eq={median_profile('stage2_joint_normal_equations_ms'):.3f} "
            f"objective={median_profile('objective_eval_ms'):.3f} "
            f"schur={median_profile('stage2_joint_schur_solve_ms'):.3f} "
            f"fk_combined_calls={median_profile('stage2_joint_fk_combined_calls'):.0f} "
            f"fk_separate_calls={median_profile('stage2_joint_fk_separate_calls'):.0f} "
            f"verdict={values[0][0].verdict} status={result.status}"
        )
    if cli.run_lbfgs_oracle and runs["combined"][0][0].verdict != "LM_MATCHES_LBFGS":
        raise RuntimeError("COMBINED_FK_LBFGS_ORACLE_MISMATCH")
    if cli.run_lbfgs_oracle:
        joint_handoff = runs["combined"][0][3]
        gate = evaluate_gate_c(
            oracle_enabled=True,
            stage_matches={
                "stage1_matches": stage1.verdict == "LM_MATCHES_LBFGS",
                "pose_matches": pose.verdict == "LM_MATCHES_LBFGS",
                "joint_matches": runs["combined"][0][0].verdict == "LM_MATCHES_LBFGS",
            },
            non_oracle_checks={
                "pose_handoff_lm": bool(pose_handoff["lm"]["ok"]),
                "joint_handoff_lm": bool(joint_handoff["lm"]["ok"]),
                "separate_combined_regression": True,
            },
            oracle_checks={
                "pose_handoff_lbfgs": bool(pose_handoff["lbfgs"]["ok"]),
                "joint_handoff_lbfgs": bool(joint_handoff["lbfgs"]["ok"]),
            },
        )
        print(f"[GATE_C] {gate['verdict']}")
        if gate["verdict"] != "GATE_C_PASS_READY_FOR_MULTIFRAME_GATE_D":
            raise RuntimeError("COMBINED_FK_GATE_C_BLOCKED")
    print("[PASS] Stage 2 joint combined-FK production regression")
    return 0

def run_all_solver_regressions(args) -> None:
    run_pose_solver_regression(args)
    run_joint_backend_regression(args)
    run_dense_schur_regression(args)
    run_combined_fk_regression(args)
    run_full_handoff_validation(args)


def run_full_handoff_validation(cli=None):
    """Run the validation-only full pose-to-joint reprojection/objective check."""
    cli = cli or parse_args()
    context = build_context(fk_make_args(cli))
    stage1 = run_stage(build_stage1_definition(context), context)
    pose_definition, _ = _build_pose_warmup_definition(context, stage1)
    pose = run_stage(pose_definition, context)
    _, handoff = _build_joint_refinement_definition(
        context, pose, pose.true_objective_fn, pose.definition.raw_groups_fn,
    )
    result = handoff["lm"]
    if not result["ok"]:
        raise RuntimeError("FULL_POSE_TO_JOINT_HANDOFF_VALIDATION_FAILED")
    print(
        "[FULL_HANDOFF] "
        f"state={result['state_max_diff']:.3e} "
        f"beta={result['physical_beta_init_max_diff_to_anchor']:.3e} "
        f"raw={result['raw_reprojection_max_diff']:.3e} "
        f"shape={result['shape_anchor_objective_at_init']:.3e} "
        f"relative={result['relative_difference_joint_init_to_pose']:.3e}"
    )
    print("[PASS] Full pose-to-joint handoff validation")
    return 0
