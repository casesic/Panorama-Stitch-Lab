"""Run the read-only staged-LM convergence/objective lifecycle audit."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from lib_pano.human_fitting.staged_lm.stage2_shared_cache import (
    build_stage2_shared_geometry_cache,
)
from lib_pano.human_fitting.staged_lm.stage_definitions.joint import (
    build_joint_stage_definition,
)
from lib_pano.human_fitting.staged_lm.stage_definitions.pose import (
    build_pose_stage_definition,
)
from lib_pano.human_fitting.staged_lm.stage_definitions.rigid import (
    build_stage1_definition,
)
from lib_pano.human_fitting.staged_lm.stage_runtime import run_stage
from lib_pano.human_fitting.staged_lm.types import StageDefinition, StageOutcome
from script.benchmark.staged_lm.lm_early_stop_audit import (
    TracedLMRun,
    assert_exact_lm_parity,
    objective_call_attribution,
    replace_accepted_budget,
    replace_initial_parameters,
    run_lm_with_convergence_trace,
    static_solver_audit,
    trace_to_json,
)
from script.human_fitting.staged_lm.app import build_context


EXTENDED_DEFAULTS = {"S1": 40, "STAGE2_POSE": 15, "STAGE2_JOINT": 5}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Read-only LM early-stop and objective-call audit",
    )
    parser.add_argument("--frames", required=True)
    parser.add_argument("--frame_sets", required=True)
    parser.add_argument("--hmr_dir", required=True)
    parser.add_argument("--sam3d_dense_dir", required=True)
    parser.add_argument("--bbox_dir", required=True)
    parser.add_argument("--image_width", type=int, default=5760)
    parser.add_argument("--image_height", type=int, default=2880)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--reproj_objective",
        choices=("raw_l2", "charbonnier_irls"),
        default="charbonnier_irls",
    )
    parser.add_argument("--profile_timing", action="store_true")
    parser.add_argument("--run_final_reinit", action="store_true")
    parser.add_argument("--run_extended_budget", action="store_true")
    parser.add_argument("--stage1_extended_acc", type=int, default=40)
    parser.add_argument("--pose_extended_acc", type=int, default=15)
    parser.add_argument("--joint_extended_acc", type=int, default=5)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--output_npz", required=True)
    return parser.parse_args()


def _frame_set_values(text: str) -> list[int]:
    text = text.strip()
    if "-" in text and "," not in text:
        first, last = (int(value) for value in text.split("-", 1))
        return list(range(first, last + 1))
    return [int(value) for value in text.split(",") if value.strip()]


def _frame_argument(values: list[int]) -> str:
    return ",".join(str(value) for value in values)


def _context_args(
    args: argparse.Namespace, frames: list[int],
) -> argparse.Namespace:
    return argparse.Namespace(
        frames=_frame_argument(frames),
        hmr_dir=args.hmr_dir,
        sam3d_dense_dir=args.sam3d_dense_dir,
        bbox_dir=args.bbox_dir,
        image_width=args.image_width,
        image_height=args.image_height,
        output_dir=str(Path(args.output_json).parent),
        device=args.device,
        validate_device_contract=False,
        reproj_objective=args.reproj_objective,
        stop_after_stage="stage2",
        profile_timing=args.profile_timing,
        export_debug029_format=False,
    )


def _trace_definition(
    definition: StageDefinition,
    context,
) -> TracedLMRun:
    if definition.linearization_builder is None:
        raise RuntimeError("INCOMPLETE_PRODUCTION_STAGE_DEFINITION")
    if definition.lm_true_objective_fn is None:
        raise RuntimeError("INCOMPLETE_PRODUCTION_STAGE_DEFINITION")
    cache = definition.analytic_cache
    if cache is not None and hasattr(cache, "reset_timing"):
        cache.reset_timing()
        if hasattr(cache, "profile_timing"):
            cache.profile_timing = context.args.profile_timing

    def make_linearization(parameters):
        return definition.linearization_builder(
            parameters, definition.aggregation_spec,
        )

    trace = run_lm_with_convergence_trace(
        label=f"{definition.name}_LM",
        make_linearization=make_linearization,
        raw_metrics_fn=definition.raw_metrics_fn,
        initial_parameters=definition.initial_lm_parameters,
        param_dim=definition.parameter_dimension,
        max_accepted_steps=definition.lm_max_accepted_steps,
        max_trials=context.solver_config.lm_max_trials,
        solver_config=context.solver_config,
        profile_timing=context.args.profile_timing,
        true_objective_fn=definition.lm_true_objective_fn,
        stage_name=definition.name,
        frame_count=context.frame_count,
        body_pose_dim=context.body_pose_dim,
        beta_dim=context.beta_dim,
        beta_abs_limit=context.objective_config.beta_abs_limit,
    )
    trace.result.eval_backend = definition.eval_backend
    return trace


def _production_and_trace(
    definition: StageDefinition,
    context,
) -> tuple[StageOutcome, TracedLMRun]:
    production = run_stage(definition, context)
    traced = _trace_definition(definition, context)
    assert_exact_lm_parity(production.lm_result, traced.result)
    cache = definition.analytic_cache
    cache_calls = getattr(cache, "objective_calls", None)
    trace_calls = len(traced.objective_calls)
    if cache_calls is not None and cache_calls != trace_calls:
        raise AssertionError(
            f"OBJECTIVE_COUNTER_MISMATCH:{definition.name}:"
            f"{cache_calls}!={trace_calls}"
        )
    return production, traced


def _trace_with_parity(
    definition: StageDefinition,
    context,
) -> TracedLMRun:
    production = run_stage(definition, context)
    traced = _trace_definition(definition, context)
    assert_exact_lm_parity(production.lm_result, traced.result)
    return traced


def _prefix_exact(
    production: TracedLMRun, extended: TracedLMRun,
) -> bool:
    trial_count = len(production.result.trial_history)
    weight_count = len(production.result.weight_history)
    return (
        extended.result.trial_history[:trial_count]
        == production.result.trial_history
        and extended.result.weight_history[:weight_count]
        == production.result.weight_history
        and all(
            torch.equal(
                production.checkpoints[key],
                extended.checkpoints[key],
            )
            for key in production.checkpoints
            if key.startswith("accepted_step_")
        )
    )


def _final_reinit_classification(trace: TracedLMRun) -> list[str]:
    result = []
    initial = trace.initial_diagnostics
    if initial["initial_gradient_pass_config"]:
        result.append("INITIAL_CONVERGENCE_CHECK_MISSING_OBSERVED")
    if (
        trace.result.counters.accepted_steps == 0
        and initial["initial_gradient_pass_hardcoded"]
        and trace.result.status == "LM_NO_ACCEPTED_TRIAL"
    ):
        result.append("NO_ACCEPTED_BRANCH_MASKS_INITIAL_CONVERGENCE")
    if trace.result.counters.accepted_steps > 1:
        result.append("SAVED_FINAL_WAS_BUDGET_LIMITED_NOT_CONVERGED")
    if trace.result.counters.accepted_steps <= 1 and (
        float(
            (
                trace.result.parameters
                - trace.checkpoints["baseline_initial"]
            ).norm()
        ) < 1e-4
    ):
        result.append("EARLY_STOP_RESPONDS_TO_HIGH_QUALITY_INITIALIZATION")
    return result


def _stage_audit(
    definition: StageDefinition,
    production: StageOutcome,
    trace: TracedLMRun,
    context,
    *,
    extended_budget: int,
    run_final_reinit: bool,
    run_extended: bool,
) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    result = {
        "production": production.lm_result.to_json_dict(),
        "trace": trace_to_json(trace),
        "trace_parity": "EXACT",
        "cache_objective_counter": getattr(
            definition.analytic_cache, "objective_calls", None,
        ),
    }
    accepted_count = production.lm_result.counters.accepted_steps
    if accepted_count:
        previous_key = (
            f"accepted_step_{accepted_count - 1}"
            if accepted_count > 1 else "baseline_initial"
        )
        metrics_before = definition.raw_metrics_fn(
            trace.checkpoints[previous_key],
        )
        metrics_after = production.lm_result.rms_metrics
        result["production_budget_last_step"] = {
            "objective_reduction":
                trace.outer_records[-1].actual_reduction,
            "relative_objective_reduction":
                trace.outer_records[-1].relative_objective_change,
            "metrics_before": metrics_before,
            "metrics_after": metrics_after,
            "metric_change": {
                key: metrics_after[key] - metrics_before[key]
                for key in ("rms", "median", "p95")
            },
            "delta_block_norms":
                trace.outer_records[-1].block_step_norms,
            "classification": "MEANINGFUL_PROGRESS_AT_BUDGET_LIMIT",
        }
    checkpoints = dict(trace.checkpoints)
    classifications = [
        "RAW_STEP_NORM_MIXES_PARAMETER_UNITS",
        trace_to_json(trace)["duplicate_objective_proof"]["status"],
    ]
    reachability = trace_to_json(trace)["predicate_reachability"]
    if reachability["first_outer_scaled_gradient_pass"] is None:
        classifications.append("SCALED_GRADIENT_TOLERANCE_NOT_REACHED")
    if reachability["first_outer_step_pass"] is None:
        classifications.append("STEP_TOLERANCE_NOT_REACHED")
    if reachability["first_outer_rel_objective_pass"] is None:
        classifications.append("REL_OBJECTIVE_TOLERANCE_NOT_REACHED")
    if production.lm_result.status == "MAX_ACCEPTED_STEPS":
        classifications.append("BUDGET_CAP_REACHED_WITH_MEANINGFUL_PROGRESS")
    if trace.outer_records and trace.outer_records[-1].is_final_budget_outer:
        classifications.append(
            "FINAL_BUDGET_RELINEARIZATION_HAS_NO_NEXT_STEP",
        )

    if run_final_reinit:
        reinit_definition = replace_initial_parameters(
            definition, production.lm_result.parameters,
        )
        reinit_trace = _trace_with_parity(reinit_definition, context)
        reinit_trace.checkpoints["final_reinit_final"] = (
            reinit_trace.result.parameters.detach().clone()
        )
        reinit_json = trace_to_json(reinit_trace)
        reinit_json.update({
            "parameter_drift_from_supplied_final": float(
                (
                    reinit_trace.result.parameters
                    - production.lm_result.parameters
                ).norm()
            ),
            "objective_drift": (
                reinit_trace.result.objective
                - production.lm_result.objective
            ),
            "classifications": _final_reinit_classification(reinit_trace),
        })
        result["final_solution_reinit"] = reinit_json
        checkpoints.update({
            "final_reinit_final":
                reinit_trace.result.parameters.detach().clone(),
        })
        classifications.extend(reinit_json["classifications"])

    if run_extended:
        extended_definition = replace_accepted_budget(
            definition, extended_budget,
        )
        extended_trace = _trace_with_parity(extended_definition, context)
        prefix = _prefix_exact(trace, extended_trace)
        if not prefix:
            raise AssertionError(
                f"EXTENDED_BUDGET_PREFIX_FAIL:{definition.name}",
            )
        extended_json = trace_to_json(extended_trace)
        extended_json.update({
            "production_prefix_exact": prefix,
            "objective_at_production_budget":
                production.lm_result.objective,
            "objective_at_extended_stop": extended_trace.result.objective,
            "metric_at_production_budget":
                production.lm_result.rms_metrics,
            "metric_at_extended_stop":
                extended_trace.result.rms_metrics,
            "parameter_drift": float(
                (
                    extended_trace.result.parameters
                    - production.lm_result.parameters
                ).norm()
            ),
            "extra_normal_equation_calls": (
                extended_trace.result.counters.normal_equation_count
                - production.lm_result.counters.normal_equation_count
            ),
            "extra_objective_calls": (
                len(extended_trace.objective_calls)
                - len(trace.objective_calls)
            ),
            "first_triggered_predicate": (
                extended_trace.result.status
                if extended_trace.result.status.startswith("CONVERGED_")
                else None
            ),
        })
        result["extended_budget"] = extended_json
        checkpoints["extended_final"] = (
            extended_trace.result.parameters.detach().clone()
        )
        if extended_trace.result.objective < production.lm_result.objective:
            classifications.append("SAVED_FINAL_IS_BUDGET_LIMITED")

    result["classifications"] = sorted(set(classifications))
    return result, checkpoints


def _recommendations(stage_reports: list[dict[str, Any]]) -> list[dict[str, Any]]:
    removable = sum(
        report["trace"]["objective_call_attribution"][
            "removable_duplicate_calls"
        ]
        for report in stage_reports
    )
    return [
        {
            "rank": 1, "option": "A. EXACT_OBJECTIVE_VALUE_REUSE",
            "reason": (
                f"{removable} duplicate calls proven reusable across the "
                "audited frame sets; preserves trajectory/status/"
                "parameters with low numerical and implementation risk."
            ),
        },
        {
            "rank": 2, "option": "B. INITIAL_POINT_GRADIENT_CHECK",
            "reason": "Saves trials for converged reinitializations but may change status.",
        },
        {
            "rank": 3, "option": "C. CONFIG_CONSISTENT_NO_ACCEPTED_STATUS",
            "reason": "Corrects status semantics; parameters stay fixed but status changes.",
        },
        {
            "rank": 4, "option": "F. ACCEPTED_STEP_BUDGET_SWEEP",
            "reason": "Evidence gathering before changing final parameters.",
        },
        {
            "rank": 5, "option": "D. SCALE_AWARE_STEP_CRITERION",
            "reason": "More meaningful units, but changes stopping behavior.",
        },
        {
            "rank": 6, "option": "E. STAGE_SPECIFIC_CONVERGENCE_TOLERANCES",
            "reason": "Potentially useful but higher tuning and numerical risk.",
        },
        {
            "rank": 7, "option": "G. REMOVE_FINAL_BUDGET_RELINEARIZATION",
            "reason": "Would remove final-gradient status and weight-history semantics.",
        },
    ]


def run_audit(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    report: dict[str, Any] = {
        "static_audit": static_solver_audit(),
        "production_baseline": {},
        "trace_parity": {},
        "frame_sets": {},
        "per_stage": {},
        "objective_call_attribution": {},
        "duplicate_objective_proof": {},
        "final_budget_relinearization": {},
        "final_solution_reinit": {},
        "extended_budget": {},
        "predicate_reachability": {},
        "parameter_block_step_norms": {},
        "classifications": {},
        "recommendations": [],
        "irls_semantics": {
            "scaled_gradient": "surrogate_scaled_gradient",
            "relative_change": "true_objective_relative_change",
            "note": (
                "The frozen-IRLS JTr/JTJ gradient is not described here as "
                "the full analytic Charbonnier gradient."
            ),
        },
    }
    npz: dict[str, np.ndarray] = {}
    recommendation_source: list[dict[str, Any]] = []

    for frame_set_text in args.frame_sets.split(";"):
        frames = _frame_set_values(frame_set_text)
        frame_key = _frame_argument(frames)
        try:
            context = build_context(_context_args(args, frames))
        except (FileNotFoundError, KeyError, ValueError, RuntimeError) as error:
            if isinstance(error, RuntimeError) and "Missing" not in str(error):
                raise
            report["frame_sets"][frame_key] = {
                "status": "FRAME_SET_SKIPPED_MISSING_DATA",
                "error": f"{type(error).__name__}: {error}",
            }
            continue
        stage1_definition = build_stage1_definition(context)
        stage1, stage1_trace = _production_and_trace(
            stage1_definition, context,
        )
        shared_cache = build_stage2_shared_geometry_cache(context)
        pose_definition, _ = build_pose_stage_definition(
            context, stage1, shared_cache,
        )
        pose, pose_trace = _production_and_trace(pose_definition, context)
        joint_definition, _ = build_joint_stage_definition(
            context, pose, shared_cache,
        )
        joint, joint_trace = _production_and_trace(
            joint_definition, context,
        )
        stage_inputs = (
            ("S1", stage1_definition, stage1, stage1_trace,
             args.stage1_extended_acc),
            ("STAGE2_POSE", pose_definition, pose, pose_trace,
             args.pose_extended_acc),
            ("STAGE2_JOINT", joint_definition, joint, joint_trace,
             args.joint_extended_acc),
        )
        stage_reports = {}
        for stage_name, definition, outcome, trace, extended_budget in stage_inputs:
            stage_report, checkpoints = _stage_audit(
                definition, outcome, trace, context,
                extended_budget=extended_budget,
                run_final_reinit=args.run_final_reinit,
                run_extended=args.run_extended_budget,
            )
            stage_reports[stage_name] = stage_report
            recommendation_source.append(stage_report)
            for candidate, tensor in checkpoints.items():
                key = f"{frame_key}__{stage_name}__{candidate}"
                array = tensor.detach().cpu().numpy()
                if array.dtype == object:
                    raise TypeError("AUDIT_CHECKPOINT_OBJECT_DTYPE")
                npz[key] = array
        report["frame_sets"][frame_key] = {
            "status": "PASS",
            "stages": stage_reports,
        }
        report["per_stage"][frame_key] = stage_reports
        report["production_baseline"][frame_key] = {
            name: value["production"] for name, value in stage_reports.items()
        }
        report["trace_parity"][frame_key] = {
            name: value["trace_parity"] for name, value in stage_reports.items()
        }
        report["objective_call_attribution"][frame_key] = {
            name: value["trace"]["objective_call_attribution"]
            for name, value in stage_reports.items()
        }
        report["duplicate_objective_proof"][frame_key] = {
            name: value["trace"]["duplicate_objective_proof"]
            for name, value in stage_reports.items()
        }
        report["final_budget_relinearization"][frame_key] = {
            name: value["trace"]["final_budget_relinearization"]
            for name, value in stage_reports.items()
        }
        report["predicate_reachability"][frame_key] = {
            name: value["trace"]["predicate_reachability"]
            for name, value in stage_reports.items()
        }
        report["parameter_block_step_norms"][frame_key] = {
            name: [
                record["block_step_norms"]
                for record in value["trace"]["outer_records"]
            ]
            for name, value in stage_reports.items()
        }
        if args.run_final_reinit:
            report["final_solution_reinit"][frame_key] = {
                name: value["final_solution_reinit"]
                for name, value in stage_reports.items()
            }
        if args.run_extended_budget:
            report["extended_budget"][frame_key] = {
                name: value["extended_budget"]
                for name, value in stage_reports.items()
            }

    report["recommendations"] = _recommendations(recommendation_source)
    all_classifications = sorted({
        classification
        for frame in report["per_stage"].values()
        for stage in frame.values()
        for classification in stage["classifications"]
    })
    report["classifications"]["all"] = all_classifications
    if any(
        "INITIAL_CONVERGENCE_CHECK_MISSING_OBSERVED"
        in classifications
        for frame in report["per_stage"].values()
        for classifications in (
            stage.get("final_solution_reinit", {}).get(
                "classifications", [],
            )
            for stage in frame.values()
        )
    ):
        main = "EARLY_STOP_MISSING_INITIAL_POINT_CHECK"
    elif any(
        stage["production"]["status"] == "MAX_ACCEPTED_STEPS"
        for frame in report["per_stage"].values()
        for stage in frame.values()
    ):
        main = "CURRENT_BUDGET_ENDS_BEFORE_CONVERGENCE"
    else:
        main = "EARLY_STOP_EFFECTIVE_BUT_TOLERANCES_NOT_REACHED"
    report["classifications"]["main"] = main
    return report, npz


def main() -> int:
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA_NOT_AVAILABLE")
        print("CUDA_AUDIT_SKIPPED")
        return 0
    report, checkpoints = run_audit(args)
    output_json = Path(args.output_json)
    output_npz = Path(args.output_npz)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_npz.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(report, indent=2, allow_nan=False), encoding="utf-8",
    )
    np.savez(output_npz, **checkpoints)
    print(f"[AUDIT_JSON] {output_json}")
    print(f"[AUDIT_NPZ] {output_npz}")
    print(f"[MAIN] {report['classifications']['main']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
