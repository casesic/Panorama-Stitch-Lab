"""Benchmark legacy objective lifecycle against exact local value reuse."""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import math
import statistics
import sys
import time
from pathlib import Path
from typing import Any

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
from lib_pano.human_fitting.staged_lm.types import StageOutcome
from script.benchmark.staged_lm.lm_early_stop_audit import (
    assert_exact_lm_parity,
    run_lm_with_convergence_trace,
)
from script.human_fitting.staged_lm.app import build_context


STAGES = ("S1", "STAGE2_POSE", "STAGE2_JOINT")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Exact objective-value reuse benchmark",
    )
    parser.add_argument("--frames", default="513")
    parser.add_argument("--frame_sets", required=True)
    parser.add_argument("--hmr_dir", required=True)
    parser.add_argument("--sam3d_dense_dir", required=True)
    parser.add_argument("--bbox_dir", required=True)
    parser.add_argument("--image_width", type=int, default=5760)
    parser.add_argument("--image_height", type=int, default=2880)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--reproj_objective",
        choices=("charbonnier_irls", "raw_l2"),
        default="charbonnier_irls",
    )
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--output_json", required=True)
    return parser.parse_args()


def _frames(text: str) -> list[int]:
    if "-" in text and "," not in text:
        first, last = (int(value) for value in text.split("-", 1))
        return list(range(first, last + 1))
    return [int(value) for value in text.split(",")]


def _context_args(args, frames):
    return argparse.Namespace(
        frames=",".join(map(str, frames)),
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
        profile_timing=True,
        export_debug029_format=False,
    )


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _legacy_stage(definition, context) -> StageOutcome:
    cache = definition.analytic_cache
    if cache is not None:
        cache.reset_timing()
        cache.profile_timing = True

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
        profile_timing=True,
        true_objective_fn=definition.lm_true_objective_fn,
        stage_name=definition.name,
        frame_count=context.frame_count,
        body_pose_dim=context.body_pose_dim,
        beta_dim=context.beta_dim,
        beta_abs_limit=context.objective_config.beta_abs_limit,
    )
    trace.result.eval_backend = definition.eval_backend
    return StageOutcome(
        trace.result, None, None, definition.lm_true_objective_fn, definition,
    )


def _stage_record(outcome, definition, wall_ms):
    cache = definition.analytic_cache
    result = outcome.lm_result
    return {
        "wall_ms": wall_ms,
        "objective_calls": cache.objective_calls,
        "objective_time_ms": cache.objective_ms,
        "status": result.status,
        "accepted": result.counters.accepted_steps,
        "rejected": result.counters.rejected_trials,
        "normal_equation_calls": result.counters.normal_equation_count,
        "solve_calls": result.counters.solve_attempt_count,
        "result": result,
    }


def _run_variant(args, frames, variant):
    context = build_context(_context_args(args, frames))
    runner = _legacy_stage if variant == "legacy" else run_stage
    _sync(context.device)
    segment_start = time.perf_counter()

    definitions = {}
    records = {}
    stage1_definition = build_stage1_definition(context)
    definitions["S1"] = stage1_definition
    _sync(context.device)
    start = time.perf_counter()
    stage1 = runner(stage1_definition, context)
    _sync(context.device)
    records["S1"] = _stage_record(
        stage1, stage1_definition, (time.perf_counter() - start) * 1000.0,
    )

    shared = build_stage2_shared_geometry_cache(context)
    pose_definition, _ = build_pose_stage_definition(context, stage1, shared)
    definitions["STAGE2_POSE"] = pose_definition
    _sync(context.device)
    start = time.perf_counter()
    pose = runner(pose_definition, context)
    _sync(context.device)
    records["STAGE2_POSE"] = _stage_record(
        pose, pose_definition, (time.perf_counter() - start) * 1000.0,
    )

    joint_definition, _ = build_joint_stage_definition(context, pose, shared)
    definitions["STAGE2_JOINT"] = joint_definition
    _sync(context.device)
    start = time.perf_counter()
    joint = runner(joint_definition, context)
    _sync(context.device)
    records["STAGE2_JOINT"] = _stage_record(
        joint, joint_definition, (time.perf_counter() - start) * 1000.0,
    )
    _sync(context.device)
    segment_ms = (time.perf_counter() - segment_start) * 1000.0
    return {
        "stages": records,
        "segment_ms": segment_ms,
        "ms_per_frame": segment_ms / len(frames),
        "fps": len(frames) * 1000.0 / segment_ms,
        "objective_calls_total": sum(
            record["objective_calls"] for record in records.values()
        ),
        "objective_time_total_ms": sum(
            record["objective_time_ms"] for record in records.values()
        ),
    }


def _parity(legacy, optimized):
    for stage in STAGES:
        old = legacy["stages"][stage]["result"]
        new = optimized["stages"][stage]["result"]
        try:
            assert_exact_lm_parity(old, new)
        except AssertionError as error:
            raise AssertionError(f"TRAJECTORY_PARITY_FAIL:{stage}") from error


def _percentile(values, fraction):
    ordered = sorted(values)
    index = min(len(ordered) - 1, math.ceil(fraction * len(ordered)) - 1)
    return ordered[index]


def _stats(values):
    return {
        "p50": statistics.median(values),
        "p90": _percentile(values, 0.9),
        "min": min(values),
        "max": max(values),
        "mean": statistics.mean(values),
        "std": statistics.pstdev(values),
    }


def _classification(reduction):
    if reduction >= 0.15:
        return "SIGNIFICANT"
    if reduction >= 0.05:
        return "MEASURABLE"
    if reduction > 0:
        return "SMALL"
    if reduction >= -0.05:
        return "NO_MEASURABLE_GAIN"
    return "REGRESSION"


def _serializable_run(run):
    return {
        key: value for key, value in run.items() if key != "stages"
    } | {
        "stages": {
            name: {
                key: value for key, value in record.items()
                if key != "result"
            }
            for name, record in run["stages"].items()
        },
    }


def benchmark_frame_set(args, frames):
    orders = []
    measured = {"legacy": [], "optimized": []}
    total = args.warmup + args.repeats
    for repeat in range(total):
        order = (
            ("legacy", "optimized")
            if repeat % 2 == 0 else ("optimized", "legacy")
        )
        pair = {}
        with contextlib.redirect_stdout(io.StringIO()):
            for variant in order:
                pair[variant] = _run_variant(args, frames, variant)
        _parity(pair["legacy"], pair["optimized"])
        if repeat >= args.warmup:
            orders.append(list(order))
            for variant in order:
                measured[variant].append(pair[variant])

    result = {
        "frames": frames,
        "frame_count": len(frames),
        "orders": orders,
        "trajectory_parity": "EXACT",
        "variants": {},
        "stages": {},
    }
    for variant in ("legacy", "optimized"):
        runs = measured[variant]
        result["variants"][variant] = {
            "segment_ms": _stats([run["segment_ms"] for run in runs]),
            "ms_per_frame": _stats([run["ms_per_frame"] for run in runs]),
            "fps": _stats([run["fps"] for run in runs]),
            "objective_calls_total": sorted({
                run["objective_calls_total"] for run in runs
            }),
            "objective_time_total_ms": _stats([
                run["objective_time_total_ms"] for run in runs
            ]),
        }
    for stage in STAGES:
        legacy_records = [
            run["stages"][stage] for run in measured["legacy"]
        ]
        optimized_records = [
            run["stages"][stage] for run in measured["optimized"]
        ]
        old_calls = sorted({record["objective_calls"] for record in legacy_records})
        new_calls = sorted({record["objective_calls"] for record in optimized_records})
        if len(old_calls) != 1 or len(new_calls) != 1:
            raise AssertionError("OBJECTIVE_CALL_REUSE_GATE_FAIL:unstable")
        old_wall = _stats([record["wall_ms"] for record in legacy_records])
        new_wall = _stats([record["wall_ms"] for record in optimized_records])
        reduction = 1.0 - new_wall["p50"] / old_wall["p50"]
        result["stages"][stage] = {
            "legacy_objective_calls": old_calls[0],
            "optimized_objective_calls": new_calls[0],
            "removed_calls": old_calls[0] - new_calls[0],
            "reduction_fraction":
                (old_calls[0] - new_calls[0]) / old_calls[0],
            "legacy_wall_ms": old_wall,
            "optimized_wall_ms": new_wall,
            "p50_wall_speedup": old_wall["p50"] / new_wall["p50"],
            "p90_wall_speedup": old_wall["p90"] / new_wall["p90"],
            "p50_reduction_fraction": reduction,
            "performance_classification": _classification(reduction),
            "accepted_rejected": [
                legacy_records[0]["accepted"],
                legacy_records[0]["rejected"],
            ],
            "normal_equation_calls":
                legacy_records[0]["normal_equation_calls"],
            "solve_calls": legacy_records[0]["solve_calls"],
            "parameter_exact": True,
            "objective_exact": True,
            "metrics_exact": True,
            "history_exact": True,
        }
    old_segment = result["variants"]["legacy"]["segment_ms"]["p50"]
    new_segment = result["variants"]["optimized"]["segment_ms"]["p50"]
    reduction = 1.0 - new_segment / old_segment
    result["pipeline"] = {
        "legacy_p50_ms": old_segment,
        "optimized_p50_ms": new_segment,
        "p50_speedup": old_segment / new_segment,
        "p50_reduction_fraction": reduction,
        "classification": _classification(reduction),
        "legacy_p50_fps": len(frames) * 1000.0 / old_segment,
        "optimized_p50_fps": len(frames) * 1000.0 / new_segment,
    }
    return result


def main() -> int:
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA_NOT_AVAILABLE")
        print("CUDA_BENCHMARK_SKIPPED")
        return 0
    report: dict[str, Any] = {
        "device": args.device,
        "objective": args.reproj_objective,
        "warmup": args.warmup,
        "repeats": args.repeats,
        "timing_schema_version": 5,
        "frame_sets": {},
        "t8": {"status": "T8_SKIPPED_MISSING_DATA"},
        "side_effect_audit": {
            "observable_effects": [
                "objective_calls", "objective_elapsed_time",
                "nested_kernel_profile_counters_and_elapsed_time",
            ],
            "status": "OBJECTIVE_SIDE_EFFECTS_TIMING_ONLY",
        },
    }
    for text in args.frame_sets.split(";"):
        frames = _frames(text)
        key = ",".join(map(str, frames))
        try:
            result = benchmark_frame_set(args, frames)
        except (FileNotFoundError, KeyError, ValueError, RuntimeError) as error:
            if isinstance(error, RuntimeError) and "Missing" not in str(error):
                raise
            result = {
                "status": "FRAME_SET_SKIPPED_MISSING_DATA",
                "error": f"{type(error).__name__}: {error}",
            }
        report["frame_sets"][key] = result

    frame513 = report["frame_sets"].get("513")
    if frame513 and "stages" in frame513:
        expected = {
            "S1": (67, 27), "STAGE2_POSE": (16, 6),
            "STAGE2_JOINT": (4, 2),
        }
        for stage, pair in expected.items():
            actual = frame513["stages"][stage]
            if (
                actual["legacy_objective_calls"],
                actual["optimized_objective_calls"],
            ) != pair:
                raise AssertionError(
                    f"OBJECTIVE_CALL_REUSE_GATE_FAIL:{stage}",
                )
        old_total = sum(pair[0] for pair in expected.values())
        new_total = sum(pair[1] for pair in expected.values())
        report["frame513_gate"] = {
            "status": "PASS",
            "legacy_calls": old_total,
            "optimized_calls": new_total,
            "removed_calls": old_total - new_total,
            "reduction_fraction": (old_total - new_total) / old_total,
        }
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, allow_nan=False), encoding="utf-8",
    )
    print(f"[OBJECTIVE_REUSE_BENCHMARK] {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
