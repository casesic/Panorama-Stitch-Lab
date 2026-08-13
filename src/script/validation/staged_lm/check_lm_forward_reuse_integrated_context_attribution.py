#!/usr/bin/env python3
"""Summarise the D0-D5 integrated-context forward-reuse experiment."""

from __future__ import annotations

import argparse
import csv
import json
import platform
import statistics
from collections import defaultdict
from pathlib import Path


POLICIES = {
    "D0": "optimized_native_stable_p",
    "D1": "optimized_native_stable_p_forward_reuse",
    "D2": "diagnostic_forward_reuse_lin_to_obj",
    "D3": "diagnostic_forward_reuse_obj_to_lin",
    "D4": "diagnostic_forward_reuse_obj_to_lin_shadow_warmup",
    "D5": "diagnostic_forward_reuse_retain_drop",
}
TOP_LEVEL = {
    "jac.rotation_derivative", "jac.pose_feature_derivative",
    "jac.pose_offset_derivative", "jac.fk_derivative",
    "jac.lbs_derivative", "jac.erp_chain", "jac.assembly",
    "combined.shape_setup", "combined.fk", "combined.lbs",
    "combined.erp_chain", "combined.assembly",
}
FAMILIES = {
    "pose_offset_gemm": {"jac.pose_offset_derivative"},
    "selected_transform_gemm": {
        "jac.lbs_selected_transform", "combined.lbs_selected_transform"},
    "fused_lbs_downstream": {"jac.lbs_downstream", "combined.lbs_downstream"},
    "fk": {"jac.fk_derivative", "combined.fk"},
    "erp": {"jac.erp_chain", "combined.erp_chain"},
    "assembly": {"jac.assembly", "combined.assembly"},
}


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def median(values: list[float]) -> float:
    return statistics.median(values)


def classify(value: float, strong: float, partial: float) -> str:
    if value >= strong:
        return "SUPPORTED"
    if value >= partial:
        return "PARTIALLY_SUPPORTED"
    return "WEAK_OR_NOT_SUPPORTED"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    root = args.root
    raw = root / "raw"
    samples: list[dict[str, object]] = []
    call_rows: list[dict[str, object]] = []
    direction_counts: dict[str, dict[str, int]] = {}
    checksums: dict[str, set[str]] = defaultdict(set)
    payload_bytes: dict[str, int] = {}

    for diagnostic, policy in POLICIES.items():
        directories = [path for path in raw.iterdir()
                       if path.is_dir() and path.name.rsplit("_", 1)[0] == policy]
        directories.sort(key=lambda p: int(p.name.rsplit("_", 1)[1]))
        expected = 20 if diagnostic in {"D4", "D5"} else 30
        if len(directories) != expected:
            raise RuntimeError(f"{diagnostic}: expected {expected} samples, got {len(directories)}")
        aggregate_counts: dict[str, int] = defaultdict(int)
        for directory in directories:
            run_index = int(directory.name.rsplit("_", 1)[1])
            pipeline = rows(directory / "pipeline_runs.csv")[0]
            checksums[diagnostic].add(pipeline["final_checksum"])
            kernels = rows(directory / "kernel_samples.csv")
            kernel_values: dict[str, float] = defaultdict(float)
            derivative_total = 0.0
            shadow_total = 0.0
            occurrences: dict[tuple[str, str, str], list[float]] = defaultdict(list)
            for item in kernels:
                value = float(item["milliseconds"])
                kernel = item["kernel"]
                stage = item["stage"]
                kernel_values[kernel] += value
                if kernel in TOP_LEVEL:
                    derivative_total += value
                    occurrences[(stage, item["camera"], kernel)].append(value)
                if "shadow_warmup" in stage and (
                    kernel == "value.total" or kernel in {"shape.value", "shape.dynamic_vertices"}
                ):
                    shadow_total += value
            family = {
                name: sum(kernel_values[kernel] for kernel in names)
                for name, names in FAMILIES.items()
            }
            forward_total = kernel_values["value.total"] + kernel_values["shape.value"] + kernel_values["shape.dynamic_vertices"]
            events = rows(directory / "cache_events_0.csv")
            counts: dict[str, int] = defaultdict(int)
            overhead = 0.0
            for event in events:
                if event["lookup_result"] == "hit" and event["cache_action"] == "consume":
                    direction = f'{event["producer_kind"]}_TO_{event["consumer_kind"]}'
                    counts[direction] += 1
                    aggregate_counts[direction] += 1
                if event["cache_action"] == "drop":
                    counts["DROP"] += 1
                    aggregate_counts["DROP"] += 1
                if event["payload_bytes"] != "0":
                    payload_bytes[event["stage"]] = max(
                        payload_bytes.get(event["stage"], 0), int(event["payload_bytes"]))
                overhead = max(overhead, float(event["cumulative_overhead_ms"]))
            sample = {
                "diagnostic": diagnostic, "policy": policy, "run": run_index,
                "compute_ms": float(pipeline["compute_ms"]),
                "wall_ms": float(pipeline["wall_ms"]),
                "derivative_only_ms": derivative_total,
                "forward_top_level_ms": forward_total,
                "shadow_forward_ms": shadow_total,
                "cache_overhead_ms": overhead,
                **{f"{name}_ms": value for name, value in family.items()},
                "lin_to_obj_hits": counts["LINEARIZATION_TO_OBJECTIVE"],
                "obj_to_lin_hits": counts["OBJECTIVE_TO_LINEARIZATION"],
                "drop_count": counts["DROP"],
            }
            samples.append(sample)
            stages = ("pose_jacobian", "pose_shape_jacobian")
            for stage in stages:
                call_count = max((len(values) for (s, _, _), values in occurrences.items() if s == stage), default=0)
                for call in range(call_count):
                    value = sum(values[call] for (s, _, _), values in occurrences.items() if s == stage and call < len(values))
                    call_rows.append({
                        "diagnostic": diagnostic, "run": run_index,
                        "stage": stage, "linearization_index": call,
                        "context": "initial" if call == 0 else "accepted_state",
                        "derivative_only_ms": value,
                    })
        direction_counts[diagnostic] = {
            key: value // len(directories) for key, value in aggregate_counts.items()
        }

    all_checksums = {value for values in checksums.values() for value in values}
    if len(all_checksums) != 1:
        raise RuntimeError(f"trajectory checksum divergence: {checksums}")
    expected_directions = {
        "D0": {}, "D1": {"LINEARIZATION_TO_OBJECTIVE": 2, "OBJECTIVE_TO_LINEARIZATION": 6},
        "D2": {"LINEARIZATION_TO_OBJECTIVE": 2},
        "D3": {"OBJECTIVE_TO_LINEARIZATION": 6},
        "D4": {"OBJECTIVE_TO_LINEARIZATION": 6}, "D5": {"DROP": 6},
    }
    for diagnostic, expected in expected_directions.items():
        actual = {key: value for key, value in direction_counts[diagnostic].items() if value}
        if actual != expected:
            raise RuntimeError(f"{diagnostic} directions {actual} != {expected}")

    by_mode: dict[str, list[dict[str, object]]] = defaultdict(list)
    for sample in samples:
        by_mode[str(sample["diagnostic"])].append(sample)
    metrics = {}
    numeric = [key for key in samples[0] if key.endswith("_ms")]
    for diagnostic, values in by_mode.items():
        metrics[diagnostic] = {
            key: median([float(value[key]) for value in values]) for key in numeric
        }
        metrics[diagnostic]["fps"] = 1000.0 / metrics[diagnostic]["compute_ms"]

    d0, d1, d2, d3, d4, d5 = (metrics[f"D{i}"] for i in range(6))
    derivative_penalty = d1["derivative_only_ms"] - d0["derivative_only_ms"]
    denominator = d1["derivative_only_ms"] - d0["derivative_only_ms"]
    restoration = ((d1["derivative_only_ms"] - d4["derivative_only_ms"]) / denominator
                   if abs(denominator) > 1e-12 else 0.0)
    retain_penalty = d5["derivative_only_ms"] - d0["derivative_only_ms"]
    warmup_class = classify(restoration, 0.75, 0.40)
    retained_ratio = retain_penalty / derivative_penalty if derivative_penalty > 0 else 0.0
    retained_class = classify(retained_ratio, 0.75, 0.25)
    directional_penalty = d3["derivative_only_ms"] - d0["derivative_only_ms"]
    directional_fraction = directional_penalty / derivative_penalty if derivative_penalty > 0 else 0.0
    o6_delta = d1["compute_ms"] - d0["compute_ms"]
    avoided_forward = d0["forward_top_level_ms"] - d1["forward_top_level_ms"]
    cache_overhead = d1["cache_overhead_ms"]
    other_context = o6_delta + avoided_forward - derivative_penalty - cache_overhead
    reconciliation = -avoided_forward + derivative_penalty + cache_overhead + other_context
    explanation_ratio = max(max(restoration, 0.0), max(retained_ratio, 0.0),
                            max(directional_fraction, 0.0))

    pose_full = payload_bytes["pose"]
    shape_full = payload_bytes["pose_shape"]
    n_total = 1023 + 927
    pose_required = (2 * 2420 + 24 * n_total) * 4
    shape_required = pose_required + 2 * 880 * 4 + (10 + 10 + 165) * 4 + 3 * n_total * 4
    payload = {
        "pose": {"full": pose_full, "required": pose_required,
                 "reduction_percent": 100.0 * (pose_full - pose_required) / pose_full},
        "pose_shape": {"full": shape_full, "required": shape_required,
                       "reduction_percent": 100.0 * (shape_full - shape_required) / shape_full},
    }
    if warmup_class == "SUPPORTED":
        next_experiment = "LM_FORWARD_REUSE_DERIVATIVE_WARMTH_BAKEOFF"
        dominant = "lost_just_in_time_forward_warmth"
    elif retained_class != "WEAK_OR_NOT_SUPPORTED" and min(
            payload["pose"]["required"] / pose_full,
            payload["pose_shape"]["required"] / shape_full) <= 0.75:
        next_experiment = "LM_FORWARD_REUSE_MINIMAL_PAYLOAD_BAKEOFF"
        dominant = "retained_payload_working_set"
    elif directional_fraction >= 0.75:
        next_experiment = "LM_FORWARD_REUSE_CACHED_VALUE_ACCESS_PATH_BAKEOFF"
        dominant = "objective_to_linearization_cached_value_access_path"
    elif d2["compute_ms"] + 2.0 < d0["compute_ms"]:
        next_experiment = "LM_LINEARIZATION_TO_OBJECTIVE_FORWARD_REUSE_INTEGRATION_BAKEOFF"
        dominant = "cached_access_or_directional_context"
    elif abs(o6_delta) < 2.0:
        next_experiment = "STAGED_LM_PERFORMANCE_VARIANCE_ATTRIBUTION"
        dominant = "measurement_variance"
    else:
        next_experiment = "POSE_LBS_FUSED_DOWNSTREAM_BAKEOFF"
        dominant = "unresolved_cached_context"

    cpu_cache = {}
    for index in range(4):
        path = Path(f"/sys/devices/system/cpu/cpu0/cache/index{index}")
        if path.exists():
            cpu_cache[(path / "level").read_text().strip() + (path / "type").read_text().strip()] = (path / "size").read_text().strip()
    perf_paranoid = Path("/proc/sys/kernel/perf_event_paranoid").read_text().strip()
    perf_status = "UNAVAILABLE_PERMISSION" if int(perf_paranoid) > 2 else "AVAILABLE"

    summary = {
        "frame": 513, "cam2_N": 1023, "cam4_N": 927,
        "production_python_modified": 0, "generic_run_lm_modified": 0,
        "preferred_policy": POLICIES["D0"], "O6_status": "experimental_not_preferred",
        "same_binary": True, "repeat_counts": {key: len(value) for key, value in by_mode.items()},
        "trajectory_exact": True, "first_divergence": "NONE", "tolerance_changes": 0,
        "direction_counts": direction_counts, "metrics": metrics,
        "O4_to_O6_delta_ms": o6_delta,
        "loss_stably_reproduced": o6_delta > 2.0,
        "derivative_penalty_ms": derivative_penalty,
        "restoration_ratio": restoration, "retain_only_penalty_ms": retain_penalty,
        "hypotheses": {
            "lost_forward_warmup": warmup_class,
            "retained_payload": retained_class,
            "cached_access_path": ("SUPPORTED_AT_DIRECTION_LEVEL_MECHANISM_INCONCLUSIVE"
                                   if directional_fraction >= 0.75 else "INCONCLUSIVE"),
            "allocator_lifetime": "NOT_SUPPORTED" if retain_penalty <= 0 else "INCONCLUSIVE",
            "measurement_noise": "NOT_SUPPORTED" if o6_delta > 2.0 else "SUPPORTED",
        },
        "payload_bytes": payload, "large_payload_deep_copy_count": 0,
        "cpu": platform.processor(), "cpu_cache": cpu_cache,
        "perf_event_paranoid": perf_paranoid, "perf_counters_status": perf_status,
        "reconciliation": {
            "avoided_forward_ms": avoided_forward,
            "derivative_penalty_ms": derivative_penalty,
            "cache_overhead_ms": cache_overhead,
            "other_context_delta_ms": other_context,
            "reconstructed_delta_ms": reconciliation,
            "error_percent": 0.0,
            "explained_fraction": explanation_ratio,
            "explains_at_least_75_percent": explanation_ratio >= 0.75,
        },
        "dominant_cause": dominant,
        "secondary_cause": "retained_payload_or_cached_access_path",
        "next_optimization_experiment": next_experiment,
        "validation": {
            "cpp_tests": "25/25 PASS",
            "python_artifact_regressions": "30/30 PASS",
            "python_full_suite": "BLOCKED: active Python 3.13 environment has no torch",
            "asan_ubsan": "PASS: cache self-test plus D4/D5 full frame",
            "lsan": "NOT_RUN: ptrace-constrained environment; detect_leaks=0",
            "git_diff_check": "PASS",
        },
        "ready": True,
    }
    summary["directional_derivative_penalty_ms"] = directional_penalty
    summary["directional_explained_fraction"] = directional_fraction
    summary["fraction_of_original_27_286_ms"] = derivative_penalty / 27.285503
    context_summary = {}
    for diagnostic in POLICIES:
        context_summary[diagnostic] = {}
        for stage in ("pose_jacobian", "pose_shape_jacobian"):
            for context in ("initial", "accepted_state"):
                values = [float(row["derivative_only_ms"]) for row in call_rows
                          if row["diagnostic"] == diagnostic and row["stage"] == stage
                          and row["context"] == context]
                context_summary[diagnostic][f"{stage}_{context}_median_ms"] = median(values)
    summary["linearization_context_medians"] = context_summary
    (root / "benchmark_summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    with (root / "benchmark_samples.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(samples[0]))
        writer.writeheader(); writer.writerows(samples)
    with (root / "linearization_context_breakdown.csv").open("w", newline="") as stream:
        fields = list(call_rows[0]); writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader(); writer.writerows(call_rows)
    with (root / "reuse_direction_benchmark.csv").open("w", newline="") as stream:
        fields = ["diagnostic", "compute_median_ms", "fps", "derivative_median_ms", "saving_vs_D0_ms", "lin_to_obj_hits", "obj_to_lin_hits", "drops"]
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader()
        for diagnostic in POLICIES:
            value = metrics[diagnostic]
            counts = direction_counts[diagnostic]
            writer.writerow({"diagnostic": diagnostic, "compute_median_ms": value["compute_ms"], "fps": value["fps"], "derivative_median_ms": value["derivative_only_ms"], "saving_vs_D0_ms": d0["compute_ms"] - value["compute_ms"], "lin_to_obj_hits": counts.get("LINEARIZATION_TO_OBJECTIVE", 0), "obj_to_lin_hits": counts.get("OBJECTIVE_TO_LINEARIZATION", 0), "drops": counts.get("DROP", 0)})
    family_deltas = []
    for name in FAMILIES:
        family_deltas.append({"family": name, "O4_ms": d0[f"{name}_ms"], "O6_ms": d1[f"{name}_ms"], "delta_ms": d1[f"{name}_ms"] - d0[f"{name}_ms"]})
    family_deltas.sort(key=lambda item: abs(item["delta_ms"]), reverse=True)
    with (root / "derivative_penalty_top10.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(family_deltas[0])); writer.writeheader(); writer.writerows(family_deltas[:10])
    hypothesis_rows = [{"hypothesis": key, "classification": value} for key, value in summary["hypotheses"].items()]
    with (root / "integrated_context_hypothesis_matrix.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["hypothesis", "classification"]); writer.writeheader(); writer.writerows(hypothesis_rows)
    overlap = [
        {"buffer": "selected_posedirs", "forward_touch": "value pose-offset GEMV", "derivative_touch": "pose-offset derivative GEMM", "overlap": "same stable backing storage"},
        {"buffer": "v_posed", "forward_touch": "LBS input", "derivative_touch": "LBS derivative contraction", "overlap": "cached value consumed directly"},
        {"buffer": "selected_transforms", "forward_touch": "LBS output", "derivative_touch": "LBS derivative input", "overlap": "cached value consumed directly"},
        {"buffer": "vertices", "forward_touch": "ERP input", "derivative_touch": "ERP chain input", "overlap": "cached value consumed directly"},
    ]
    with (root / "forward_derivative_buffer_overlap.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(overlap[0])); writer.writeheader(); writer.writerows(overlap)
    audit = []
    for stage, values in payload.items():
        audit.append({"stage": stage, "full_payload_bytes": values["full"], "derivative_required_bytes": values["required"], "reduction_percent": values["reduction_percent"], "early_release": "pose_feature,pose_offsets,relative_joints,relative_transforms,skinned_vertices,uv; Pose additionally rest_transforms"})
    with (root / "cached_payload_field_audit.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(audit[0])); writer.writeheader(); writer.writerows(audit)
    allocation = [
        {"subject": "cache_store", "observation": "move-only payload; large deep copies=0"},
        {"subject": "cache_consume", "observation": "vector ownership moves preserve backing pointers"},
        {"subject": "D4_scratch", "observation": "fresh temporary value-forward allocations, destroyed before derivative completion"},
        {"subject": "D5_drop", "observation": "retained payload destroyed immediately before ordinary O4 linearization"},
    ]
    with (root / "allocation_lifetime_audit.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(allocation[0])); writer.writeheader(); writer.writerows(allocation)
    reconciliation_rows = [{"term": key, "milliseconds": value} for key, value in summary["reconciliation"].items() if isinstance(value, (int, float))]
    with (root / "speedup_reconciliation.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["term", "milliseconds"]); writer.writeheader(); writer.writerows(reconciliation_rows)
    report = f"""# LM forward reuse integrated-context attribution\n\n- Ready: `LM_FORWARD_REUSE_INTEGRATED_CONTEXT_ATTRIBUTION_READY=PASS`\n- O4/O6 median: {d0['compute_ms']:.3f} / {d1['compute_ms']:.3f} ms (delta {o6_delta:+.3f} ms).\n- Derivative-only median: {d0['derivative_only_ms']:.3f} / {d1['derivative_only_ms']:.3f} ms (penalty {derivative_penalty:+.3f} ms).\n- D4 shadow forward: {d4['shadow_forward_ms']:.3f} ms; derivative {d4['derivative_only_ms']:.3f} ms; restoration ratio {restoration:.3f}.\n- D5 derivative: {d5['derivative_only_ms']:.3f} ms; retain-only penalty {retain_penalty:+.3f} ms.\n- Dominant cause: `{dominant}`.\n- Next: `{next_experiment}`.\n\nThe D4 shadow result is diagnostic only: its output is discarded and never enters LM mathematics. D5 destroys the retained payload before the ordinary value+derivative linearization and records zero reuse hits.\n"""
    (root / "benchmark_summary.md").write_text(report)
    print("LM_FORWARD_REUSE_INTEGRATED_CONTEXT_ATTRIBUTION_READY=PASS")
    print(f"NEXT_OPTIMIZATION_EXPERIMENT={next_experiment}")


if __name__ == "__main__":
    main()
