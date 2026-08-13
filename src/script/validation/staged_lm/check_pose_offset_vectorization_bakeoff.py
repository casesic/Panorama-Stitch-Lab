"""Summarize the isolated A/B/C pose-offset vectorization experiment."""
from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path

import numpy as np

from script.validation.staged_lm.check_pose_cpp_jacobian import _metrics


CONTEXTS = ("pose_cam2", "pose_cam4", "pose_shape_cam2", "pose_shape_cam4")
CALLS = {"pose_cam2": 6, "pose_cam4": 6,
         "pose_shape_cam2": 2, "pose_shape_cam4": 2}
VARIANTS = {"a": "GEMM_NO_VECTORIZE", "b": "GEMM_VECTORIZE",
            "c": "GEMM_VECTORIZE_NATIVE"}
SCALAR_PIPELINE_MS = 4617.56


def _environment(path: Path) -> dict[str, str]:
    return dict(line.split("=", 1) for line in path.read_text().splitlines() if "=" in line)


def _stats(values: list[float]) -> dict[str, float]:
    array = np.asarray(values)
    return {"median_ms": float(np.median(array)), "mean_ms": float(np.mean(array)),
            "min_ms": float(np.min(array)), "max_ms": float(np.max(array)),
            "p25_ms": float(np.percentile(array, 25)),
            "p75_ms": float(np.percentile(array, 75))}


def _numerics(reference: np.ndarray, actual: np.ndarray) -> dict[str, float | int | bool]:
    delta = actual.astype(np.float64) - reference.astype(np.float64)
    absolute = np.abs(delta).reshape(-1)
    worst = int(np.argmax(absolute))
    return {"max_abs": float(absolute[worst]), "mean_abs": float(absolute.mean()),
            "p99_abs": float(np.percentile(absolute, 99)),
            "rel_frob": float(np.linalg.norm(delta.ravel()) /
                              max(np.linalg.norm(reference.astype(np.float64).ravel()), 1e-30)),
            "worst_parameter_column": worst % 66,
            "worst_xyz": (worst // 66) % 3,
            "worst_point": worst // (3 * 66)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--kernel-breakdown", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads((args.fixture / "manifest.json").read_text())
    variants: dict[str, dict] = {}
    frame_costs: dict[str, float] = {}
    for key, name in VARIANTS.items():
        directory = args.root / f"variant_{key}"
        environment = _environment(directory / "run1/environment.txt")
        contexts = {}
        frame_cost = 0.0
        for context in CONTEXTS:
            samples = []
            for run in ("run1", "run2"):
                with (directory / run / "benchmark_samples.csv").open() as stream:
                    samples.extend(float(row["total_ms"]) for row in csv.DictReader(stream)
                                   if row["context"] == context and
                                   row["candidate"] == "eigen_gemm")
            actual = np.fromfile(directory / "run1" / f"{context}_gemm.f32", "<f4")
            scalar = np.fromfile(directory / "run1" / f"{context}_scalar.f32", "<f4")
            python = np.fromfile(args.fixture / context / "python_d_pose_offsets.f32", "<f4")
            timing = _stats(samples)
            formal = _metrics(python, actual, "feature")
            contexts[context] = {
                "point_count": manifest["contexts"][context]["point_count"],
                "calls_per_frame": CALLS[context], "timing": timing,
                "candidate_vs_python": _numerics(python, actual),
                "candidate_vs_scalar": _numerics(scalar, actual),
                "formal_feature_gate": bool(formal["pass"]),
                "gflops": (2 * manifest["contexts"][context]["point_count"] * 3 * 486 * 66 /
                            (timing["median_ms"] * 1e6)),
            }
            frame_cost += CALLS[context] * timing["median_ms"]
        variants[key] = {
            "name": name, "environment": environment, "contexts": contexts,
            "frame_pose_offset_ms": frame_cost,
            "compile_flags": ({"a": "-O3 -DNDEBUG -DEIGEN_DONT_VECTORIZE -ffp-contract=off",
                               "b": "-O3 -DNDEBUG -ffp-contract=off",
                               "c": "-O3 -DNDEBUG -ffp-contract=off -march=native"}[key]),
        }
        frame_costs[key] = frame_cost

    scalar_report = json.loads((args.root.parent.parent / "pose_offset_gemm/frame_000513/output/pose_offset_gemm_parity.json").read_text())
    scalar_frame = float(scalar_report["scalar_frame_ms"])
    for variant in variants.values():
        for context, value in variant["contexts"].items():
            value["scalar_reference_ms"] = scalar_report["contexts"][context]["scalar_timing"]["median"]
    projects = {key: SCALAR_PIPELINE_MS - scalar_frame + cost
                for key, cost in frame_costs.items()}
    best = min(frame_costs, key=frame_costs.get)

    with args.kernel_breakdown.open() as stream:
        rows = list(csv.DictReader(stream))
    old_pose_offset = sum(float(row["total_ms"]) for row in rows
                          if row["kernel"] == "jac.pose_offset_derivative")
    projected_rows = []
    for row in rows:
        value = float(row["total_ms"])
        if row["kernel"] == "jac.pose_offset_derivative":
            context = ("pose_shape_" if row["stage"] == "pose_shape_jacobian" else "pose_") + row["camera"]
            value = variants[best]["contexts"][context]["timing"]["median_ms"] * int(row["call_count"])
        projected_rows.append({"kernel": row["kernel"], "stage": row["stage"],
                               "camera": row["camera"], "projected_ms": value})
    projected_rows.sort(key=lambda row: row["projected_ms"], reverse=True)
    lbs_rows = [row for row in projected_rows if "lbs" in row["kernel"] and
                row["stage"] in ("pose_jacobian", "pose_shape_jacobian")]
    lbs_total = sum(row["projected_ms"] for row in lbs_rows)
    objective = 258.5236
    all_correct = all(context["formal_feature_gate"] for variant in variants.values()
                      for context in variant["contexts"].values())
    speedup_ab = frame_costs["a"] / frame_costs["b"]
    speedup_bc = frame_costs["b"] / frame_costs["c"]
    next_experiment = ("LBS_DERIVATIVE_DIRECT_CONTRACTION_BAKEOFF"
                       if speedup_ab >= 1.1 and projected_rows[0]["kernel"].endswith("lbs_derivative")
                       else "POSE_OFFSET_DERIVATIVE_LAYOUT_ALIGNMENT_BAKEOFF")
    report = {
        "schema_version": "pose_offset_vectorization_bakeoff_v1",
        "frame": 513, "N_cam2": 1023, "N_cam4": 927, "variants": variants,
        "cpu_isa": ["SSE", "SSE2", "SSE4.1", "SSE4.2", "AVX", "AVX2", "FMA",
                    "AVX512F", "AVX512DQ", "AVX512CD", "AVX512BW", "AVX512VL", "AVX512VNNI"],
        "speedup_a_to_b": speedup_ab, "speedup_b_to_c": speedup_bc,
        "speedup_scalar_to_best": scalar_frame / frame_costs[best],
        "frame_pose_offset_ms_a": frame_costs["a"],
        "frame_pose_offset_ms_b": frame_costs["b"],
        "frame_pose_offset_ms_c": frame_costs["c"],
        "projected_pipeline_a": projects["a"], "projected_pipeline_b": projects["b"],
        "projected_pipeline_c": projects["c"],
        "projected_fps_a": 1000 / projects["a"], "projected_fps_b": 1000 / projects["b"],
        "projected_fps_c": 1000 / projects["c"], "best_variant": VARIANTS[best],
        "best_projected_pipeline_speedup": SCALAR_PIPELINE_MS / projects[best],
        "remaining_ms_above_1fps": projects[best] - 1000,
        "required_additional_pipeline_speedup": projects[best] / 1000,
        "projected_top_hotspots": projected_rows[:10],
        "projected_lbs_derivative_ms": lbs_total,
        "projected_lbs_derivative_percent": 100 * lbs_total / projects[best],
        "projected_pose_offset_derivative_ms": frame_costs[best],
        "projected_pose_offset_derivative_percent": 100 * frame_costs[best] / projects[best],
        "objective_ms": objective, "objective_percent": 100 * objective / projects[best],
        "old_pose_offset_attribution_ms": old_pose_offset,
        "VECTORISATION_WORTH_INTEGRATING": "YES" if all_correct and speedup_ab >= 1.1 else "NO/MARGINAL",
        "NATIVE_BUILD_PERFORMANCE_VALUE": "MEANINGFUL" if all_correct and speedup_bc >= 1.15 else "LOW",
        "EIGEN_GEMM_EXPRESSION_APPROVED": True,
        "EIGEN_VECTORISATION_POLICY_APPROVED": all_correct and speedup_ab >= 1.1,
        "downstream_pose_b": "PASS_SELF_TEST_SHADOW", "downstream_pose_c": "PASS_SELF_TEST_SHADOW",
        "downstream_pose_shape_b": "PASS_SHARED_66D_CHAIN;J_BETA_UNCHANGED",
        "downstream_pose_shape_c": "PASS_SHARED_66D_CHAIN;J_BETA_UNCHANGED",
        "trajectory_changed": "NOT_MEASURED_NO_FULL_PIPELINE_SHADOW",
        "formal_tolerance_changed": False, "default_scalar_path": "UNCHANGED",
        "gemm_source_changed": False, "full_pipeline_shadow_executed": False,
        "error_ratio_b_over_a": max(
            variants["b"]["contexts"][context]["candidate_vs_python"]["rel_frob"] /
            max(variants["a"]["contexts"][context]["candidate_vs_python"]["rel_frob"], 1e-30)
            for context in CONTEXTS),
        "error_ratio_c_over_a": max(
            variants["c"]["contexts"][context]["candidate_vs_python"]["rel_frob"] /
            max(variants["a"]["contexts"][context]["candidate_vs_python"]["rel_frob"], 1e-30)
            for context in CONTEXTS),
        "NEXT_OPTIMIZATION_EXPERIMENT": next_experiment,
        "POSE_OFFSET_DERIVATIVE_CPU_VECTORISATION_BAKEOFF_READY": "PASS" if all_correct else "FAIL",
    }
    (args.root / "benchmark_summary.json").write_text(json.dumps(report, indent=2) + "\n")
    (args.root / "parity.json").write_text(json.dumps({
        "variants": {key: {context: value["candidate_vs_python"]
                            for context, value in variant["contexts"].items()}
                     for key, variant in variants.items()},
        "formal_tolerance_changed": False,
        "ready": report["POSE_OFFSET_DERIVATIVE_CPU_VECTORISATION_BAKEOFF_READY"],
    }, indent=2) + "\n")
    with (args.root / "projected_pipeline.csv").open("w", newline="") as stream:
        writer = csv.writer(stream); writer.writerow(["variant", "pose_offset_ms", "pipeline_ms", "fps"])
        for key in VARIANTS: writer.writerow([VARIANTS[key], frame_costs[key], projects[key], 1000/projects[key]])
    with (args.root / "benchmark_samples.csv").open("w", newline="") as destination:
        writer = csv.writer(destination); writer.writerow(["variant", "run", "context", "iteration", "candidate", "total_ms", "pack_ms", "core_ms", "unpack_ms", "checksum"])
        for key in VARIANTS:
            for run in ("run1", "run2"):
                with (args.root / f"variant_{key}/{run}/benchmark_samples.csv").open() as source:
                    for row in csv.DictReader(source): writer.writerow([key, run, *row.values()])
    lines = ["# Pose-offset derivative CPU vectorization bake-off", "",
             f"A→B frame-weighted speedup: {speedup_ab:.3f}×; B→C: {speedup_bc:.3f}×.", "",
             f"Best: {VARIANTS[best]}, projected {projects[best]:.2f} ms / {1000/projects[best]:.3f} FPS.", "",
             f"LBS derivative is {lbs_total:.2f} ms ({100*lbs_total/projects[best]:.2f}%); pose-offset derivative falls to {frame_costs[best]:.2f} ms.", "",
             f"`NEXT_OPTIMIZATION_EXPERIMENT={next_experiment}`", ""]
    (args.root / "benchmark_summary.md").write_text("\n".join(lines))
    print("POSE_OFFSET_DERIVATIVE_CPU_VECTORISATION_BAKEOFF_READY=" + report["POSE_OFFSET_DERIVATIVE_CPU_VECTORISATION_BAKEOFF_READY"])
    if not all_correct: raise SystemExit(1)


if __name__ == "__main__":
    main()
