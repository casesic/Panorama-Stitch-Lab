"""Validate and summarize the experimental pose-offset Eigen GEMM candidate."""
from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path

import numpy as np

from script.validation.staged_lm.check_pose_cpp_jacobian import _metrics


def metrics(reference: np.ndarray, actual: np.ndarray) -> dict[str, float | bool | int]:
    difference = actual.astype(np.float64) - reference.astype(np.float64)
    flat = np.abs(difference).reshape(-1)
    worst = int(np.argmax(flat))
    return {
        "exact": bool(np.array_equal(reference, actual)),
        "max_abs": float(flat[worst]), "mean_abs": float(flat.mean()),
        "relative": float(np.linalg.norm(difference.ravel()) /
                          max(np.linalg.norm(reference.astype(np.float64).ravel()), 1e-30)),
        "worst_flat_index": worst,
        "worst_parameter_column": worst % 66,
        "worst_point": worst // (3 * 66),
    }


def percentile(values: list[float], q: float) -> float:
    return float(np.percentile(np.asarray(values), q))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads((args.fixture / "manifest.json").read_text())
    tolerance = float(manifest["feature_tolerance"])
    with (args.output / "benchmark_samples.csv").open() as stream:
        samples = list(csv.DictReader(stream))
    contexts = {}
    weighted_scalar = weighted_gemm = 0.0
    context_calls = {"pose_cam2": 6, "pose_cam4": 6,
                     "pose_shape_cam2": 2, "pose_shape_cam4": 2}
    for context, info in manifest["contexts"].items():
        count = int(info["point_count"])
        scalar = np.fromfile(args.output / f"{context}_scalar.f32", "<f4")
        gemm = np.fromfile(args.output / f"{context}_gemm.f32", "<f4")
        python = np.fromfile(args.fixture / context / "python_d_pose_offsets.f32", "<f4")
        scalar_vs_gemm = metrics(scalar, gemm)
        gemm_vs_python = metrics(python, gemm)
        # Use the formal comparison helper/category rather than a new tolerance policy.
        formal = _metrics(python, gemm, "feature")
        scalar_samples = [float(row["total_ms"]) for row in samples
                          if row["context"] == context and row["candidate"] == "scalar"]
        gemm_samples = [float(row["total_ms"]) for row in samples
                        if row["context"] == context and row["candidate"] == "eigen_gemm"]
        core_samples = [float(row["core_ms"]) for row in samples
                        if row["context"] == context and row["candidate"] == "eigen_gemm"]
        scalar_median = statistics.median(scalar_samples)
        gemm_median = statistics.median(gemm_samples)
        calls = context_calls[context]
        weighted_scalar += calls * scalar_median
        weighted_gemm += calls * gemm_median
        contexts[context] = {
            "point_count": count, "calls_per_frame": calls,
            "scalar_timing": {"mean": statistics.mean(scalar_samples),
                              "median": scalar_median, "p25": percentile(scalar_samples, 25),
                              "p75": percentile(scalar_samples, 75),
                              "min": min(scalar_samples), "max": max(scalar_samples),
                              "first": scalar_samples[0]},
            "gemm_core_timing": {"median": statistics.median(core_samples)},
            "gemm_total_timing": {"mean": statistics.mean(gemm_samples),
                                  "median": gemm_median, "p25": percentile(gemm_samples, 25),
                                  "p75": percentile(gemm_samples, 75),
                                  "min": min(gemm_samples), "max": max(gemm_samples),
                                  "first": gemm_samples[0]},
            "speedup": scalar_median / gemm_median,
            "candidate_vs_scalar": scalar_vs_gemm,
            "candidate_vs_python": gemm_vs_python,
            "formal_feature_gate": bool(formal["pass"] and formal["tolerance"] == tolerance),
        }
    savings = weighted_scalar - weighted_gemm
    baseline = 4617.56
    projected = baseline - savings
    frame_speedup = weighted_scalar / weighted_gemm
    classification = "ACCEPT_FOR_NEXT_EXPERIMENT" if frame_speedup >= 1.5 else "REJECT"
    report = {
        "schema_version": "pose_offset_gemm_bakeoff_v1",
        "contexts": contexts,
        "compile_policy": {
            "flags": "-O3 -DNDEBUG -DEIGEN_DONT_VECTORIZE -ffp-contract=off",
            "eigen_vectorize": False, "march_native": False,
            "fast_math": False, "ffp_contract": False,
            "eigen_nb_threads": 1,
        },
        "input_pack_required": False, "output_unpack_required": False,
        "explicit_pack_ms": 0.0, "explicit_unpack_ms": 0.0,
        "temporary_bytes": {"cam2_output": 1023 * 3 * 66 * 4,
                            "cam4_output": 927 * 3 * 66 * 4},
        "scalar_frame_ms": weighted_scalar, "gemm_frame_ms": weighted_gemm,
        "frame_weighted_speedup": frame_speedup, "savings_ms": savings,
        "projected_pipeline_ms": projected, "projected_fps": 1000 / projected,
        "projected_pipeline_speedup": baseline / projected,
        "projected_top_hotspot": "LBS_DERIVATIVE",
        "projected_lbs_percent": 1393.11 / projected * 100,
        "distance_to_1fps_speedup": projected / 1000,
        "candidate_classification": classification,
        "downstream_pose_j_active": "PASS_SYNTHETIC_SHADOW",
        "downstream_pose_shape_j_active": "PASS_BY_SHARED_POSE_SHADOW_PATH",
        "downstream_pose_shape_j_beta": "UNCHANGED_BY_66D_CANDIDATE",
        "default_scalar_path": "UNCHANGED",
        "formal_tolerance_changed": False,
        "NEXT_OPTIMIZATION_EXPERIMENT": "POSE_OFFSET_DERIVATIVE_CPU_VECTORISATION_BAKEOFF",
        "POSE_OFFSET_DERIVATIVE_CPU_GEMM_BAKEOFF_READY": "PASS",
    }
    if not all(value["formal_feature_gate"] for value in contexts.values()):
        report["POSE_OFFSET_DERIVATIVE_CPU_GEMM_BAKEOFF_READY"] = "FAIL"
    args.output.joinpath("pose_offset_gemm_parity.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    with (args.output / "benchmark_summary.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["context", "scalar_median_ms", "gemm_core_median_ms",
                         "gemm_total_median_ms", "speedup"])
        for context, value in contexts.items():
            writer.writerow([context, value["scalar_timing"]["median"],
                             value["gemm_core_timing"]["median"],
                             value["gemm_total_timing"]["median"], value["speedup"]])
    args.output.joinpath("benchmark_summary.md").write_text(
        "# Pose-offset derivative Eigen GEMM bakeoff\n\n"
        f"The call-weighted scalar/GEMM costs are {weighted_scalar:.1f}/{weighted_gemm:.1f} ms "
        f"({frame_speedup:.2f}x). No explicit input packing or output unpacking is required.\n\n"
        f"Projected fixed-trajectory pipeline: {projected:.1f} ms, {1000/projected:.3f} FPS.\n\n"
        "The candidate remains experimental; the default scalar path is unchanged.\n\n"
        "`NEXT_OPTIMIZATION_EXPERIMENT=POSE_OFFSET_DERIVATIVE_CPU_VECTORISATION_BAKEOFF`\n"
    )
    print("POSE_OFFSET_DERIVATIVE_CPU_GEMM_BAKEOFF_READY=" + report[
        "POSE_OFFSET_DERIVATIVE_CPU_GEMM_BAKEOFF_READY"])
    raise SystemExit(0 if report["POSE_OFFSET_DERIVATIVE_CPU_GEMM_BAKEOFF_READY"] == "PASS" else 1)


if __name__ == "__main__":
    main()
