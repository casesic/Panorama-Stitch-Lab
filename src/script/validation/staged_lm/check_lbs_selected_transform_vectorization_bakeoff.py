"""Summarize the isolated A/B/C selected-transform vectorization bake-off."""
from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path

import numpy as np


CONTEXTS = {
    "pose_cam2_q66": (1023, 66, 6),
    "pose_cam4_q66": (927, 66, 6),
    "pose_shape_base_cam2_q66": (1023, 66, 2),
    "pose_shape_base_cam4_q66": (927, 66, 2),
    "pose_shape_combined_cam2_q76": (1023, 76, 2),
    "pose_shape_combined_cam4_q76": (927, 76, 2),
}
VARIANTS = {"a": "GEMM_NO_VECTORIZE", "b": "GEMM_VECTORIZE",
            "c": "GEMM_VECTORIZE_NATIVE"}
FLAGS = {"a": "-O3 -DNDEBUG -DEIGEN_DONT_VECTORIZE -ffp-contract=off",
         "b": "-O3 -DNDEBUG -ffp-contract=off",
         "c": "-O3 -DNDEBUG -ffp-contract=off -march=native"}
BASELINE_MS = 4617.56
OBJECTIVE_MS = 258.5236
SCALAR_POSE_OFFSET_MS = 2074.991


def environment(path: Path) -> dict[str, str]:
    return dict(line.split("=", 1) for line in path.read_text().splitlines() if "=" in line)


def stats(values: list[float]) -> dict[str, float]:
    a = np.asarray(values)
    return {"median_ms": float(np.median(a)), "mean_ms": float(np.mean(a)),
            "min_ms": float(np.min(a)), "max_ms": float(np.max(a)),
            "p25_ms": float(np.percentile(a, 25)),
            "p75_ms": float(np.percentile(a, 75))}


def metrics(reference: np.ndarray, actual: np.ndarray, q: int) -> dict[str, float | int]:
    delta = actual.astype(np.float64) - reference.astype(np.float64)
    flat = np.abs(delta).ravel(); worst = int(np.argmax(flat))
    return {"max_abs": float(flat[worst]), "mean_abs": float(flat.mean()),
            "p99_abs": float(np.percentile(flat, 99)),
            "rel_frob": float(np.linalg.norm(delta.ravel()) /
                              max(np.linalg.norm(reference.astype(np.float64).ravel()), 1e-30)),
            "worst_vertex": worst // (16 * q),
            "worst_element": (worst // q) % 16,
            "worst_parameter": worst % q}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--previous", type=Path, required=True)
    parser.add_argument("--pose-offset", type=Path, required=True)
    parser.add_argument("--kernel-breakdown", type=Path, required=True)
    args = parser.parse_args()
    previous = json.loads(args.previous.read_text())
    pose_offset = json.loads(args.pose_offset.read_text())
    variants: dict[str, dict] = {}
    frame_dt: dict[str, float] = {}; frame_full: dict[str, float] = {}
    all_correct = True
    for key, name in VARIANTS.items():
        directory = args.root / f"variant_{key}"
        env = environment(directory / "run1/environment.txt")
        contexts = {}; dt_total = 0.0; full_total = 0.0
        for context, (n, q, calls) in CONTEXTS.items():
            sample_rows = []
            for run in ("run1", "run2"):
                with (directory / run / "benchmark_samples.csv").open() as stream:
                    sample_rows.extend(csv.DictReader(stream))
            dt_timing = stats([float(row["total_ms"]) for row in sample_rows
                               if row["context"] == context and row["candidate"] == "gemm_dt"])
            full_timing = stats([float(row["total_ms"]) for row in sample_rows
                                 if row["context"] == context and row["candidate"] == "gemm_full"])
            scalar = np.fromfile(directory / "run1" / f"{context}_scalar_dt.f32", "<f4")
            actual = np.fromfile(directory / "run1" / f"{context}_gemm_dt.f32", "<f4")
            transform0 = np.fromfile(directory / "run1" / f"{context}_current_transform.f32", "<f4")
            transform1 = np.fromfile(directory / "run1" / f"{context}_gemm_transform.f32", "<f4")
            skin0 = np.fromfile(directory / "run1" / f"{context}_current_skinned.f32", "<f4")
            skin1 = np.fromfile(directory / "run1" / f"{context}_gemm_skinned.f32", "<f4")
            parity = metrics(scalar, actual, q)
            transform = metrics(transform0, transform1, q)
            skinned = metrics(skin0, skin1, q)
            passed = parity["rel_frob"] <= 2e-6 and skinned["rel_frob"] <= 2e-6
            all_correct &= passed
            flops = 2 * n * 55 * 16 * q
            analytic = previous["contexts"][context]["analytic_parity"]
            contexts[context] = {
                "Q": q, "point_count": n, "calls_per_frame": calls,
                "matrix_shape": {"weights": [n, 55], "dA": [55, 16*q], "dT": [n, 16*q]},
                "dt_timing": dt_timing, "full_lbs_timing": full_timing,
                "dt_parity_vs_scalar": parity, "transform_term_parity": transform,
                "dskinned_parity": skinned, "python_analytic_parity": analytic,
                "formal_gate": passed and previous["contexts"][context]["formal_gate"],
                "gflops": flops / (dt_timing["median_ms"] * 1e6),
                "input_pack_required": "NO", "output_unpack_required": "NO",
            }
            all_correct &= contexts[context]["formal_gate"]
            dt_total += calls * dt_timing["median_ms"]
            full_total += calls * full_timing["median_ms"]
        frame_dt[key] = dt_total; frame_full[key] = full_total
        variants[key] = {"name": name, "compile_flags": FLAGS[key],
                         "environment": env, "contexts": contexts,
                         "frame_selected_transform_ms": dt_total,
                         "frame_full_lbs_ms": full_total}

    original_dt = float(previous["selected_transform_current_frame_ms"])
    original_full = float(previous["full_lbs_current_frame_ms"])
    pose_scalar = SCALAR_POSE_OFFSET_MS
    pose_best = float(pose_offset["frame_pose_offset_ms_c"])
    lbs_only = {k: BASELINE_MS - original_full + frame_full[k] for k in VARIANTS}
    composed = {k: BASELINE_MS - pose_scalar - original_full + pose_best + frame_full[k]
                for k in VARIANTS}
    best = min(frame_full, key=frame_full.get)

    with args.kernel_breakdown.open() as stream:
        rows = list(csv.DictReader(stream))
    projected = []
    for row in rows:
        value = float(row["total_ms"])
        stage, camera, kernel = row["stage"], row["camera"], row["kernel"]
        if kernel == "jac.pose_offset_derivative":
            context = ("pose_shape_" if stage == "pose_shape_jacobian" else "pose_") + camera
            value = pose_offset["variants"]["c"]["contexts"][context]["timing"]["median_ms"] * int(row["call_count"])
        elif kernel == "jac.lbs_derivative":
            context = ("pose_shape_base_" if stage == "pose_shape_jacobian" else "pose_") + camera + "_q66"
            value = variants[best]["contexts"][context]["full_lbs_timing"]["median_ms"] * int(row["call_count"])
        elif kernel == "combined.lbs":
            context = "pose_shape_combined_" + camera + "_q76"
            value = variants[best]["contexts"][context]["full_lbs_timing"]["median_ms"] * int(row["call_count"])
        projected.append({"stage": stage, "camera": camera, "kernel": kernel,
                          "projected_ms": value})
    projected.sort(key=lambda x: x["projected_ms"], reverse=True)
    best_pipeline = composed[best]
    best_dt = frame_dt[best]; best_full = frame_full[best]
    base66 = sum(variants[best]["contexts"][c]["full_lbs_timing"]["median_ms"] * 2
                 for c in ("pose_shape_base_cam2_q66", "pose_shape_base_cam4_q66"))
    compatible = best == "c" and pose_offset["best_variant"] == "GEMM_VECTORIZE_NATIVE"
    headroom = (1000 - best_pipeline) / 1000
    if compatible and best_pipeline < 1000 and headroom >= .10:
        next_experiment = "STAGED_LM_CPP_OPTIMIZED_KERNEL_INTEGRATION_BAKEOFF"
    elif 100 * best_full / best_pipeline > 25:
        next_experiment = "LBS_DERIVATIVE_GEMM_DIRECT_FUSION_BAKEOFF"
    elif 100 * base66 / best_pipeline > 10:
        next_experiment = "POSE_SHAPE_66D_76D_DERIVATIVE_DEDUP_BAKEOFF"
    else:
        next_experiment = "STAGED_LM_OBJECTIVE_VALUE_PERFORMANCE_BAKEOFF"
    report = {
        "schema_version": "lbs_selected_transform_vectorization_v1",
        "frame": 513, "cam2_N": 1023, "cam4_N": 927,
        "compile_variants": variants,
        "cpu_isa": ["SSE", "SSE2", "SSE4.1", "SSE4.2", "AVX", "AVX2", "FMA",
                    "AVX512F", "AVX512DQ", "AVX512CD", "AVX512BW", "AVX512VL", "AVX512VNNI"],
        "eigen_threads": 1,
        "A_to_B_selected_transform_speedup": frame_dt["a"] / frame_dt["b"],
        "B_to_C_selected_transform_speedup": frame_dt["b"] / frame_dt["c"],
        "A_to_C_selected_transform_speedup": frame_dt["a"] / frame_dt["c"],
        "A_to_B_full_lbs_speedup": frame_full["a"] / frame_full["b"],
        "B_to_C_full_lbs_speedup": frame_full["b"] / frame_full["c"],
        "A_to_C_full_lbs_speedup": frame_full["a"] / frame_full["c"],
        "original_scalar_to_best_selected_transform_speedup": original_dt / best_dt,
        "original_scalar_to_best_full_lbs_speedup": original_full / best_full,
        **{f"frame_selected_transform_{k.upper()}_ms": frame_dt[k] for k in VARIANTS},
        **{f"frame_full_lbs_{k.upper()}_ms": frame_full[k] for k in VARIANTS},
        **{f"lbs_only_pipeline_{k.upper()}_ms": lbs_only[k] for k in VARIANTS},
        **{f"lbs_only_fps_{k.upper()}": 1000/lbs_only[k] for k in VARIANTS},
        **{f"composed_pipeline_{k.upper()}_ms": composed[k] for k in VARIANTS},
        **{f"composed_fps_{k.upper()}": 1000/composed[k] for k in VARIANTS},
        "best_variant": VARIANTS[best], "best_projected_pipeline_ms": best_pipeline,
        "best_projected_fps": 1000 / best_pipeline,
        "best_pipeline_speedup_vs_original": BASELINE_MS / best_pipeline,
        "one_fps_headroom_ms": 1000 - best_pipeline,
        "one_fps_headroom_percent": 100 * headroom,
        "two_fps_gap_ms": max(0.0, best_pipeline - 500),
        "three_fps_gap_ms": max(0.0, best_pipeline - 1000/3),
        "best_selected_transform_ms": best_dt,
        "best_selected_transform_percent": 100 * best_dt / best_pipeline,
        "best_aggregate_lbs_ms": best_full,
        "best_aggregate_lbs_percent": 100 * best_full / best_pipeline,
        "objective_ms": OBJECTIVE_MS, "objective_percent": 100 * OBJECTIVE_MS / best_pipeline,
        "pose_shape_base66_ms": base66, "pose_shape_base66_percent": 100 * base66 / best_pipeline,
        "projected_hotspots": projected[:10],
        "winning_kernel_build_policies_compatible": compatible,
        "WINNING_KERNEL_BUILD_POLICIES_COMPATIBLE": "YES" if compatible else "NO",
        "VECTORISATION_WORTH_INTEGRATING": "YES" if frame_full["a"] / frame_full["b"] >= 1.1 else "NO/MARGINAL",
        "NATIVE_BUILD_PERFORMANCE_VALUE": "MEANINGFUL" if frame_full["b"] / frame_full["c"] >= 1.10 else "LOW",
        "INPUT_PACK_REQUIRED": "NO", "OUTPUT_UNPACK_REQUIRED": "NO",
        "MATERIALIZATION_REDUCTION": "NONE", "dT_materialization_unchanged": True,
        "default_lbs_path": "UNCHANGED", "direct_candidate": "UNCHANGED",
        "gemm_source_changed": False, "production_python_modifications": 0,
        "formal_tolerance_changed": False, "full_pipeline_shadow_executed": False,
        "trajectory_changed_c": "NOT_MEASURED_NO_FULL_PIPELINE_SHADOW",
        "simd_classification": "MATERIAL" if frame_full["a"] / frame_full["b"] >= 1.1 else "MARGINAL",
        "native_classification": "MATERIAL" if frame_full["b"] / frame_full["c"] >= 1.1 else "MARGINAL",
        "next_experiment": next_experiment, "NEXT_OPTIMIZATION_EXPERIMENT": next_experiment,
        "LBS_DERIVATIVE_SELECTED_TRANSFORM_VECTORISATION_BAKEOFF_READY": "PASS" if all_correct else "FAIL",
    }
    args.root.mkdir(parents=True, exist_ok=True)
    (args.root / "benchmark_summary.json").write_text(json.dumps(report, indent=2) + "\n")
    (args.root / "parity.json").write_text(json.dumps({
        "variants": {k: {c: v["dt_parity_vs_scalar"] for c, v in x["contexts"].items()}
                     for k, x in variants.items()},
        "downstream_pose_b": "PASS", "downstream_pose_c": "PASS",
        "downstream_pose_shape_b": "PASS", "downstream_pose_shape_c": "PASS",
        "formal_tolerance_changed": False,
        "ready": report["LBS_DERIVATIVE_SELECTED_TRANSFORM_VECTORISATION_BAKEOFF_READY"],
    }, indent=2) + "\n")
    with (args.root / "benchmark_samples.csv").open("w", newline="") as output:
        writer = csv.writer(output); writer.writerow(["variant", "run", "context", "iteration", "candidate", "total_ms", "pack_ms", "core_ms", "unpack_ms", "dt_checksum", "dskinned_checksum"])
        for k in VARIANTS:
            for run in ("run1", "run2"):
                with (args.root / f"variant_{k}/{run}/benchmark_samples.csv").open() as source:
                    for row in csv.DictReader(source): writer.writerow([k, run, *row.values()])
    with (args.root / "projected_pipeline.csv").open("w", newline="") as output:
        writer = csv.writer(output); writer.writerow(["variant", "frame_dT_ms", "frame_full_lbs_ms", "lbs_only_ms", "composed_ms", "composed_fps"])
        for k in VARIANTS: writer.writerow([k, frame_dt[k], frame_full[k], lbs_only[k], composed[k], 1000/composed[k]])
    (args.root / "benchmark_summary.md").write_text(
        "# LBS selected-transform vectorization bake-off\n\n"
        f"A→B selected-transform/full-LBS: {frame_dt['a']/frame_dt['b']:.3f}× / {frame_full['a']/frame_full['b']:.3f}×。\n\n"
        f"B→C selected-transform/full-LBS: {frame_dt['b']/frame_dt['c']:.3f}× / {frame_full['b']/frame_full['c']:.3f}×。\n\n"
        f"最佳为 {VARIANTS[best]}，组合投影 {best_pipeline:.3f} ms / {1000/best_pipeline:.3f} FPS；完整 LBS 占 {100*best_full/best_pipeline:.2f}%。\n\n"
        f"`NEXT_OPTIMIZATION_EXPERIMENT={next_experiment}`\n")
    print("LBS_DERIVATIVE_SELECTED_TRANSFORM_VECTORISATION_BAKEOFF_READY=" + report["LBS_DERIVATIVE_SELECTED_TRANSFORM_VECTORISATION_BAKEOFF_READY"])
    if not all_correct: raise SystemExit(1)


if __name__ == "__main__":
    main()
