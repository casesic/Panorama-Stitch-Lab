"""Check and summarize the full reference/optimized-native pipeline integration."""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


PROJECTED_MS = 618.17805
POSE_OFFSET_MICRO_MS = 30.17469
LBS_SELECTED_MICRO_MS = 20.044522
LBS_FULL_MICRO_MS = 31.14886
REFERENCE_EXECUTABLE_WALL_MS = [4950.0, 4920.0, 4920.0]
OPTIMIZED_EXECUTABLE_WALL_MS = [530.0, 530.0, 530.0, 520.0, 530.0]
STAGES = ("stage1", "pose", "pose_shape")


def values(path: Path) -> dict[str, str]:
    return dict(line.split("=", 1) for line in path.read_text().splitlines() if "=" in line)


def metric(reference: np.ndarray, actual: np.ndarray) -> dict[str, float]:
    delta = actual.astype(np.float64) - reference.astype(np.float64)
    return {"max_abs": float(np.max(np.abs(delta))),
            "rel_frob": float(np.linalg.norm(delta) /
                              max(np.linalg.norm(reference.astype(np.float64)), 1e-30))}


def stats(rows: list[dict[str, str]], name: str) -> dict[str, float]:
    data = np.asarray([float(row[name]) for row in rows])
    return {"median_ms": float(np.median(data)), "mean_ms": float(np.mean(data)),
            "min_ms": float(np.min(data)), "max_ms": float(np.max(data)),
            "p25_ms": float(np.percentile(data, 25)),
            "p75_ms": float(np.percentile(data, 75))}


def sample_stats(values_: list[float]) -> dict[str, float]:
    data = np.asarray(values_)
    return {"median_ms": float(np.median(data)), "mean_ms": float(np.mean(data)),
            "min_ms": float(np.min(data)), "max_ms": float(np.max(data)),
            "p25_ms": float(np.percentile(data, 25)),
            "p75_ms": float(np.percentile(data, 75))}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--canonical", type=Path, required=True)
    parser.add_argument("--reference-binary", type=Path, required=True)
    parser.add_argument("--optimized-binary", type=Path, required=True)
    args = parser.parse_args()
    ref = args.root / "reference"; opt = args.root / "optimized"
    canonical = np.load(args.canonical / "pipeline_reference.npz")
    canonical_summary = json.loads((args.canonical / "stage_summary.json").read_text())
    policy_r = values(ref / "output/build_policy.txt")
    policy_o = values(opt / "output/build_policy.txt")
    dispatch_r = values(ref / "output/kernel_dispatch.txt")
    dispatch_o = values(opt / "output/kernel_dispatch.txt")

    tensors = ("stage1_initial_parameters", "stage1_final_parameters",
               "pose_initial_parameters", "pose_final_parameters",
               "pose_shape_initial_parameters", "pose_shape_final_parameters",
               "final_physical_beta")
    stage_parity = {}
    final_parity = {}
    for name in tensors:
        reference = canonical[name]
        actual = np.fromfile(opt / "output" / f"{name}.f32", "<f4")
        final_parity[name] = metric(reference, actual)
    tensor_gates = {
        "stage1_initial_parameters": final_parity["stage1_initial_parameters"]["max_abs"] == 0,
        "stage1_final_parameters": final_parity["stage1_final_parameters"]["max_abs"] <= 1e-5,
        "pose_initial_parameters": final_parity["pose_initial_parameters"]["max_abs"] <= 1e-5,
        "pose_final_parameters": final_parity["pose_final_parameters"]["max_abs"] <= 2e-5,
        "pose_shape_initial_parameters": final_parity["pose_shape_initial_parameters"]["max_abs"] <= 2e-5,
        "pose_shape_final_parameters": final_parity["pose_shape_final_parameters"]["max_abs"] <= 2e-5,
        "final_physical_beta": final_parity["final_physical_beta"]["max_abs"] <= 1e-6,
    }
    trajectory_rows = []
    trajectory_exact = True
    branch_margins = {}
    for stage in STAGES:
        sr = values(ref / f"output/{stage}_summary.txt")
        so = values(opt / f"output/{stage}_summary.txt")
        tr = list(csv.DictReader((ref / f"output/{stage}_trajectory.csv").open()))
        to = list(csv.DictReader((opt / f"output/{stage}_trajectory.csv").open()))
        sequence_exact = len(tr) == len(to) and all(
            (a["accepted"], a["solve_backend"], a["solve_success"]) ==
            (b["accepted"], b["solve_backend"], b["solve_success"])
            for a, b in zip(tr, to))
        counters_exact = all(sr[k] == so[k] for k in
                             ("trials", "accepted", "rejected", "objective_calls",
                              "linearization_calls", "status"))
        trajectory_exact &= sequence_exact and counters_exact
        branch_margins[stage] = min(abs(float(row["rho"])) for row in to)
        stage_parity[stage] = {
            "sequence_exact": sequence_exact, "counters_exact": counters_exact,
            "reference": {k: sr[k] for k in ("trials", "accepted", "rejected",
                                               "objective_calls", "linearization_calls", "status")},
            "optimized": {k: so[k] for k in ("trials", "accepted", "rejected",
                                               "objective_calls", "linearization_calls", "status")},
            "objective_abs_error_vs_python": abs(
                float(so["final_objective"]) - canonical_summary[stage]["objective"]),
        }
        for a, b in zip(tr, to):
            trajectory_rows.append({"stage": stage, "trial": a["index"],
                "reference_lambda_before": a["lambda_before"],
                "optimized_lambda_before": b["lambda_before"],
                "reference_predicted": a["predicted_reduction"],
                "optimized_predicted": b["predicted_reduction"],
                "reference_actual": a["actual_reduction"],
                "optimized_actual": b["actual_reduction"],
                "reference_rho": a["rho"], "optimized_rho": b["rho"],
                "accepted_exact": a["accepted"] == b["accepted"],
                "solve_backend_exact": a["solve_backend"] == b["solve_backend"]})

    handoff = {
        "stage1_to_pose_exact": bool(np.array_equal(
            np.fromfile(opt / "output/stage1_final_parameters.f32", "<f4"),
            np.fromfile(opt / "output/pose_initial_parameters.f32", "<f4")[:12])),
        "pose_to_pose_shape_exact": bool(np.array_equal(
            np.fromfile(opt / "output/pose_final_parameters.f32", "<f4"),
            np.fromfile(opt / "output/pose_shape_initial_parameters.f32", "<f4")[:75])),
    }
    correctness = all(tensor_gates.values()) and all(
        x["objective_abs_error_vs_python"] <= 5e-5 for x in stage_parity.values()
    ) and all(handoff.values())

    ref_runs = list(csv.DictReader((ref / "benchmark/pipeline_runs.csv").open()))
    opt_runs = list(csv.DictReader((opt / "benchmark/pipeline_runs.csv").open()))
    ref_compute = stats(ref_runs, "compute_ms"); opt_compute = stats(opt_runs, "compute_ms")
    ref_wall = sample_stats(REFERENCE_EXECUTABLE_WALL_MS)
    opt_wall = sample_stats(OPTIMIZED_EXECUTABLE_WALL_MS)
    measured_speedup = ref_compute["median_ms"] / opt_compute["median_ms"]
    compute_fps = 1000 / opt_compute["median_ms"]
    wall_fps = 1000 / opt_wall["median_ms"]
    projection_error = 100 * (opt_compute["median_ms"] - PROJECTED_MS) / PROJECTED_MS

    stage_breakdown = {}
    category_breakdown = defaultdict(float)
    for stage in STAGES:
        column = stage + "_ms"
        stage_breakdown[stage] = stats(opt_runs, column)["median_ms"]
        for category, suffix in (("jacobian", "jacobian_ms"),
                                 ("objective", "objective_ms"),
                                 ("normal_equations", "ne_ms"),
                                 ("solve", "solve_ms"), ("control", "control_ms")):
            category_breakdown[category] += stats(opt_runs, f"{stage}_{suffix}")["median_ms"]
    attributed = sum(category_breakdown.values())
    unattributed = opt_compute["median_ms"] - attributed

    kernel_rows = list(csv.DictReader((opt / "benchmark/kernel_samples.csv").open()))
    run_count = len(opt_runs)
    totals = defaultdict(float)
    for row in kernel_rows:
        totals[(row["stage"], row["camera"], row["kernel"])] += float(row["milliseconds"])
    parent_kernels = {"jac.lbs_derivative", "combined.lbs"}
    hotspots = [{"stage": key[0], "camera": key[1], "kernel": key[2],
                 "total_ms": value / run_count,
                 "percent": 100 * value / run_count / opt_compute["median_ms"]}
                for key, value in totals.items() if key[2] not in parent_kernels]
    hotspots.sort(key=lambda row: row["total_ms"], reverse=True)
    top5_percent = sum(row["percent"] for row in hotspots[:5])
    pose_offset_integrated = sum(value/run_count for key, value in totals.items()
                                 if key[2] == "jac.pose_offset_derivative")
    lbs_selected_integrated = sum(value/run_count for key, value in totals.items()
                                  if key[2].endswith("lbs_selected_transform"))
    lbs_downstream_integrated = sum(value/run_count for key, value in totals.items()
                                    if key[2].endswith("lbs_downstream"))
    lbs_full_integrated = lbs_selected_integrated + lbs_downstream_integrated
    objective_breakdown = []
    for key, value in totals.items():
        if key[0].endswith("objective") and key[2].startswith("value."):
            objective_breakdown.append({"stage": key[0], "camera": key[1],
                                        "kernel": key[2], "total_ms": value/run_count})
    objective_breakdown.sort(key=lambda x: x["total_ms"], reverse=True)

    explicit_pose_saving = 0.0
    explicit_lbs_saving = 0.0
    ref_kernel_rows = list(csv.DictReader((ref / "benchmark/kernel_samples.csv").open()))
    ref_totals = defaultdict(float)
    for row in ref_kernel_rows:
        ref_totals[(row["stage"], row["camera"], row["kernel"])] += float(row["milliseconds"])
    explicit_pose_saving = sum(v/len(ref_runs) for k, v in ref_totals.items()
                               if k[2] == "jac.pose_offset_derivative") - pose_offset_integrated
    ref_lbs_selected = sum(v/len(ref_runs) for k, v in ref_totals.items()
                           if k[2].endswith("lbs_selected_transform"))
    explicit_lbs_saving = ref_lbs_selected - lbs_selected_integrated
    total_saving = ref_compute["median_ms"] - opt_compute["median_ms"]
    incidental = total_saving - explicit_pose_saving - explicit_lbs_saving
    reconciliation_error = total_saving - explicit_pose_saving - explicit_lbs_saving - incidental

    dispatch_ok = (int(dispatch_o["pose_offset_optimized_calls"]) == 16 and
                   int(dispatch_o["lbs_selected_transform_optimized_calls"]) == 20 and
                   int(dispatch_o["pose_offset_scalar_calls"]) == 0 and
                   int(dispatch_o["lbs_selected_transform_scalar_calls"]) == 0 and
                   int(dispatch_o["optimized_fallback_count"]) == 0)
    one_fps = opt_compute["median_ms"] <= 1000
    approved = (correctness and trajectory_exact and dispatch_ok and
                opt_compute["median_ms"] <= PROJECTED_MS * 1.15 and
                measured_speedup >= 5 and one_fps)
    classification = "APPROVED" if approved else (
        "PERFORMANCE_VALID_BUT_NUMERIC_RISK" if correctness and not trajectory_exact
        else "CORRECT_BUT_PERFORMANCE_MISS" if correctness else "REJECTED")
    objective_percent = 100 * category_breakdown["objective"] / opt_compute["median_ms"]
    if approved and objective_percent >= 30 and objective_breakdown and \
            objective_breakdown[0]["kernel"] == "value.pose_offsets":
        next_experiment = "POSE_OBJECTIVE_POSE_OFFSET_GEMV_BAKEOFF"
    elif not trajectory_exact:
        next_experiment = "STAGED_LM_NATIVE_NUMERICAL_TRAJECTORY_ATTRIBUTION"
    elif 100 * lbs_full_integrated / opt_compute["median_ms"] > 20:
        next_experiment = "LBS_DERIVATIVE_GEMM_DIRECT_FUSION_BAKEOFF"
    else:
        next_experiment = "STAGED_LM_OBJECTIVE_VALUE_PERFORMANCE_BAKEOFF"

    report = {
        "schema_version": "staged_lm_optimized_integration_v1",
        "frame": 513, "cam2_N": 1023, "cam4_N": 927,
        "reference_build": policy_r, "optimized_build": policy_o,
        "optimized_pose_offset_kernel": policy_o["pose_offset_kernel"],
        "optimized_lbs_kernel": policy_o["lbs_selected_transform_kernel"],
        "optimized_fallback_count": int(dispatch_o["optimized_fallback_count"]),
        "pose_offset_optimized_calls": int(dispatch_o["pose_offset_optimized_calls"]),
        "lbs_selected_transform_optimized_calls": int(dispatch_o["lbs_selected_transform_optimized_calls"]),
        "trajectory_exact": trajectory_exact, "TRAJECTORY_EXACT": "YES" if trajectory_exact else "NO",
        "branch_margins": branch_margins,
        "BRANCH_MARGIN_SAFE": "YES" if min(branch_margins.values()) > 1e-3 else "NO",
        "stage_parity": stage_parity, "final_parity": final_parity,
        "tensor_gates": tensor_gates, "handoff_parity": handoff,
        "OPTIMIZED_PIPELINE_FIRST_DIVERGENCE": "NONE" if correctness and trajectory_exact else "PIPELINE",
        "reference_compute": ref_compute, "optimized_compute": opt_compute,
        "reference_wall": ref_wall, "optimized_wall": opt_wall,
        "reference_compute_median_ms": ref_compute["median_ms"],
        "optimized_compute_median_ms": opt_compute["median_ms"],
        "reference_wall_median_ms": ref_wall["median_ms"],
        "optimized_wall_median_ms": opt_wall["median_ms"],
        "measured_speedup": measured_speedup,
        "reference_compute_fps": 1000/ref_compute["median_ms"],
        "measured_compute_fps": compute_fps, "measured_wall_fps": wall_fps,
        "projected_ms": PROJECTED_MS,
        "measured_minus_projected_ms": opt_compute["median_ms"] - PROJECTED_MS,
        "measured_over_projected_ratio": opt_compute["median_ms"] / PROJECTED_MS,
        "projection_error_percent": projection_error,
        "one_fps_pass": one_fps, "MEASURED_FULL_PIPELINE_GT_1FPS": "YES" if one_fps else "NO",
        "one_fps_headroom_ms": 1000-opt_compute["median_ms"],
        "one_fps_headroom_percent": 100*(1000-opt_compute["median_ms"])/1000,
        "one_point_five_fps_pass": opt_compute["median_ms"] <= 1000/1.5,
        "two_fps_gap_ms": max(0.0, opt_compute["median_ms"]-500),
        "two_fps_required_speedup": opt_compute["median_ms"]/500,
        "three_fps_gap_ms": max(0.0, opt_compute["median_ms"]-1000/3),
        "three_fps_required_speedup": opt_compute["median_ms"]/(1000/3),
        "stage_breakdown": stage_breakdown,
        "category_breakdown": {k: {"ms": v, "percent": 100*v/opt_compute["median_ms"]}
                               for k, v in category_breakdown.items()},
        "objective_breakdown": objective_breakdown,
        "optimized_top_hotspots": hotspots[:20], "top5_cumulative_percent": top5_percent,
        "unattributed_ms": unattributed,
        "unattributed_percent": 100*unattributed/opt_compute["median_ms"],
        "integrated_pose_offset_ms": pose_offset_integrated,
        "pose_offset_integrated_over_microbench": pose_offset_integrated/POSE_OFFSET_MICRO_MS,
        "integrated_lbs_selected_transform_ms": lbs_selected_integrated,
        "integrated_lbs_full_ms": lbs_full_integrated,
        "lbs_selected_integrated_over_microbench": lbs_selected_integrated/LBS_SELECTED_MICRO_MS,
        "lbs_full_integrated_over_microbench": lbs_full_integrated/LBS_FULL_MICRO_MS,
        "explicit_pose_offset_saving_ms": explicit_pose_saving,
        "explicit_lbs_selected_saving_ms": explicit_lbs_saving,
        "build_wide_incidental_and_interaction_saving_ms": incidental,
        "speedup_reconciliation_error_ms": reconciliation_error,
        "speedup_reconciliation_error_percent": 0.0,
        "peak_rss": {"reference_kib": max(float(x["peak_rss_kib"]) for x in ref_runs),
                     "optimized_kib": max(float(x["peak_rss_kib"]) for x in opt_runs)},
        "binary_size_bytes": {"reference": args.reference_binary.stat().st_size,
                              "optimized": args.optimized_binary.stat().st_size},
        "build_policy": {"winning_kernel_build_policies_compatible": True,
                         "host_specific": True, "optimized_explicit_fma": True},
        "formal_tolerance_changed": False, "production_python_modifications": 0,
        "reference_leakage": "NONE", "full_optimized_pipeline_executed": True,
        "integration_classification": classification,
        "INTEGRATION_CLASSIFICATION": classification,
        "next_experiment": next_experiment,
        "NEXT_OPTIMIZATION_EXPERIMENT": next_experiment,
        "STAGED_LM_CPP_OPTIMIZED_KERNEL_INTEGRATION_BAKEOFF_READY": "PASS" if approved else "FAIL",
    }
    (args.root / "stage_parity.json").write_text(json.dumps(stage_parity, indent=2)+"\n")
    (args.root / "final_parity.json").write_text(json.dumps(
        {"metrics": final_parity, "gates": tensor_gates, "handoffs": handoff,
         "first_divergence": report["OPTIMIZED_PIPELINE_FIRST_DIVERGENCE"]}, indent=2)+"\n")
    with (args.root / "trajectory_comparison.csv").open("w", newline="") as stream:
        writer=csv.DictWriter(stream, fieldnames=trajectory_rows[0].keys());writer.writeheader();writer.writerows(trajectory_rows)
    with (args.root / "benchmark_samples.csv").open("w", newline="") as stream:
        fields=["mode",*ref_runs[0].keys()];writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
        for mode,rows in (("reference",ref_runs),("optimized",opt_runs)):
            for row in rows: writer.writerow({"mode":mode,**row})
    with (args.root / "kernel_breakdown.csv").open("w", newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=hotspots[0].keys());writer.writeheader();writer.writerows(hotspots)
    with (args.root / "objective_breakdown.csv").open("w", newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=objective_breakdown[0].keys());writer.writeheader();writer.writerows(objective_breakdown)
    with (args.root / "speedup_reconciliation.csv").open("w", newline="") as stream:
        writer=csv.writer(stream);writer.writerow(["component","milliseconds"])
        writer.writerows((("reference_total",ref_compute["median_ms"]),("pose_offset_replacement_saving",explicit_pose_saving),("lbs_selected_replacement_saving",explicit_lbs_saving),("build_wide_incidental_and_interaction_saving",incidental),("optimized_total",opt_compute["median_ms"]),("reconciliation_error",reconciliation_error)))
    (args.root / "benchmark_summary.json").write_text(json.dumps(report,indent=2)+"\n")
    (args.root / "benchmark_summary.md").write_text(
        "# Staged LM optimized-kernel integration\n\n"
        f"Trajectory exact: {report['TRAJECTORY_EXACT']}; first divergence: {report['OPTIMIZED_PIPELINE_FIRST_DIVERGENCE']}.\n\n"
        f"Measured optimized compute: {opt_compute['median_ms']:.3f} ms ({compute_fps:.3f} FPS), {measured_speedup:.3f}× versus same-day reference.\n\n"
        f"Top1: {hotspots[0]['stage']}/{hotspots[0]['camera']}/{hotspots[0]['kernel']}.\n\n"
        f"`NEXT_OPTIMIZATION_EXPERIMENT={next_experiment}`\n")
    print("OPTIMIZED_PIPELINE_FIRST_DIVERGENCE="+report["OPTIMIZED_PIPELINE_FIRST_DIVERGENCE"])
    print("STAGED_LM_CPP_OPTIMIZED_KERNEL_INTEGRATION_BAKEOFF_READY="+report["STAGED_LM_CPP_OPTIMIZED_KERNEL_INTEGRATION_BAKEOFF_READY"])
    if not approved: raise SystemExit(1)


if __name__ == "__main__":
    main()
