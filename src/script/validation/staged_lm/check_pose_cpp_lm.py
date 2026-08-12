"""Check frame-513 C++ Pose LM trajectory against canonical Python LM."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from script.validation.staged_lm.check_pose_cpp_normal_equations import check_pose_normal_equations


def _meta(path: Path) -> dict[str, str]:
    return dict(line.split("=", 1) for line in path.read_text().splitlines() if line)


def _metric(reference: np.ndarray, actual: np.ndarray, rel_tol: float, abs_tol: float) -> dict[str, Any]:
    difference = actual.astype(np.float64) - reference.astype(np.float64)
    relative = float(np.linalg.norm(difference.ravel()) / max(float(np.linalg.norm(reference.astype(np.float64).ravel())), 1e-300))
    maximum = float(np.max(np.abs(difference))) if difference.size else 0.0
    passed = bool(actual.shape == reference.shape and np.isfinite(actual).all() and relative <= rel_tol and maximum <= abs_tol)
    return {"exact": bool(np.array_equal(reference, actual)), "max_abs": maximum,
            "mean_abs": float(np.mean(np.abs(difference))) if difference.size else 0.0,
            "relative": relative, "rel_tol": rel_tol, "abs_tol": abs_tol, "pass": passed}


def _scalar(reference: float, actual: float, rel_tol: float = 1e-5, abs_tol: float = 1e-4) -> dict[str, Any]:
    return _metric(np.asarray([reference]), np.asarray([actual]), rel_tol, abs_tol)


def check_pose_lm(reference_dir: Path, cpp_output: Path, pose_cpp_output: Path) -> dict[str, Any]:
    upstream = check_pose_normal_equations(reference_dir.parent.parent / "pose/frame_000513/pose_reference.npz", pose_cpp_output, pose_cpp_output)
    if not upstream["pass"]: raise RuntimeError("POSE_NE_MUST_PASS_BEFORE_LM")
    trajectory = json.loads((reference_dir / "pose_lm_reference_trajectory.json").read_text())
    meta = _meta(cpp_output / "cpp_output_meta.txt")
    rows = list(csv.DictReader((cpp_output / "trajectory.tsv").open(), delimiter="\t"))
    cpp_trials = [row for row in rows if row["type"] == "trial"]
    comparisons: dict[str, Any] = {}
    order: list[str] = []
    with np.load(reference_dir / "pose_lm_reference.npz") as reference:
        final_parameters_reference = reference["final_parameters"].copy()
        comparisons["initial_parameters"] = _metric(reference["initial_parameters"], np.fromfile(cpp_output / "initial_parameters.f32", "<f4"), 0.0, 0.0)
        order.append("initial_parameters")
        comparisons["initial_objective"] = _scalar(trajectory["initial_objective"], float(np.fromfile(cpp_output / "initial_objective.f64", "<f8")[0]))
        order.append("initial_objective")
        for index, linearization in enumerate(trajectory["linearizations"]):
            for field, rel, absolute in (("H", 2e-6, 5e-2), ("g", 2e-4, 2e-3)):
                name = f"linearization_{index}_{field}"
                actual = np.fromfile(cpp_output / f"{name}.f64", "<f8").reshape(reference[name].shape)
                comparisons[name] = _metric(reference[name], actual, rel, absolute); order.append(name)
        for index, (python, cpp) in enumerate(zip(trajectory["trials"], cpp_trials)):
            for field, suffix, dtype, rel, absolute in (
                ("delta_double", "f64", "<f8", 1e-4, 1e-5),
                ("delta_float", "f32", "<f4", 1e-4, 1e-5),
                ("candidate", "f32", "<f4", 1e-5, 1e-5),
            ):
                name = f"trial_{index}_{field}"
                comparisons[name] = _metric(reference[name], np.fromfile(cpp_output / f"{name}.{suffix}", dtype), rel, absolute); order.append(name)
            for field, cpp_name, rel, absolute in (
                ("candidate_objective", "objective", 1e-5, 1e-3),
                ("predicted_reduction", "predicted", 1e-5, 1e-3),
                ("actual_reduction", "actual", 1e-5, 1e-3),
                ("rho", "rho", 1e-5, 1e-4),
            ):
                name = f"trial_{index}_{field}"
                comparisons[name] = _scalar(python[field], float(cpp[cpp_name]), rel, absolute); order.append(name)
    python_accept = [bool(trial["accepted"]) for trial in trajectory["trials"]]
    cpp_accept = [row["accepted"] == "1" for row in cpp_trials]
    lambda_python = [(trial["lambda_before"], trial["lambda_after"]) for trial in trajectory["trials"]]
    lambda_cpp = [(float(row["lambda_before"]), float(row["lambda_after"])) for row in cpp_trials]
    branch = {
        "trial_count_exact": len(cpp_trials) == len(trajectory["trials"]),
        "accept_reject_exact": cpp_accept == python_accept,
        "lambda_branch_sequence_exact": all(
            abs(a - c) / max(abs(a), 1.0) <= 1e-6
            for pair_p, pair_c in zip(lambda_python, lambda_cpp) for a, c in zip(pair_p, pair_c)
        ) and len(lambda_cpp) == len(lambda_python),
        "convergence_branch_exact": meta["convergence_branch"] == trajectory["convergence_branch"],
        "status_exact": meta["status"] == trajectory["status"],
    }
    counters = {key: int(meta[key]) == trajectory[reference_key] for key, reference_key in (
        ("accepted_steps", "accepted_steps"), ("rejected_trials", "rejected_trials"),
        ("objective_call_count", "objective_call_count"),
        ("normal_equation_call_count", "normal_equation_call_count"),
        ("solve_call_count", "solve_call_count"), ("candidate_trial_count", "candidate_trial_count"),
    )}
    final_parameters = np.fromfile(cpp_output / "final_parameters.f32", "<f4")
    comparisons["final_parameters"] = _metric(final_parameters_reference, final_parameters, 1e-5, 1e-5); order.append("final_parameters")
    comparisons["final_objective"] = _scalar(trajectory["final_objective"], float(np.fromfile(cpp_output / "final_objective.f64", "<f8")[0]), 1e-5, 1e-3); order.append("final_objective")
    first = next((name for name in order if not comparisons[name]["pass"]), "NONE")
    if first == "NONE": first = next((f"branch.{key}" for key, value in branch.items() if not value), "NONE")
    if first == "NONE": first = next((f"counter.{key}" for key, value in counters.items() if not value), "NONE")
    rho_margins = [min(abs(float(row["rho"]) - value) for value in (0.0, 0.25, 0.75)) for row in cpp_trials]
    rho_differences = [abs(float(row["rho"]) - trial["rho"]) for row, trial in zip(cpp_trials, trajectory["trials"])]
    timing = {key: float(value) for key, value in meta.items() if key.endswith("_ms")}
    total = timing["total_lm_ms"]
    result = {"schema_version": "pose_lm_cpp_parity_v1", "comparisons": comparisons,
              "branch_parity": branch, "counter_parity": counters,
              "accept_reject_sequence": {"python": python_accept, "cpp": cpp_accept},
              "lambda_sequence": {"python": lambda_python, "cpp": lambda_cpp},
              "MIN_RHO_BRANCH_MARGIN": min(rho_margins), "max_rho_difference": max(rho_differences),
              "LM_FIRST_DIVERGENCE": first, "pass": first == "NONE",
              "POSE_LM_CPP_CPU_REFERENCE_READY": first == "NONE", "timing_ms": timing,
              "timing_percent": {key: value / total * 100 for key, value in timing.items() if key.startswith("total_") and key != "total_lm_ms"}}
    json_path = cpp_output / "pose_lm_cpp_parity.json"; md_path = cpp_output / "pose_lm_cpp_parity.md"
    json_path.write_text(json.dumps(result, indent=2) + "\n")
    md_path.write_text(f"# Pose LM C++ CPU parity\n\n`LM_FIRST_DIVERGENCE={first}`\n\n`POSE_LM_CPP_CPU_REFERENCE_READY={'PASS' if result['pass'] else 'FAIL'}`\n\nMinimum rho branch margin: `{result['MIN_RHO_BRANCH_MARGIN']:.6e}`.\n")
    print(f"LM_FIRST_DIVERGENCE={first}")
    print(f"POSE_LM_CPP_CPU_REFERENCE_READY={'PASS' if result['pass'] else 'FAIL'}")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference_dir", type=Path, required=True)
    parser.add_argument("--cpp_output", type=Path, required=True)
    parser.add_argument("--pose_cpp_output", type=Path, required=True)
    args = parser.parse_args(argv)
    return 0 if check_pose_lm(args.reference_dir, args.cpp_output, args.pose_cpp_output)["pass"] else 1

if __name__ == "__main__": raise SystemExit(main())
