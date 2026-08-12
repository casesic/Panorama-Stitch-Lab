"""Check C++ frame-513 staged-pipeline boundaries against production."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _metric(reference: np.ndarray, actual: np.ndarray) -> dict[str, float]:
    difference = np.asarray(actual, np.float64) - np.asarray(reference, np.float64)
    return {
        "max_abs": float(np.max(np.abs(difference))),
        "rel_frob": float(
            np.linalg.norm(difference.ravel()) /
            max(np.linalg.norm(np.asarray(reference, np.float64).ravel()), 1e-30)
        ),
    }


def _values(path: Path) -> dict[str, str]:
    return dict(line.strip().split("=", 1) for line in path.read_text().splitlines())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference_dir", type=Path, required=True)
    parser.add_argument("--cpp_output", type=Path, required=True)
    args = parser.parse_args()
    reference = np.load(args.reference_dir / "pipeline_reference.npz")
    summary = json.loads((args.reference_dir / "stage_summary.json").read_text())
    names = [
        "stage1_initial_parameters", "stage1_final_parameters",
        "pose_initial_parameters", "pose_final_parameters",
        "pose_shape_initial_parameters", "pose_shape_final_parameters",
        "final_physical_beta",
    ]
    metrics = {
        name: _metric(reference[name], np.fromfile(
            args.cpp_output / f"{name}.f32", dtype="<f4"))
        for name in names
    }
    gates = {
        "stage1_initial_parameters": metrics["stage1_initial_parameters"]["max_abs"] == 0,
        "stage1_final_parameters": metrics["stage1_final_parameters"]["max_abs"] <= 1e-5,
        "pose_initial_parameters": metrics["pose_initial_parameters"]["max_abs"] <= 1e-5,
        "pose_final_parameters": metrics["pose_final_parameters"]["max_abs"] <= 2e-5,
        "pose_shape_initial_parameters": metrics["pose_shape_initial_parameters"]["max_abs"] <= 2e-5,
        "pose_shape_final_parameters": metrics["pose_shape_final_parameters"]["max_abs"] <= 2e-5,
        "final_physical_beta": metrics["final_physical_beta"]["max_abs"] <= 1e-6,
    }
    stage_parity = {}
    objective_errors = {}
    for stage in ("stage1", "pose", "pose_shape"):
        cpp = _values(args.cpp_output / f"{stage}_summary.txt")
        py = summary[stage]
        objective_errors[stage] = abs(float(cpp["final_objective"]) - py["objective"])
        stage_parity[stage] = {
            "status": cpp["status"] == py["status"],
            "trials": int(cpp["trials"]) == py["trials"],
            "accepted": int(cpp["accepted"]) == py["accepted_steps"],
            "rejected": int(cpp["rejected"]) == py["rejected_trials"],
            "objective_calls": int(cpp["objective_calls"]) == py["objective_calls"],
            "linearization_calls": (
                int(cpp["linearization_calls"]) == py["linearization_calls"]
            ),
            "objective": objective_errors[stage] <= 5e-5,
        }
    cpp_stage1 = np.fromfile(
        args.cpp_output / "stage1_final_parameters.f32", "<f4")
    cpp_pose_initial = np.fromfile(
        args.cpp_output / "pose_initial_parameters.f32", "<f4")
    cpp_pose_final = np.fromfile(
        args.cpp_output / "pose_final_parameters.f32", "<f4")
    cpp_shape_initial = np.fromfile(
        args.cpp_output / "pose_shape_initial_parameters.f32", "<f4")
    beta_anchor = reference["beta_anchor"]
    beta_latent = cpp_shape_initial[-10:]
    handoff = {
        "stage1_to_pose_rigid_exact": bool(np.array_equal(cpp_stage1, cpp_pose_initial[:12])),
        "stage1_to_pose_body_anchor_exact": bool(np.array_equal(
            cpp_pose_initial[12:], reference["pose_initial_parameters"][12:])),
        "pose_to_pose_shape_frame_exact": bool(np.array_equal(
            cpp_pose_final, cpp_shape_initial[:75])),
        "beta_reverse_max_abs": float(np.max(np.abs(
            2.5 * np.tanh(beta_latent) - beta_anchor))),
    }
    handoff["beta_reverse_consistent"] = handoff["beta_reverse_max_abs"] <= 1e-6
    stage_first = {
        "STAGE1_FIRST_DIVERGENCE": "NONE" if gates["stage1_final_parameters"] and all(stage_parity["stage1"].values()) else "STAGE1",
        "STAGE1_TO_POSE_HANDOFF_FIRST_DIVERGENCE": "NONE" if handoff["stage1_to_pose_rigid_exact"] and handoff["stage1_to_pose_body_anchor_exact"] else "HANDOFF",
        "POSE_FIRST_DIVERGENCE": "NONE" if gates["pose_final_parameters"] and all(stage_parity["pose"].values()) else "POSE",
        "POSE_TO_POSE_SHAPE_HANDOFF_FIRST_DIVERGENCE": "NONE" if handoff["pose_to_pose_shape_frame_exact"] and handoff["beta_reverse_consistent"] else "HANDOFF",
        "POSE_SHAPE_FIRST_DIVERGENCE": "NONE" if gates["pose_shape_final_parameters"] and all(stage_parity["pose_shape"].values()) else "POSE_SHAPE",
    }
    ready = all(gates.values()) and all(
        all(values.values()) for values in stage_parity.values()
    ) and all(value == "NONE" for value in stage_first.values())
    report = {
        "metrics": metrics, "metric_gates": gates,
        "objective_errors": objective_errors, "stage_parity": stage_parity,
        "handoff_invariants": handoff, **stage_first,
        "PIPELINE_FIRST_DIVERGENCE": "NONE" if ready else "STAGE_BOUNDARY",
        "STAGED_LM_PIPELINE_CPP_CPU_REFERENCE_READY": "PASS" if ready else "FAIL",
    }
    output_json = args.cpp_output / "staged_lm_pipeline_cpp_parity.json"
    output_json.write_text(json.dumps(report, indent=2) + "\n")
    (args.cpp_output / "staged_lm_pipeline_cpp_parity.md").write_text(
        "# Staged LM C++ pipeline parity\n\n"
        f"- PIPELINE_FIRST_DIVERGENCE: `{report['PIPELINE_FIRST_DIVERGENCE']}`\n"
        f"- READY: `{report['STAGED_LM_PIPELINE_CPP_CPU_REFERENCE_READY']}`\n"
    )
    for name, value in stage_first.items():
        print(f"{name}={value}")
    print(f"PIPELINE_FIRST_DIVERGENCE={report['PIPELINE_FIRST_DIVERGENCE']}")
    print("STAGED_LM_PIPELINE_CPP_CPU_REFERENCE_READY=" + report[
        "STAGED_LM_PIPELINE_CPP_CPU_REFERENCE_READY"])
    raise SystemExit(0 if ready else 1)


if __name__ == "__main__":
    main()
