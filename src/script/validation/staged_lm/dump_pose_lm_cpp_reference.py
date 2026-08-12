"""Dump the canonical frame-513 Pose LM trajectory for the C++ port."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from lib_pano.human_fitting.staged_lm.lm.convergence import (
    check_convergence, classify_no_accepted_trial,
    increase_damping_after_rejection, initialise_damping_from_max_diagonal,
    update_damping_after_acceptance,
)
from lib_pano.human_fitting.staged_lm.lm.equations import (
    _evaluate_linearization, compute_scaled_gradient,
    normal_equations_max_diagonal, solve_normal_equations,
)
from lib_pano.human_fitting.staged_lm.lm.trial import evaluate_trial_reduction
from lib_pano.human_fitting.staged_lm.stage2_shared_cache import build_stage2_shared_geometry_cache
from lib_pano.human_fitting.staged_lm.stage_definitions.pose import build_pose_stage_definition
from lib_pano.human_fitting.staged_lm.stage_definitions.rigid import build_stage1_definition
from lib_pano.human_fitting.staged_lm.stage_runtime import run_stage
from script.human_fitting.staged_lm.app import build_context


REPOSITORY_ROOT = Path(__file__).resolve().parents[4]


def _numpy(value: torch.Tensor) -> np.ndarray:
    return value.detach().cpu().contiguous().numpy()


def trace_pose_lm(stage: Any, context: Any) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Validation-only transcription of production runner with checkpoints."""
    solver = context.solver_config
    max_steps = stage.lm_max_accepted_steps
    max_trials = solver.lm_max_trials
    x = stage.initial_lm_parameters.detach().clone()
    arrays: dict[str, np.ndarray] = {"initial_parameters": _numpy(x)}
    linearizations: list[dict[str, Any]] = []
    trials: list[dict[str, Any]] = []
    objective_calls = 0
    normal_equation_calls = 0
    solve_calls = 0
    accepted_steps = 0
    rejected_trials = 0
    objective_ms = 0.0
    linearization_ms = 0.0
    solve_ms = 0.0

    def objective(parameters: torch.Tensor) -> float:
        nonlocal objective_calls, objective_ms
        started = time.perf_counter()
        value = float(stage.lm_true_objective_fn(parameters).detach().item())
        objective_ms += (time.perf_counter() - started) * 1000.0
        objective_calls += 1
        return value

    def linearize(parameters: torch.Tensor, outer: int, objective_value: float, damping_value: float | None):
        nonlocal normal_equation_calls, linearization_ms
        started = time.perf_counter()
        model = stage.linearization_builder(parameters, stage.aggregation_spec)
        equations, _ = _evaluate_linearization(model, parameters)
        linearization_ms += (time.perf_counter() - started) * 1000.0
        index = len(linearizations)
        arrays[f"linearization_{index}_parameters"] = _numpy(parameters)
        arrays[f"linearization_{index}_H"] = _numpy(equations.hessian)
        arrays[f"linearization_{index}_g"] = _numpy(equations.gradient)
        linearizations.append({
            "index": index, "outer": outer, "objective": objective_value,
            "lambda_before": damping_value,
            "max_diagonal": normal_equations_max_diagonal(equations),
            "scaled_gradient": compute_scaled_gradient(equations),
        })
        normal_equation_calls += 1
        return equations

    # Production evaluates the initial H/g before the initial objective.
    equations = linearize(x, -1, float("nan"), None)
    damping = initialise_damping_from_max_diagonal(normal_equations_max_diagonal(equations))
    current_objective = objective(x)
    linearizations[0]["objective"] = current_objective
    linearizations[0]["lambda_before"] = damping
    initial_objective = current_objective
    initial_lambda = damping
    status = "RUNNING"
    convergence_branch = "NONE"

    for outer in range(max_steps):
        gradient_64 = equations.gradient
        objective_before = current_objective
        accepted_this = False
        last_delta_norm = None
        for trial in range(max_trials):
            trial_damping = damping
            started = time.perf_counter()
            solve_result = solve_normal_equations(equations, damping)
            solve_ms += (time.perf_counter() - started) * 1000.0
            solve_calls += 1
            record: dict[str, Any] = {
                "index": len(trials), "outer": outer, "trial": trial,
                "lambda_before": trial_damping,
                "solve_success": solve_result.success,
                "solve_backend": solve_result.backend,
            }
            if not solve_result.success:
                damping = increase_damping_after_rejection(damping)
                rejected_trials += 1
                record.update({"accepted": False, "lambda_after": damping})
                trials.append(record)
                continue
            delta = solve_result.delta
            delta_float = delta.to(dtype=torch.float32)
            candidate = (x + delta_float).detach()
            candidate_objective = objective(candidate)
            predicted, actual, rho, accepted = evaluate_trial_reduction(
                equations, gradient_64, delta.to(torch.float64),
                objective_before, candidate_objective,
            )
            index = len(trials)
            arrays[f"trial_{index}_delta_double"] = _numpy(delta.to(torch.float64))
            arrays[f"trial_{index}_delta_float"] = _numpy(delta_float)
            arrays[f"trial_{index}_candidate"] = _numpy(candidate)
            last_delta_norm = float(delta.norm())
            record.update({
                "candidate_objective": candidate_objective,
                "predicted_reduction": predicted, "actual_reduction": actual,
                "rho": rho, "accepted": accepted,
                "delta_norm": last_delta_norm,
                "min_diagonal": float(torch.diagonal(equations.hessian).min()),
                "max_diagonal": float(torch.diagonal(equations.hessian).max()),
                "gradient_norm": float(equations.gradient.norm()),
            })
            if accepted:
                x = candidate
                current_objective = candidate_objective
                accepted_steps += 1
                accepted_this = True
                damping = update_damping_after_acceptance(damping, rho)
            else:
                damping = increase_damping_after_rejection(damping)
                rejected_trials += 1
            record["lambda_after"] = damping
            trials.append(record)
            if accepted:
                break

        if not accepted_this:
            scaled = compute_scaled_gradient(equations)
            status = classify_no_accepted_trial(accepted_steps, scaled)
            convergence_branch = "NO_ACCEPTED_TRIAL"
            break

        next_equations = linearize(x, outer, current_objective, damping)
        scaled = compute_scaled_gradient(next_equations)
        relative_objective = (
            (objective_before - current_objective) / max(abs(objective_before), 1e-16)
        )
        convergence = check_convergence(
            scaled, last_delta_norm, relative_objective, solver,
        )
        linearizations[-1].update({
            "relative_objective_change": relative_objective,
            "step_norm": last_delta_norm,
            "convergence": convergence or "NONE",
        })
        if convergence is not None:
            status = convergence
            convergence_branch = convergence
            break
        equations = next_equations

    if status == "RUNNING":
        status = "MAX_ACCEPTED_STEPS"
        convergence_branch = status
    arrays["final_parameters"] = _numpy(x)
    arrays["final_objective"] = np.asarray(current_objective, dtype=np.float64)
    trajectory = {
        "schema_version": "pose_lm_cpp_reference_v1",
        "initial_objective": initial_objective, "initial_lambda": initial_lambda,
        "linearizations": linearizations, "trials": trials,
        "final_objective": current_objective, "final_lambda": damping,
        "status": status, "convergence_branch": convergence_branch,
        "accepted_steps": accepted_steps, "rejected_trials": rejected_trials,
        "outer_iterations": len({trial["outer"] for trial in trials}),
        "objective_call_count": objective_calls,
        "normal_equation_call_count": normal_equation_calls,
        "solve_call_count": solve_calls, "candidate_trial_count": sum(
            int(trial["solve_success"]) for trial in trials
        ),
        "timing_ms": {
            "objective": objective_ms, "linearization": linearization_ms,
            "solve": solve_ms,
        },
    }
    return trajectory, arrays


def dump_from_context(context: Any, output: Path) -> dict[str, Any]:
    stage1 = run_stage(build_stage1_definition(context), context)
    shared = build_stage2_shared_geometry_cache(context)
    pose, _ = build_pose_stage_definition(context, stage1, shared)
    trajectory, arrays = trace_pose_lm(pose, context)
    # Independent production runner parity gate; no production instrumentation.
    production = run_stage(pose, context).lm_result
    parity = {
        "final_parameters_exact": bool(torch.equal(
            torch.from_numpy(arrays["final_parameters"]), production.parameters.cpu(),
        )),
        "final_objective_exact": trajectory["final_objective"] == production.objective,
        "final_lambda_exact": trajectory["final_lambda"] == production.final_lambda,
        "status_exact": trajectory["status"] == production.status,
        "accepted_steps_exact": trajectory["accepted_steps"] == production.counters.accepted_steps,
        "rejected_trials_exact": trajectory["rejected_trials"] == production.counters.rejected_trials,
        "linearization_count_exact": (
            trajectory["normal_equation_call_count"] == production.counters.linearization_count
        ),
        "solve_count_exact": trajectory["solve_call_count"] == production.counters.solve_attempt_count,
        "trial_history_scalar_exact": all(
            trial["outer"] == actual["outer"]
            and trial["trial"] == actual["trial"]
            and trial["lambda_before"] == actual["damping"]
            and trial["accepted"] == actual["accepted"]
            and trial.get("predicted_reduction") == actual["predicted_reduction"]
            and trial.get("actual_reduction") == actual["actual_reduction"]
            and trial.get("rho") == actual["rho"]
            for trial, actual in zip(trajectory["trials"], production.trial_history)
        ) and len(trajectory["trials"]) == len(production.trial_history),
    }
    if not all(parity.values()):
        raise RuntimeError(f"POSE_LM_TRACE_PRODUCTION_PARITY_FAILED: {parity}")
    output.mkdir(parents=True, exist_ok=True)
    np.savez(output / "pose_lm_reference.npz", **arrays)
    (output / "pose_lm_reference_trajectory.json").write_text(
        json.dumps(trajectory, indent=2, allow_nan=True) + "\n"
    )
    manifest = {
        "schema_version": "pose_lm_cpp_reference_v1", "frame": context.frame_indices[0],
        "device": str(context.device), "parameter_dtype": str(context.dtype),
        "normal_equation_dtype": "torch.float64", "delta_dtype": "torch.float64",
        "candidate_dtype": "torch.float32", "objective_scalar_dtype": "Python float",
        "lambda_scalar_dtype": "Python float", "rho_scalar_dtype": "Python float",
        "solver_config": vars(context.solver_config),
        "max_accepted_steps": pose.lm_max_accepted_steps,
        "fixture_source": "../../pose/frame_000513/cpp_fixture",
        "reference_leakage": "NONE", "production_parity": parity,
        "arrays": {name: {"shape": list(value.shape), "dtype": str(value.dtype)}
                   for name, value in arrays.items()},
    }
    (output / "pose_lm_reference_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (output / "pose_lm_reference_report.md").write_text(
        "# Pose LM Python reference\n\n"
        f"Status: `{trajectory['status']}`; accepted `{trajectory['accepted_steps']}`; "
        f"rejected `{trajectory['rejected_trials']}`; trials `{len(trajectory['trials'])}`.\n\n"
        "The validation-only trace is scalar- and tensor-exact with an independent "
        "call to the production LM runner. Objective values are reused exactly.\n"
    )
    return {"trajectory": trajectory, "manifest": manifest}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frame", type=int, default=513)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument("--hmr_dir", required=True)
    parser.add_argument("--sam3d_dense_dir", required=True)
    parser.add_argument("--bbox_dir", required=True)
    parser.add_argument("--image_width", type=int, default=5760)
    parser.add_argument("--image_height", type=int, default=2880)
    parser.add_argument("--reproj_objective", default="charbonnier_irls")
    args = parser.parse_args(argv)
    args.frames = str(args.frame); args.profile_timing = False
    args.validate_device_contract = False; args.stop_after_stage = "stage2_pose"
    output = args.output_dir or REPOSITORY_ROOT / "data_process/Vis/cpp_reference_dump/pose_lm" / f"frame_{args.frame:06d}"
    result = dump_from_context(build_context(args), output)
    print(f"POSE_LM_CPP_REFERENCE_DUMP output={output} status={result['trajectory']['status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
