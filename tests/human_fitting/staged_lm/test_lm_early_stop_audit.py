"""Synthetic contracts for the read-only LM convergence audit."""

from __future__ import annotations

import json
from dataclasses import FrozenInstanceError
from pathlib import Path

import numpy as np
import torch

try:
    from ._bootstrap import ensure_test_paths
except ImportError:
    from _bootstrap import ensure_test_paths
ensure_test_paths(__file__)

from lib_pano.human_fitting.staged_lm.arrowhead_schur import (  # noqa: E402
    ArrowheadNormalEquationSystem,
)
from lib_pano.human_fitting.staged_lm.config import SolverConfig  # noqa: E402
from lib_pano.human_fitting.staged_lm.linearization import (  # noqa: E402
    Linearization,
    LinearizationProvenance,
)
from lib_pano.human_fitting.staged_lm.lm.runner import (  # noqa: E402
    run_lm_optimization,
)
from lib_pano.human_fitting.staged_lm.types import (  # noqa: E402
    NormalEquations,
    StageDefinition,
)
from script.benchmark.staged_lm.lm_early_stop_audit import (  # noqa: E402
    assert_exact_lm_parity,
    duplicate_objective_proof,
    objective_call_attribution,
    replace_accepted_budget,
    replace_initial_parameters,
    run_lm_with_convergence_trace,
    split_step_norms,
    static_solver_audit,
    trace_to_json,
)


def _dense_problem():
    target = torch.tensor([1.5, -0.5], dtype=torch.float32)

    def make_linearization(_x):
        def residual(x):
            return x - target

        def equations(x):
            residual_value = residual(x)
            return NormalEquations(
                residual_value,
                torch.eye(2, dtype=torch.float64),
                residual_value.to(torch.float64),
            )

        return Linearization(
            residual, normal_equations=equations,
            weight_stats={"weight": 1.0},
        )

    def objective(x):
        return 0.5 * (x - target).square().sum()

    def metrics(x):
        value = float((x - target).norm())
        return {"rms": value, "median": value, "p95": value}

    return make_linearization, objective, metrics


def _run_pair(max_steps=4):
    make_linearization, objective, metrics = _dense_problem()
    kwargs = dict(
        label="SYNTHETIC_LM", make_linearization=make_linearization,
        raw_metrics_fn=metrics, initial_parameters=torch.zeros(2),
        param_dim=2, max_accepted_steps=max_steps, max_trials=8,
        solver_config=SolverConfig(), profile_timing=False,
        true_objective_fn=objective,
    )
    production = run_lm_optimization(**kwargs)
    trace = run_lm_with_convergence_trace(**kwargs)
    return production, trace


def test_dense_trace_exact_parity_and_objective_roles() -> None:
    production, trace = _run_pair()
    assert_exact_lm_parity(production, trace.result)
    counts = objective_call_attribution(trace)
    assert counts["current_total_calls"] == len(trace.objective_calls)
    assert counts["minimal_exact_objective_calls"] == (
        1 + counts["successful_solve_trial_calls"]
    )
    proof = duplicate_objective_proof(trace)
    assert proof["status"] == "EXACT_OBJECTIVE_REUSE_PROVEN"
    assert all(item["parameters_equal"] for item in proof["comparisons"])


def test_structured_arrowhead_smoke() -> None:
    target = torch.tensor([0.4, -0.2], dtype=torch.float32)

    def make_linearization(_x):
        def residual(x):
            return x - target

        def equations(x):
            gradient = (x - target).to(torch.float64)
            system = ArrowheadNormalEquationSystem(
                frame_hessians=torch.ones(1, 1, 1, dtype=torch.float64),
                frame_beta_cross=torch.zeros(1, 1, 1, dtype=torch.float64),
                beta_hessian=torch.ones(1, 1, dtype=torch.float64),
                frame_gradients=gradient[:1].reshape(1, 1),
                beta_gradient=gradient[1:],
            )
            return NormalEquations(
                residual(x), None, system.flat_gradient(), system,
            )

        return Linearization(
            residual, normal_equations=equations,
            provenance=LinearizationProvenance(
                linear_solver_backend="arrowhead_schur",
            ),
        )

    kwargs = dict(
        label="ARROWHEAD_LM", make_linearization=make_linearization,
        raw_metrics_fn=lambda x: {
            "rms": float((x - target).norm()), "median": 0.0, "p95": 0.0,
        },
        initial_parameters=torch.zeros(2), param_dim=2,
        max_accepted_steps=3, max_trials=8,
        solver_config=SolverConfig(), profile_timing=False,
        true_objective_fn=lambda x: 0.5 * (x - target).square().sum(),
    )
    production = run_lm_optimization(**kwargs)
    trace = run_lm_with_convergence_trace(**kwargs)
    assert_exact_lm_parity(production, trace.result)
    assert trace.result.counters.schur_success_count > 0


def test_static_behavior_audit() -> None:
    audit = static_solver_audit()
    assert not audit["initial_convergence_check_present"]
    assert audit["post_accept_convergence_check_present"]
    assert not audit["no_accepted_branch_uses_solver_config"]
    assert audit["no_accepted_branch_min_accepted_steps"] == 5
    assert audit["no_accepted_branch_scaled_grad_threshold"] == 1e-4
    assert audit["convergence_predicate_order"] == [
        "scaled_gradient", "step_norm", "relative_objective",
    ]
    assert audit["final_budget_step_relinearized"]


def test_converged_initial_point_enters_trials_and_is_misclassified() -> None:
    make_linearization, objective, metrics = _dense_problem()
    target = torch.tensor([1.5, -0.5], dtype=torch.float32)
    trace = run_lm_with_convergence_trace(
        label="INITIAL_FINAL_LM",
        make_linearization=make_linearization,
        raw_metrics_fn=metrics,
        initial_parameters=target,
        param_dim=2,
        max_accepted_steps=4,
        max_trials=8,
        solver_config=SolverConfig(),
        profile_timing=False,
        true_objective_fn=objective,
    )
    assert trace.initial_diagnostics["initial_gradient_pass_config"]
    assert trace.result.counters.solve_attempt_count == 8
    assert trace.result.counters.accepted_steps == 0
    assert trace.result.status == "LM_NO_ACCEPTED_TRIAL"
    branch = trace.no_accepted_diagnostics[0]
    assert branch["config_gradient_pass"]
    assert not branch["min_accepted_count_pass"]
    assert branch["hypothetical_config_consistent_status"] == (
        "CONVERGED_SCALED_GRADIENT"
    )


def test_step_norm_block_decomposition() -> None:
    stage1 = split_step_norms(
        "S1", torch.arange(12, dtype=torch.float32),
        torch.zeros(12), frame_count=1,
    )
    assert "orient_norm_total" in stage1
    assert "translation_norm_total" in stage1
    pose = split_step_norms(
        "STAGE2_POSE", torch.ones(75), torch.ones(75), frame_count=1,
    )
    assert pose["pose_norm"] > pose["rigid_norm"]
    joint = split_step_norms(
        "STAGE2_JOINT", torch.ones(85), torch.ones(85), frame_count=1,
        beta_dim=10,
    )
    assert joint["beta_latent_norm"] == float(torch.ones(10).norm())
    assert joint["physical_beta_delta_norm"] > 0


def test_reinitialization_and_extended_budget_are_immutable() -> None:
    make_linearization, objective, metrics = _dense_problem()
    original = StageDefinition(
        "S1", 2, torch.zeros(2), lambda _x: [],
        None, metrics, 2, linearization_builder=lambda x, _spec:
        make_linearization(x), lm_true_objective_fn=objective,
    )
    reinitialized = replace_initial_parameters(original, torch.ones(2))
    extended = replace_accepted_budget(original, 5)
    assert torch.equal(original.initial_lm_parameters, torch.zeros(2))
    assert original.lm_max_accepted_steps == 2
    assert reinitialized is not original
    assert extended.lm_max_accepted_steps == 5


def test_extended_budget_prefix_and_serialization(tmp_path=None) -> None:
    _, short = _run_pair(2)
    _, extended = _run_pair(4)
    assert extended.result.trial_history[:len(short.result.trial_history)] == (
        short.result.trial_history
    )
    payload = trace_to_json(extended)
    json.dumps(payload, allow_nan=False)
    arrays = {
        key: value.detach().cpu().numpy()
        for key, value in extended.checkpoints.items()
    }
    assert arrays
    assert all(value.dtype != object for value in arrays.values())


def test_audit_does_not_modify_production_sources() -> None:
    audit_path = Path(__file__).resolve().parents[3] / (
        "src/script/benchmark/staged_lm/lm_early_stop_audit.py"
    )
    text = audit_path.read_text()
    assert "write_text" not in text
    assert "apply_patch" not in text


def main() -> int:
    for name, function in sorted(globals().items()):
        if name.startswith("test_") and callable(function):
            function()
    print("LM_EARLY_STOP_AUDIT_TEST_PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
