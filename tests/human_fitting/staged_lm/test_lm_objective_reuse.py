"""Exact trajectory and lifecycle tests for local objective-value reuse."""

from __future__ import annotations

import ast
import inspect
from pathlib import Path
from unittest.mock import patch

import torch

try:
    from ._bootstrap import ensure_test_paths
except ImportError:
    from _bootstrap import ensure_test_paths
ensure_test_paths(__file__)

from lib_pano.human_fitting.staged_lm.lm import runner as lm_runner  # noqa: E402
from lib_pano.human_fitting.staged_lm.arrowhead_schur import (  # noqa: E402
    ArrowheadNormalEquationSystem,
)
from lib_pano.human_fitting.staged_lm.config import SolverConfig  # noqa: E402
from lib_pano.human_fitting.staged_lm.linearization import (  # noqa: E402
    Linearization,
    LinearizationProvenance,
)
from lib_pano.human_fitting.staged_lm.types import (  # noqa: E402
    LinearSolveResult,
    NormalEquations,
)
from script.benchmark.staged_lm.lm_early_stop_audit import (  # noqa: E402
    assert_exact_lm_parity,
    run_lm_with_convergence_trace,
)


class CountingObjective:
    def __init__(self, target: torch.Tensor):
        self.target = target
        self.calls: list[torch.Tensor] = []

    def __call__(self, parameters: torch.Tensor) -> torch.Tensor:
        self.calls.append(parameters.detach().clone())
        return 0.5 * (parameters - self.target).square().sum()


def _problem(dimension: int = 2, *, structured: bool = False):
    target = torch.linspace(0.25, 0.75, dimension)

    def residual(parameters):
        return parameters - target

    def make_linearization(_parameters):
        def equations(parameters):
            gradient = residual(parameters).to(torch.float64)
            if not structured:
                return NormalEquations(
                    residual(parameters),
                    torch.eye(dimension, dtype=torch.float64),
                    gradient,
                )
            system = ArrowheadNormalEquationSystem(
                frame_hessians=torch.ones(1, 1, 1, dtype=torch.float64),
                frame_beta_cross=torch.zeros(1, 1, 1, dtype=torch.float64),
                beta_hessian=torch.ones(1, 1, dtype=torch.float64),
                frame_gradients=gradient[:1].reshape(1, 1),
                beta_gradient=gradient[1:],
            )
            return NormalEquations(
                residual(parameters), None, system.flat_gradient(), system,
            )

        return Linearization(
            residual,
            normal_equations=equations,
            weight_stats={"stable": 1},
            provenance=LinearizationProvenance(
                linear_solver_backend=(
                    "arrowhead_schur" if structured else "dense"
                ),
            ),
        )

    def metrics(parameters):
        value = float(residual(parameters).norm())
        return {"rms": value, "median": value, "p95": value}

    return target, make_linearization, metrics


def _kwargs(
    objective, make_linearization, metrics, initial, *,
    max_steps=4, config=None,
):
    return dict(
        label="OBJECTIVE_REUSE_LM",
        make_linearization=make_linearization,
        raw_metrics_fn=metrics,
        initial_parameters=initial,
        param_dim=int(initial.numel()),
        max_accepted_steps=max_steps,
        max_trials=8,
        solver_config=config or SolverConfig(),
        profile_timing=False,
        true_objective_fn=objective,
    )


def _legacy_and_optimized(*, structured=False, initial=None, max_steps=4, config=None):
    dimension = 2
    target, make_linearization, metrics = _problem(
        dimension, structured=structured,
    )
    initial = torch.zeros(dimension) if initial is None else initial
    legacy_objective = CountingObjective(target)
    optimized_objective = CountingObjective(target)
    legacy = run_lm_with_convergence_trace(**_kwargs(
        legacy_objective, make_linearization, metrics, initial,
        max_steps=max_steps, config=config,
    ))
    optimized = lm_runner.run_lm_optimization(**_kwargs(
        optimized_objective, make_linearization, metrics, initial,
        max_steps=max_steps, config=config,
    ))
    assert_exact_lm_parity(legacy.result, optimized)
    return legacy, optimized, legacy_objective, optimized_objective


def test_dense_multi_outer_exact_and_call_formula() -> None:
    legacy, optimized, old_objective, new_objective = _legacy_and_optimized()
    successful_candidates = sum(
        item["solve_success"] for item in optimized.trial_history
    )
    assert len(new_objective.calls) == 1 + successful_candidates
    assert len(old_objective.calls) > len(new_objective.calls)
    assert torch.equal(legacy.result.parameters, optimized.parameters)
    assert legacy.result.objective == optimized.objective
    assert legacy.result.trial_history == optimized.trial_history
    assert legacy.result.weight_history == optimized.weight_history
    assert legacy.result.counters.normal_equation_count == (
        optimized.counters.normal_equation_count
    )
    assert optimized.counters.linearization_count == (
        optimized.counters.accepted_steps + 1
    )


def test_structured_arrowhead_path_is_exact() -> None:
    legacy, optimized, _, new_objective = _legacy_and_optimized(
        structured=True,
    )
    assert optimized.counters.schur_success_count > 0
    assert vars(legacy.result.counters) == vars(optimized.counters)
    assert len(new_objective.calls) == (
        1 + optimized.counters.successful_solve_count
    )


def test_first_trial_accept_and_convergence_statuses() -> None:
    for config, expected in (
        (SolverConfig(), "CONVERGED_SCALED_GRADIENT"),
        (SolverConfig(scaled_grad_tol=1e-30, step_tol=2.0),
         "CONVERGED_STEP"),
        (SolverConfig(
            scaled_grad_tol=1e-30, step_tol=1e-30, rel_obj_tol=1.0,
        ), "CONVERGED_REL_OBJECTIVE"),
    ):
        _, optimized, _, objective = _legacy_and_optimized(config=config)
        assert optimized.status == expected
        assert optimized.trial_history[0]["accepted"]
        assert len(objective.calls) == (
            1 + optimized.counters.successful_solve_count
        )


def test_reject_then_accept_does_not_pollute_current_value() -> None:
    target, make_linearization, metrics = _problem(1)
    objective = CountingObjective(target)
    deltas = iter((
        torch.tensor([-1.0], dtype=torch.float64),
        torch.tensor([0.25], dtype=torch.float64),
    ))

    def solve(_equations, _damping):
        return LinearSolveResult(next(deltas), "cholesky")

    with patch.object(lm_runner, "solve_normal_equations", solve):
        result = lm_runner.run_lm_optimization(**_kwargs(
            objective, make_linearization, metrics, torch.zeros(1),
            max_steps=1,
        ))
    assert result.status == "CONVERGED_SCALED_GRADIENT"
    assert result.counters.rejected_trials == 1
    assert result.counters.accepted_steps == 1
    assert len(objective.calls) == 3
    assert torch.equal(objective.calls[0], torch.zeros(1))
    assert torch.equal(objective.calls[1], torch.tensor([-1.0]))
    assert torch.equal(objective.calls[2], target)
    assert result.objective == 0.0


def test_all_rejected_and_no_accepted_status_reuse_initial() -> None:
    target, make_linearization, metrics = _problem()
    objective = CountingObjective(target)
    result = lm_runner.run_lm_optimization(**_kwargs(
        objective, make_linearization, metrics, target.clone(),
    ))
    assert result.status == "LM_NO_ACCEPTED_TRIAL"
    assert result.counters.accepted_steps == 0
    assert result.counters.solve_attempt_count == 8
    assert len(objective.calls) == 9
    assert result.objective == 0.0


def test_solve_failures_do_not_evaluate_candidates() -> None:
    target, make_linearization, metrics = _problem()
    objective = CountingObjective(target)
    failed = LinearSolveResult(None, "dense_solve")
    with patch.object(
        lm_runner, "solve_normal_equations", return_value=failed,
    ):
        result = lm_runner.run_lm_optimization(**_kwargs(
            objective, make_linearization, metrics, torch.zeros(2),
        ))
    assert result.status == "LM_NO_ACCEPTED_TRIAL"
    assert result.counters.solve_attempt_count == 8
    assert len(objective.calls) == 1


def test_zero_budget_evaluates_initial_once_and_returns_it() -> None:
    target, make_linearization, metrics = _problem()
    objective = CountingObjective(target)
    result = lm_runner.run_lm_optimization(**_kwargs(
        objective, make_linearization, metrics, torch.zeros(2),
        max_steps=0,
    ))
    assert result.status == "MAX_ACCEPTED_STEPS"
    assert result.counters.solve_attempt_count == 0
    assert result.counters.linearization_count == 1
    assert len(objective.calls) == 1
    assert result.objective == float(
        0.5 * target.square().sum()
    )


def test_solver_invocations_do_not_share_values() -> None:
    target, make_linearization, metrics = _problem()
    objective = CountingObjective(target)
    kwargs = _kwargs(
        objective, make_linearization, metrics, torch.zeros(2),
        max_steps=0,
    )
    lm_runner.run_lm_optimization(**kwargs)
    lm_runner.run_lm_optimization(**kwargs)
    assert len(objective.calls) == 2


def test_static_local_scalar_without_general_cache_or_extra_sync() -> None:
    path = Path(inspect.getsourcefile(lm_runner.run_lm_optimization))
    file_source = path.read_text()
    source = inspect.getsource(lm_runner.run_lm_optimization)
    tree = ast.parse(file_source)
    run_node = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "run_lm_optimization"
    )
    assigned_names = {
        target.id
        for node in ast.walk(run_node)
        if isinstance(node, (ast.Assign, ast.AnnAssign))
        for target in (
            node.targets if isinstance(node, ast.Assign) else [node.target]
        )
        if isinstance(target, ast.Name)
    }
    assert "current_true_objective" in assigned_names
    forbidden = (
        "lru_cache", "ObjectiveCache", "MemoizedObjective",
        "ObjectiveStateManager", "data_ptr", "torch.equal",
    )
    assert not any(token in source for token in forbidden)
    assert source.count("true_objective_fn(") == 2
    assert source.count(".item()") == 2
    assert not any(
        isinstance(node, (ast.ClassDef, ast.Global, ast.Nonlocal))
        for node in ast.walk(run_node)
    )


def main() -> int:
    for name, function in sorted(globals().items()):
        if name.startswith("test_") and callable(function):
            function()
    print("LM_OBJECTIVE_REUSE_TEST_PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
