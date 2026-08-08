"""Architecture and scalar-policy tests for the LM control-flow split."""

from __future__ import annotations

import ast
import importlib
import importlib.util
import inspect
from pathlib import Path
from types import SimpleNamespace

import torch

try:
    from ._bootstrap import ensure_test_paths
except ImportError:
    from _bootstrap import ensure_test_paths
ensure_test_paths(__file__)

PACKAGE = "lib_pano.human_fitting.staged_lm"
ROOT = Path(__file__).resolve().parents[3]
PRODUCTION = ROOT / "src/lib_pano/human_fitting/staged_lm"


def _module(name: str):
    return importlib.import_module(f"{PACKAGE}.{name}")


def test_modules_and_removed_legacy_facade() -> None:
    _module("lm.equations")
    trial = _module("lm.trial")
    _module("lm.convergence")
    _module("lm.runner")
    assert callable(trial.evaluate_trial_reduction)
    assert not (PRODUCTION / "lm_solver.py").exists()
    assert importlib.util.find_spec(f"{PACKAGE}.lm_solver") is None


def test_runner_signature_and_objective_reuse_source() -> None:
    runner = _module("lm.runner")
    signature = inspect.signature(runner.run_lm_optimization)
    assert list(signature.parameters) == [
        "label", "make_linearization", "raw_metrics_fn", "initial_parameters",
        "param_dim", "max_accepted_steps", "max_trials", "solver_config",
        "profile_timing", "true_objective_fn",
    ]
    source = inspect.getsource(runner.run_lm_optimization)
    assert "current_true_objective" in source
    assert source.count("true_objective_fn(") == 2
    assert source.count(".detach().item()") == 2
    tree = ast.parse(source)
    assert not any(isinstance(node, (ast.ClassDef, ast.Global, ast.Nonlocal))
                   for node in ast.walk(tree))


def test_import_direction_and_canonical_stage_runtime() -> None:
    for path in (PRODUCTION / "lm").glob("*.py"):
        source = path.read_text()
        assert "lm_solver" not in source
        assert "script.benchmark" not in source
        assert "script.validation" not in source
        tree = ast.parse(source)
        if path.name in {"equations.py", "trial.py", "convergence.py"}:
            assert not any(
                isinstance(node, ast.ImportFrom) and (node.module or "").endswith("runner")
                for node in ast.walk(tree)
            )
    runtime_source = (PRODUCTION / "stage_runtime.py").read_text()
    assert "from .lm.runner import run_lm_optimization" in runtime_source
    for path in (ROOT / "src").rglob("*.py"):
        assert "staged_lm.lm_solver import" not in path.read_text(), path


def test_legacy_no_accepted_and_convergence_policy() -> None:
    convergence = _module("lm.convergence")
    assert convergence._LAMBDA_MIN == 1e-10
    assert convergence._LAMBDA_MAX == 1e12
    assert convergence.classify_no_accepted_trial(5, 0.999e-4) == (
        "CONVERGED_SCALED_GRADIENT"
    )
    assert convergence.classify_no_accepted_trial(4, 0.0) == "LM_NO_ACCEPTED_TRIAL"
    source = inspect.getsource(convergence.check_convergence)
    assert source.index("scaled_grad_tol") < source.index("step_tol") < source.index("rel_obj_tol")


def test_damping_boundaries() -> None:
    convergence = _module("lm.convergence")
    assert convergence.update_damping_after_acceptance(3.0, 0.75) == 3.0
    assert convergence.update_damping_after_acceptance(3.0, 0.25) == 3.0
    assert convergence.update_damping_after_acceptance(3.0, 0.750001) == 1.0
    assert convergence.update_damping_after_acceptance(3.0, 0.249999) == 9.0
    assert convergence.increase_damping_after_rejection(1e12) == 1e12


def test_trial_reduction_and_record_order() -> None:
    equations_module = _module("lm.equations")
    trial = _module("lm.trial")
    types = _module("types")
    hessian = torch.eye(2, dtype=torch.float64)
    gradient = torch.tensor([1.0, -2.0], dtype=torch.float64)
    equations = types.NormalEquations(torch.zeros(2), hessian, gradient)
    delta = torch.tensor([-0.1, 0.2], dtype=torch.float64)
    predicted, actual, rho, accepted = trial.evaluate_trial_reduction(
        equations, gradient, delta, 10.0, 9.0,
    )
    expected = float(
        -gradient.dot(delta)
        - 0.5 * equations_module.normal_equations_quadratic_form(equations, delta)
    )
    assert predicted == expected and actual == 1.0
    assert rho == actual / max(predicted, 1e-12) and accepted
    record = trial.build_evaluated_trial_record(
        1, 2, 3.0, "dense", delta, predicted, actual, rho, accepted, False,
    )
    assert list(record) == [
        "outer", "trial", "damping", "linear_solver_backend", "solve_success",
        "delta_norm", "predicted_reduction", "actual_reduction", "rho",
        "accepted", "used_dense_fallback",
    ]


def test_structured_contract_validation_does_not_flatten_gradient() -> None:
    equations_module = _module("lm.equations")
    types = _module("types")
    frame_hessians = torch.eye(2, dtype=torch.float64).reshape(1, 2, 2)
    system = SimpleNamespace(
        parameter_dimension=3,
        frame_hessians=frame_hessians,
        frame_beta_cross=torch.zeros(1, 2, 1, dtype=torch.float64),
        beta_hessian=torch.ones(1, 1, dtype=torch.float64),
        frame_gradients=torch.zeros(1, 2, dtype=torch.float64),
        beta_gradient=torch.zeros(1, dtype=torch.float64),
        flat_gradient=lambda: (_ for _ in ()).throw(
            AssertionError("production validation must not flatten gradients")
        ),
    )
    equations = types.NormalEquations(
        residual=torch.zeros(2),
        hessian=None,
        gradient=torch.zeros(3, dtype=torch.float64),
        structured_system=system,
    )
    equations_module._validate_normal_equations(equations, torch.zeros(3))


if __name__ == "__main__":
    for name, value in sorted(globals().items()):
        if name.startswith("test_") and callable(value):
            value()
            print(f"{name}: PASS")
