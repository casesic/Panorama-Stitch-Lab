"""Architecture checks for the Stage 2 pose/shape readability split."""

from __future__ import annotations

import ast
import importlib
import importlib.util
import inspect
from pathlib import Path

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


def test_joint_modules_import_and_own_expected_entrypoints() -> None:
    expected = {
        "pose_shape.cache": "build_stage2_pose_shape_fast_cache",
        "pose_shape.shape": "evaluate_pose_shape_state",
        "pose_shape.forward": "pose_shape_fast_raw_groups",
        "pose_shape.beta_derivatives": "pose_shape_group_raw_residual_and_beta_jacobian",
        "pose_shape.derivatives": "pose_shape_group_raw_residual_and_jacobians",
        "pose_shape.objective": "stage2_pose_shape_fast_true_objective",
        "pose_shape.arrowhead": "stage2_pose_shape_arrowhead_normal_equations",
        "pose_shape.batched": "build_stage2_pose_shape_batched_linearization",
    }
    for module_name, symbol in expected.items():
        assert callable(getattr(_module(module_name), symbol))


def test_legacy_joint_modules_are_removed() -> None:
    for name in (
        "stage2_pose_shape_fast",
        "stage2_pose_shape_beta_jacobian",
        "stage2_pose_shape_jacobian",
        "stage2_pose_shape_batched",
    ):
        assert not (PRODUCTION / f"{name}.py").exists()
        assert importlib.util.find_spec(f"{PACKAGE}.{name}") is None


def test_joint_package_has_no_facade_or_script_dependency() -> None:
    forbidden_suffixes = tuple(f"stage2_pose_shape_{name}" for name in (
        "fast", "beta_jacobian", "jacobian", "batched",
    ))
    for path in (PRODUCTION / "pose_shape").glob("*.py"):
        tree = ast.parse(path.read_text())
        modules = [
            node.module or "" for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)
        ] + [
            alias.name for node in ast.walk(tree) if isinstance(node, ast.Import)
            for alias in node.names
        ]
        assert not any(
            module.endswith(forbidden_suffixes)
            or module.startswith(("script.", "tests"))
            for module in modules
        ), path


def test_src_has_no_legacy_joint_facade_imports() -> None:
    needles = tuple(f"stage2_pose_shape_{name} import" for name in (
        "fast", "beta_jacobian", "jacobian", "batched",
    ))
    for path in (ROOT / "src").rglob("*.py"):
        assert not any(needle in path.read_text() for needle in needles), path


def test_stage_definition_and_solver_boundaries_are_canonical() -> None:
    source = (PRODUCTION / "stage_definitions/pose_shape.py").read_text()
    assert "from ..pose_shape.cache import" in source
    assert "from ..pose_shape.objective import" in source
    assert "from ..pose_shape.batched import" in source
    assert "stage2_pose_shape_fast import" not in source
    arrowhead = _module("pose_shape.arrowhead")
    assert arrowhead.ArrowheadNormalEquationSystem is _module(
        "arrowhead_schur"
    ).ArrowheadNormalEquationSystem
    batched_source = inspect.getsource(
        _module("pose_shape.batched").build_stage2_pose_shape_batched_linearization
    )
    assert "analytic_stage2_pose_shape_batched" in batched_source
    assert "direct_arrowhead_normal_equations_batched_full_group" in batched_source
    assert 'linear_solver_backend="arrowhead_schur"' in batched_source


if __name__ == "__main__":
    for name, value in sorted(globals().items()):
        if name.startswith("test_") and callable(value):
            value()
            print(f"{name}: PASS")
