"""Architecture checks for the Stage 2 Pose readability split."""

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


def test_pose_modules_import_and_expose_responsibilities() -> None:
    cache = _module("pose.cache")
    forward = _module("pose.forward")
    derivatives = _module("pose.derivatives")
    objective = _module("pose.objective")
    batched = _module("pose.batched")
    assert callable(cache.build_stage2_pose_fast_cache)
    assert callable(forward.fast_raw_groups)
    assert callable(derivatives.stage2_pose_explicit_raw_jacobian)
    assert callable(objective.stage2_pose_fast_true_objective)
    assert callable(batched.build_stage2_pose_batched_linearization)


def test_legacy_pose_modules_are_removed() -> None:
    for name in (
        "stage2_pose_fast",
        "stage2_pose_jacobian",
        "stage2_pose_batched",
    ):
        assert not (PRODUCTION / f"{name}.py").exists()
        assert importlib.util.find_spec(f"{PACKAGE}.{name}") is None


def test_pose_package_does_not_depend_on_facades_or_scripts() -> None:
    for path in (PRODUCTION / "pose").glob("*.py"):
        source = path.read_text()
        tree = ast.parse(source)
        imported_modules = [
            node.module or ""
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom)
        ] + [
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, ast.Import)
            for alias in node.names
        ]
        assert not any(
            module.endswith(("stage2_pose_fast", "stage2_pose_jacobian", "stage2_pose_batched"))
            or module.startswith(("script.benchmark", "script.validation", "tests"))
            for module in imported_modules
        ), path


def test_src_uses_no_pose_facade_imports() -> None:
    needles = (
        "stage2_pose_fast import",
        "stage2_pose_jacobian import",
        "stage2_pose_batched import",
    )
    for path in (ROOT / "src").rglob("*.py"):
        source = path.read_text()
        assert not any(needle in source for needle in needles), path


def test_stage_definition_uses_canonical_pose_modules() -> None:
    source = (PRODUCTION / "stage_definitions/pose.py").read_text()
    assert "from ..pose.cache import" in source
    assert "from ..pose.batched import" in source
    assert "stage2_pose_fast import" not in source


def test_public_signatures_and_linearization_provenance_source_are_stable() -> None:
    cache = _module("pose.cache")
    derivatives = _module("pose.derivatives")
    batched = _module("pose.batched")
    assert list(inspect.signature(cache.build_stage2_pose_fast_cache).parameters) == [
        "context", "shared_cache",
    ]
    assert list(inspect.signature(derivatives.stage2_pose_explicit_raw_jacobian).parameters) == [
        "cache", "parameters",
    ]
    assert list(inspect.signature(batched.build_stage2_pose_batched_linearization).parameters) == [
        "fast_cache", "batch_cache", "linearisation_point", "body_pose_anchors",
        "pose_anchor_scale", "is_irls", "charbonnier_eps", "aggregation_spec",
        "rep_scale",
    ]
    source = inspect.getsource(batched.build_stage2_pose_batched_linearization)
    assert "analytic_stage2_pose_batched" in source
    assert "direct_normal_equations_batched_full_group" in source
    assert 'linear_solver_backend="dense"' in source


if __name__ == "__main__":
    for name, value in sorted(globals().items()):
        if name.startswith("test_") and callable(value):
            value()
            print(f"{name}: PASS")
