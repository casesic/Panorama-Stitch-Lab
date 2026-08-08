"""Architecture contracts for the staged-LM stage module split."""

from __future__ import annotations

import ast
import importlib
import importlib.util
import inspect
from pathlib import Path
from types import SimpleNamespace

try:
    from ._bootstrap import ensure_test_paths
except ImportError:
    from _bootstrap import ensure_test_paths
ensure_test_paths(__file__)

from lib_pano.human_fitting.staged_lm import (  # noqa: E402
    pipeline,
    stage_handoff,
    stage_runtime,
    stage_timing,
)
from lib_pano.human_fitting.staged_lm.stage_definitions import (  # noqa: E402
    joint,
    pose,
    rigid,
)
ROOT = Path(stage_runtime.__file__).resolve().parent
SRC_ROOT = ROOT.parents[2]


def _tree(module) -> ast.Module:
    path = Path(module.__file__)
    return ast.parse(path.read_text(), filename=str(path))


def test_legacy_stage_module_is_removed() -> None:
    assert not (ROOT / "stages.py").exists()
    assert importlib.util.find_spec(
        "lib_pano.human_fitting.staged_lm.stages"
    ) is None


def test_module_responsibilities_and_exports() -> None:
    assert callable(stage_runtime.run_stage)
    assert callable(rigid.build_stage1_definition)
    assert callable(pose.build_pose_stage_definition)
    assert callable(joint.build_joint_stage_definition)
    for name in (
        "check_s1_to_pose", "handoff_s1_to_pose", "check_pose_to_joint_state",
    ):
        assert callable(getattr(stage_handoff, name))
    assert callable(stage_timing.record_preparation_timing)
    assert callable(stage_timing.record_stage2_cache_totals)
    assert callable(stage_timing.build_stage_run_profile)
    top_level_pipeline_functions = [
        node.name for node in _tree(pipeline).body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]
    assert top_level_pipeline_functions == ["run_stage2"]


def test_stage_run_profile_serialization() -> None:
    raw = {
        "solver_jacobian_ms": 2.0,
        "objective_eval_ms": 3.0,
        "linear_solve_ms": 5.0,
        "final_metrics_ms": 7.0,
    }
    stage = SimpleNamespace(
        eval_backend="analytic",
        linearization_backend="direct_normal_equations",
        linear_solver_backend="dense",
    )
    result = stage_timing.build_stage_run_profile(raw, 20.0, stage)
    assert result == {
        "units": "ms",
        "normal_equations_ms": 2.0,
        "true_objective_ms": 3.0,
        "linear_solve_ms": 5.0,
        "final_metrics_ms": 7.0,
        "other_runner_ms": 3.0,
        "accounting": {
            "approximately_sums_to": "wall_time_ms",
            "exclusive_buckets": [
                "normal_equations_ms", "true_objective_ms",
                "linear_solve_ms", "final_metrics_ms", "other_runner_ms",
            ],
            "other_runner_includes": [
                "linearization_construction", "python_overhead",
            ],
        },
        "eval_backend": "analytic",
        "linearization_backend": "direct_normal_equations",
        "linear_solver_backend": "dense",
    }


def test_builder_signatures_and_return_contracts() -> None:
    builders = (
        rigid.build_stage1_definition,
        pose.build_pose_stage_definition,
        joint.build_joint_stage_definition,
    )
    for builder in builders:
        assert not any(
            "backend" in name for name in inspect.signature(builder).parameters
        )
    assert str(inspect.signature(
        rigid.build_stage1_definition,
    ).return_annotation) == "StageDefinition"
    for builder in (pose.build_pose_stage_definition, joint.build_joint_stage_definition):
        annotation = inspect.signature(builder).return_annotation
        assert "tuple[StageDefinition, dict[str, Any]]" in str(annotation)


def test_inner_linearization_builder_annotations() -> None:
    for module in (rigid, pose, joint):
        matches = [
            node for node in ast.walk(_tree(module))
            if isinstance(node, ast.FunctionDef)
            and node.name == "linearization_builder"
        ]
        assert len(matches) == 1
        assert isinstance(matches[0].returns, ast.Name)
        assert matches[0].returns.id == "Linearization"


def test_production_uses_canonical_imports() -> None:
    app = importlib.import_module("script.human_fitting.staged_lm.app")
    app_source = Path(app.__file__).read_text()
    assert "staged_lm.pipeline import run_stage2" in app_source
    assert "staged_lm.stage_runtime import run_stage" in app_source
    assert "staged_lm.stage_definitions.rigid import" in app_source
    violations = []
    for path in SRC_ROOT.rglob("*.py"):
        text = path.read_text()
        if "human_fitting.staged_lm.stages import" in text:
            violations.append(path)
    assert not violations, violations


def test_import_graph_has_no_cycle() -> None:
    names = (
        "stage_runtime", "stage_handoff", "stage_timing", "pipeline",
        "stage_definitions.common", "stage_definitions.rigid",
        "stage_definitions.pose", "stage_definitions.joint",
    )
    for name in names:
        importlib.import_module(f"lib_pano.human_fitting.staged_lm.{name}")


def main() -> int:
    for name, function in sorted(globals().items()):
        if name.startswith("test_") and callable(function):
            function()
    print("STAGED_LM_STAGE_MODULE_SPLIT_PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
