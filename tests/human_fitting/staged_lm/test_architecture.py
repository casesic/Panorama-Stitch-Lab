"""Import architecture and fixed production-path tests."""

from __future__ import annotations

import ast
import importlib
import inspect
from dataclasses import fields
from pathlib import Path

try:
    from ._bootstrap import ensure_test_paths
except ImportError:
    from _bootstrap import ensure_test_paths
ensure_test_paths(__file__)

from lib_pano.human_fitting.staged_lm import config, pipeline, stage_handoff
from lib_pano.human_fitting.staged_lm.stage_definitions import joint, pose, rigid
from lib_pano.human_fitting.staged_lm.joint import (
    build_stage2_joint_analytic_linearization,
    compute_joint_fk_with_combined_jacobian,
)
from lib_pano.human_fitting.staged_lm.types import StageDefinition
from script.human_fitting.staged_lm.cli import build_argument_parser
from script.validation.staged_lm.references import gate_c, joint_backends, lbfgs, oracles
from tests.human_fitting.staged_lm.helpers import _cache, _parameters


PRODUCTION_MODULES = [
    "lib_pano.human_fitting.staged_lm",
    *[
        f"lib_pano.human_fitting.staged_lm.{name}"
        for name in (
            "config", "types", "device", "data_loading", "parameter_layout",
            "residuals", "irls", "linearization",
            "lm", "lm.equations", "lm.trial", "lm.convergence", "lm.runner",
            "geometry", "geometry.so3", "geometry.erp", "arrowhead_schur",
            "analytic_stage1", "stage2_shared_cache",
            "pose", "pose.cache", "pose.forward", "pose.derivatives",
            "pose.objective", "pose.batched",
            "joint", "joint.cache", "joint.shape", "joint.forward",
            "joint.beta_derivatives", "joint.derivatives", "joint.objective",
            "joint.arrowhead", "joint.batched",
            "stage_runtime", "stage_handoff", "stage_timing", "pipeline",
            "stage_definitions", "stage_definitions.common",
            "stage_definitions.rigid", "stage_definitions.pose",
            "stage_definitions.joint", "output",
        )
    ],
]
REMOVED_OPTIONS = {
    "--jacobian_backend", "--stage1_eval_backend",
    "--stage2_pose_linearization_backend", "--stage2_pose_eval_backend",
    "--stage2_joint_linearization_backend", "--stage2_joint_eval_backend",
    "--stage2_joint_linear_solver", "--stage2_joint_fk_jacobian_backend",
    "--run_lbfgs_oracle", "--validate_stage1_fast_eval",
    "--validate_stage2_pose_fast_eval", "--validate_stage2_joint_fast_eval",
    "--validate_irls_first_order", "--no_validate_irls_first_order",
}


def _production_root() -> Path:
    return Path(pipeline.__file__).resolve().parent


def test_production_and_benchmark_import_locations() -> None:
    root = _production_root()
    for name in PRODUCTION_MODULES:
        module = importlib.import_module(name)
        assert module.__name__ == name
        assert Path(module.__file__).resolve().is_relative_to(root)
    benchmark = importlib.import_module(
        "script.benchmark.staged_lm.joint_kernel_candidates"
    )
    expected = Path(__file__).resolve().parents[3] / "src/script/benchmark/staged_lm"
    assert Path(benchmark.__file__).resolve().is_relative_to(expected.resolve())


def test_production_has_no_script_or_benchmark_dependency() -> None:
    violations = []
    for path in _production_root().rglob("*.py"):
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            modules = []
            if isinstance(node, ast.Import):
                modules.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                modules.append(node.module)
            for module in modules:
                if module == "script" or module.startswith("script."):
                    violations.append((path, module))
    assert not violations, violations


def test_validation_reference_imports() -> None:
    names = (
        "oracles", "joint_backends", "lbfgs", "gate_c", "types",
        "stage_backends",
    )
    root = Path(__file__).resolve().parents[3] / "src/script/validation/staged_lm/references"
    for name in names:
        module = importlib.import_module(
            f"script.validation.staged_lm.references.{name}"
        )
        assert Path(module.__file__).resolve().is_relative_to(root.resolve())


def test_cli_has_no_implementation_selector() -> None:
    options = {
        option
        for action in build_argument_parser()._actions
        for option in action.option_strings
    }
    assert not options.intersection(REMOVED_OPTIONS)
    for required in ("--device", "--frames", "--reproj_objective", "--profile_timing"):
        assert required in options


def test_production_builder_signatures_and_labels() -> None:
    for function in (
        rigid.build_stage1_definition,
        pose.build_pose_stage_definition,
        joint.build_joint_stage_definition,
    ):
        assert not any("backend" in name for name in inspect.signature(function).parameters)
    assert config.PRODUCTION_STAGE1_LINEARIZATION == "analytic_stage1"
    assert config.PRODUCTION_STAGE1_EVALUATION == "analytic"
    assert config.PRODUCTION_POSE_LINEARIZATION == "analytic_batched_full_group"
    assert config.PRODUCTION_POSE_EVALUATION == "fast"
    assert config.PRODUCTION_JOINT_LINEARIZATION == "analytic_batched_full_group"
    assert config.PRODUCTION_JOINT_EVALUATION == "fast"
    assert config.PRODUCTION_JOINT_LINEAR_SOLVER == "arrowhead_schur"
    assert config.PRODUCTION_JOINT_FK == "combined_batched_depth"
    assert "argparse" not in Path(config.__file__).read_text()
    validation_only = {
        "initial_lbfgs_parameters", "lbfgs_outer_steps",
        "analytic_linearization", "lbfgs_true_objective_fn",
        "fast_eval_validator", "full_forward_oracle_validation_ms",
    }
    assert not ({field.name for field in fields(StageDefinition)} & validation_only)


def test_production_has_no_reference_selector() -> None:
    stage_source = "\n".join(
        inspect.getsource(module) for module in (rigid, pose, joint, pipeline)
    )
    joint_module = importlib.import_module(
        "lib_pano.human_fitting.staged_lm.joint.derivatives"
    )
    joint_source = inspect.getsource(joint_module)
    assert "context.args.jacobian_backend" not in stage_source
    assert "stage2_joint_linear_solver" not in stage_source
    assert "compute_joint_fk_with_pose_and_beta_jacobians(" not in joint_source
    assert "compute_joint_fk_with_combined_jacobian(" in joint_source


def test_production_preparation_has_no_full_forward_fallback() -> None:
    stage_source = "\n".join(
        inspect.getsource(module) for module in (rigid, pose, joint, pipeline)
    )
    app_module = importlib.import_module("script.human_fitting.staged_lm.app")
    output_module = importlib.import_module(
        "lib_pano.human_fitting.staged_lm.output"
    )
    assert "_make_full_true_objective" not in stage_source
    assert "sample_groups =" not in stage_source
    assert "build_primary_aggregation_spec(" not in stage_source
    handoff_source = inspect.getsource(stage_handoff.check_pose_to_joint_state)
    assert "raw_groups_fn" not in handoff_source
    assert "make_raw_reproj_groups" not in handoff_source
    assert "true_objective" not in handoff_source
    assert "raw_groups_fn" not in inspect.getsource(app_module._stage_json)
    breakdown_source = inspect.getsource(output_module.compute_objective_breakdown)
    assert "raw_groups_fn" not in breakdown_source
    assert "true_objective_fn" not in breakdown_source


def test_joint_production_linearization_is_structured() -> None:
    from lib_pano.human_fitting.staged_lm.irls import build_primary_aggregation_spec
    from lib_pano.human_fitting.staged_lm.joint import joint_fast_raw_groups

    cache, parameters = _cache(), _parameters()
    groups = joint_fast_raw_groups(cache, parameters, 3.0)
    linearization = build_stage2_joint_analytic_linearization(
        cache, parameters, 3.0, parameters.new_zeros(2),
        [parameters.new_zeros(63)], 0.2, 0.4,
        is_irls=True, charbonnier_eps=1.0,
        aggregation_spec=build_primary_aggregation_spec(groups), rep_scale=0.02,
    )
    equations = linearization.normal_equations(parameters)
    assert equations.hessian is None
    assert equations.structured_system is not None
    assert linearization.provenance.linear_solver_backend == "arrowhead_schur"


def test_reference_capabilities_are_importable() -> None:
    assert callable(oracles.make_irls_linearization)
    assert callable(oracles.build_full_true_objective)
    assert callable(joint_backends.compute_joint_fk_with_pose_and_beta_jacobians_separate_reference)
    assert callable(joint_backends.stage2_joint_normal_equations_reference)
    assert callable(lbfgs.run_lbfgs_oracle)
    assert callable(gate_c.evaluate_gate_c)
    assert callable(compute_joint_fk_with_combined_jacobian)


def main() -> int:
    for name, function in sorted(globals().items()):
        if name.startswith("test_") and callable(function):
            function()
    print("STAGED_LM_ARCHITECTURE_PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
