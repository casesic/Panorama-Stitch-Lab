"""Unified command-line entry point for staged-LM validation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from script.validation.staged_lm.common import add_data_arguments, add_profile_argument
from script.validation.staged_lm.pose_shape import (
    run_all_joint_jacobian_validations,
    run_joint_analytic_validation,
    run_joint_beta_validation,
    run_joint_production_shadow,
)
from script.validation.staged_lm.objective_breakdown import (
    run_objective_breakdown_validation,
)
from script.validation.staged_lm.pose import (
    run_all_pose_validations,
    run_pose_forward_validation,
    run_pose_jacobian_validation,
    run_pose_production_regression,
    run_pose_solver_regression,
)
from script.validation.staged_lm.solver import (
    run_all_solver_regressions,
    run_combined_fk_regression,
    run_dense_schur_regression,
    run_full_handoff_validation,
    run_joint_backend_regression,
)
from script.validation.staged_lm.stage1 import run_all_stage1_validations, run_stage1_batch_shadow


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Staged-LM validation")
    subparsers = parser.add_subparsers(dest="command", required=True)
    commands = (
        "stage1", "stage1-batch-shadow", "pose-forward", "pose-jacobian", "pose-solver", "pose-production",
        "joint-beta", "joint-jacobian", "joint-production-shadow", "joint-solver", "joint-schur",
        "joint-fk", "handoff", "objective-breakdown", "all",
    )
    for command in commands:
        current = subparsers.add_parser(command)
        add_data_arguments(current)
        if command != "stage1":
            add_profile_argument(current)
        if command in ("pose-forward", "pose-jacobian", "all"):
            current.add_argument("--stage2_pose_npz", default=None)
            current.add_argument("--check_intermediates", action="store_true")
        if command in ("joint-beta", "joint-jacobian", "all"):
            current.add_argument("--joint_npz", default=None)
            if not any(action.dest == "check_intermediates" for action in current._actions):
                current.add_argument("--check_intermediates", action="store_true")
        if command in ("joint-schur", "all"):
            current.add_argument("--scaling_only", action="store_true")
        if command in ("joint-fk", "all"):
            current.add_argument("--run_lbfgs_oracle", action="store_true")
    return parser


def main() -> int:
    args = _parser().parse_args()
    handlers = {
        "stage1": run_all_stage1_validations,
        "stage1-batch-shadow": run_stage1_batch_shadow,
        "pose-forward": run_pose_forward_validation,
        "pose-jacobian": run_pose_jacobian_validation,
        "pose-solver": run_pose_solver_regression,
        "pose-production": run_pose_production_regression,
        "joint-beta": run_joint_beta_validation,
        "joint-jacobian": run_joint_analytic_validation,
        "joint-production-shadow": run_joint_production_shadow,
        "joint-solver": run_joint_backend_regression,
        "joint-schur": run_dense_schur_regression,
        "joint-fk": run_combined_fk_regression,
        "handoff": run_full_handoff_validation,
        "objective-breakdown": run_objective_breakdown_validation,
    }
    if args.command == "all":
        run_all_stage1_validations(args)
        run_all_pose_validations(args)
        run_all_joint_jacobian_validations(args)
        run_all_solver_regressions(args)
    else:
        handlers[args.command](args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
