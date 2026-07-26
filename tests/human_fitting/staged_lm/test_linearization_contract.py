"""Tests for the typed stage-to-solver linearization contract."""

from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

try:
    from ._bootstrap import ensure_test_paths
except ImportError:
    from _bootstrap import ensure_test_paths

ensure_test_paths(__file__)

from lib_pano.human_fitting.staged_lm.analytic_stage1 import (
    build_stage1_analytic_linearization,
)
from lib_pano.human_fitting.staged_lm.linearization import (
    Linearization,
    LinearizationProvenance,
)
from lib_pano.human_fitting.staged_lm.lm.equations import (
    _evaluate_linearization,
    compute_jacobian,
)
from lib_pano.human_fitting.staged_lm.joint import (
    build_stage2_joint_analytic_linearization,
)
from lib_pano.human_fitting.staged_lm.pose.derivatives import (
    build_stage2_pose_analytic_linearization,
)
from lib_pano.human_fitting.staged_lm.types import NormalEquations


def _normal_equations(x: torch.Tensor) -> NormalEquations:
    dimension = x.numel()
    return NormalEquations(
        residual=x.new_empty(0),
        hessian=torch.eye(dimension, dtype=torch.float64, device=x.device),
        gradient=torch.zeros(dimension, dtype=torch.float64, device=x.device),
    )


class LinearizationContractTests(unittest.TestCase):
    def test_surrogate_only_and_default_provenance(self) -> None:
        linearization = Linearization(surrogate=lambda x: x.square())
        self.assertFalse(linearization.has_analytic_jacobian)
        self.assertFalse(linearization.uses_direct_normal_equations)
        self.assertEqual(
            linearization.provenance,
            LinearizationProvenance(),
        )

    def test_analytic_jacobian_path(self) -> None:
        linearization = Linearization(
            surrogate=lambda x: x.square(),
            jacobian=lambda x: 2.0 * torch.diag(x),
        )
        x = torch.tensor([2.0, 3.0])
        self.assertTrue(linearization.has_analytic_jacobian)
        self.assertTrue(torch.equal(
            compute_jacobian(linearization, x),
            2.0 * torch.diag(x),
        ))

    def test_direct_normal_equation_path(self) -> None:
        x = torch.tensor([0.0, 0.0])
        linearization = Linearization(
            surrogate=lambda value: value,
            normal_equations=_normal_equations,
        )
        equations, direct = _evaluate_linearization(linearization, x)
        self.assertTrue(direct)
        self.assertTrue(torch.equal(
            equations.hessian,
            torch.eye(2, dtype=torch.float64),
        ))

    def test_frozen_and_replace(self) -> None:
        linearization = Linearization(surrogate=lambda x: x)
        with self.assertRaises(FrozenInstanceError):
            linearization.jacobian = lambda x: x
        changed = replace(
            linearization,
            normal_equations=_normal_equations,
            provenance=replace(
                linearization.provenance,
                linear_solver_backend="arrowhead_schur",
            ),
        )
        self.assertTrue(changed.uses_direct_normal_equations)
        self.assertEqual(
            changed.provenance.linear_solver_backend,
            "arrowhead_schur",
        )

    def test_invalid_callables_are_rejected(self) -> None:
        with self.assertRaises(TypeError):
            Linearization(surrogate=3)
        with self.assertRaises(TypeError):
            Linearization(surrogate=lambda x: x, jacobian=3)
        with self.assertRaises(TypeError):
            Linearization(surrogate=lambda x: x, normal_equations=3)

    def test_contract_has_no_mapping_interface(self) -> None:
        linearization = Linearization(surrogate=lambda x: x)
        for attribute in ("__getitem__", "get", "keys"):
            self.assertFalse(hasattr(linearization, attribute))

    def test_stage1_builder_returns_contract(self) -> None:
        cache = SimpleNamespace(groups=[])
        result = build_stage1_analytic_linearization(
            cache, torch.zeros(6), is_irls=False, charbonnier_eps=1e-3,
            aggregation_spec=None, rep_scale=1.0,
        )
        self.assertIsInstance(result, Linearization)

    def test_pose_builder_returns_contract(self) -> None:
        common = SimpleNamespace(
            surrogate_fn=lambda x: x,
            jacobian_fn=lambda x: torch.eye(x.numel()),
            row_scales_by_group=[],
            weight_stats={},
            spec={},
        )
        with patch(
            "lib_pano.human_fitting.staged_lm.pose.derivatives."
            "build_stage2_pose_linearization_common",
            return_value=common,
        ):
            result = build_stage2_pose_analytic_linearization(
                object(), torch.zeros(2), [], 1.0, is_irls=False,
                charbonnier_eps=1e-3, aggregation_spec=None, rep_scale=1.0,
            )
        self.assertIsInstance(result, Linearization)

    def test_joint_builder_returns_contract(self) -> None:
        common = SimpleNamespace(
            surrogate_fn=lambda x: x,
            jacobian_fn=lambda x: torch.eye(x.numel()),
            row_scales_by_group=[],
            weight_stats={},
            spec={},
        )
        with patch(
            "lib_pano.human_fitting.staged_lm.joint.arrowhead."
            "build_stage2_joint_linearization_common",
            return_value=common,
        ):
            result = build_stage2_joint_analytic_linearization(
                object(), torch.zeros(2), 2.5, torch.zeros(1), [], 1.0, 1.0,
                is_irls=False, charbonnier_eps=1e-3,
                aggregation_spec=None, rep_scale=1.0,
            )
        self.assertIsInstance(result, Linearization)


if __name__ == "__main__":
    unittest.main()
