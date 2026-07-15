"""Pure-math tests for dense-equivalent arrowhead Schur solves."""

from __future__ import annotations

import unittest
import sys
from pathlib import Path

import torch

try:
    from ._bootstrap import ensure_test_paths
except ImportError:
    from _bootstrap import ensure_test_paths
ensure_test_paths(__file__)

from lib_pano.human_fitting.staged_lm.arrowhead_schur import ArrowheadNormalEquationSystem
from lib_pano.human_fitting.staged_lm.lm_solver import (
    normal_equations_hessian_norm, normal_equations_max_diagonal,
    normal_equations_quadratic_form, solve_damped_normal_equations,
    solve_normal_equations,
)
from lib_pano.human_fitting.staged_lm.types import NormalEquations


def make_system(frame_count: int, p: int = 5, b: int = 2) -> ArrowheadNormalEquationSystem:
    generator = torch.Generator().manual_seed(4100 + frame_count)
    frame_h, cross, frame_g = [], [], []
    beta_h = torch.zeros(b, b, dtype=torch.float64)
    beta_g = torch.zeros(b, dtype=torch.float64)
    for frame in range(frame_count):
        rows = 9 + frame
        jf = torch.randn(rows, p, generator=generator, dtype=torch.float64)
        jb = torch.randn(rows, b, generator=generator, dtype=torch.float64)
        residual = torch.randn(rows, generator=generator, dtype=torch.float64)
        frame_h.append(jf.T @ jf)
        cross.append(jf.T @ jb)
        frame_g.append(jf.T @ residual)
        beta_h += jb.T @ jb
        beta_g += jb.T @ residual
    return ArrowheadNormalEquationSystem(
        torch.stack(frame_h), torch.stack(cross), beta_h,
        torch.stack(frame_g), beta_g,
    )


class ArrowheadSchurTests(unittest.TestCase):
    def test_dense_properties_and_damping_sweep(self) -> None:
        for frame_count in (1, 2, 4, 8):
            system = make_system(frame_count)
            dense = system.to_dense(count_call=False)
            equations = NormalEquations(
                torch.empty(0, dtype=torch.float64), None,
                system.flat_gradient(), system,
            )
            self.assertEqual(system.parameter_dimension, frame_count * 5 + 2)
            torch.testing.assert_close(system.max_diagonal(), torch.diagonal(dense).max(), rtol=0, atol=0)
            torch.testing.assert_close(system.hessian_frobenius_norm(), dense.norm(), rtol=1e-15, atol=1e-15)
            self.assertEqual(normal_equations_max_diagonal(equations), float(torch.diagonal(dense).max()))
            torch.testing.assert_close(normal_equations_hessian_norm(equations), dense.norm(), rtol=1e-15, atol=1e-15)
            delta_test = torch.randn(system.parameter_dimension, dtype=torch.float64)
            expected_q = delta_test @ dense @ delta_test
            torch.testing.assert_close(system.quadratic_form(delta_test), expected_q, rtol=1e-14, atol=1e-12)
            torch.testing.assert_close(normal_equations_quadratic_form(equations, delta_test), expected_q, rtol=1e-14, atol=1e-12)
            p = system.frame_dimension
            for first in range(frame_count):
                for second in range(frame_count):
                    if first != second:
                        block = dense[first*p:(first+1)*p, second*p:(second+1)*p]
                        self.assertEqual(float(block.abs().max()), 0.0)
            for damping in (1e-10, 1e-7, 1e-4, 1e-2, 1.0, 1e3):
                schur = system.solve_damped(damping)
                dense_delta, _, _ = solve_damped_normal_equations(
                    dense, system.flat_gradient(), damping, system.parameter_dimension,
                )
                self.assertIsNotNone(schur)
                self.assertIsNotNone(dense_delta)
                torch.testing.assert_close(schur, dense_delta, rtol=1e-10, atol=1e-10)
                pred_s = -system.flat_gradient().dot(schur) - .5 * system.quadratic_form(schur)
                pred_d = -system.flat_gradient().dot(dense_delta) - .5 * (dense_delta @ dense @ dense_delta)
                torch.testing.assert_close(pred_s, pred_d, rtol=1e-12, atol=1e-12)

    def test_normal_path_never_materializes_dense(self) -> None:
        system = make_system(4)
        equations = NormalEquations(torch.empty(0), None, system.flat_gradient(), system)
        delta, chol, fallback, schur, schur_fallback, materialized = solve_normal_equations(
            equations, 1e-3, system.parameter_dimension,
        )
        self.assertIsNotNone(delta)
        self.assertTrue(chol and schur)
        self.assertFalse(fallback or schur_fallback or materialized)
        self.assertEqual(system.to_dense_calls, 0)

    def test_schur_failure_dense_fallback_once(self) -> None:
        # Indefinite local block: Schur Cholesky fails, while sufficiently
        # damped dense Cholesky succeeds.
        system = make_system(2)
        system.frame_hessians[0, 0, 0] = -0.5
        equations = NormalEquations(torch.empty(0), None, system.flat_gradient(), system)
        result = solve_normal_equations(equations, 1.0, system.parameter_dimension)
        self.assertIsNotNone(result[0])
        self.assertTrue(result[4] and result[5])
        self.assertEqual(system.to_dense_calls, 1)

    def test_nonfinite_rejected_and_invalid_representation(self) -> None:
        system = make_system(1)
        bad_h = system.frame_hessians.clone(); bad_h[0, 0, 0] = torch.nan
        with self.assertRaisesRegex(ValueError, "NONFINITE_ARROWHEAD_SYSTEM"):
            ArrowheadNormalEquationSystem(
                bad_h, system.frame_beta_cross, system.beta_hessian,
                system.frame_gradients, system.beta_gradient,
            )
        with self.assertRaisesRegex(RuntimeError, "INVALID_NORMAL_EQUATION_REPRESENTATION"):
            normal_equations_max_diagonal(NormalEquations(torch.empty(0), None, torch.empty(0)))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_cuda_device_dtype(self) -> None:
        system = make_system(2)
        system = ArrowheadNormalEquationSystem(*[
            value.cuda() for value in (
                system.frame_hessians, system.frame_beta_cross, system.beta_hessian,
                system.frame_gradients, system.beta_gradient,
            )
        ])
        delta = system.solve_damped(1e-3)
        self.assertEqual(delta.device.type, "cuda")
        self.assertEqual(delta.dtype, torch.float64)


if __name__ == "__main__":
    unittest.main()
