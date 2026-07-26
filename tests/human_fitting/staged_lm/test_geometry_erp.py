"""Unit tests for the shared staged-LM ERP projection kernels."""

from __future__ import annotations

import unittest

import torch

try:
    from ._bootstrap import ensure_test_paths
except ImportError:
    from _bootstrap import ensure_test_paths

ensure_test_paths(__file__)

from lib_pano.human_fitting.projection import ROTATION_MATRIX
from lib_pano.human_fitting.staged_lm.geometry.erp import (
    erp_projection,
    erp_projection_with_jacobian,
    project_cam2_erp,
    project_cam2_erp_with_jacobian,
    project_cam2_with_jacobian,
    project_unified_erp,
    project_unified_erp_with_jacobian,
)


class GeometryERPTests(unittest.TestCase):
    def setUp(self) -> None:
        self.points = torch.tensor(
            [[0.7, -0.2, 1.4], [-0.5, 0.3, 2.0], [1.2, 0.4, -0.8]],
            dtype=torch.float64,
        )
        self.width = 5760
        self.height = 2880

    def test_uv_and_jacobian_shapes(self) -> None:
        uv, jacobian = erp_projection_with_jacobian(
            self.points, self.width, self.height,
        )
        self.assertEqual(uv.shape, (3, 2))
        self.assertEqual(jacobian.shape, (3, 2, 3))

    def test_normal_and_near_pole_points_are_finite(self) -> None:
        pole_points = torch.tensor(
            [[1e-5, 1.0, 1e-5], [-1e-5, -1.0, 2e-5]],
            dtype=torch.float64,
        )
        for points in (self.points, pole_points):
            uv, jacobian = erp_projection_with_jacobian(
                points, self.width, self.height,
            )
            self.assertTrue(torch.isfinite(uv).all())
            self.assertTrue(torch.isfinite(jacobian).all())

    def test_jacobian_matches_float64_finite_difference(self) -> None:
        _, analytic = erp_projection_with_jacobian(
            self.points, self.width, self.height,
        )
        epsilon = 1e-6
        columns = []
        for index in range(3):
            step = torch.zeros_like(self.points)
            step[:, index] = epsilon
            plus = erp_projection(self.points + step, self.width, self.height)
            minus = erp_projection(self.points - step, self.width, self.height)
            columns.append((plus - minus) / (2.0 * epsilon))
        finite_difference = torch.stack(columns, dim=-1)
        self.assertTrue(torch.allclose(
            analytic, finite_difference, atol=2e-6, rtol=2e-8,
        ))

    def test_compatibility_aliases_are_exact(self) -> None:
        canonical_uv = project_unified_erp(
            self.points, self.width, self.height,
        )
        canonical_pair = project_unified_erp_with_jacobian(
            self.points, self.width, self.height,
        )
        self.assertTrue(torch.equal(
            canonical_uv,
            project_cam2_erp(self.points, self.width, self.height),
        ))
        for alias in (
            project_cam2_erp_with_jacobian,
            project_cam2_with_jacobian,
        ):
            alias_pair = alias(self.points, self.width, self.height)
            self.assertTrue(torch.equal(canonical_pair[0], alias_pair[0]))
            self.assertTrue(torch.equal(canonical_pair[1], alias_pair[1]))

    def test_fixed_rotation_projection_and_chain_rule_are_exact(self) -> None:
        fixed_rotation = ROTATION_MATRIX.to(
            device=self.points.device, dtype=self.points.dtype,
        )
        camera_points = self.points @ fixed_rotation.T
        expected_uv, camera_jacobian = erp_projection_with_jacobian(
            camera_points, self.width, self.height,
        )
        expected_jacobian = torch.einsum(
            "nij,jk->nik", camera_jacobian, fixed_rotation,
        )
        actual_uv, actual_jacobian = project_unified_erp_with_jacobian(
            self.points, self.width, self.height,
        )
        self.assertTrue(torch.equal(actual_uv, expected_uv))
        self.assertTrue(torch.equal(actual_jacobian, expected_jacobian))

    def test_dtype_and_device_are_preserved(self) -> None:
        for dtype in (torch.float32, torch.float64):
            points = self.points.to(dtype=dtype)
            uv, jacobian = project_unified_erp_with_jacobian(
                points, self.width, self.height,
            )
            self.assertEqual(uv.dtype, dtype)
            self.assertEqual(jacobian.dtype, dtype)
            self.assertEqual(uv.device, points.device)
            self.assertEqual(jacobian.device, points.device)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_cuda_smoke(self) -> None:
        points = self.points.cuda()
        uv, jacobian = project_unified_erp_with_jacobian(
            points, self.width, self.height,
        )
        self.assertTrue(uv.is_cuda)
        self.assertTrue(jacobian.is_cuda)
        self.assertTrue(torch.isfinite(uv).all())
        self.assertTrue(torch.isfinite(jacobian).all())


if __name__ == "__main__":
    unittest.main()
