"""Unit tests for the shared staged-LM SO(3) kernels."""

from __future__ import annotations

import unittest

import torch

try:
    from ._bootstrap import ensure_test_paths
except ImportError:
    from _bootstrap import ensure_test_paths

ensure_test_paths(__file__)

from lib_pano.human_fitting.staged_lm.geometry.so3 import (
    _skew,
    axis_angle_to_matrix,
    rotated_point_jacobian,
    skew,
    so3_right_jacobian,
)


def _reference_axis_angle_to_matrix(axis_angle: torch.Tensor) -> torch.Tensor:
    theta2 = torch.sum(axis_angle * axis_angle, dim=-1, keepdim=True)
    theta = torch.sqrt(theta2)
    small = theta2 < 1e-8
    safe_theta = torch.where(small, torch.ones_like(theta), theta)
    a_regular = torch.sin(theta) / safe_theta
    b_regular = (1.0 - torch.cos(theta)) / torch.where(
        small, torch.ones_like(theta2), theta2,
    )
    a = torch.where(
        small, 1.0 - theta2 / 6.0 + theta2 * theta2 / 120.0, a_regular,
    )
    b = torch.where(
        small, 0.5 - theta2 / 24.0 + theta2 * theta2 / 720.0, b_regular,
    )
    phi_hat = skew(axis_angle)
    identity = torch.eye(3, dtype=axis_angle.dtype, device=axis_angle.device)
    return identity + a[..., None] * phi_hat + b[..., None] * (phi_hat @ phi_hat)


class GeometrySO3Tests(unittest.TestCase):
    def test_zero_axis_angle_is_identity(self) -> None:
        actual = axis_angle_to_matrix(torch.zeros(3, dtype=torch.float64))
        self.assertTrue(torch.equal(actual, torch.eye(3, dtype=torch.float64)))

    def test_very_small_angle_is_finite(self) -> None:
        actual = axis_angle_to_matrix(
            torch.tensor([1e-12, -2e-12, 3e-12], dtype=torch.float64),
        )
        self.assertTrue(torch.isfinite(actual).all())

    def test_general_axis_angle_matches_extracted_formula(self) -> None:
        value = torch.tensor([0.3, -0.5, 0.2], dtype=torch.float64)
        self.assertTrue(torch.equal(
            axis_angle_to_matrix(value),
            _reference_axis_angle_to_matrix(value),
        ))

    def test_batch_shape(self) -> None:
        value = torch.randn(2, 4, 3, dtype=torch.float64)
        self.assertEqual(axis_angle_to_matrix(value).shape, (2, 4, 3, 3))

    def test_right_jacobian_near_zero_is_finite(self) -> None:
        values = torch.tensor(
            [[0.0, 0.0, 0.0], [1e-12, -2e-12, 1e-12]],
            dtype=torch.float64,
        )
        result = so3_right_jacobian(values)
        self.assertEqual(result.shape, (2, 3, 3))
        self.assertTrue(torch.isfinite(result).all())

    def test_rotated_point_jacobian_shape(self) -> None:
        result = rotated_point_jacobian(
            torch.tensor([0.2, -0.1, 0.4], dtype=torch.float64),
            torch.randn(7, 3, dtype=torch.float64),
        )
        self.assertEqual(result.shape, (7, 3, 3))

    def test_rotated_point_jacobian_finite_difference(self) -> None:
        phi = torch.tensor([0.2, -0.1, 0.4], dtype=torch.float64)
        points = torch.tensor(
            [[0.3, -0.7, 1.1], [-1.0, 0.2, 0.5]],
            dtype=torch.float64,
        )
        analytic = rotated_point_jacobian(phi, points)
        epsilon = 1e-6
        columns = []
        for index in range(3):
            step = torch.zeros_like(phi)
            step[index] = epsilon
            plus = points @ axis_angle_to_matrix(phi + step).T
            minus = points @ axis_angle_to_matrix(phi - step).T
            columns.append((plus - minus) / (2.0 * epsilon))
        finite_difference = torch.stack(columns, dim=-1)
        self.assertTrue(torch.allclose(
            analytic, finite_difference, atol=2e-9, rtol=2e-8,
        ))

    def test_skew_matches_cross_product(self) -> None:
        vector = torch.tensor([0.2, -0.4, 0.7], dtype=torch.float64)
        point = torch.tensor([-0.3, 1.2, 0.5], dtype=torch.float64)
        self.assertTrue(torch.allclose(
            skew(vector) @ point,
            torch.cross(vector, point, dim=0),
            atol=1e-15,
            rtol=0.0,
        ))

    def test_compatibility_skew_alias(self) -> None:
        values = torch.tensor(
            [[1.0, 2.0, 3.0], [-2.0, 1.0, 0.5]],
            dtype=torch.float64,
        )
        self.assertTrue(torch.equal(_skew(values), skew(values)))


if __name__ == "__main__":
    unittest.main()
