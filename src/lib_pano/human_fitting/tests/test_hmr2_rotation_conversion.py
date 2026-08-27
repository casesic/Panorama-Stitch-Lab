from __future__ import annotations

import numpy as np
import pytest

from lib_pano.human_fitting.hmr2_staged_init import (
    axis_angle_to_rotation_matrices,
    rotation_matrices_to_axis_angle,
)


@pytest.mark.parametrize("axis_angle", [
    [0.0, 0.0, 0.0],
    [1e-8, -2e-8, 3e-8],
    [0.4, -0.7, 1.1],
    [np.pi - 1e-6, 0.0, 0.0],
])
def test_rotation_matrix_axis_angle_round_trip(axis_angle):
    source = np.asarray(axis_angle, dtype=np.float64)[None]
    rotation = axis_angle_to_rotation_matrices(source)
    recovered = rotation_matrices_to_axis_angle(rotation)
    reconstructed = axis_angle_to_rotation_matrices(recovered)
    np.testing.assert_allclose(reconstructed, rotation, atol=2e-6, rtol=2e-6)
