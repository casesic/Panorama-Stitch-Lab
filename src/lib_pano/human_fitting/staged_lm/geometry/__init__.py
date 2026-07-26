"""Shared mathematical kernels for staged LM fitting."""

from .erp import (
    erp_projection,
    erp_projection_with_jacobian,
    project_unified_erp,
    project_unified_erp_with_jacobian,
)
from .so3 import (
    axis_angle_to_matrix,
    rotated_point_jacobian,
    skew,
    so3_right_jacobian,
)

__all__ = [
    "skew",
    "axis_angle_to_matrix",
    "so3_right_jacobian",
    "rotated_point_jacobian",
    "erp_projection",
    "erp_projection_with_jacobian",
    "project_unified_erp",
    "project_unified_erp_with_jacobian",
]
