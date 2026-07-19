"""Compatibility primitives for SAM 3D Body initialization audits.

This module deliberately does not transplant MHR parameters into SMPL-X.
It provides checked geometry/schema operations used by benchmark-only audits.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np

from .sam3dbody_project import project_vertices_opencv_style


class BodyPoseCompatibility(str, Enum):
    DIRECT_SMPLX_AXIS_ANGLE_63 = "DIRECT_SMPLX_AXIS_ANGLE_63"
    CONVERTIBLE_ROTATION_REPRESENTATION = "CONVERTIBLE_ROTATION_REPRESENTATION"
    MHR_TO_SMPLX_JOINT_MAPPING_REQUIRED = "MHR_TO_SMPLX_JOINT_MAPPING_REQUIRED"
    MODEL_SPECIFIC_LATENT_NOT_DIRECTLY_COMPATIBLE = "MODEL_SPECIFIC_LATENT_NOT_DIRECTLY_COMPATIBLE"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True)
class ProjectionClosure:
    vertices_cam_max_abs: float
    vertices_cam_rms: float
    uv_max_abs: float
    uv_rms: float
    passed: bool


def audit_crop_projection(recon: Any) -> ProjectionClosure:
    points, vertices_cam, _ = project_vertices_opencv_style(
        recon.pred_vertices, recon.pred_cam_t, recon.focal_length, recon.image_shape
    )
    dc = np.asarray(vertices_cam, dtype=np.float64) - np.asarray(recon.vertices_cam, dtype=np.float64)
    duv = np.asarray(points, dtype=np.float64) - np.asarray(recon.vertices_2d, dtype=np.float64)
    return ProjectionClosure(
        vertices_cam_max_abs=float(np.max(np.abs(dc))),
        vertices_cam_rms=float(np.sqrt(np.mean(dc * dc))),
        uv_max_abs=float(np.max(np.abs(duv))),
        uv_rms=float(np.sqrt(np.mean(duv * duv))),
        passed=bool(np.max(np.abs(duv)) <= 1e-4 and np.sqrt(np.mean(duv * duv)) <= 1e-5),
    )


def classify_body_pose(array: np.ndarray | None) -> BodyPoseCompatibility:
    if array is None:
        return BodyPoseCompatibility.UNKNOWN
    if tuple(array.shape) == (63,):
        return BodyPoseCompatibility.DIRECT_SMPLX_AXIS_ANGLE_63
    if array.size == 133:
        return BodyPoseCompatibility.MODEL_SPECIFIC_LATENT_NOT_DIRECTLY_COMPATIBLE
    return BodyPoseCompatibility.UNKNOWN


def validate_smplx_pose_candidate(array: np.ndarray) -> np.ndarray:
    value = np.asarray(array)
    if value.shape != (63,):
        raise ValueError(
            f"SMPL-X body pose initializer requires exact shape (63,), got {value.shape}; "
            "MHR parameters must not be sliced"
        )
    return value


def validate_smplx_beta_candidate(array: np.ndarray) -> np.ndarray:
    value = np.asarray(array)
    if value.shape != (10,):
        raise ValueError(
            f"SMPL-X beta initializer requires exact shape (10,), got {value.shape}; "
            "MHR shape parameters must not be truncated"
        )
    return value


def orthonormal_basis_from_rays(center: np.ndarray, right: np.ndarray, down: np.ndarray) -> np.ndarray:
    forward = np.asarray(center, dtype=np.float64)
    forward /= np.linalg.norm(forward)
    x = np.asarray(right, dtype=np.float64) - forward * np.dot(right, forward)
    x /= np.linalg.norm(x)
    y_hint = np.asarray(down, dtype=np.float64) - forward * np.dot(down, forward)
    y = y_hint - x * np.dot(y_hint, x)
    y /= np.linalg.norm(y)
    basis = np.stack([x, y, forward], axis=1)
    if np.linalg.det(basis) < 0:
        y = -y
        basis = np.stack([x, y, forward], axis=1)
    return basis


def basis_diagnostics(basis: np.ndarray) -> dict[str, float | bool]:
    value = np.asarray(basis, dtype=np.float64)
    error = value.T @ value - np.eye(3)
    determinant = float(np.linalg.det(value))
    return {
        "determinant": determinant,
        "orthogonality_max_abs": float(np.max(np.abs(error))),
        "passed": bool(abs(determinant - 1.0) <= 1e-8 and np.max(np.abs(error)) <= 1e-8),
    }
