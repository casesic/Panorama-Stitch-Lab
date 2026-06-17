"""
SAM 3D Body vertex projection (camera frame -> 2D image).

Pure numpy, no torch. Extracted from debug066_batch_export_sam3dbody_recon.py.
"""

from __future__ import annotations

import numpy as np


def project_vertices_opencv_style(
    vertices: np.ndarray,
    cam_t: np.ndarray,
    focal_length: float,
    image_shape: tuple[int, int] | tuple[int, int, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Project SAM 3D Body predicted vertices to 2D image coordinates.

    Formula:
        verts_cam = vertices + cam_t[None, :]
        u = focal_length * verts_cam[:, 0] / z + W / 2
        v = focal_length * verts_cam[:, 1] / z + H / 2
        valid_z = z > 1e-6

    Returns:
        vertices_2d  [N, 2] float32
        vertices_cam [N, 3] float32
        valid_z      [N] bool
    """
    h, w = image_shape[:2]
    verts_cam = vertices + cam_t[None, :]
    z = verts_cam[:, 2]
    eps = 1e-6
    valid_z = z > eps
    u = focal_length * verts_cam[:, 0] / np.maximum(z, eps) + w / 2.0
    v = focal_length * verts_cam[:, 1] / np.maximum(z, eps) + h / 2.0
    pts2d = np.stack([u, v], axis=1).astype(np.float32)
    return pts2d, verts_cam.astype(np.float32), valid_z


def compute_inside_mask(
    points_2d: np.ndarray,
    image_shape: tuple[int, int] | tuple[int, int, int],
    valid_z: np.ndarray | None = None,
) -> np.ndarray:
    """
    Compute boolean mask of points that fall inside the image bounds.

    Checks: finite, x in [0, W), y in [0, H).
    If valid_z is provided, also requires valid_z == True.
    """
    h, w = image_shape[:2]
    mask = (
        np.isfinite(points_2d).all(axis=1)
        & (points_2d[:, 0] >= 0)
        & (points_2d[:, 0] < w)
        & (points_2d[:, 1] >= 0)
        & (points_2d[:, 1] < h)
    )
    if valid_z is not None:
        mask = mask & valid_z
    return mask
