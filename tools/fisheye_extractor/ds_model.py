from __future__ import annotations

from typing import Any

import numpy as np


def _validate_params(params: dict[str, Any]) -> tuple[float, float, float, float, float, float]:
    """Validate required DS parameters and return them as float64-compatible scalars."""
    required_keys = ("xi", "alpha", "fx", "fy", "cx", "cy")
    missing = [key for key in required_keys if key not in params]
    if missing:
        raise KeyError(f"Missing DS parameters: {missing}")

    xi = float(params["xi"])
    alpha = float(params["alpha"])
    fx = float(params["fx"])
    fy = float(params["fy"])
    cx = float(params["cx"])
    cy = float(params["cy"])
    return xi, alpha, fx, fy, cx, cy


def _as_points_3d(points_3d: np.ndarray) -> tuple[np.ndarray, bool]:
    """Normalize 3D point input to shape (N, 3) and report whether input was a single point."""
    points = np.asarray(points_3d, dtype=np.float64)
    if points.ndim == 1:
        if points.shape[0] != 3:
            raise ValueError(f"Expected shape (3,), got {points.shape}")
        return points.reshape(1, 3), True
    if points.ndim == 2 and points.shape[1] == 3:
        return points, False
    raise ValueError(f"Expected shape (N, 3) or (3,), got {points.shape}")


def _as_pixels(pixels: np.ndarray) -> tuple[np.ndarray, bool]:
    """Normalize pixel input to shape (N, 2) and report whether input was a single pixel."""
    pixels_arr = np.asarray(pixels, dtype=np.float64)
    if pixels_arr.ndim == 1:
        if pixels_arr.shape[0] != 2:
            raise ValueError(f"Expected shape (2,), got {pixels_arr.shape}")
        return pixels_arr.reshape(1, 2), True
    if pixels_arr.ndim == 2 and pixels_arr.shape[1] == 2:
        return pixels_arr, False
    raise ValueError(f"Expected shape (N, 2) or (2,), got {pixels_arr.shape}")


def _finalize_output(values: np.ndarray, is_single: bool) -> np.ndarray:
    """Restore output shape to match single-point or batched input convention."""
    return values[0] if is_single else values


def project(
    points_3d: np.ndarray,
    params: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    """Project 3D camera-frame points onto fisheye pixels with the Double Sphere model.

    Args:
        points_3d: Array of shape (N, 3) or a single point with shape (3,).
        params: DS parameter dictionary containing xi, alpha, fx, fy, cx, cy.

    Returns:
        A tuple of:
        - pixels: Projected pixel coordinates with shape (N, 2) or (2,), dtype float32.
        - valid_mask: Boolean mask of shape (N,) or scalar-like shape (), indicating whether
          each 3D point lies inside the valid projection domain.
    """
    xi, alpha, fx, fy, cx, cy = _validate_params(params)
    points, is_single = _as_points_3d(points_3d)

    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]

    d1 = np.sqrt(np.maximum(x * x + y * y + z * z, 0.0))
    z_shifted = xi * d1 + z
    d2 = np.sqrt(np.maximum(x * x + y * y + z_shifted * z_shifted, 0.0))

    if alpha <= 0.5:
        w1 = alpha / (1.0 - alpha)
    else:
        w1 = (1.0 - alpha) / alpha
    w2 = (w1 + xi) / np.sqrt(np.maximum(2.0 * w1 * xi + xi * xi + 1.0, 0.0))
    valid_mask = z_shifted > (-w2 * d1)

    denom = alpha * d2 + (1.0 - alpha) * z_shifted
    safe_denom = np.where(np.abs(denom) > 0.0, denom, 1.0)

    u = fx * x / safe_denom + cx
    v = fy * y / safe_denom + cy

    pixels = np.stack([u, v], axis=1).astype(np.float32)
    valid_mask = valid_mask.astype(bool)
    return _finalize_output(pixels, is_single), _finalize_output(valid_mask, is_single)


def unproject(
    pixels: np.ndarray,
    params: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    """Unproject fisheye pixels to unit-norm 3D rays with the Double Sphere model.

    Args:
        pixels: Array of shape (N, 2) or a single pixel with shape (2,).
        params: DS parameter dictionary containing xi, alpha, fx, fy, cx, cy.

    Returns:
        A tuple of:
        - points_3d: Unit-sphere rays with shape (N, 3) or (3,), dtype float32.
        - valid_mask: Boolean mask of shape (N,) or scalar-like shape (), indicating whether
          each pixel lies inside the valid unprojection domain.
    """
    xi, alpha, fx, fy, cx, cy = _validate_params(params)
    pixels_arr, is_single = _as_pixels(pixels)

    u = pixels_arr[:, 0]
    v = pixels_arr[:, 1]

    mx = (u - cx) / fx
    my = (v - cy) / fy
    r2 = mx * mx + my * my

    valid_mask = np.ones(r2.shape, dtype=bool)
    if alpha > 0.5:
        valid_mask &= r2 < (1.0 / (2.0 * alpha - 1.0))

    sqrt_term = np.sqrt(np.maximum(1.0 - (2.0 * alpha - 1.0) * r2, 0.0))
    mz_denom = alpha * sqrt_term + 1.0 - alpha
    safe_mz_denom = np.where(np.abs(mz_denom) > 0.0, mz_denom, 1.0)
    mz = (1.0 - alpha * alpha * r2) / safe_mz_denom

    k_sqrt_term = np.sqrt(np.maximum(mz * mz + (1.0 - xi * xi) * r2, 0.0))
    k_denom = mz * mz + r2
    safe_k_denom = np.where(np.abs(k_denom) > 0.0, k_denom, 1.0)
    k = (mz * xi + k_sqrt_term) / safe_k_denom

    x = k * mx
    y = k * my
    z = k * mz - xi

    points = np.stack([x, y, z], axis=1)
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    safe_norms = np.where(norms > 0.0, norms, 1.0)
    points = points / safe_norms

    points = points.astype(np.float32)
    valid_mask = valid_mask.astype(bool)
    return _finalize_output(points, is_single), _finalize_output(valid_mask, is_single)


def incidence_angle_deg(
    pixels: np.ndarray,
    params: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the incidence angle in degrees between each ray and the optical axis.

    Args:
        pixels: Array of shape (N, 2) or a single pixel with shape (2,).
        params: DS parameter dictionary containing xi, alpha, fx, fy, cx, cy.

    Returns:
        A tuple of:
        - theta_deg: Angles in degrees with shape (N,) or scalar-like shape (), dtype float32.
        - valid_mask: Boolean mask of shape (N,) or scalar-like shape (), indicating whether
          each pixel lies inside the valid unprojection domain.
    """
    rays, valid_mask = unproject(pixels, params)
    rays_arr = np.asarray(rays, dtype=np.float64)

    if rays_arr.ndim == 1:
        z = np.clip(rays_arr[2], -1.0, 1.0)
        theta_deg = np.float32(np.degrees(np.arccos(z)))
        return theta_deg, valid_mask

    z = np.clip(rays_arr[:, 2], -1.0, 1.0)
    theta_deg = np.degrees(np.arccos(z)).astype(np.float32)
    return theta_deg, valid_mask
