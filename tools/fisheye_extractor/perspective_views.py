from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from ds_model import project


def _validate_rotation_matrix(rotation: np.ndarray) -> np.ndarray:
    """Validate and normalize a rotation matrix input to shape (3, 3)."""
    matrix = np.asarray(rotation, dtype=np.float64)
    if matrix.shape != (3, 3):
        raise ValueError(f"Expected rotation matrix shape (3, 3), got {matrix.shape}")
    return matrix


def _infer_fisheye_size(ds_params: dict[str, Any]) -> tuple[int, int]:
    """Infer the fisheye image size from parameters or fall back to the project default."""
    width = int(ds_params.get("image_width", 2880))
    height = int(ds_params.get("image_height", 2880))
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid fisheye image size inferred from ds_params: {(width, height)}")
    return width, height


def _as_persp_pixels(persp_pixels: np.ndarray) -> np.ndarray:
    """Validate perspective pixel input and normalize it to shape (N, 2)."""
    pixels = np.asarray(persp_pixels, dtype=np.float64)
    if pixels.ndim != 2 or pixels.shape[1] != 2:
        raise ValueError(f"Expected perspective pixels with shape (N, 2), got {pixels.shape}")
    return pixels


def _perspective_intrinsics(fov_deg: float, out_size: tuple[int, int]) -> tuple[float, float, float]:
    """Compute symmetric perspective focal length and principal point."""
    width, height = out_size
    if width <= 0 or height <= 0:
        raise ValueError(f"Output size must be positive, got {out_size}")
    if fov_deg <= 0.0 or fov_deg >= 180.0:
        raise ValueError(f"FOV must be in (0, 180), got {fov_deg}")

    f_persp = (width / 2.0) / np.tan(np.deg2rad(fov_deg / 2.0))
    cx_persp = (width - 1.0) / 2.0
    cy_persp = (height - 1.0) / 2.0
    return float(f_persp), float(cx_persp), float(cy_persp)


def _perspective_pixels_to_rays(
    pixels: np.ndarray,
    fov_deg: float,
    out_size: tuple[int, int],
) -> np.ndarray:
    """Convert perspective pixel coordinates to unit rays in the virtual camera frame."""
    f_persp, cx_persp, cy_persp = _perspective_intrinsics(fov_deg, out_size)

    x_c = (pixels[:, 0] - cx_persp) / f_persp
    y_c = (pixels[:, 1] - cy_persp) / f_persp
    z_c = np.ones_like(x_c, dtype=np.float64)

    rays = np.stack([x_c, y_c, z_c], axis=1)
    norms = np.linalg.norm(rays, axis=1, keepdims=True)
    safe_norms = np.where(norms > 0.0, norms, 1.0)
    return rays / safe_norms


def build_rotation_matrix(yaw_deg: float, pitch_deg: float) -> np.ndarray:
    """Build the view rotation matrix R = R_yaw @ R_pitch as specified by the project."""
    yaw_rad = np.deg2rad(float(yaw_deg))
    pitch_rad = np.deg2rad(float(pitch_deg))

    cos_yaw = np.cos(yaw_rad)
    sin_yaw = np.sin(yaw_rad)
    cos_pitch = np.cos(pitch_rad)
    sin_pitch = np.sin(pitch_rad)

    r_yaw = np.array(
        [
            [cos_yaw, 0.0, sin_yaw],
            [0.0, 1.0, 0.0],
            [-sin_yaw, 0.0, cos_yaw],
        ],
        dtype=np.float64,
    )
    r_pitch = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, cos_pitch, -sin_pitch],
            [0.0, sin_pitch, cos_pitch],
        ],
        dtype=np.float64,
    )
    return r_yaw @ r_pitch


def build_perspective_lut(
    R: np.ndarray,
    fov_deg: float,
    out_size: tuple[int, int],
    ds_params: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build remap lookup tables from a fisheye image to a virtual perspective camera."""
    rotation = _validate_rotation_matrix(R)
    width, height = out_size

    u_coords = np.arange(width, dtype=np.float64)
    v_coords = np.arange(height, dtype=np.float64)
    uu, vv = np.meshgrid(u_coords, v_coords, indexing="xy")
    persp_pixels = np.stack([uu.reshape(-1), vv.reshape(-1)], axis=1)

    rays_camera = _perspective_pixels_to_rays(persp_pixels, fov_deg, out_size)
    rays_fisheye = rays_camera @ rotation.T

    fisheye_pixels, valid_mask = project(rays_fisheye, ds_params)
    fisheye_pixels = np.asarray(fisheye_pixels, dtype=np.float32)
    valid_mask = np.asarray(valid_mask, dtype=bool)

    fisheye_width, fisheye_height = _infer_fisheye_size(ds_params)
    in_bounds = (
        (fisheye_pixels[:, 0] >= 0.0)
        & (fisheye_pixels[:, 0] <= fisheye_width - 1)
        & (fisheye_pixels[:, 1] >= 0.0)
        & (fisheye_pixels[:, 1] <= fisheye_height - 1)
    )
    valid_mask &= in_bounds

    map_x = fisheye_pixels[:, 0].reshape(height, width)
    map_y = fisheye_pixels[:, 1].reshape(height, width)
    valid_mask = valid_mask.reshape(height, width)

    map_x = np.where(valid_mask, map_x, 0.0).astype(np.float32)
    map_y = np.where(valid_mask, map_y, 0.0).astype(np.float32)
    return map_x, map_y, valid_mask


def build_all_luts(
    ds_params: dict[str, Any],
    views: list[dict[str, Any]],
    fov_deg: float,
    out_size: tuple[int, int],
) -> dict[str, dict[str, np.ndarray]]:
    """Build perspective remap LUTs for all configured views."""
    lut_dict: dict[str, dict[str, np.ndarray]] = {}
    for view in views:
        if "name" not in view or "yaw" not in view or "pitch" not in view:
            raise KeyError(f"Each view must contain name, yaw, and pitch: {view}")
        rotation = build_rotation_matrix(view["yaw"], view["pitch"])
        map_x, map_y, valid_mask = build_perspective_lut(rotation, fov_deg, out_size, ds_params)
        lut_dict[str(view["name"])] = {
            "map_x": map_x,
            "map_y": map_y,
            "valid_mask": valid_mask,
            "R": rotation,
        }
    return lut_dict


def remap_to_perspective(
    fisheye_img: np.ndarray,
    map_x: np.ndarray,
    map_y: np.ndarray,
    valid_mask: np.ndarray,
) -> np.ndarray:
    """Remap a fisheye image into a perspective view and black out invalid pixels."""
    image = np.asarray(fisheye_img)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"Expected fisheye image shape (H, W, 3), got {image.shape}")

    remapped = cv2.remap(image, map_x, map_y, interpolation=cv2.INTER_LINEAR)
    remapped[~valid_mask] = 0
    return remapped


def persp_pixel_to_fisheye(
    persp_pixels: np.ndarray,
    R: np.ndarray,
    fov_deg: float,
    out_size: tuple[int, int],
    ds_params: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    """Map perspective-image pixels back to fisheye-image pixel coordinates."""
    rotation = _validate_rotation_matrix(R)
    pixels = _as_persp_pixels(persp_pixels)

    rays_camera = _perspective_pixels_to_rays(pixels, fov_deg, out_size)
    rays_fisheye = rays_camera @ rotation.T
    fisheye_pixels, valid_mask = project(rays_fisheye, ds_params)
    fisheye_pixels = np.asarray(fisheye_pixels, dtype=np.float32)
    valid_mask = np.asarray(valid_mask, dtype=bool)

    fisheye_width, fisheye_height = _infer_fisheye_size(ds_params)
    in_bounds = (
        (fisheye_pixels[:, 0] >= 0.0)
        & (fisheye_pixels[:, 0] <= fisheye_width - 1)
        & (fisheye_pixels[:, 1] >= 0.0)
        & (fisheye_pixels[:, 1] <= fisheye_height - 1)
    )
    valid_mask &= in_bounds
    return fisheye_pixels, valid_mask
