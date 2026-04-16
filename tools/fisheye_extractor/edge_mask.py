from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np

from ds_model import incidence_angle_deg


def compute_theta_map(
    ds_params: dict[str, Any],
    image_size: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    """Compute per-pixel incidence angles and the DS validity mask for a fisheye image."""
    height, width = image_size
    if height <= 0 or width <= 0:
        raise ValueError(f"image_size must contain positive values, got {image_size}")

    start_time = time.time()

    x_coords = np.arange(width, dtype=np.float64)
    y_coords = np.arange(height, dtype=np.float64)
    xx, yy = np.meshgrid(x_coords, y_coords, indexing="xy")
    pixels = np.stack([xx.reshape(-1), yy.reshape(-1)], axis=1)

    theta_deg, valid_mask = incidence_angle_deg(pixels, ds_params)
    theta_map = np.asarray(theta_deg, dtype=np.float32).reshape(height, width)
    valid_mask = np.asarray(valid_mask, dtype=bool).reshape(height, width)
    theta_map = np.where(valid_mask, theta_map, 0.0).astype(np.float32)

    elapsed = time.time() - start_time
    print(f"compute_theta_map took {elapsed:.3f}s")
    return theta_map, valid_mask


def compute_edge_mask(
    theta_map: np.ndarray,
    valid_mask: np.ndarray,
    theta_threshold: float = 60.0,
    exclude_center_bottom_x: tuple[float, float] | None = (0.4, 0.6),
    exclude_center_bottom_y: tuple[float, float] | None = (0.8, 1.0),
) -> np.ndarray:
    """Compute the valid edge-distortion region mask from a theta map."""
    theta_map_arr = np.asarray(theta_map, dtype=np.float32)
    valid_mask_arr = np.asarray(valid_mask, dtype=bool)

    if theta_map_arr.shape != valid_mask_arr.shape:
        raise ValueError(
            f"theta_map and valid_mask must have the same shape, got "
            f"{theta_map_arr.shape} and {valid_mask_arr.shape}"
        )
    edge_mask = valid_mask_arr & (theta_map_arr > float(theta_threshold))

    if exclude_center_bottom_x is not None and exclude_center_bottom_y is not None:
        h, w = theta_map.shape
        x_min = int(w * exclude_center_bottom_x[0])
        x_max = int(w * exclude_center_bottom_x[1])
        y_min = int(h * exclude_center_bottom_y[0])
        y_max = int(h * exclude_center_bottom_y[1])
        exclude_rect = np.zeros((h, w), dtype=bool)
        exclude_rect[y_min:y_max, x_min:x_max] = True
        edge_mask = edge_mask & ~exclude_rect

    return edge_mask


def get_or_build_cache(
    ds_params: dict[str, Any],
    image_size: tuple[int, int],
    cache_path: str,
    theta_threshold: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load cached theta/valid maps from disk or build and save them if missing."""
    cache_file = Path(cache_path)

    if cache_file.exists():
        with np.load(cache_file) as data:
            theta_map = np.asarray(data["theta_map"], dtype=np.float32)
            valid_mask = np.asarray(data["valid_mask"], dtype=bool)
        edge_mask = compute_edge_mask(theta_map, valid_mask, theta_threshold)
        print("Cache loaded")
        return theta_map, valid_mask, edge_mask

    theta_map, valid_mask = compute_theta_map(ds_params, image_size)
    edge_mask = compute_edge_mask(theta_map, valid_mask, theta_threshold)

    cache_file.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_file, theta_map=theta_map, valid_mask=valid_mask)
    print(f"Cache built and saved to {cache_file}")
    return theta_map, valid_mask, edge_mask
