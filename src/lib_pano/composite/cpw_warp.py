"""
CPW (Control Point Weighted) warp, directional erosion, and compositing.

Extracted from debug067_sam3dbody_cached_full_cpw.py.
Pure numpy + opencv, no torch.
"""

from __future__ import annotations

import cv2
import numpy as np


def generate_cpw_backward_map(
    src_pts: np.ndarray,
    dst_pts: np.ndarray,
    image_shape: tuple[int, int] | tuple[int, int, int],
    k: int = 12,
    power: float = 2.0,
    max_points: int | None = None,
) -> tuple[np.ndarray, dict]:
    """
    Generate a CPW backward warp map from source to target.

    Given control point pairs (src_pts on source image, dst_pts on target image),
    produce a backward map that, when used with cv2.remap(source, map_x, map_y),
    warps the source image such that dst_pts[i] lands on src_pts[i].

    Args:
        src_pts: [N, 2] control points on the source image.
        dst_pts: [N, 2] control points on the target (destination) image.
        image_shape: (H, W) of the target crop region.
        k: number of nearest neighbours for interpolation.
        power: distance weighting power.
        max_points: if set, randomly sample up to this many points.

    Returns:
        backward_map: [H, W, 2] float32, channels ordered (x, y).
        stats: dict with num_points, k, power, disp_mean, disp_median, disp_max.
    """
    h, w = image_shape[:2]

    if max_points is not None and len(src_pts) > max_points:
        rng = np.random.RandomState(0)
        idx = rng.choice(len(src_pts), max_points, replace=False)
        s_pts = src_pts[idx]
        d_pts = dst_pts[idx]
    else:
        s_pts = src_pts
        d_pts = dst_pts

    disp = s_pts - d_pts
    N = len(d_pts)
    knn = min(k, N)
    eps = 1e-8

    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    q_grid = np.stack([xx, yy], axis=-1)
    q_flat = q_grid.reshape(-1, 2)

    diff = q_flat[:, None, :] - d_pts[None, :, :]
    dist2 = np.sum(diff * diff, axis=-1)

    if knn < N:
        knn_idx = np.argpartition(dist2, knn - 1, axis=1)[:, :knn]
        knn_dist2 = np.take_along_axis(dist2, knn_idx, axis=1)
        knn_disp = disp[knn_idx]
    else:
        knn_dist2 = dist2
        knn_disp = disp[None, :, :]

    exact_mask = knn_dist2 < 1e-8
    weights = 1.0 / (knn_dist2 ** (power * 0.5) + eps)
    weights = weights / (weights.sum(axis=1, keepdims=True) + eps)

    exact_any = exact_mask.any(axis=1)
    weighted_disp = np.sum(weights[..., None] * knn_disp, axis=1)
    if exact_any.any():
        first_exact = np.argmax(exact_mask, axis=1)
        exact_disp = knn_disp[np.arange(knn_disp.shape[0]), first_exact]
        weighted_disp[exact_any] = exact_disp[exact_any]

    q_src = q_flat + weighted_disp
    map_x = q_src[:, 0].reshape(h, w).astype(np.float32)
    map_y = q_src[:, 1].reshape(h, w).astype(np.float32)
    backward_map = np.stack([map_x, map_y], axis=-1)

    disp_norm = np.linalg.norm(disp, axis=1)
    stats = {
        "num_points": N,
        "k": knn,
        "power": power,
        "disp_mean": float(disp_norm.mean()),
        "disp_median": float(np.median(disp_norm)),
        "disp_max": float(disp_norm.max()),
    }

    return backward_map, stats


def warp_image_with_backward_map(
    image: np.ndarray,
    backward_map: np.ndarray,
    interpolation: int = cv2.INTER_LINEAR,
    border_mode: int = cv2.BORDER_CONSTANT,
    border_value: int | tuple[int, int, int] = 0,
) -> np.ndarray:
    """Apply a backward warp map to an image via cv2.remap.

    Args:
        image: source image [H, W, C].
        backward_map: [H_out, W_out, 2] float32, channels (x, y).
    """
    map_x = backward_map[..., 0].astype(np.float32)
    map_y = backward_map[..., 1].astype(np.float32)
    return cv2.remap(image, map_x, map_y, interpolation, borderMode=border_mode, borderValue=border_value)


def directional_erode_valid_mask(
    mask: np.ndarray,
    direction: str,
    erode_px: int,
) -> np.ndarray:
    """Erode a boolean valid mask from one boundary only.

    direction="up":    top boundary eroded inward.
    direction="down":  bottom boundary eroded inward.
    """
    if erode_px <= 1:
        return mask.astype(bool)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, erode_px))
    if direction == "down":
        anchor = (0, 0)
    elif direction == "up":
        anchor = (0, erode_px - 1)
    else:
        raise ValueError(f"Unknown direction: {direction}")
    eroded = cv2.erode(mask.astype(np.uint8), kernel, anchor=anchor)
    return eroded > 0


def target_priority_composite(
    target_img: np.ndarray,
    warped_source_img: np.ndarray,
    target_valid_mask: np.ndarray,
    source_valid_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Composite warped source into target using target-priority logic.

    target valid pixels -> use target.
    target invalid but source valid -> use warped source.
    both invalid -> use target (or zero).

    Args:
        target_img: target image.
        warped_source_img: warped source image (same shape as target_img).
        target_valid_mask: bool mask of valid target pixels.
        source_valid_mask: bool mask of valid source pixels. If None, all are valid.
    """
    if source_valid_mask is None:
        source_valid_mask = np.ones(target_img.shape[:2], dtype=bool)
    fill_mask = (~target_valid_mask) & source_valid_mask
    result = target_img.copy()
    result[fill_mask] = warped_source_img[fill_mask]
    return result
