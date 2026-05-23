from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def clamp_int(value: int, low: int, high: int) -> int:
    return max(low, min(high, value))


def maybe_load_rgba(rgba_path: Path) -> np.ndarray | None:
    rgba = cv2.imread(str(rgba_path), cv2.IMREAD_UNCHANGED)
    if rgba is None:
        return None
    if rgba.ndim != 3 or rgba.shape[2] != 4:
        return None
    return rgba


def place_rgba_on_canvas(
    rgba: np.ndarray,
    bbox: list[int],
    canvas: np.ndarray,
) -> np.ndarray | None:
    if rgba.ndim != 3 or rgba.shape[2] != 4:
        return None

    erp_h, erp_w = canvas.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in bbox]
    clip_x1 = clamp_int(x1, 0, erp_w)
    clip_y1 = clamp_int(y1, 0, erp_h)
    clip_x2 = clamp_int(x2, 0, erp_w)
    clip_y2 = clamp_int(y2, 0, erp_h)
    if clip_x1 >= clip_x2 or clip_y1 >= clip_y2:
        return None

    rgba_h, rgba_w = rgba.shape[:2]
    src_x2 = min(rgba_w, clip_x2 - clip_x1)
    src_y2 = min(rgba_h, clip_y2 - clip_y1)
    if src_x2 <= 0 or src_y2 <= 0:
        return None

    rgba = rgba[:src_y2, :src_x2]
    result = canvas.copy()
    alpha_mask = rgba[..., 3] > 0
    if not np.any(alpha_mask):
        return result

    region = result[clip_y1:clip_y1 + src_y2, clip_x1:clip_x1 + src_x2]
    region[alpha_mask] = rgba[..., :3][alpha_mask]
    return result


def place_rgba_on_black(
    rgba_path: Path,
    bbox: list[int],
    erp_h: int,
    erp_w: int,
) -> np.ndarray | None:
    rgba = maybe_load_rgba(rgba_path)
    if rgba is None:
        return None
    canvas = np.zeros((erp_h, erp_w, 3), dtype=np.uint8)
    return place_rgba_on_canvas(rgba, bbox, canvas)
