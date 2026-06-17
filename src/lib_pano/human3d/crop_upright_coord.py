"""Crop-upright coordinate restore to equirect_rotate full-image coordinates.

Extracted from debug064 and debug067. Rotation rules are hardcoded for
20260311_riding for now.
"""

from __future__ import annotations

import numpy as np


def get_20260311_rotate_rule(frame_idx: int) -> str:
    """Return the upright rotation rule for a given frame in 20260311_riding."""
    if 0 <= frame_idx <= 3494:
        return "ccw90"
    if 3495 <= frame_idx <= 5381:
        return "cw90"
    if 5951 <= frame_idx <= 5954:
        return "ccw90"
    return "none"


def upright_points_to_equirect_rotate(
    points_upright: np.ndarray,
    bbox: list[int] | tuple[int, int, int, int],
    frame_idx: int,
    rotate_rule: str | None = None,
) -> np.ndarray:
    """Restore points from crop_upright coords to equirect_rotate full-image coords.

    Args:
        points_upright: [N, 2] points in crop_upright image coordinates.
        bbox: [x1, y1, x2, y2] crop bbox in equirect_rotate coordinates.
        frame_idx: Frame index (used to determine rotation rule if not given).
        rotate_rule: Override rotation rule ("ccw90", "cw90", "180", "none").
            If None, uses get_20260311_rotate_rule(frame_idx).

    Returns:
        [N, 2] float32 points in equirect_rotate full-image coordinates.
    """
    x1, y1, x2, y2 = bbox
    crop_w = x2 - x1
    crop_h = y2 - y1

    pts = points_upright.astype(np.float32)
    out = np.empty_like(pts)

    if rotate_rule is None:
        rotate_rule = get_20260311_rotate_rule(frame_idx)

    if rotate_rule == "ccw90":
        out[:, 0] = crop_w - 1 - pts[:, 1]
        out[:, 1] = pts[:, 0]
    elif rotate_rule == "cw90":
        out[:, 0] = pts[:, 1]
        out[:, 1] = crop_h - 1 - pts[:, 0]
    elif rotate_rule == "180":
        out[:, 0] = crop_w - 1 - pts[:, 0]
        out[:, 1] = crop_h - 1 - pts[:, 1]
    elif rotate_rule == "none":
        out[:, 0] = pts[:, 0]
        out[:, 1] = pts[:, 1]
    else:
        raise ValueError(f"Unknown rotate rule: {rotate_rule}")

    out[:, 0] += x1
    out[:, 1] += y1
    return out
