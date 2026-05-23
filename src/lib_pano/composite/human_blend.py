from __future__ import annotations

import numpy as np


def maybe_make_valid_mask(image: np.ndarray | None) -> np.ndarray | None:
    if image is None:
        return None
    return np.any(image > 0, axis=2)


def blend_two_humans(warp: np.ndarray, target_origin: np.ndarray, direction: str = "top") -> np.ndarray:
    if warp.shape != target_origin.shape:
        raise ValueError("warp and target_origin must have the same shape")
    if direction not in ("top", "bottom"):
        raise ValueError(f"Unsupported blend direction: {direction}")

    target_valid = np.any(target_origin > 0, axis=2)
    rows = np.arange(target_valid.shape[0], dtype=np.int32)[:, None]
    if direction == "top":
        top_boundary_per_col = np.where(
            target_valid.any(axis=0),
            target_valid.argmax(axis=0),
            target_valid.shape[0],
        ).astype(np.int32)
        boundary_mask = rows < top_boundary_per_col[None, :]
    else:
        bottom_boundary_per_col = np.where(
            target_valid.any(axis=0),
            target_valid.shape[0] - 1 - target_valid[::-1].argmax(axis=0),
            -1,
        ).astype(np.int32)
        boundary_mask = rows > bottom_boundary_per_col[None, :]

    warp_valid = np.any(warp > 0, axis=2)
    completion_mask = warp_valid & (~target_valid) & boundary_mask

    result = target_origin.copy()
    result[completion_mask] = warp[completion_mask]
    return result
