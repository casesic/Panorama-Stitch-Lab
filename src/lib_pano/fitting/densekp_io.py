from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass
class DenseKPResult:
    keypoints_2d: np.ndarray  # [138, 2]
    values: np.ndarray        # [138]
    valid: np.ndarray         # [138] bool


def get_rotate_code(frame_id: int):
    if frame_id < 3495:
        return cv2.ROTATE_90_COUNTERCLOCKWISE
    if 3495 <= frame_id <= 5381:
        return cv2.ROTATE_90_CLOCKWISE
    if 5951 <= frame_id <= 5954:
        return cv2.ROTATE_90_COUNTERCLOCKWISE
    return None


def upright_points_to_crop_local(
    points_xy: np.ndarray,
    crop_width: int,
    crop_height: int,
    frame_id: int,
) -> np.ndarray:
    """将 upright crop 坐标逆旋转回 crop local 坐标。"""
    rotate_code = get_rotate_code(frame_id)
    out = points_xy.copy()
    if rotate_code == cv2.ROTATE_90_COUNTERCLOCKWISE:
        out[:, 0] = crop_width - 1 - points_xy[:, 1]
        out[:, 1] = points_xy[:, 0]
    elif rotate_code == cv2.ROTATE_90_CLOCKWISE:
        out[:, 0] = points_xy[:, 1]
        out[:, 1] = crop_height - 1 - points_xy[:, 0]
    return out


def load_densekp_result(
    frame_idx: int,
    densekp_dir: Path,
    cam_id: str,
    image_width: int,
    image_height: int,
    crop_bbox: list[int] | None = None,
) -> DenseKPResult | None:
    if crop_bbox is None:
        return None

    npz_path = densekp_dir / cam_id / f"{frame_idx:05d}_0.npz"
    if not npz_path.exists():
        return None

    data = np.load(npz_path)
    keypoints_upright = data["keypoints_orig_xy"].astype(np.float32)
    keypoint_value = data["keypoint_value"].astype(np.float32)

    x1, y1, x2, y2 = [int(v) for v in crop_bbox]
    crop_width = x2 - x1
    crop_height = y2 - y1

    keypoints_local = upright_points_to_crop_local(
        keypoints_upright, crop_width, crop_height, frame_idx
    )
    keypoints_full = keypoints_local + np.array([x1, y1], dtype=np.float32)

    finite = np.isfinite(keypoints_full).all(axis=1)
    x_in = (keypoints_full[:, 0] >= 0) & (keypoints_full[:, 0] < image_width)
    y_in = (keypoints_full[:, 1] >= 0) & (keypoints_full[:, 1] < image_height)
    valid = finite & x_in & y_in

    return DenseKPResult(
        keypoints_2d=keypoints_full,
        values=keypoint_value,
        valid=valid,
    )
