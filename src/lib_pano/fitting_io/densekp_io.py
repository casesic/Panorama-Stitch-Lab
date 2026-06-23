"""DenseKP 138 点读取：从 CameraHMR npz 加载关键点，还原到全图坐标并筛选 valid 点。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from lib_pano.human3d.crop_upright_coord import upright_points_to_equirect_rotate


@dataclass
class DenseKPResult:
    """DenseKP 单帧结果：138 个关键点的全图坐标、heatmap value 和 validity mask。"""
    keypoints_2d: np.ndarray
    values: np.ndarray
    valid: np.ndarray

def load_densekp_result(
    frame_idx: int,
    densekp_dir: Path,
    cam_id: str,
    image_width: int,
    image_height: int,
    crop_bbox: list[int] | None = None,
) -> DenseKPResult | None:
    """从 CameraHMR npz 加载 DenseKP 关键点，还原到 equirect_rotate 全图坐标并标记 validity。"""
    if crop_bbox is None:
        return None

    npz_path = densekp_dir / cam_id / f"{frame_idx:05d}_0.npz"
    if not npz_path.exists():
        return None

    data = np.load(npz_path)
    keypoints_upright = data["keypoints_orig_xy"].astype(np.float32)
    keypoint_value = data["keypoint_value"].astype(np.float32)

    # CameraHMR bbox in upright crop coordinates — used to filter implausible points
    densekp_bbox = data.get("bbox_xyxy")
    if densekp_bbox is not None:
        densekp_bbox = densekp_bbox.astype(np.float32)

    keypoints_full = upright_points_to_equirect_rotate(
        keypoints_upright, crop_bbox, frame_idx
    )

    finite = np.isfinite(keypoints_full).all(axis=1)
    x_in = (keypoints_full[:, 0] >= 0) & (keypoints_full[:, 0] < image_width)
    y_in = (keypoints_full[:, 1] >= 0) & (keypoints_full[:, 1] < image_height)
    valid = finite & x_in & y_in

    return DenseKPResult(
        keypoints_2d=keypoints_full,
        values=keypoint_value,
        valid=valid,
    )
