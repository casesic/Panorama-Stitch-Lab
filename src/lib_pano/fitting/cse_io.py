from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from lib_pano.fitting.densekp_io import upright_points_to_crop_local


@dataclass
class CSEResult:
    xy_full: np.ndarray   # [N, 2] full ERP coordinates
    vertex_ids: np.ndarray  # [N] SMPL vertex IDs
    valid: np.ndarray       # [N] bool


def load_cse_result(
    frame_idx: int,
    cse_npz_dir: Path,
    cam_id: str,
    image_width: int,
    image_height: int,
    crop_bbox: list[int] | None = None,
) -> CSEResult | None:
    if crop_bbox is None:
        return None

    npz_path = cse_npz_dir / cam_id / f"{frame_idx:05d}.npz"
    if not npz_path.exists():
        return None

    data = np.load(npz_path)
    xs = data["xs"].astype(np.float32)
    ys = data["ys"].astype(np.float32)
    vertex_ids = data["vertex_ids"].astype(np.int32)

    x1, y1, x2, y2 = [int(v) for v in crop_bbox]
    crop_width = x2 - x1
    crop_height = y2 - y1

    points_upright = np.stack([xs, ys], axis=1).astype(np.float32)
    points_local = upright_points_to_crop_local(
        points_upright, crop_width, crop_height, frame_idx
    )
    points_full = points_local + np.array([x1, y1], dtype=np.float32)

    finite = np.isfinite(points_full).all(axis=1)
    x_in = (points_full[:, 0] >= 0) & (points_full[:, 0] < image_width)
    y_in = (points_full[:, 1] >= 0) & (points_full[:, 1] < image_height)
    valid = (vertex_ids >= 0) & finite & x_in & y_in

    return CSEResult(
        xy_full=points_full,
        vertex_ids=vertex_ids,
        valid=valid,
    )
