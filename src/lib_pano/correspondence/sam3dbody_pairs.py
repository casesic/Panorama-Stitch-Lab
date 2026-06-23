"""SAM 3D Body 跨相机顶点配对、坐标还原和全图范围内的点过滤。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from lib_pano.human3d.sam3dbody_io import Sam3DBodyRecon
from lib_pano.human3d.crop_upright_coord import upright_points_to_equirect_rotate


@dataclass
class Sam3DBodyPairResult:
    """Result of pairing and restoring SAM 3D Body vertices across two cameras."""

    frame_idx: int
    src_cam: str
    dst_cam: str
    src_pts_upright: np.ndarray
    dst_pts_upright: np.ndarray
    src_pts_full: np.ndarray | None
    dst_pts_full: np.ndarray | None
    vertex_ids: np.ndarray


def pair_sam3dbody_vertices(
    src_recon: Sam3DBodyRecon,
    dst_recon: Sam3DBodyRecon,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """跨视角同名 vertex 配对：筛选 valid_z 且 inside 的顶点，输出 src/dst 对应点数组和 vertex id。"""
    valid = src_recon.valid_z & dst_recon.valid_z & src_recon.inside & dst_recon.inside
    vertex_ids = np.where(valid)[0].astype(np.int64)
    src_pts = src_recon.vertices_2d[vertex_ids].astype(np.float32)
    dst_pts = dst_recon.vertices_2d[vertex_ids].astype(np.float32)
    return src_pts, dst_pts, vertex_ids


def restore_sam3dbody_pairs_to_full(
    src_pts_upright: np.ndarray,
    dst_pts_upright: np.ndarray,
    src_bbox: list[int],
    dst_bbox: list[int],
    frame_idx: int,
) -> tuple[np.ndarray, np.ndarray]:
    """将 crop_upright 坐标下的点对还原到 equirect_rotate 全图坐标。"""
    src_full = upright_points_to_equirect_rotate(src_pts_upright, src_bbox, frame_idx)
    dst_full = upright_points_to_equirect_rotate(dst_pts_upright, dst_bbox, frame_idx)
    return src_full, dst_full


def filter_pairs_inside_image(
    src_pts: np.ndarray,
    dst_pts: np.ndarray,
    vertex_ids: np.ndarray,
    src_shape: tuple[int, int] | tuple[int, int, int],
    dst_shape: tuple[int, int] | tuple[int, int, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """过滤点对：保留 finite 且在两张全图范围内的点。"""
    src_h, src_w = src_shape[:2]
    dst_h, dst_w = dst_shape[:2]

    has_finite = np.isfinite(src_pts).all(axis=1) & np.isfinite(dst_pts).all(axis=1)
    src_in = (src_pts[:, 0] >= 0) & (src_pts[:, 0] < src_w) & (src_pts[:, 1] >= 0) & (src_pts[:, 1] < src_h)
    dst_in = (dst_pts[:, 0] >= 0) & (dst_pts[:, 0] < dst_w) & (dst_pts[:, 1] >= 0) & (dst_pts[:, 1] < dst_h)
    good = has_finite & src_in & dst_in

    good_idx = np.where(good)[0]
    return src_pts[good_idx], dst_pts[good_idx], vertex_ids[good_idx]
