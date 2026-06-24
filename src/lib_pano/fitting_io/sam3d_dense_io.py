"""SAM 3D Body dense SMPL-X observation 读取：从导出的 npz 加载 10475 个 SMPL-X 表面点，
还原到 equirect_rotate 全图坐标并标记 validity。

输入 npz 已在 exporter 中完成 MHR→SMPL-X 的 barycentric 映射，本 loader 仅负责
`equirect_crop_rotate_upright` → `equirect_rotate` 的坐标还原，不读取图像像素、不做任何
黑色区域/segmentation mask 过滤、不做下采样。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from lib_pano.human3d.crop_upright_coord import upright_points_to_equirect_rotate

SMPLX_NUM_VERTS = 10475


@dataclass
class SAM3DDenseResult:
    """
    单帧 SAM 3D Body dense SMPL-X surface observation，
    读取后统一输出为 equirect_rotate 全图坐标。
    """

    vertex_ids: np.ndarray      # [N], int32，SMPL-X vertex ids
    points_2d: np.ndarray       # [N, 2]，equirect_rotate full coordinates
    valid: np.ndarray           # [N] bool
    valid_z: np.ndarray         # [N] bool，来自输入 observation
    input_inside: np.ndarray    # [N] bool，upright crop 内有效


def load_sam3d_dense_result(
    frame_idx: int,
    sam3d_dense_dir: Path,
    cam_id: str,
    image_width: int,
    image_height: int,
    crop_bbox: list[int] | None = None,
    rotate_rule: str | None = None,
) -> SAM3DDenseResult | None:
    """从 SAM 3D Body dense npz 加载 SMPL-X 表面点，还原到 equirect_rotate 全图坐标并标记 validity。"""
    if crop_bbox is None:
        return None

    npz_path = sam3d_dense_dir / cam_id / f"{frame_idx:05d}.npz"
    if not npz_path.exists():
        return None

    data = np.load(npz_path, allow_pickle=True)

    vertex_ids = data["vertex_ids"].astype(np.int32)
    points_upright = data["points_2d"].astype(np.float32)
    valid_z = data["valid_z"].astype(bool)
    input_inside = data["inside"].astype(bool)
    input_valid = data["valid"].astype(bool)

    expected = (SMPLX_NUM_VERTS,)
    assert vertex_ids.shape == expected, (
        f"{npz_path}: vertex_ids shape {vertex_ids.shape} != {expected}"
    )
    assert points_upright.shape == (SMPLX_NUM_VERTS, 2), (
        f"{npz_path}: points_2d shape {points_upright.shape} != {(SMPLX_NUM_VERTS, 2)}"
    )
    assert valid_z.shape == expected, (
        f"{npz_path}: valid_z shape {valid_z.shape} != {expected}"
    )
    assert input_inside.shape == expected, (
        f"{npz_path}: inside shape {input_inside.shape} != {expected}"
    )
    assert input_valid.shape == expected, (
        f"{npz_path}: valid shape {input_valid.shape} != {expected}"
    )

    # upright crop → equirect_rotate full coordinates（复用已有还原函数）
    points_full = upright_points_to_equirect_rotate(
        points_upright,
        crop_bbox,
        frame_idx,
        rotate_rule=rotate_rule,
    )

    # full ERP 坐标合法性：仅几何判定，不读取像素、不做黑色区域/mask 过滤
    finite = np.isfinite(points_full).all(axis=1)
    x_in = (points_full[:, 0] >= 0) & (points_full[:, 0] < image_width)
    y_in = (points_full[:, 1] >= 0) & (points_full[:, 1] < image_height)

    valid = input_valid & finite & x_in & y_in

    return SAM3DDenseResult(
        vertex_ids=vertex_ids,
        points_2d=points_full.astype(np.float32),
        valid=valid,
        valid_z=valid_z,
        input_inside=input_inside,
    )


# Example:
# result = load_sam3d_dense_result(
#     frame_idx=513,
#     sam3d_dense_dir=Path(
#         "data_process/Output/20260311_riding/sam3dbody/smplx_dense"
#     ),
#     cam_id="cam2",
#     image_width=5760,
#     image_height=2880,
#     crop_bbox=[3614, 1091, 4323, 1560],
# )
