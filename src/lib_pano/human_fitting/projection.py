"""SMPL-X 拟合用的 ERP 投影：世界系点投影到等距柱状图，含 cam2/cam4 的相机旋转。"""

from __future__ import annotations

import math

import torch

ROTATION_MATRIX = torch.tensor(
    [1, 0, 0, 0, 0, 1, 0, -1, 0], dtype=torch.float32
).reshape(3, 3)

DEFAULT_T_CAM4_INIT = torch.tensor([0.0, 0.0, -0.17355913120508845], dtype=torch.float32)


def project_erp(points_3d: torch.Tensor, W: int, H: int) -> torch.Tensor:
    """将 3D 点投影为 ERP 像素坐标 (u,v)（torch 版，经度→u，纬度→v）。"""
    x = points_3d[..., 0]
    y = points_3d[..., 1]
    z = points_3d[..., 2]
    r = torch.norm(points_3d, dim=-1).clamp_min(1e-6)
    lon = torch.atan2(x, z)
    lat = torch.asin((-y / r).clamp(-1 + 1e-6, 1 - 1e-6))
    u = (lon / (2.0 * math.pi) + 0.5) * W
    v = (0.5 - lat / math.pi) * H
    return torch.stack([u, v], dim=-1)


def project_cam2(points_3d: torch.Tensor, W: int, H: int) -> torch.Tensor:
    """先施加固定 cam2 旋转再投影到 ERP。"""
    R = ROTATION_MATRIX.to(points_3d.device, dtype=points_3d.dtype)
    return project_erp(points_3d @ R.T, W, H)


def project_cam4(points_3d: torch.Tensor, W: int, H: int, R_cam4: torch.Tensor) -> torch.Tensor:
    """先施加 cam4 外参旋转和固定旋转，再投影到 ERP。"""
    R_cam4 = R_cam4.to(points_3d.device, dtype=points_3d.dtype)
    R_rot = ROTATION_MATRIX.to(points_3d.device, dtype=points_3d.dtype)
    pts_cam4 = points_3d @ R_cam4.T
    return project_erp(pts_cam4 @ R_rot.T, W, H)
