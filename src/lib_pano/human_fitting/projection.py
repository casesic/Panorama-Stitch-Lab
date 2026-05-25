from __future__ import annotations

import math

import torch

ROTATION_MATRIX = torch.tensor(
    [1, 0, 0, 0, 0, 1, 0, -1, 0], dtype=torch.float32
).reshape(3, 3)

DEFAULT_T_CAM4_INIT = torch.tensor([0.0, 0.0, -0.17355913120508845], dtype=torch.float32)


def project_erp(points_3d: torch.Tensor, W: int, H: int) -> torch.Tensor:
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
    R = ROTATION_MATRIX.to(points_3d.device, dtype=points_3d.dtype)
    return project_erp(points_3d @ R.T, W, H)


def project_cam4(points_3d: torch.Tensor, W: int, H: int, R_cam4: torch.Tensor) -> torch.Tensor:
    R_cam4 = R_cam4.to(points_3d.device, dtype=points_3d.dtype)
    R_rot = ROTATION_MATRIX.to(points_3d.device, dtype=points_3d.dtype)
    pts_cam4 = points_3d @ R_cam4.T
    return project_erp(pts_cam4 @ R_rot.T, W, H)
