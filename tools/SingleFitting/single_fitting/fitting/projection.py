from __future__ import annotations

import math

import torch

from single_fitting.debug_shared import DS_PARAMS, R_CAM4

R_CAM4_TORCH = torch.tensor(R_CAM4, dtype=torch.float32)
DEFAULT_T_CAM4_INIT = torch.tensor([0.0, 0.0, -0.17355913120508845], dtype=torch.float32)


def ds_project_torch(
    points_3d: torch.Tensor,
    xi: float = DS_PARAMS["xi"],
    alpha: float = DS_PARAMS["alpha"],
    fx: float = DS_PARAMS["fx"],
    fy: float = DS_PARAMS["fy"],
    cx: float = DS_PARAMS["cx"],
    cy: float = DS_PARAMS["cy"],
) -> torch.Tensor:
    x = points_3d[..., 0]
    y = points_3d[..., 1]
    z = points_3d[..., 2]
    r2 = x**2 + y**2
    d1 = torch.sqrt(r2 + z**2).clamp_min(1e-6)
    d2 = torch.sqrt(r2 + (xi * d1 + z) ** 2).clamp_min(1e-6)
    w1 = (alpha * d2 + (1.0 - alpha) * (xi * d1 + z)).clamp_min(1e-6)
    return torch.stack([fx * x / w1 + cx, fy * y / w1 + cy], dim=-1)


def world_to_cam4_torch(points_world: torch.Tensor, delta_t: torch.Tensor) -> torch.Tensor:
    r = R_CAM4_TORCH.to(points_world.device, dtype=points_world.dtype)
    pts_flat = points_world.reshape(-1, 3)
    delta_t = delta_t.to(points_world.device, dtype=points_world.dtype).reshape(1, 3)
    pts_cam4 = pts_flat @ r.T - delta_t
    return pts_cam4.reshape(points_world.shape)


def project_world_to_cam2_torch(points_world: torch.Tensor) -> torch.Tensor:
    return ds_project_torch(points_world)


def project_world_to_cam4_torch(points_world: torch.Tensor, delta_t: torch.Tensor) -> torch.Tensor:
    return ds_project_torch(world_to_cam4_torch(points_world, delta_t))


ROTATION_MATRIX = torch.tensor(
    [1, 0, 0, 0, 0, 1, 0, -1, 0], dtype=torch.float32
).reshape(3, 3)


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
