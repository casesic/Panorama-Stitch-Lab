"""Omni-radtan 鱼眼相机投影模型（NumPy / PyTorch 可微）"""

import numpy as np
import torch


def omni_project_np(pts: np.ndarray,
                    fx, fy, cx, cy, k1, k2, p1, p2, xi) -> np.ndarray:
    """(N,3) → (N,2) Omni-radtan 投影 (NumPy)"""
    X, Y, Z = pts[:, 0], pts[:, 1], pts[:, 2]
    s  = Z + xi
    mx = X / s; my = Y / s
    r2 = mx*mx + my*my
    D  = 1 + k1*r2 + k2*r2*r2
    u  = fx*(D*mx + 2*p1*mx*my + p2*(r2+2*mx*mx)) + cx
    v  = fy*(D*my + p1*(r2+2*my*my) + 2*p2*mx*my) + cy
    return np.stack([u, v], axis=-1)


def omni_project_torch(pts3d: torch.Tensor,
                       fx, fy, cx, cy, k1, k2, p1, p2, xi) -> torch.Tensor:
    """(B,N,3) → (B,N,2) Omni-radtan 投影 (PyTorch 可微)"""
    X, Y, Z = pts3d[..., 0], pts3d[..., 1], pts3d[..., 2]
    s  = Z + xi
    mx = X / s; my = Y / s
    r2 = mx*mx + my*my
    D  = 1 + k1*r2 + k2*r2*r2
    u  = fx*(D*mx + 2*p1*mx*my + p2*(r2+2*mx*mx)) + cx
    v  = fy*(D*my + p1*(r2+2*my*my) + 2*p2*mx*my) + cy
    return torch.stack([u, v], dim=-1)
