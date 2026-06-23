"""3D 空间点投影：将相机坐标系下的空间点投影到等距柱状或鱼眼像素坐标（torch 实现）。"""

import torch
import numpy as np


def _project_to_equirect(pts_3d, out_w, out_h, R=None, mask_back=True):
    """Project 3D camera-frame points (B,N,3) to equirectangular pixel coordinates (B,N,2)."""
    if R is not None:
        pts_3d = torch.einsum('ij, bnj -> bni', R, pts_3d)

    X, Y, Z = pts_3d[..., 0], pts_3d[..., 1], pts_3d[..., 2]
    eps = torch.tensor(1e-8, dtype=pts_3d.dtype, device=pts_3d.device)
    norm = torch.sqrt(X * X + Y * Y + Z * Z).clamp_min(eps)

    x = X / norm
    y = Y / norm
    z = Z / norm

    lon = torch.atan2(x, z)
    lat = torch.asin(torch.clamp(-y, -1.0, 1.0))

    out_w_t = torch.as_tensor(out_w, dtype=pts_3d.dtype, device=pts_3d.device)
    out_h_t = torch.as_tensor(out_h, dtype=pts_3d.dtype, device=pts_3d.device)

    u = (lon / (2.0 * np.pi) + 0.5) * out_w_t
    v = (0.5 - lat / np.pi) * out_h_t

    uv = torch.stack([u, v], dim=-1)
    if mask_back:
        valid = z > 0
        invalid_value = torch.tensor(-1.0, dtype=uv.dtype, device=uv.device)
        uv = torch.where(valid.unsqueeze(-1), uv, invalid_value)
    return uv

def omni_project(pts_3d, K, D, xi, R):
    """Project 3D points (B,N,3) to fisheye pixels using the Omni + RadTan lens model."""
    if(R is not None):
        pts_3d = torch.einsum('ij, bnj -> bni', R, pts_3d) 
    X, Y, Z = pts_3d[..., 0], pts_3d[..., 1], pts_3d[..., 2]
    s = Z + xi
    mx = X / s
    my = Y / s
    r2 = mx * mx + my * my
    k1, k2, p1, p2 = D
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    d = 1 + k1*r2 + k2*r2*r2
    u = fx*(d*mx + p1*(r2 + 2*mx*mx) + p2*(2*mx*my)) + cx
    v = fy*(d*my + p1*(2*mx*my) + p2*(r2 + 2*my*my)) + cy
    return torch.stack([u, v], dim=-1)

def double_sphere_project(pts_3d, K, alpha, xi, R):
    """Project 3D points (B,N,3) to fisheye pixels using the Double Sphere lens model."""
    if(R is not None):
        pts_3d = torch.einsum('ij, bnj -> bni', R, pts_3d) 
    X, Y, Z = pts_3d[..., 0], pts_3d[..., 1], pts_3d[..., 2]
    
    r2 = X*X + Y*Y
    d1 = torch.sqrt(r2 + Z*Z)
    k = xi * d1 + Z
    d2 = torch.sqrt(r2 + k*k)
    norm = alpha * d2 + (1 - alpha) * k
    mx = X / norm
    my = Y / norm
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    u = fx*mx + cx
    v = fy*my + cy
    return torch.stack([u, v], dim=-1)


def omni_radtan_project_to_equirect(pts_3d, out_w, out_h, R=None, mask_back=True):
    """Shortcut: project 3D points to equirect pixels via Omni+RadTan model (delegates to _project_to_equirect)."""
    return _project_to_equirect(pts_3d, out_w=out_w, out_h=out_h, R=R, mask_back=mask_back)


def double_sphere_project_to_equirect(pts_3d, out_w, out_h, R=None, mask_back=True):
    """Shortcut: project 3D points to equirect pixels via Double Sphere model (delegates to _project_to_equirect)."""
    return _project_to_equirect(pts_3d, out_w=out_w, out_h=out_h, R=R, mask_back=mask_back)
    