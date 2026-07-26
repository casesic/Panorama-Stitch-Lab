"""ERP projection kernels shared by rigid, pose and joint staged-LM stages."""

from __future__ import annotations

import math

import torch

from lib_pano.human_fitting.projection import ROTATION_MATRIX


def _erp_projection_values(
    points: torch.Tensor,
    image_width: int,
    image_height: int,
) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
    """Return ERP UV ``[..., 2]`` and shared values for points ``[..., 3]``."""
    x, y, z = points.unbind(-1)
    horizontal2 = x * x + z * z
    horizontal = torch.sqrt(horizontal2.clamp_min(1e-12))
    radius2 = horizontal2 + y * y
    longitude = torch.atan2(x, z)
    # Match production ``project_erp`` operation order for FP32 parity.
    radius = torch.norm(points, dim=-1).clamp_min(1e-6)
    latitude_argument = (-y / radius).clamp(-1.0 + 1e-6, 1.0 - 1e-6)
    latitude = torch.asin(latitude_argument)
    u = (longitude / (2.0 * math.pi) + 0.5) * image_width
    v = (0.5 - latitude / math.pi) * image_height
    return torch.stack((u, v), dim=-1), (x, y, z, horizontal2, horizontal, radius2, radius)


def erp_projection(
    points: torch.Tensor,
    image_width: int,
    image_height: int,
) -> torch.Tensor:
    """Project camera-space points ``[..., 3]`` to ERP UV ``[..., 2]``."""
    uv, _ = _erp_projection_values(points, image_width, image_height)
    return uv


def erp_projection_with_jacobian(
    points: torch.Tensor,
    image_width: int,
    image_height: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ERP UV ``[N, 2]`` and ``d(uv)/d(xyz) [N, 2, 3]``."""
    uv, values = _erp_projection_values(points, image_width, image_height)
    x, y, z, horizontal2, horizontal, radius2, radius = values
    safe_radius2 = radius2.clamp_min(1e-12)

    safe_horizontal2 = horizontal2.clamp_min(1e-12)
    du_scale = image_width / (2.0 * math.pi)
    dv_scale = image_height / math.pi
    du = torch.stack((
        du_scale * z / safe_horizontal2,
        torch.zeros_like(x),
        -du_scale * x / safe_horizontal2,
    ), dim=-1)
    dv = torch.stack((
        -dv_scale * y * x / (horizontal * safe_radius2),
        dv_scale * horizontal / safe_radius2,
        -dv_scale * y * z / (horizontal * safe_radius2),
    ), dim=-1)
    latitude_active = (
        (radius2 > 1e-12)
        & ((-y / radius) > -1.0 + 1e-6)
        & ((-y / radius) < 1.0 - 1e-6)
    )
    dv = torch.where(latitude_active.unsqueeze(-1), dv, torch.zeros_like(dv))
    return uv, torch.stack((du, dv), dim=-2)


def project_unified_erp(
    points: torch.Tensor,
    image_width: int,
    image_height: int,
) -> torch.Tensor:
    """Project unified cam2-coordinate points ``[N, 3]`` to ERP UV ``[N, 2]``."""
    fixed_rotation = ROTATION_MATRIX.to(device=points.device, dtype=points.dtype)
    return erp_projection(points @ fixed_rotation.T, image_width, image_height)


def project_unified_erp_with_jacobian(
    points: torch.Tensor,
    image_width: int,
    image_height: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return unified ERP UV ``[N, 2]`` and point Jacobian ``[N, 2, 3]``."""
    fixed_rotation = ROTATION_MATRIX.to(device=points.device, dtype=points.dtype)
    camera_points = points @ fixed_rotation.T
    uv, jacobian_camera = erp_projection_with_jacobian(camera_points, image_width, image_height)
    return uv, torch.einsum("nij,jk->nik", jacobian_camera, fixed_rotation)


# Compatibility aliases for external diagnostics during migration.
project_cam2_erp = project_unified_erp
project_cam2_erp_with_jacobian = project_unified_erp_with_jacobian
project_cam2_with_jacobian = project_unified_erp_with_jacobian
