"""SO(3) kernels shared by rigid, pose and joint staged-LM stages."""

from __future__ import annotations

import torch


def skew(vectors: torch.Tensor) -> torch.Tensor:
    """Return ``[v]x`` for vectors shaped ``[..., 3]`` as ``[..., 3, 3]``."""
    zeros = torch.zeros_like(vectors[..., 0])
    x, y, z = vectors.unbind(-1)
    return torch.stack((
        zeros, -z, y,
        z, zeros, -x,
        -y, x, zeros,
    ), dim=-1).reshape(vectors.shape[:-1] + (3, 3))


def axis_angle_to_matrix(axis_angle: torch.Tensor) -> torch.Tensor:
    """Map ``axis_angle [..., 3]`` to rotation ``[..., 3, 3]``.

    Implements
    ``R(phi) = I + sin(theta)/theta [phi]x
    + (1-cos(theta))/theta^2 [phi]x^2``.
    """
    theta2 = torch.sum(axis_angle * axis_angle, dim=-1, keepdim=True)
    theta = torch.sqrt(theta2)
    small = theta2 < 1e-8
    safe_theta = torch.where(small, torch.ones_like(theta), theta)
    a_regular = torch.sin(theta) / safe_theta
    b_regular = (1.0 - torch.cos(theta)) / torch.where(small, torch.ones_like(theta2), theta2)
    a = torch.where(small, 1.0 - theta2 / 6.0 + theta2 * theta2 / 120.0, a_regular)
    b = torch.where(small, 0.5 - theta2 / 24.0 + theta2 * theta2 / 720.0, b_regular)
    phi_hat = skew(axis_angle)
    identity = torch.eye(3, dtype=axis_angle.dtype, device=axis_angle.device)
    return identity + a[..., None] * phi_hat + b[..., None] * (phi_hat @ phi_hat)


def so3_right_jacobian(axis_angle: torch.Tensor) -> torch.Tensor:
    """Return ``Jr(phi) [..., 3, 3]`` for ``axis_angle [..., 3]``."""
    theta2 = torch.sum(axis_angle * axis_angle, dim=-1, keepdim=True)
    theta = torch.sqrt(theta2)
    small = theta2 < 1e-8
    safe_theta2 = torch.where(small, torch.ones_like(theta2), theta2)
    a_regular = (1.0 - torch.cos(theta)) / safe_theta2
    b_regular = (theta - torch.sin(theta)) / torch.where(
        small, torch.ones_like(theta2), safe_theta2 * torch.where(small, torch.ones_like(theta), theta)
    )
    a = torch.where(small, 0.5 - theta2 / 24.0 + theta2 * theta2 / 720.0, a_regular)
    b = torch.where(small, 1.0 / 6.0 - theta2 / 120.0 + theta2 * theta2 / 5040.0, b_regular)
    phi_hat = skew(axis_angle)
    identity = torch.eye(3, dtype=axis_angle.dtype, device=axis_angle.device)
    return identity - a[..., None] * phi_hat + b[..., None] * (phi_hat @ phi_hat)


def rotated_point_jacobian(
    axis_angle: torch.Tensor,
    local_points: torch.Tensor,
) -> torch.Tensor:
    """Return ``d(Rp)/dphi = -R [p]x Jr(phi)`` with shape ``[N, 3, 3]``.

    ``axis_angle`` has shape ``[3]`` and ``local_points`` has shape ``[N, 3]``.
    """
    rotation = axis_angle_to_matrix(axis_angle.reshape(3))
    right_jacobian = so3_right_jacobian(axis_angle.reshape(3))
    return -torch.einsum("ij,njk,kl->nil", rotation, skew(local_points), right_jacobian)


# Compatibility alias for callers outside the staged-LM package.
_skew = skew
