"""Dynamic shared shape state for Stage 2 Joint.

Purpose: map shared beta latent to physical beta, shaped joints, and their
beta/latent derivatives at the current LM point. The state is computed once
per beta latent and shared by all frame/camera groups.
This module does not implement rotations, FK, LBS, objectives, or H/g.
Read :mod:`.forward` next.
"""

from __future__ import annotations

from dataclasses import dataclass
import time

import torch

from .cache import Stage2JointFastCache, _sync


@dataclass
class Stage2JointShapeState:
    """Dynamic shape values with beta dimension ``B`` and joints ``[J,3]``."""

    beta: torch.Tensor
    d_beta_d_latent: torch.Tensor
    joints: torch.Tensor
    d_joints_d_beta: torch.Tensor
    d_joints_d_latent: torch.Tensor


def physical_beta_from_latent(
    beta_latent: torch.Tensor, beta_abs_limit: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Map latent ``[B]`` to physical beta ``[B]`` and diagonal derivative."""
    latent = beta_latent.reshape(-1)
    tanh = torch.tanh(latent)
    beta = beta_abs_limit * tanh
    derivative = beta_abs_limit * (1 - tanh.square())
    if not torch.isfinite(beta).all() or not torch.isfinite(derivative).all():
        raise RuntimeError("NONFINITE_STAGE2_JOINT_BETA_MAPPING")
    return beta, derivative


def evaluate_joint_shape_state(
    cache: Stage2JointFastCache, beta_latent: torch.Tensor, beta_abs_limit: float,
) -> Stage2JointShapeState:
    """Evaluate joints ``[J,3]`` and beta/latent derivatives ``[J,3,B]``."""
    if beta_latent.numel() != cache.model.beta_dim:
        raise ValueError("Stage 2 joint beta latent dimension mismatch")
    _sync(cache, beta_latent); start = time.perf_counter()
    beta, derivative = physical_beta_from_latent(beta_latent, beta_abs_limit)
    d_joints = cache.model.d_joints_d_beta.to(beta_latent)
    joints = cache.model.base_joints.to(beta_latent) + torch.einsum("jcb,b->jc", d_joints, beta)
    result = Stage2JointShapeState(
        beta, derivative, joints, d_joints,
        d_joints * derivative.reshape(1, 1, -1),
    )
    _sync(cache, beta_latent)
    cache.shape_state_ms += (time.perf_counter() - start) * 1000.0
    cache.shape_state_calls += 1
    return result

