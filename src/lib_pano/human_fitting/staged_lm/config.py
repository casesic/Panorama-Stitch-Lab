"""Numerical configuration and constants for staged LM fitting."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

# Rotation applied to cam4 extrinsics.  The fish-eye rig mounts cam4 facing the
# opposite direction of cam2, so a 180-degree flip about the world Z axis is
# required before both cameras share a common ERP frame.
R_CAM4: torch.Tensor = torch.tensor([-1, 0, 0, 0, 1, 0, 0, 0, -1], dtype=torch.float32).reshape(3, 3)

# Minimum number of valid SAM3D dense correspondences a single camera must
# contribute for a frame to be usable.  Below this the reprojection problem is
# under-determined.
MIN_VALID_POINTS = 16

# Per-frame parameter count for Stage 1: cam2 orient(3) + transl(3) +
# cam4 orient(3) + transl(3).
STAGE1_PER_FRAME_DIM = 12

# Numerical guard for the atanh/tanh beta parameterisation.  ``atanh`` diverges
# as its argument approaches +/-1, so the physical beta anchor must stay at
# least this far inside the ``beta_abs_limit`` boundary.
BETA_ATANH_EPS = 1e-6

# Fixed production implementation labels. These values are provenance only;
# production control flow does not select implementations from strings.
PRODUCTION_STAGE1_LINEARIZATION = "analytic_stage1"
PRODUCTION_STAGE1_EVALUATION = "analytic"
PRODUCTION_POSE_LINEARIZATION = "analytic"
PRODUCTION_POSE_EVALUATION = "fast"
PRODUCTION_JOINT_LINEARIZATION = "analytic"
PRODUCTION_JOINT_EVALUATION = "fast"
PRODUCTION_JOINT_LINEAR_SOLVER = "arrowhead_schur"
PRODUCTION_JOINT_FK = "combined"


@dataclass
class ScalingConstants:
    """Pre-computed scaling factors shared across all stages.

    Attributes
    ----------
    rep_scale:
        Normalisation factor for raw reprojection residuals.  Divides pixel
        residuals by ``sigma * sqrt(N)`` so the objective is dimensionless and
        independent of the total valid-point count.
    pose_anchor_scale:
        Per-element scale applied to the pose-anchor residual so the anchor
        term contributes ``pose_anchor_weight`` in expectation across all
        frames and pose dimensions.
    shape_anchor_scale:
        Per-element scale applied to the shape-anchor residual so the anchor
        term contributes ``shape_anchor_weight`` across all beta dimensions.
    """

    rep_scale: float
    pose_anchor_scale: float
    shape_anchor_scale: float


@dataclass(frozen=True)
class SolverConfig:
    """Numerical solver settings kept in code rather than exposed via CLI."""

    stage1_lm_max_acc: int = 20
    stage2_pose_warmup_lm_max_acc: int = 5
    stage2_joint_lm_max_acc: int = 1
    stage1_bf_outer: int = 15
    stage2_pose_warmup_bf_outer: int = 20
    stage2_joint_bf_outer: int = 20
    lm_max_trials: int = 8
    rel_obj_tol: float = 1e-7
    scaled_grad_tol: float = 1e-5
    step_tol: float = 1e-5
    lbfgs_max_iter: int = 10
    lbfgs_lr: float = 0.3
    irls_gradient_rel_tol: float = 5e-4


@dataclass(frozen=True)
class ObjectiveConfig:
    """Reprojection, anchor, and bounded-beta settings kept in code."""

    reproj_sigma_px: float = 10.0
    charbonnier_eps: float = 1.0
    pose_anchor_weight: float = 1.0
    beta_abs_limit: float = 2.5
    shape_anchor_weight: float = 10.0


def resolve_objective_config(namespace: Any) -> ObjectiveConfig:
    """Load saved nested/legacy objective values or use current code defaults."""
    # nested = getattr(namespace, "objective_config", None)
    # if isinstance(nested, dict):
    #     return ObjectiveConfig(**nested)
    # defaults = ObjectiveConfig()
    # values = {
    #     name: getattr(namespace, name, getattr(defaults, name))
    #     for name in ObjectiveConfig.__dataclass_fields__
    # }
    # return ObjectiveConfig(**values)
    return ObjectiveConfig()
