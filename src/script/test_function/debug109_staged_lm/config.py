"""CLI argument parsing and project-wide constants for the staged LM experiment.

This module centralises the command-line interface so the entry script stays
focused on orchestration. The CLI preserves the numerical defaults of the
original experiment while using the unified Stage 2 pose/joint terminology.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

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


def build_argument_parser() -> argparse.ArgumentParser:
    """Construct the argument parser for the staged LM experiment.

    Numerical defaults match the original experiment while option names use
    the unified Stage 2 pose/joint terminology.
    """
    parser = argparse.ArgumentParser(description="Staged LM with IRLS Charbonnier")
    parser.add_argument("--frames", type=str, required=True)
    parser.add_argument("--hmr_dir", type=str, required=True)
    parser.add_argument("--sam3d_dense_dir", type=str, required=True)
    parser.add_argument("--bbox_dir", type=str, required=True)
    parser.add_argument("--image_width", type=int, default=5760)
    parser.add_argument("--image_height", type=int, default=2880)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument(
        "--reproj_objective",
        choices=["raw_l2", "charbonnier_irls"],
        default="charbonnier_irls",
    )
    parser.add_argument(
        "--stop_after_stage",
        choices=["stage1", "stage2_pose", "stage2"],
        default="stage2",
    )
    parser.add_argument("--run_lbfgs_oracle", action="store_true", default=False)
    parser.add_argument(
        "--validate-irls-first-order",
        action=argparse.BooleanOptionalAction,
        dest="validate_irls_first_order",
        default=False,
    )
    parser.add_argument("--validate_irls_each_accepted", action="store_true", default=False)
    parser.add_argument("--profile_timing", action="store_true", default=False)
    parser.add_argument("--progress_every_accepted", type=int, default=0)
    return parser


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments and return the namespace."""
    return build_argument_parser().parse_args()


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
    stage2_pose_warmup_lm_max_acc: int = 10
    stage2_joint_lm_max_acc: int = 10
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


def resolve_objective_config(namespace: argparse.Namespace) -> ObjectiveConfig:
    """Load saved nested/legacy objective values or use current code defaults."""
    nested = getattr(namespace, "objective_config", None)
    if isinstance(nested, dict):
        return ObjectiveConfig(**nested)
    defaults = ObjectiveConfig()
    values = {
        name: getattr(namespace, name, getattr(defaults, name))
        for name in ObjectiveConfig.__dataclass_fields__
    }
    return ObjectiveConfig(**values)
