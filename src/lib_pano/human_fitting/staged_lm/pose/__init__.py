"""Canonical Stage 2 Pose implementation package."""

from .batched import (
    Stage2PoseBatchCache,
    build_stage2_pose_batched_linearization,
    build_stage2_pose_batch_cache,
    stage2_pose_batched_normal_equations,
)
from .cache import (
    Stage2PoseFastCache,
    build_stage2_pose_fast_cache,
)
from .derivatives import (
    build_stage2_pose_analytic_linearization,
    stage2_pose_explicit_raw_jacobian,
    stage2_pose_normal_equations,
)
from .forward import fast_raw_groups
from .objective import (
    stage2_pose_fast_raw_metrics,
    stage2_pose_fast_true_objective,
)

__all__ = [
    "Stage2PoseFastCache",
    "Stage2PoseBatchCache",
    "build_stage2_pose_fast_cache",
    "build_stage2_pose_batch_cache",
    "fast_raw_groups",
    "stage2_pose_explicit_raw_jacobian",
    "stage2_pose_normal_equations",
    "build_stage2_pose_analytic_linearization",
    "stage2_pose_batched_normal_equations",
    "build_stage2_pose_batched_linearization",
    "stage2_pose_fast_true_objective",
    "stage2_pose_fast_raw_metrics",
]
