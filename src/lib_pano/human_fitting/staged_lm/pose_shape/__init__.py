"""Canonical Stage 2 pose/shape implementation package."""

from .batched import (
    Stage2PoseShapeBatchCache,
    build_stage2_pose_shape_batch_cache,
    build_stage2_pose_shape_batched_linearization,
    stage2_pose_shape_batched_normal_equations,
)
from .cache import (
    Stage2PoseShapeFastCache,
    Stage2PoseShapeFrameCameraCache,
    Stage2PoseShapeModelCache,
    build_stage2_pose_shape_fast_cache,
    validate_stage2_pose_shape_layout,
)
from .shape import (
    Stage2PoseShapeState,
    evaluate_pose_shape_state,
    physical_beta_from_latent,
)
from .forward import (
    evaluate_pose_shape_group_state,
    pose_shape_fast_raw_groups,
    pose_shape_fast_raw_residual,
    pose_shape_fast_selected_vertices,
)
from .beta_derivatives import (
    pose_shape_group_raw_residual_and_beta_jacobian,
    pose_shape_anchor_residual_and_jacobian,
)
from .derivatives import (
    CombinedFKResult,
    PoseShapeGroupAnalyticState,
    PoseShapeGroupJacobianResult,
    PoseShapeSurrogateResult,
    compute_pose_shape_fk_with_combined_jacobian,
    evaluate_pose_shape_group_analytic_state,
    pose_shape_explicit_raw_global_jacobian,
    pose_shape_group_raw_residual_and_jacobians,
    pose_shape_pose_anchor_residual_and_jacobian,
    pose_shape_surrogate_residual_and_jacobian,
)
from .objective import (
    stage2_pose_shape_fast_objective_terms,
    stage2_pose_shape_fast_raw_metrics,
    stage2_pose_shape_fast_true_objective,
)
from .arrowhead import (
    accumulate_stage2_pose_shape_arrowhead_system,
    build_stage2_pose_shape_analytic_linearization,
    stage2_pose_shape_arrowhead_normal_equations,
)

__all__ = [
    "Stage2PoseShapeModelCache", "Stage2PoseShapeFrameCameraCache",
    "Stage2PoseShapeFastCache", "Stage2PoseShapeBatchCache",
    "build_stage2_pose_shape_fast_cache", "build_stage2_pose_shape_batch_cache",
    "validate_stage2_pose_shape_layout",
    "physical_beta_from_latent", "evaluate_pose_shape_state",
    "Stage2PoseShapeState", "evaluate_pose_shape_group_state",
    "pose_shape_fast_raw_groups", "pose_shape_fast_raw_residual",
    "pose_shape_fast_selected_vertices",
    "pose_shape_group_raw_residual_and_beta_jacobian",
    "pose_shape_group_raw_residual_and_jacobians",
    "CombinedFKResult", "compute_pose_shape_fk_with_combined_jacobian",
    "PoseShapeGroupAnalyticState", "PoseShapeGroupJacobianResult", "PoseShapeSurrogateResult",
    "evaluate_pose_shape_group_analytic_state", "pose_shape_explicit_raw_global_jacobian",
    "pose_shape_pose_anchor_residual_and_jacobian",
    "pose_shape_anchor_residual_and_jacobian",
    "pose_shape_surrogate_residual_and_jacobian",
    "accumulate_stage2_pose_shape_arrowhead_system",
    "stage2_pose_shape_arrowhead_normal_equations",
    "build_stage2_pose_shape_analytic_linearization",
    "stage2_pose_shape_batched_normal_equations",
    "build_stage2_pose_shape_batched_linearization",
    "stage2_pose_shape_fast_true_objective",
    "stage2_pose_shape_fast_objective_terms", "stage2_pose_shape_fast_raw_metrics",
]
