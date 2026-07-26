"""Canonical Stage 2 Joint implementation package."""

from .batched import (
    Stage2JointBatchCache,
    build_stage2_joint_batch_cache,
    build_stage2_joint_batched_linearization,
    stage2_joint_batched_normal_equations,
)
from .cache import (
    Stage2JointFastCache,
    Stage2JointFrameCameraCache,
    Stage2JointModelCache,
    build_stage2_joint_fast_cache,
    validate_stage2_joint_layout,
)
from .shape import (
    Stage2JointShapeState,
    evaluate_joint_shape_state,
    physical_beta_from_latent,
)
from .forward import (
    evaluate_joint_group_state,
    joint_fast_raw_groups,
    joint_fast_raw_residual,
    joint_fast_selected_vertices,
)
from .beta_derivatives import (
    joint_group_raw_residual_and_beta_jacobian,
    joint_shape_anchor_residual_and_jacobian,
)
from .derivatives import (
    CombinedFKResult,
    JointGroupAnalyticState,
    JointGroupJacobianResult,
    JointSurrogateResult,
    compute_joint_fk_with_combined_jacobian,
    evaluate_joint_group_analytic_state,
    joint_explicit_raw_global_jacobian,
    joint_group_raw_residual_and_jacobians,
    joint_pose_anchor_residual_and_jacobian,
    joint_surrogate_residual_and_jacobian,
)
from .objective import (
    stage2_joint_fast_objective_terms,
    stage2_joint_fast_raw_metrics,
    stage2_joint_fast_true_objective,
)
from .arrowhead import (
    accumulate_stage2_joint_arrowhead_system,
    build_stage2_joint_analytic_linearization,
    stage2_joint_arrowhead_normal_equations,
)

__all__ = [
    "Stage2JointModelCache", "Stage2JointFrameCameraCache",
    "Stage2JointFastCache", "Stage2JointBatchCache",
    "build_stage2_joint_fast_cache", "build_stage2_joint_batch_cache",
    "validate_stage2_joint_layout",
    "physical_beta_from_latent", "evaluate_joint_shape_state",
    "Stage2JointShapeState", "evaluate_joint_group_state",
    "joint_fast_raw_groups", "joint_fast_raw_residual",
    "joint_fast_selected_vertices",
    "joint_group_raw_residual_and_beta_jacobian",
    "joint_group_raw_residual_and_jacobians",
    "CombinedFKResult", "compute_joint_fk_with_combined_jacobian",
    "JointGroupAnalyticState", "JointGroupJacobianResult", "JointSurrogateResult",
    "evaluate_joint_group_analytic_state", "joint_explicit_raw_global_jacobian",
    "joint_pose_anchor_residual_and_jacobian",
    "joint_shape_anchor_residual_and_jacobian",
    "joint_surrogate_residual_and_jacobian",
    "accumulate_stage2_joint_arrowhead_system",
    "stage2_joint_arrowhead_normal_equations",
    "build_stage2_joint_analytic_linearization",
    "stage2_joint_batched_normal_equations",
    "build_stage2_joint_batched_linearization",
    "stage2_joint_fast_true_objective",
    "stage2_joint_fast_objective_terms", "stage2_joint_fast_raw_metrics",
]
