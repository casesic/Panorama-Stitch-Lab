"""Static and observation-dependent cache construction for Stage 2 Joint.

Purpose: own model/cache dataclasses, selected immutable rows, layout
validation, timing counters, and serial cache preparation.
Mathematical scope stops before latent-beta mapping and dynamic shape.
This module does not implement FK, LBS, ERP, Jacobians, objectives, H/g, or
Schur solving. Read :mod:`.shape` next.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import torch
from smplx.lbs import blend_shapes, vertices2joints

from ..config import STAGE1_PER_FRAME_DIM
from ..stage2_shared_cache import Stage2SharedGeometryCache
from ..types import StageContext


@dataclass
class Stage2JointModelCache:
    base_v_shaped: torch.Tensor
    beta_shapedirs: torch.Tensor
    base_joints: torch.Tensor
    d_joints_d_beta: torch.Tensor
    parents: torch.Tensor
    joint_count: int
    pose_feature_dim: int
    full_vertex_count: int
    beta_dim: int
    body_pose_dim: int
    dtype: torch.dtype
    device: torch.device


@dataclass
class Stage2JointFrameCameraCache:
    frame_position: int
    frame_index: int
    camera: str
    vertex_ids: torch.Tensor
    targets: torch.Tensor
    left_hand_pose: torch.Tensor
    right_hand_pose: torch.Tensor
    scale: torch.Tensor
    selected_base_v_shaped: torch.Tensor
    selected_beta_shapedirs: torch.Tensor
    selected_posedirs: torch.Tensor
    selected_lbs_weights: torch.Tensor


@dataclass
class Stage2JointFastCache:
    model: Stage2JointModelCache
    groups: list[Stage2JointFrameCameraCache]
    frame_count: int
    image_width: int
    image_height: int
    setup_ms: float
    batch_cache: Any | None = None
    batch_cache_setup_ms: float = 0.0
    batch_padding_ratio: float = 0.0
    batch_persistent_bytes: int = 0
    batch_group_count: int = 0
    batch_nmax: int = 0
    profile_timing: bool = False
    shape_state_ms: float = 0.0
    shape_state_calls: int = 0
    residual_ms: float = 0.0
    residual_calls: int = 0
    beta_jacobian_ms: float = 0.0
    beta_jacobian_calls: int = 0
    group_analytic_calls: int = 0
    selected_lbs_forward_calls: int = 0
    normal_equations_ms: float = 0.0
    normal_equations_calls: int = 0
    objective_ms: float = 0.0
    objective_calls: int = 0
    metrics_ms: float = 0.0
    metrics_calls: int = 0
    explicit_jacobian_ms: float = 0.0
    explicit_jacobian_calls: int = 0
    arrowhead_accumulation_ms: float = 0.0
    arrowhead_accumulation_calls: int = 0
    schur_solve_ms: float = 0.0
    schur_solve_calls: int = 0
    schur_local_factor_ms: float = 0.0
    schur_reduced_build_ms: float = 0.0
    schur_reduced_solve_ms: float = 0.0
    schur_back_substitution_ms: float = 0.0
    schur_dense_fallback_calls: int = 0
    structured_to_dense_calls: int = 0
    fk_combined_calls: int = 0
    fk_separate_calls: int = 0
    batched_normal_equations_ms: float = 0.0
    batched_normal_equations_calls: int = 0
    batched_full_analytic_calls: int = 0
    batched_shape_state_calls: int = 0

    def reset_timing(self) -> None:
        self.shape_state_ms = self.residual_ms = self.beta_jacobian_ms = 0.0
        self.normal_equations_ms = self.objective_ms = self.metrics_ms = self.explicit_jacobian_ms = 0.0
        self.arrowhead_accumulation_ms = self.schur_solve_ms = 0.0
        self.schur_local_factor_ms = self.schur_reduced_build_ms = 0.0
        self.schur_reduced_solve_ms = self.schur_back_substitution_ms = 0.0
        self.shape_state_calls = self.residual_calls = self.beta_jacobian_calls = 0
        self.normal_equations_calls = self.objective_calls = self.metrics_calls = self.explicit_jacobian_calls = 0
        self.arrowhead_accumulation_calls = self.schur_solve_calls = 0
        self.schur_dense_fallback_calls = self.structured_to_dense_calls = 0
        self.group_analytic_calls = self.selected_lbs_forward_calls = 0
        self.fk_combined_calls = self.fk_separate_calls = 0
        self.batched_normal_equations_ms = 0.0
        self.batched_normal_equations_calls = 0
        self.batched_full_analytic_calls = 0
        self.batched_shape_state_calls = 0

    def timing_dict(self) -> dict[str, float | int]:
        result: dict[str, float | int] = {
            "stage2_joint_cache_setup_ms": self.setup_ms + self.batch_cache_setup_ms,
            "stage2_joint_derived_cache_setup_ms": self.setup_ms,
            "stage2_joint_fast_cache_setup_ms": self.setup_ms,
            "stage2_joint_batch_cache_setup_ms": self.batch_cache_setup_ms,
            "stage2_joint_batch_padding_ratio": self.batch_padding_ratio,
            "stage2_joint_batch_persistent_bytes": self.batch_persistent_bytes,
            "stage2_joint_batch_group_count": self.batch_group_count,
            "stage2_joint_batch_nmax": self.batch_nmax,
        }
        for name in ("shape_state", "residual", "beta_jacobian", "normal_equations",
                     "objective", "metrics", "explicit_jacobian", "arrowhead_accumulation",
                     "schur_solve"):
            total, calls = getattr(self, f"{name}_ms"), getattr(self, f"{name}_calls")
            result[f"stage2_joint_{name}_ms"] = total
            result[f"stage2_joint_{name}_calls"] = calls
            result[f"stage2_joint_{name}_ms_per_call"] = total / calls if calls else 0.0
        result["stage2_joint_group_analytic_calls"] = self.group_analytic_calls
        result["stage2_joint_selected_lbs_forward_calls"] = self.selected_lbs_forward_calls
        result["stage2_joint_fk_combined_calls"] = self.fk_combined_calls
        result["stage2_joint_fk_separate_calls"] = self.fk_separate_calls
        for name in ("schur_local_factor", "schur_reduced_build", "schur_reduced_solve",
                     "schur_back_substitution"):
            result[f"stage2_joint_{name}_ms"] = getattr(self, f"{name}_ms")
        result["stage2_joint_schur_dense_fallback_calls"] = self.schur_dense_fallback_calls
        result["stage2_joint_structured_to_dense_calls"] = self.structured_to_dense_calls
        result["stage2_joint_batched_normal_equations_ms"] = self.batched_normal_equations_ms
        result["stage2_joint_batched_normal_equations_calls"] = self.batched_normal_equations_calls
        result["stage2_joint_batched_normal_equations_ms_per_call"] = (
            self.batched_normal_equations_ms / self.batched_normal_equations_calls
            if self.batched_normal_equations_calls else 0.0
        )
        result["stage2_joint_batched_full_analytic_calls"] = self.batched_full_analytic_calls
        result["stage2_joint_batched_shape_state_calls"] = self.batched_shape_state_calls
        return result

    def true_objective(self, parameters: torch.Tensor, beta_abs_limit: float,
                       beta_anchor: torch.Tensor, body_pose_anchors: list[torch.Tensor],
                       pose_anchor_scale: float, shape_anchor_scale: float,
                       objective_mode: str, charbonnier_eps: float, rep_scale: float,
                       aggregation_spec: dict[str, Any] | None) -> torch.Tensor:
        from .objective import stage2_joint_fast_true_objective
        return stage2_joint_fast_true_objective(
            self, parameters, beta_abs_limit, beta_anchor, body_pose_anchors,
            pose_anchor_scale, shape_anchor_scale, objective_mode,
            charbonnier_eps, rep_scale, aggregation_spec,
        )

    def raw_metrics(self, parameters: torch.Tensor, beta_abs_limit: float) -> dict[str, float]:
        from .objective import stage2_joint_fast_raw_metrics
        return stage2_joint_fast_raw_metrics(self, parameters, beta_abs_limit)


def validate_stage2_joint_layout(
    cache: Stage2JointFastCache, parameters: torch.Tensor,
) -> tuple[int, int]:
    per_frame_dim = STAGE1_PER_FRAME_DIM + cache.model.body_pose_dim
    expected = cache.frame_count * per_frame_dim + cache.model.beta_dim
    body_joint_count = cache.model.body_pose_dim // 3
    if (cache.model.body_pose_dim % 3 != 0 or body_joint_count != 21
            or parameters.numel() != expected):
        raise RuntimeError(
            "UNSUPPORTED_STAGE2_JOINT_ANALYTIC_LAYOUT: "
            f"expected parameter dimension={expected}, actual parameter dimension={parameters.numel()}, "
            f"frame_count={cache.frame_count}, per_frame_dim={per_frame_dim}, "
            f"body_pose_dim={cache.model.body_pose_dim}, beta_dim={cache.model.beta_dim}, "
            f"model_NUM_BODY_JOINTS={body_joint_count}"
        )
    return per_frame_dim, expected


def _sync(cache: Stage2JointFastCache, tensor: torch.Tensor) -> None:
    if cache.profile_timing and tensor.is_cuda:
        torch.cuda.synchronize(tensor.device)


def build_stage2_joint_fast_cache(
    context: StageContext,
    shared_cache: Stage2SharedGeometryCache,
) -> Stage2JointFastCache:
    """Build dynamic-beta derived state while reusing shared selected rows."""
    start = time.perf_counter()
    model = context.model
    if model.shapedirs.shape[-1] != context.beta_dim:
        raise RuntimeError(
            "UNSUPPORTED_STAGE2_JOINT_FORWARD_CONTRACT: "
            f"shapedirs beta dimension {model.shapedirs.shape[-1]} != {context.beta_dim}"
        )
    if context.body_pose_dim != int(model.NUM_BODY_JOINTS) * 3:
        raise RuntimeError("UNSUPPORTED_STAGE2_JOINT_FORWARD_CONTRACT: body pose dimension")
    layout = shared_cache.model
    device, dtype = layout.device, layout.dtype
    zero_beta = torch.zeros(1, context.beta_dim, dtype=dtype, device=device)
    zero_expression = torch.zeros(1, int(model.num_expression_coeffs), dtype=dtype, device=device)
    shape_components = torch.cat((zero_beta, zero_expression), -1)
    shape_dirs = torch.cat((model.shapedirs, model.expr_dirs), -1)
    base_v_shaped = model.v_template + blend_shapes(shape_components, shape_dirs).squeeze(0)
    beta_shapedirs = model.shapedirs
    base_joints = vertices2joints(model.J_regressor, base_v_shaped.unsqueeze(0)).squeeze(0)
    d_joints = torch.einsum("jv,vcb->jcb", model.J_regressor, beta_shapedirs)
    model_cache = Stage2JointModelCache(
        base_v_shaped.detach(), beta_shapedirs.detach(), base_joints.detach(),
        d_joints.detach(), layout.parents, layout.joint_count,
        layout.pose_feature_dim, layout.full_vertex_count, context.beta_dim,
        context.body_pose_dim, dtype, device,
    )
    groups: list[Stage2JointFrameCameraCache] = []
    for shared in shared_cache.groups:
        groups.append(Stage2JointFrameCameraCache(
            shared.frame_position, shared.frame_index, shared.camera,
            shared.vertex_ids, shared.targets, shared.left_hand_pose,
            shared.right_hand_pose, shared.scale,
            base_v_shaped.index_select(0, shared.vertex_ids).detach(),
            beta_shapedirs.index_select(0, shared.vertex_ids).detach(),
            shared.selected_posedirs, shared.selected_lbs_weights,
        ))
    return Stage2JointFastCache(
        model_cache, groups, context.frame_count, context.image_width, context.image_height,
        (time.perf_counter() - start) * 1000.0, context.args.profile_timing,
    )

