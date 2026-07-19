"""Dynamic-beta selected-vertex forward for Stage 2 joint diagnostics."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import torch
from smplx.lbs import batch_rigid_transform, blend_shapes, vertices2joints

from .analytic_stage1 import project_cam2_erp
from .config import STAGE1_PER_FRAME_DIM
from .irls import compute_primary_charbonnier_tensor
from .parameter_layout import unpack_frame_parameters
from .residuals import flatten_reprojection_groups, raw_reproj_metrics
from .stage2_pose_fast import (
    assemble_full_pose_rotations,
    compute_selected_lbs,
    compute_selected_pose_offsets,
)
from .stage2_shared_cache import Stage2SharedGeometryCache
from .types import ReprojectionGroup, StageContext


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
class Stage2JointShapeState:
    beta: torch.Tensor
    d_beta_d_latent: torch.Tensor
    joints: torch.Tensor
    d_joints_d_beta: torch.Tensor
    d_joints_d_latent: torch.Tensor


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
        return stage2_joint_fast_true_objective(
            self, parameters, beta_abs_limit, beta_anchor, body_pose_anchors,
            pose_anchor_scale, shape_anchor_scale, objective_mode,
            charbonnier_eps, rep_scale, aggregation_spec,
        )

    def raw_metrics(self, parameters: torch.Tensor, beta_abs_limit: float) -> dict[str, float]:
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


def physical_beta_from_latent(
    beta_latent: torch.Tensor, beta_abs_limit: float,
) -> tuple[torch.Tensor, torch.Tensor]:
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


def evaluate_joint_group_state(
    cache: Stage2JointFastCache,
    camera_cache: Stage2JointFrameCameraCache,
    shape_state: Stage2JointShapeState,
    global_orient: torch.Tensor,
    translation: torch.Tensor,
    body_pose: torch.Tensor,
) -> dict[str, torch.Tensor]:
    rotations = assemble_full_pose_rotations(
        global_orient, body_pose, camera_cache.left_hand_pose.to(global_orient),
        camera_cache.right_hand_pose.to(global_orient), cache.model,
    )
    pose_offsets, pose_feature = compute_selected_pose_offsets(
        rotations, camera_cache.selected_posedirs.to(global_orient),
    )
    selected_v_shaped = camera_cache.selected_base_v_shaped.to(global_orient) + torch.einsum(
        "ncb,b->nc", camera_cache.selected_beta_shapedirs.to(global_orient), shape_state.beta,
    )
    selected_v_posed = selected_v_shaped + pose_offsets
    posed_joints, relative_transforms = batch_rigid_transform(
        rotations.unsqueeze(0), shape_state.joints.unsqueeze(0),
        cache.model.parents.to(global_orient.device), dtype=global_orient.dtype,
    )
    skinned, selected_transforms = compute_selected_lbs(
        selected_v_posed, camera_cache.selected_lbs_weights.to(global_orient),
        relative_transforms[0],
    )
    final = skinned * camera_cache.scale.to(global_orient).reshape(1, 1) + translation.reshape(1, 3)
    return {
        "physical_beta": shape_state.beta,
        "selected_v_shaped": selected_v_shaped, "joints": shape_state.joints,
        "rotations": rotations, "pose_feature": pose_feature,
        "selected_pose_offsets": pose_offsets, "selected_v_posed": selected_v_posed,
        "relative_transforms": relative_transforms[0], "posed_joints": posed_joints[0],
        "selected_transforms": selected_transforms,
        "selected_skinned_vertices": skinned, "final_selected_vertices": final,
    }


def joint_fast_selected_vertices(
    cache: Stage2JointFastCache, camera_cache: Stage2JointFrameCameraCache,
    shape_state: Stage2JointShapeState, global_orient: torch.Tensor,
    translation: torch.Tensor, body_pose: torch.Tensor, *, return_intermediates: bool = False,
):
    state = evaluate_joint_group_state(
        cache, camera_cache, shape_state, global_orient, translation, body_pose,
    )
    return (state["final_selected_vertices"], state) if return_intermediates else state["final_selected_vertices"]


def joint_fast_raw_groups(
    cache: Stage2JointFastCache, parameters: torch.Tensor, beta_abs_limit: float,
) -> list[ReprojectionGroup]:
    per_frame_dim, _ = validate_stage2_joint_layout(cache, parameters)
    shape_state = evaluate_joint_shape_state(cache, parameters[-cache.model.beta_dim:], beta_abs_limit)
    groups = []
    for camera_cache in cache.groups:
        frame = unpack_frame_parameters(parameters, camera_cache.frame_position, per_frame_dim, True)
        orient = frame.cam2_orient if camera_cache.camera == "cam2" else frame.cam4_orient
        translation = frame.cam2_transl if camera_cache.camera == "cam2" else frame.cam4_transl
        vertices = joint_fast_selected_vertices(
            cache, camera_cache, shape_state, orient.reshape(3), translation.reshape(3),
            frame.body_pose.reshape(-1),
        )
        uv = project_cam2_erp(vertices, cache.image_width, cache.image_height)
        groups.append(ReprojectionGroup(
            camera_cache.frame_index, camera_cache.camera,
            uv - camera_cache.targets.to(parameters),
        ))
    return groups


def joint_fast_raw_residual(
    cache: Stage2JointFastCache, parameters: torch.Tensor, beta_abs_limit: float,
) -> torch.Tensor:
    _sync(cache, parameters); start = time.perf_counter()
    result = torch.cat([
        group.residual_2d.reshape(-1)
        for group in joint_fast_raw_groups(cache, parameters, beta_abs_limit)
    ])
    _sync(cache, parameters)
    cache.residual_ms += (time.perf_counter() - start) * 1000.0
    cache.residual_calls += 1
    return result


def _joint_anchor_residuals(
    cache: Stage2JointFastCache, parameters: torch.Tensor, beta_abs_limit: float,
    beta_anchor: torch.Tensor, body_pose_anchors: list[torch.Tensor],
    pose_anchor_scale: float, shape_anchor_scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    per_frame_dim, _ = validate_stage2_joint_layout(cache, parameters)
    pose = torch.cat([
        (parameters[f * per_frame_dim + STAGE1_PER_FRAME_DIM:(f + 1) * per_frame_dim]
         - body_pose_anchors[f].to(parameters)) * pose_anchor_scale
        for f in range(cache.frame_count)
    ])
    beta, _ = physical_beta_from_latent(parameters[-cache.model.beta_dim:], beta_abs_limit)
    shape = (beta - beta_anchor.to(parameters)) * shape_anchor_scale
    return pose, shape


def stage2_joint_fast_true_objective(
    cache: Stage2JointFastCache, parameters: torch.Tensor, beta_abs_limit: float,
    beta_anchor: torch.Tensor, body_pose_anchors: list[torch.Tensor],
    pose_anchor_scale: float, shape_anchor_scale: float,
    objective_mode: str, charbonnier_eps: float, rep_scale: float,
    aggregation_spec: dict[str, Any] | None,
) -> torch.Tensor:
    return _stage2_joint_fast_objective_terms(
        cache, parameters, beta_abs_limit, beta_anchor, body_pose_anchors,
        pose_anchor_scale, shape_anchor_scale, objective_mode,
        charbonnier_eps, rep_scale, aggregation_spec, record_timing=True,
    )["total"]


def stage2_joint_fast_objective_terms(
    cache: Stage2JointFastCache, parameters: torch.Tensor, beta_abs_limit: float,
    beta_anchor: torch.Tensor, body_pose_anchors: list[torch.Tensor],
    pose_anchor_scale: float, shape_anchor_scale: float,
    objective_mode: str, charbonnier_eps: float, rep_scale: float,
    aggregation_spec: dict[str, Any] | None,
) -> dict[str, torch.Tensor]:
    """Evaluate all joint objective terms with one dynamic selected forward."""
    return _stage2_joint_fast_objective_terms(
        cache, parameters, beta_abs_limit, beta_anchor, body_pose_anchors,
        pose_anchor_scale, shape_anchor_scale, objective_mode,
        charbonnier_eps, rep_scale, aggregation_spec, record_timing=False,
    )


def _stage2_joint_fast_objective_terms(
    cache: Stage2JointFastCache, parameters: torch.Tensor, beta_abs_limit: float,
    beta_anchor: torch.Tensor, body_pose_anchors: list[torch.Tensor],
    pose_anchor_scale: float, shape_anchor_scale: float,
    objective_mode: str, charbonnier_eps: float, rep_scale: float,
    aggregation_spec: dict[str, Any] | None,
    *, record_timing: bool,
) -> dict[str, torch.Tensor]:
    if record_timing:
        _sync(cache, parameters)
    start = time.perf_counter()
    groups = joint_fast_raw_groups(cache, parameters, beta_abs_limit)
    if objective_mode == "charbonnier_irls":
        if aggregation_spec is None: raise ValueError("aggregation spec required")
        primary = compute_primary_charbonnier_tensor(groups, charbonnier_eps, aggregation_spec)
    elif objective_mode == "raw_l2":
        raw = flatten_reprojection_groups(groups) * rep_scale
        primary = .5 * torch.sum(raw.square())
    else: raise ValueError(objective_mode)
    pose, shape = _joint_anchor_residuals(
        cache, parameters, beta_abs_limit, beta_anchor, body_pose_anchors,
        pose_anchor_scale, shape_anchor_scale,
    )
    pose_anchor = .5 * torch.sum(pose.square())
    shape_anchor = .5 * torch.sum(shape.square())
    result = {
        "primary": primary,
        "pose_anchor": pose_anchor,
        "shape_anchor": shape_anchor,
        "total": primary + pose_anchor + shape_anchor,
    }
    if record_timing:
        _sync(cache, parameters)
        cache.objective_ms += (time.perf_counter()-start)*1000
        cache.objective_calls += 1
    return result


def stage2_joint_fast_raw_metrics(
    cache: Stage2JointFastCache, parameters: torch.Tensor, beta_abs_limit: float,
) -> dict[str, float]:
    _sync(cache, parameters); start=time.perf_counter()
    result=raw_reproj_metrics(flatten_reprojection_groups(
        joint_fast_raw_groups(cache,parameters,beta_abs_limit)).detach())
    _sync(cache, parameters); cache.metrics_ms+=(time.perf_counter()-start)*1000; cache.metrics_calls+=1
    return result
