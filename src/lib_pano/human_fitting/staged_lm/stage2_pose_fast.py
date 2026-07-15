"""Selected-vertex SMPL-X forward and evaluation for Stage 2 pose.

It reproduces the installed SMPL-X Layer/LBS contract while materializing only
selected pose offsets, vertex transforms, and final vertices. The full model
forward remains the differentiable L-BFGS oracle.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import torch
from smplx.lbs import batch_rigid_transform, blend_shapes, vertices2joints

from lib_pano.human_fitting import smplx_sequence as _smplx_sequence  # noqa: F401
from lib_dynamic_hvip.smpl.smpl_lib import _aa_to_rotmat

from .analytic_stage1 import project_cam2_erp
from .config import STAGE1_PER_FRAME_DIM
from .irls import compute_primary_charbonnier_tensor
from .parameter_layout import unpack_frame_parameters
from .residuals import flatten_reprojection_groups, raw_reproj_metrics
from .stage2_shared_cache import Stage2SharedGeometryCache
from .types import ReprojectionGroup, StageContext


def _unsupported(message: str) -> RuntimeError:
    return RuntimeError(f"UNSUPPORTED_SMPLX_FORWARD_CONTRACT: {message}")


@dataclass
class Stage2PoseModelCache:
    v_shaped: torch.Tensor
    joints: torch.Tensor
    parents: torch.Tensor
    joint_count: int
    pose_feature_dim: int
    full_vertex_count: int
    dtype: torch.dtype
    device: torch.device


@dataclass
class Stage2PoseFrameCameraCache:
    frame_position: int
    frame_index: int
    camera: str
    vertex_ids: torch.Tensor
    targets: torch.Tensor
    left_hand_pose: torch.Tensor
    right_hand_pose: torch.Tensor
    scale: torch.Tensor
    selected_v_shaped: torch.Tensor
    selected_posedirs: torch.Tensor
    selected_lbs_weights: torch.Tensor


@dataclass
class Stage2PoseFastCache:
    model: Stage2PoseModelCache
    groups: list[Stage2PoseFrameCameraCache]
    frame_count: int
    body_pose_dim: int
    image_width: int
    image_height: int
    setup_ms: float
    profile_timing: bool = False
    residual_ms: float = 0.0
    residual_calls: int = 0
    normal_equations_ms: float = 0.0
    normal_equations_calls: int = 0
    objective_ms: float = 0.0
    objective_calls: int = 0
    metrics_ms: float = 0.0
    metrics_calls: int = 0
    explicit_jacobian_ms: float = 0.0
    explicit_jacobian_calls: int = 0
    group_analytic_calls: int = 0

    def reset_timing(self) -> None:
        self.residual_ms = self.normal_equations_ms = 0.0
        self.objective_ms = self.metrics_ms = self.explicit_jacobian_ms = 0.0
        self.residual_calls = self.normal_equations_calls = 0
        self.objective_calls = self.metrics_calls = self.explicit_jacobian_calls = 0
        self.group_analytic_calls = 0

    @staticmethod
    def _per_call(total: float, calls: int) -> float:
        return total / calls if calls else 0.0

    def timing_dict(self) -> dict[str, float | int]:
        prefix = "stage2_pose_"
        result: dict[str, float | int] = {
            f"{prefix}cache_setup_ms": self.setup_ms,
            f"{prefix}derived_cache_setup_ms": self.setup_ms,
        }
        for name in ("residual", "normal_equations", "objective", "metrics", "explicit_jacobian"):
            total = float(getattr(self, f"{name}_ms"))
            calls = int(getattr(self, f"{name}_calls"))
            result[f"{prefix}{name}_ms"] = total
            result[f"{prefix}{name}_calls"] = calls
            result[f"{prefix}{name}_ms_per_call"] = self._per_call(total, calls)
        result[f"{prefix}group_analytic_calls"] = self.group_analytic_calls
        return result

    def true_objective(
        self,
        parameters: torch.Tensor,
        body_pose_anchors: list[torch.Tensor],
        pose_anchor_scale: float,
        objective_mode: str,
        charbonnier_eps: float,
        rep_scale: float,
        aggregation_spec: dict[str, Any] | None,
    ) -> torch.Tensor:
        return stage2_pose_fast_true_objective(
            self, parameters, body_pose_anchors, pose_anchor_scale,
            objective_mode, charbonnier_eps, rep_scale, aggregation_spec,
        )

    def raw_metrics(self, parameters: torch.Tensor) -> dict[str, float]:
        return stage2_pose_fast_raw_metrics(self, parameters)


def _sync(cache: Stage2PoseFastCache, parameters: torch.Tensor) -> None:
    if cache.profile_timing and parameters.is_cuda:
        torch.cuda.synchronize(parameters.device)


def compute_selected_pose_offsets(
    rotation_matrices: torch.Tensor,
    selected_posedirs: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute selected pose offsets from SMPL-X's non-root pose feature."""
    if rotation_matrices.ndim != 3 or rotation_matrices.shape[-2:] != (3, 3):
        raise _unsupported(f"rotation matrices have shape {tuple(rotation_matrices.shape)}")
    identity = torch.eye(3, dtype=rotation_matrices.dtype, device=rotation_matrices.device)
    pose_feature = (rotation_matrices[1:] - identity).reshape(-1)
    if selected_posedirs.shape[0] != pose_feature.numel() or selected_posedirs.shape[-1] != 3:
        raise _unsupported(
            f"selected posedirs {tuple(selected_posedirs.shape)} vs pose feature {pose_feature.numel()}"
        )
    offsets = torch.einsum("p,pnc->nc", pose_feature, selected_posedirs)
    return offsets, pose_feature


def compute_selected_lbs(
    selected_v_posed: torch.Tensor,
    selected_lbs_weights: torch.Tensor,
    relative_transforms: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Skin selected vertices without constructing transforms for all vertices."""
    joint_count = relative_transforms.shape[0]
    if selected_lbs_weights.shape != (selected_v_posed.shape[0], joint_count):
        raise _unsupported("selected LBS weight shape mismatch")
    selected_transforms = (
        selected_lbs_weights @ relative_transforms.reshape(joint_count, 16)
    ).reshape(-1, 4, 4)
    homogeneous = torch.cat((
        selected_v_posed,
        torch.ones(
            selected_v_posed.shape[0], 1,
            dtype=selected_v_posed.dtype, device=selected_v_posed.device,
        ),
    ), dim=1)
    vertices = (selected_transforms @ homogeneous.unsqueeze(-1))[:, :3, 0]
    return vertices, selected_transforms


def build_stage2_pose_fast_cache(
    context: StageContext,
    shared_cache: Stage2SharedGeometryCache,
) -> Stage2PoseFastCache:
    """Build the frozen-beta Stage 2 derived cache from shared geometry."""
    start = time.perf_counter()
    model = context.model
    layout = shared_cache.model
    device, dtype = layout.device, layout.dtype
    beta = context.beta_anchor.to(device=device, dtype=dtype).reshape(1, -1)
    expression = torch.zeros(
        1, int(model.num_expression_coeffs), dtype=dtype, device=device,
    )
    shape_components = torch.cat((beta, expression), dim=-1)
    shape_dirs = torch.cat((model.shapedirs, model.expr_dirs), dim=-1)
    v_shaped = model.v_template + blend_shapes(shape_components, shape_dirs).squeeze(0)
    joints = vertices2joints(model.J_regressor, v_shaped.unsqueeze(0)).squeeze(0)
    model_cache = Stage2PoseModelCache(
        v_shaped=v_shaped.detach(), joints=joints.detach(),
        parents=layout.parents, joint_count=layout.joint_count,
        pose_feature_dim=layout.pose_feature_dim,
        full_vertex_count=layout.full_vertex_count, dtype=dtype, device=device,
    )
    groups: list[Stage2PoseFrameCameraCache] = []
    for shared in shared_cache.groups:
        groups.append(Stage2PoseFrameCameraCache(
            frame_position=shared.frame_position,
            frame_index=shared.frame_index,
            camera=shared.camera,
            vertex_ids=shared.vertex_ids,
            targets=shared.targets,
            left_hand_pose=shared.left_hand_pose,
            right_hand_pose=shared.right_hand_pose,
            scale=shared.scale,
            selected_v_shaped=v_shaped.index_select(
                0, shared.vertex_ids,
            ).detach(),
            selected_posedirs=shared.selected_posedirs,
            selected_lbs_weights=shared.selected_lbs_weights,
        ))
    return Stage2PoseFastCache(
        model=model_cache, groups=groups, frame_count=context.frame_count,
        body_pose_dim=context.body_pose_dim,
        image_width=context.image_width, image_height=context.image_height,
        setup_ms=(time.perf_counter() - start) * 1000.0,
        profile_timing=context.args.profile_timing,
    )


def assemble_full_pose_rotations(
    global_orient: torch.Tensor,
    body_pose: torch.Tensor,
    left_hand_pose: torch.Tensor,
    right_hand_pose: torch.Tensor,
    model_cache: Stage2PoseModelCache,
) -> torch.Tensor:
    """Assemble root/body/jaw/eyes/hands in the installed SMPL-X order."""
    if global_orient.numel() != 3 or body_pose.numel() != 63:
        raise _unsupported("global/body pose dimensions")
    if left_hand_pose.numel() != 45 or right_hand_pose.numel() != 45:
        raise _unsupported("hand pose dimensions")
    zeros = torch.zeros(3, dtype=global_orient.dtype, device=global_orient.device)
    axis_angles = torch.cat((
        global_orient.reshape(1, 3), body_pose.reshape(21, 3),
        zeros.reshape(1, 3), zeros.reshape(1, 3), zeros.reshape(1, 3),
        left_hand_pose.reshape(15, 3), right_hand_pose.reshape(15, 3),
    ), dim=0)
    if axis_angles.shape[0] != model_cache.joint_count:
        raise _unsupported(f"assembled joint count {axis_angles.shape[0]}")
    return _aa_to_rotmat(axis_angles)


def fast_selected_vertices(
    cache: Stage2PoseFastCache,
    camera_cache: Stage2PoseFrameCameraCache,
    global_orient: torch.Tensor,
    translation: torch.Tensor,
    body_pose: torch.Tensor,
    *,
    return_intermediates: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute only selected final vertices for one frame/camera."""
    reference = global_orient
    rotations = assemble_full_pose_rotations(
        global_orient.reshape(3), body_pose.reshape(-1),
        camera_cache.left_hand_pose.to(reference), camera_cache.right_hand_pose.to(reference),
        cache.model,
    )
    selected_v_shaped = camera_cache.selected_v_shaped.to(reference)
    selected_posedirs = camera_cache.selected_posedirs.to(reference)
    pose_offsets, pose_feature = compute_selected_pose_offsets(rotations, selected_posedirs)
    selected_v_posed = selected_v_shaped + pose_offsets
    posed_joints, relative_transforms = batch_rigid_transform(
        rotations.unsqueeze(0), cache.model.joints.to(reference).unsqueeze(0),
        cache.model.parents.to(device=reference.device), dtype=reference.dtype,
    )
    skinned, selected_transforms = compute_selected_lbs(
        selected_v_posed, camera_cache.selected_lbs_weights.to(reference),
        relative_transforms.squeeze(0),
    )
    # forward_smplx_tensor_batch applies scale after model LBS, then translation.
    final_vertices = skinned * camera_cache.scale.to(reference).reshape(1, 1) + translation.reshape(1, 3)
    if final_vertices.shape[0] != camera_cache.vertex_ids.numel():
        raise _unsupported("fast path produced an unexpected vertex count")
    if not return_intermediates:
        return final_vertices
    return final_vertices, {
        "v_shaped": cache.model.v_shaped.to(reference),
        "joints": cache.model.joints.to(reference),
        "rotation_matrices": rotations,
        "pose_feature": pose_feature,
        "selected_pose_offsets": pose_offsets,
        "fk_transforms": relative_transforms.squeeze(0),
        "selected_v_posed": selected_v_posed,
        "selected_transforms": selected_transforms,
        "selected_skinned_vertices": skinned,
        "root_center": cache.model.joints[0].to(reference) * camera_cache.scale.to(reference),
        "final_selected_vertices": final_vertices,
        "posed_joints": posed_joints.squeeze(0),
    }


def fast_raw_groups(cache: Stage2PoseFastCache, parameters: torch.Tensor) -> list[ReprojectionGroup]:
    """Build Stage 2 pose raw groups in production frame/camera order."""
    per_frame_dim = STAGE1_PER_FRAME_DIM + cache.body_pose_dim
    expected = cache.frame_count * per_frame_dim
    if parameters.numel() != expected:
        raise ValueError(f"Stage 2 pose parameter dimension: expected {expected}, got {parameters.numel()}")
    groups: list[ReprojectionGroup] = []
    for camera_cache in cache.groups:
        frame = unpack_frame_parameters(
            parameters, camera_cache.frame_position, per_frame_dim, include_body_pose=True,
        )
        orient = frame.cam2_orient if camera_cache.camera == "cam2" else frame.cam4_orient
        translation = frame.cam2_transl if camera_cache.camera == "cam2" else frame.cam4_transl
        vertices = fast_selected_vertices(
            cache, camera_cache, orient.reshape(3), translation.reshape(3), frame.body_pose.reshape(-1),
        )
        projected = project_cam2_erp(vertices, cache.image_width, cache.image_height)
        groups.append(ReprojectionGroup(
            frame_index=camera_cache.frame_index, camera=camera_cache.camera,
            residual_2d=projected - camera_cache.targets.to(parameters),
        ))
    return groups


def fast_raw_residual(cache: Stage2PoseFastCache, parameters: torch.Tensor) -> torch.Tensor:
    _sync(cache, parameters)
    start = time.perf_counter()
    result = flatten_reprojection_groups(fast_raw_groups(cache, parameters))
    _sync(cache, parameters)
    cache.residual_ms += (time.perf_counter() - start) * 1000.0
    cache.residual_calls += 1
    return result


def _pose_anchor_residual(
    cache: Stage2PoseFastCache,
    parameters: torch.Tensor,
    body_pose_anchors: list[torch.Tensor],
    pose_anchor_scale: float,
) -> torch.Tensor:
    per_frame_dim = STAGE1_PER_FRAME_DIM + cache.body_pose_dim
    return torch.cat([
        (parameters[frame * per_frame_dim + STAGE1_PER_FRAME_DIM:(frame + 1) * per_frame_dim]
         - body_pose_anchors[frame].to(parameters)) * pose_anchor_scale
        for frame in range(cache.frame_count)
    ])


def stage2_pose_fast_true_objective(
    cache: Stage2PoseFastCache,
    parameters: torch.Tensor,
    body_pose_anchors: list[torch.Tensor],
    pose_anchor_scale: float,
    objective_mode: str,
    charbonnier_eps: float,
    rep_scale: float,
    aggregation_spec: dict[str, Any] | None,
) -> torch.Tensor:
    _sync(cache, parameters)
    start = time.perf_counter()
    groups = fast_raw_groups(cache, parameters)
    if objective_mode == "charbonnier_irls":
        if aggregation_spec is None:
            raise ValueError("aggregation_spec is required for Charbonnier objective")
        primary = compute_primary_charbonnier_tensor(groups, charbonnier_eps, aggregation_spec)
    elif objective_mode == "raw_l2":
        residual = flatten_reprojection_groups(groups) * rep_scale
        primary = 0.5 * torch.sum(residual ** 2)
    else:
        raise ValueError(objective_mode)
    anchor = _pose_anchor_residual(cache, parameters, body_pose_anchors, pose_anchor_scale)
    result = primary + 0.5 * torch.sum(anchor ** 2)
    _sync(cache, parameters)
    cache.objective_ms += (time.perf_counter() - start) * 1000.0
    cache.objective_calls += 1
    return result


def stage2_pose_fast_raw_metrics(
    cache: Stage2PoseFastCache, parameters: torch.Tensor,
) -> dict[str, float]:
    _sync(cache, parameters)
    start = time.perf_counter()
    result = raw_reproj_metrics(
        flatten_reprojection_groups(fast_raw_groups(cache, parameters)).detach(),
    )
    _sync(cache, parameters)
    cache.metrics_ms += (time.perf_counter() - start) * 1000.0
    cache.metrics_calls += 1
    return result
