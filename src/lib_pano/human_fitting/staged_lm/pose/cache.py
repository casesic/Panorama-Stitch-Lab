"""Static and derived cache construction for Stage 2 Pose.

Purpose
-------
Own the cache types and preparation of tensors that remain fixed across Pose
LM outer iterations. Model-dependent tensors, observation-dependent selected
rows, and segment-dependent batch metadata are explicit here.

Mathematical scope
------------------
Frozen-beta shaped vertices/joints, selected posedirs/LBS rows, camera
metadata, and FULL_BATCH layout/cache types.

This module does not implement FK, LBS, ERP projection, derivatives, H/g, or
LM orchestration. Read :mod:`pose.forward` next.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import torch
from smplx.lbs import blend_shapes, vertices2joints

from ..stage2_shared_cache import Stage2SharedGeometryCache
from ..types import StageContext


@dataclass
class Stage2PoseModelCache:
    """Frozen-beta model state shared by all frame/camera groups."""

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
    """Selected immutable rows for one frame/camera observation group."""

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
    """Serial/value cache plus unchanged production timing counters."""

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
    batch_cache: Any | None = None
    batch_cache_setup_ms: float = 0.0
    batched_normal_equations_ms: float = 0.0
    batched_normal_equations_calls: int = 0
    batched_group_analytic_calls: int = 0
    batch_padding_ratio: float = 0.0
    batch_persistent_bytes: int = 0
    batch_group_count: int = 0
    batch_nmax: int = 0

    def reset_timing(self) -> None:
        self.residual_ms = self.normal_equations_ms = 0.0
        self.objective_ms = self.metrics_ms = self.explicit_jacobian_ms = 0.0
        self.residual_calls = self.normal_equations_calls = 0
        self.objective_calls = self.metrics_calls = self.explicit_jacobian_calls = 0
        self.group_analytic_calls = 0
        self.batched_normal_equations_ms = 0.0
        self.batched_normal_equations_calls = 0
        self.batched_group_analytic_calls = 0

    @staticmethod
    def _per_call(total: float, calls: int) -> float:
        return total / calls if calls else 0.0

    def timing_dict(self) -> dict[str, float | int]:
        prefix = "stage2_pose_"
        result: dict[str, float | int] = {
            f"{prefix}cache_setup_ms": self.setup_ms + self.batch_cache_setup_ms,
            f"{prefix}fast_cache_setup_ms": self.setup_ms,
            f"{prefix}batch_cache_setup_ms": self.batch_cache_setup_ms,
            f"{prefix}derived_cache_setup_ms": self.setup_ms,
        }
        for name in ("residual", "normal_equations", "objective", "metrics", "explicit_jacobian"):
            total = float(getattr(self, f"{name}_ms"))
            calls = int(getattr(self, f"{name}_calls"))
            result[f"{prefix}{name}_ms"] = total
            result[f"{prefix}{name}_calls"] = calls
            result[f"{prefix}{name}_ms_per_call"] = self._per_call(total, calls)
        result[f"{prefix}group_analytic_calls"] = self.group_analytic_calls
        result[f"{prefix}batched_normal_equations_ms"] = self.batched_normal_equations_ms
        result[f"{prefix}batched_normal_equations_calls"] = self.batched_normal_equations_calls
        result[f"{prefix}batched_normal_equations_ms_per_call"] = self._per_call(
            self.batched_normal_equations_ms, self.batched_normal_equations_calls,
        )
        result[f"{prefix}batched_group_analytic_calls"] = self.batched_group_analytic_calls
        result[f"{prefix}batch_padding_ratio"] = self.batch_padding_ratio
        result[f"{prefix}batch_persistent_bytes"] = self.batch_persistent_bytes
        result[f"{prefix}batch_group_count"] = self.batch_group_count
        result[f"{prefix}batch_nmax"] = self.batch_nmax
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
        # Compatibility method: objective mathematics lives in objective.py.
        from .objective import stage2_pose_fast_true_objective
        return stage2_pose_fast_true_objective(
            self, parameters, body_pose_anchors, pose_anchor_scale,
            objective_mode, charbonnier_eps, rep_scale, aggregation_spec,
        )

    def raw_metrics(self, parameters: torch.Tensor) -> dict[str, float]:
        # Compatibility method: metric mathematics lives in objective.py.
        from .objective import stage2_pose_fast_raw_metrics
        return stage2_pose_fast_raw_metrics(self, parameters)


def build_stage2_pose_fast_cache(
    context: StageContext,
    shared_cache: Stage2SharedGeometryCache,
) -> Stage2PoseFastCache:
    """Build frozen-beta tensors and selected rows.

    Model-dependent: shaped vertices, joints, parents.
    Observation-dependent: vertex ids, targets, posedirs/LBS selected rows.
    Segment-dependent: ordered frame/camera group list.
    """
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


# FULL_BATCH cache types/builders are implemented in batched.py because the
# builder shares its exact padding/reduction layout. They are exposed lazily
# through pose.__init__ to avoid a cache↔batched import cycle.

__all__ = [
    "Stage2PoseModelCache",
    "Stage2PoseFrameCameraCache",
    "Stage2PoseFastCache",
    "build_stage2_pose_fast_cache",
]
