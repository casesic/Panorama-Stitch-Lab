"""Immutable geometry and observation rows shared by both Stage 2 caches."""

from __future__ import annotations

import time
from dataclasses import dataclass

import torch

from .types import StageContext


def _unsupported(message: str) -> RuntimeError:
    return RuntimeError(f"UNSUPPORTED_SMPLX_FORWARD_CONTRACT: {message}")


@dataclass(frozen=True)
class Stage2SharedModelLayout:
    parents: torch.Tensor
    joint_count: int
    pose_feature_dim: int
    full_vertex_count: int
    body_pose_dim: int
    beta_dim: int
    dtype: torch.dtype
    device: torch.device


@dataclass(frozen=True)
class Stage2SharedFrameCameraCache:
    frame_position: int
    frame_index: int
    camera: str
    vertex_ids: torch.Tensor
    targets: torch.Tensor
    left_hand_pose: torch.Tensor
    right_hand_pose: torch.Tensor
    scale: torch.Tensor
    selected_posedirs: torch.Tensor
    selected_lbs_weights: torch.Tensor


@dataclass
class Stage2SharedGeometryCache:
    model: Stage2SharedModelLayout
    groups: list[Stage2SharedFrameCameraCache]
    frame_count: int
    image_width: int
    image_height: int
    setup_ms: float
    metadata_setup_ms: float
    posedirs_select_ms: float
    lbs_weights_select_ms: float
    posedirs_select_calls: int
    lbs_weights_select_calls: int

    def timing_dict(self) -> dict[str, float | int]:
        return {
            "stage2_shared_cache_setup_ms": self.setup_ms,
            "stage2_shared_metadata_setup_ms": self.metadata_setup_ms,
            "stage2_shared_posedirs_select_ms": self.posedirs_select_ms,
            "stage2_shared_lbs_weights_select_ms": self.lbs_weights_select_ms,
            "stage2_shared_posedirs_select_calls": self.posedirs_select_calls,
            "stage2_shared_lbs_weights_select_calls": self.lbs_weights_select_calls,
        }


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _validate_model_contract(context: StageContext) -> Stage2SharedModelLayout:
    model = context.model
    required = (
        "v_template", "shapedirs", "expr_dirs", "posedirs", "J_regressor",
        "parents", "lbs_weights",
    )
    missing = [name for name in required if not hasattr(model, name)]
    if missing:
        raise _unsupported(f"missing model fields: {missing}")
    if getattr(model, "use_pca", None) is not False:
        raise _unsupported("hand pose PCA is enabled")
    if context.body_pose_dim != int(model.NUM_BODY_JOINTS) * 3:
        raise _unsupported(f"body pose dimension {context.body_pose_dim}")
    if int(model.NUM_HAND_JOINTS) != 15:
        raise _unsupported(f"hand joint count {model.NUM_HAND_JOINTS}")
    joint_count = int(model.J_regressor.shape[0])
    expected_joint_count = (
        1 + int(model.NUM_BODY_JOINTS) + 3 + 2 * int(model.NUM_HAND_JOINTS)
    )
    if joint_count != expected_joint_count:
        raise _unsupported(f"joint count {joint_count} != {expected_joint_count}")
    pose_feature_dim = (joint_count - 1) * 9
    if model.posedirs.shape[0] != pose_feature_dim:
        raise _unsupported(f"posedirs shape {tuple(model.posedirs.shape)}")
    vertex_count = int(model.v_template.shape[0])
    device, dtype = model.v_template.device, model.v_template.dtype
    return Stage2SharedModelLayout(
        parents=model.parents.detach(), joint_count=joint_count,
        pose_feature_dim=pose_feature_dim, full_vertex_count=vertex_count,
        body_pose_dim=context.body_pose_dim, beta_dim=context.beta_dim,
        dtype=dtype, device=device,
    )


def _validate_shared_input(
    name: str,
    tensor: torch.Tensor,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> None:
    if tensor.device != device or tensor.dtype != dtype:
        raise RuntimeError(
            "INVALID_STAGE2_SHARED_CACHE_INPUT: "
            f"{name} expected device={device} dtype={dtype}, "
            f"actual device={tensor.device} dtype={tensor.dtype}"
        )


def build_stage2_shared_geometry_cache(
    context: StageContext,
) -> Stage2SharedGeometryCache:
    """Select common model rows once and retain read-only context tensors."""
    model = context.model
    device = model.v_template.device
    _sync(device)
    setup_start = time.perf_counter()

    metadata_start = time.perf_counter()
    layout = _validate_model_contract(context)
    posedirs_by_vertex = model.posedirs.reshape(
        layout.pose_feature_dim, layout.full_vertex_count, 3,
    )
    metadata: list[tuple[int, int, str, torch.Tensor, torch.Tensor,
                             torch.Tensor, torch.Tensor, torch.Tensor]] = []
    for position, frame_index in enumerate(context.frame_indices):
        frozen = context.frozen_params_by_frame[position]
        cameras = (
            (
                "cam2", context.cam2_vertex_ids_by_frame[position],
                context.cam2_targets_by_frame[position],
                frozen.cam2_left_hand_pose, frozen.cam2_right_hand_pose,
                frozen.cam2_scale,
            ),
            (
                "cam4", context.cam4_vertex_ids_by_frame[position],
                context.cam4_targets_by_frame[position],
                frozen.cam4_left_hand_pose, frozen.cam4_right_hand_pose,
                frozen.cam4_scale,
            ),
        )
        for camera, ids, targets, left, right, scale in cameras:
            _validate_shared_input(
                f"{frame_index}/{camera}/vertex_ids", ids,
                device=layout.device, dtype=torch.long,
            )
            for name, value in (
                ("targets", targets), ("left_hand_pose", left),
                ("right_hand_pose", right), ("scale", scale),
            ):
                _validate_shared_input(
                    f"{frame_index}/{camera}/{name}", value,
                    device=layout.device, dtype=layout.dtype,
                )
            if left.numel() != 45 or right.numel() != 45 or scale.numel() != 1:
                raise _unsupported(f"{camera} frozen parameter dimensions")
            metadata.append((
                position, int(frame_index), camera, ids.detach(),
                targets.detach(), left.reshape(45).detach(),
                right.reshape(45).detach(), scale.reshape(1).detach(),
            ))
    _sync(device)
    metadata_setup_ms = (time.perf_counter() - metadata_start) * 1000.0

    posedirs_start = time.perf_counter()
    selected_posedirs = [
        posedirs_by_vertex.index_select(1, item[3]).detach()
        for item in metadata
    ]
    _sync(device)
    posedirs_select_ms = (time.perf_counter() - posedirs_start) * 1000.0

    weights_start = time.perf_counter()
    selected_lbs_weights = [
        model.lbs_weights.index_select(0, item[3]).detach()
        for item in metadata
    ]
    _sync(device)
    lbs_weights_select_ms = (time.perf_counter() - weights_start) * 1000.0

    groups = [
        Stage2SharedFrameCameraCache(
            frame_position=item[0], frame_index=item[1], camera=item[2],
            vertex_ids=item[3], targets=item[4], left_hand_pose=item[5],
            right_hand_pose=item[6], scale=item[7],
            selected_posedirs=selected_posedirs[index],
            selected_lbs_weights=selected_lbs_weights[index],
        )
        for index, item in enumerate(metadata)
    ]
    _sync(device)
    return Stage2SharedGeometryCache(
        model=layout, groups=groups, frame_count=context.frame_count,
        image_width=context.image_width, image_height=context.image_height,
        setup_ms=(time.perf_counter() - setup_start) * 1000.0,
        metadata_setup_ms=metadata_setup_ms,
        posedirs_select_ms=posedirs_select_ms,
        lbs_weights_select_ms=lbs_weights_select_ms,
        posedirs_select_calls=len(groups),
        lbs_weights_select_calls=len(groups),
    )
