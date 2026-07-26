"""Production FULL_BATCH equivalent of the Pose value/derivative chain.

IRLS weights, the true objective, metrics, and the LM solver remain on their
existing serial production paths.  This module batches only pose geometry,
analytic Jacobians, and FP64 normal-equation accumulation.
"""

from __future__ import annotations

from dataclasses import dataclass
import time

import torch

from ..geometry.erp import project_unified_erp_with_jacobian
from ..geometry.so3 import axis_angle_to_matrix, skew, so3_right_jacobian
from ..linearization import Linearization, LinearizationProvenance
from ..types import NormalEquations
from .cache import Stage2PoseFastCache
from .derivatives import build_stage2_pose_linearization_common


@dataclass(frozen=True)
class Stage2PoseBatchLayout:
    frame_positions: torch.Tensor
    camera_indices: torch.Tensor
    point_counts: torch.Tensor
    valid_mask: torch.Tensor
    targets: torch.Tensor
    frame_positions_host: tuple[int, ...]
    camera_indices_host: tuple[int, ...]
    point_counts_host: tuple[int, ...]
    nmax: int
    group_count: int
    frame_count: int
    padding_ratio: float
    padded_bytes: int


@dataclass(frozen=True)
class Stage2PoseBatchParameters:
    group_orientations: torch.Tensor
    group_translations: torch.Tensor
    group_body_poses: torch.Tensor
    frame_body_poses: torch.Tensor


@dataclass
class Stage2PoseBatchCache:
    layout: Stage2PoseBatchLayout
    selected_v_shaped: torch.Tensor
    selected_posedirs: torch.Tensor
    selected_lbs_weights: torch.Tensor
    left_hand_pose: torch.Tensor
    right_hand_pose: torch.Tensor
    scale: torch.Tensor
    parents: torch.Tensor
    joints: torch.Tensor
    derivative_mapping: torch.Tensor
    fk_depth_indices: tuple[torch.Tensor, ...]
    active_indices_cam2: torch.Tensor
    active_indices_cam4: torch.Tensor
    pose_anchor_identity_64: torch.Tensor
    setup_ms: float
    padding_layout_ms: float
    padding_selected_v_shaped_ms: float
    padding_selected_posedirs_ms: float
    padding_weights_ms: float
    stack_hands_scale_ms: float
    static_layout_ms: float
    persistent_bytes: int

    @property
    def preparation_timing(self) -> dict[str, float]:
        return {
            "total_ms": self.setup_ms,
            "padding_targets_ms": self.padding_layout_ms,
            "padding_selected_v_shaped_ms": self.padding_selected_v_shaped_ms,
            "padding_selected_posedirs_ms": self.padding_selected_posedirs_ms,
            "padding_weights_ms": self.padding_weights_ms,
            "stack_hands_scale_ms": self.stack_hands_scale_ms,
            "static_mapping_depth_layout_ms": self.static_layout_ms,
        }


def _tensor_bytes(value: torch.Tensor) -> int:
    return value.numel() * value.element_size()


def _depth_layout(parents: torch.Tensor) -> tuple[torch.Tensor, ...]:
    parent_host = tuple(int(value) for value in parents.detach().cpu().tolist())
    depths = [0] * len(parent_host)
    for joint in range(1, len(parent_host)):
        depths[joint] = depths[parent_host[joint]] + 1
    return tuple(torch.tensor(
        [joint for joint, depth in enumerate(depths) if depth == level],
        dtype=torch.long, device=parents.device,
    ) for level in range(max(depths) + 1))


def _active_indices(camera_index: int, device: torch.device) -> torch.Tensor:
    camera_start = 0 if camera_index == 0 else 6
    return torch.cat((
        torch.arange(camera_start, camera_start + 6, device=device),
        torch.arange(12, 75, device=device),
    ))


def build_stage2_pose_batch_cache(
    fast_cache: Stage2PoseFastCache,
    *,
    pose_anchor_scale: float,
) -> Stage2PoseBatchCache:
    """Pad immutable pose-cache rows and build all static batch metadata once."""
    if not fast_cache.groups:
        raise ValueError("EMPTY_STAGE2_POSE_BATCH_CACHE")
    started = time.perf_counter()
    reference = fast_cache.groups[0].selected_v_shaped
    dtype, device = reference.dtype, reference.device
    groups = fast_cache.groups
    counts = tuple(int(group.targets.shape[0]) for group in groups)
    frames = tuple(int(group.frame_position) for group in groups)
    cameras = tuple(0 if group.camera == "cam2" else 1 for group in groups)
    group_count, nmax = len(groups), max(counts)

    section = time.perf_counter()
    valid_mask = torch.zeros(group_count, nmax, dtype=torch.bool, device=device)
    targets = torch.zeros(group_count, nmax, 2, dtype=dtype, device=device)
    for index, (group, count) in enumerate(zip(groups, counts)):
        valid_mask[index, :count] = True
        targets[index, :count] = group.targets
    frame_positions = torch.tensor(frames, dtype=torch.long, device=device)
    camera_indices = torch.tensor(cameras, dtype=torch.long, device=device)
    point_counts = torch.tensor(counts, dtype=torch.long, device=device)
    layout_ms = (time.perf_counter() - section) * 1000.0
    layout_tensors = (valid_mask, targets, frame_positions, camera_indices, point_counts)
    total_slots = group_count * nmax
    layout = Stage2PoseBatchLayout(
        frame_positions=frame_positions, camera_indices=camera_indices,
        point_counts=point_counts, valid_mask=valid_mask, targets=targets,
        frame_positions_host=frames, camera_indices_host=cameras,
        point_counts_host=counts, nmax=nmax, group_count=group_count,
        frame_count=fast_cache.frame_count,
        padding_ratio=(total_slots - sum(counts)) / max(total_slots, 1),
        padded_bytes=sum(_tensor_bytes(value) for value in layout_tensors),
    )

    section = time.perf_counter()
    selected_v_shaped = torch.zeros(group_count, nmax, 3, dtype=dtype, device=device)
    for index, (group, count) in enumerate(zip(groups, counts)):
        selected_v_shaped[index, :count] = group.selected_v_shaped
    vertices_ms = (time.perf_counter() - section) * 1000.0

    section = time.perf_counter()
    selected_posedirs = torch.zeros(
        group_count, fast_cache.model.pose_feature_dim, nmax, 3,
        dtype=dtype, device=device,
    )
    for index, (group, count) in enumerate(zip(groups, counts)):
        selected_posedirs[index, :, :count] = group.selected_posedirs
    posedirs_ms = (time.perf_counter() - section) * 1000.0

    section = time.perf_counter()
    selected_lbs_weights = torch.zeros(
        group_count, nmax, fast_cache.model.joint_count,
        dtype=dtype, device=device,
    )
    for index, (group, count) in enumerate(zip(groups, counts)):
        selected_lbs_weights[index, :count] = group.selected_lbs_weights
    weights_ms = (time.perf_counter() - section) * 1000.0

    section = time.perf_counter()
    left_hand_pose = torch.stack([group.left_hand_pose.reshape(-1) for group in groups])
    right_hand_pose = torch.stack([group.right_hand_pose.reshape(-1) for group in groups])
    scale = torch.stack([group.scale.reshape(-1) for group in groups])
    stack_ms = (time.perf_counter() - section) * 1000.0

    section = time.perf_counter()
    derivative_mapping = torch.zeros(22, 66, 3, dtype=dtype, device=device)
    joints = torch.arange(22, device=device)
    components = torch.arange(3, device=device)
    derivative_mapping[
        joints[:, None], joints[:, None] * 3 + components[None], components[None]
    ] = 1
    fk_depth_indices = _depth_layout(fast_cache.model.parents)
    active_indices_cam2 = _active_indices(0, device)
    active_indices_cam4 = _active_indices(1, device)
    pose_anchor_identity_64 = torch.eye(
        fast_cache.body_pose_dim, dtype=torch.float64, device=device,
    ) * float(pose_anchor_scale) ** 2
    static_ms = (time.perf_counter() - section) * 1000.0

    persistent_tensors = (
        *layout_tensors, selected_v_shaped, selected_posedirs,
        selected_lbs_weights, left_hand_pose, right_hand_pose, scale,
        derivative_mapping, active_indices_cam2, active_indices_cam4,
        pose_anchor_identity_64, *fk_depth_indices,
    )
    return Stage2PoseBatchCache(
        layout=layout, selected_v_shaped=selected_v_shaped,
        selected_posedirs=selected_posedirs,
        selected_lbs_weights=selected_lbs_weights,
        left_hand_pose=left_hand_pose, right_hand_pose=right_hand_pose,
        scale=scale, parents=fast_cache.model.parents, joints=fast_cache.model.joints,
        derivative_mapping=derivative_mapping, fk_depth_indices=fk_depth_indices,
        active_indices_cam2=active_indices_cam2,
        active_indices_cam4=active_indices_cam4,
        pose_anchor_identity_64=pose_anchor_identity_64,
        setup_ms=(time.perf_counter() - started) * 1000.0,
        padding_layout_ms=layout_ms,
        padding_selected_v_shaped_ms=vertices_ms,
        padding_selected_posedirs_ms=posedirs_ms,
        padding_weights_ms=weights_ms, stack_hands_scale_ms=stack_ms,
        static_layout_ms=static_ms,
        persistent_bytes=sum(_tensor_bytes(value) for value in persistent_tensors),
    )


def gather_pose_batch_parameters(
    parameters: torch.Tensor,
    layout: Stage2PoseBatchLayout,
) -> Stage2PoseBatchParameters:
    framed = parameters.reshape(layout.frame_count, 75)
    grouped = framed.index_select(0, layout.frame_positions)
    offsets = layout.camera_indices * 6
    components = torch.arange(3, device=parameters.device)
    frame_body_poses = framed[:, 12:75]
    return Stage2PoseBatchParameters(
        group_orientations=grouped.gather(1, offsets[:, None] + components[None]),
        group_translations=grouped.gather(1, offsets[:, None] + 3 + components[None]),
        group_body_poses=frame_body_poses.index_select(0, layout.frame_positions),
        frame_body_poses=frame_body_poses,
    )


def stage2_pose_batched_rotations_and_jacobians(
    orientations: torch.Tensor,
    body_poses: torch.Tensor,
    left_hand_pose: torch.Tensor,
    right_hand_pose: torch.Tensor,
    derivative_mapping: torch.Tensor | None = None,
    *,
    vectorized_scatter: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    group_count = orientations.shape[0]
    zeros = torch.zeros(group_count, 3, 3, dtype=orientations.dtype, device=orientations.device)
    axis_angles = torch.cat((
        orientations[:, None], body_poses.reshape(group_count, 21, 3), zeros,
        left_hand_pose.reshape(group_count, 15, 3),
        right_hand_pose.reshape(group_count, 15, 3),
    ), dim=1)
    rotations = axis_angle_to_matrix(axis_angles)
    right = so3_right_jacobian(axis_angles[:, :22])
    basis = skew(right.transpose(-1, -2))
    blocks = torch.einsum("...ij,...kjl->...ilk", rotations[:, :22], basis)
    derivatives = torch.zeros(
        group_count, 55, 3, 3, 66, dtype=rotations.dtype, device=rotations.device,
    )
    if vectorized_scatter:
        if derivative_mapping is None:
            raise ValueError("MISSING_POSE_DERIVATIVE_MAPPING")
        derivatives[:, :22] = torch.einsum(
            "gjabk,jqk->gjabq", blocks, derivative_mapping,
        )
    else:
        derivatives[:, 0, :, :, :3] = blocks[:, 0]
        for joint in range(21):
            derivatives[:, joint + 1, :, :, 3 + joint * 3:6 + joint * 3] = blocks[:, joint + 1]
    return rotations, derivatives


def stage2_pose_batched_feature_offsets_from_rotations(
    cache: Stage2PoseBatchCache,
    rotations: torch.Tensor,
    d_rotations: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    identity = torch.eye(3, dtype=rotations.dtype, device=rotations.device)
    feature = (rotations[:, 1:] - identity).reshape(rotations.shape[0], -1)
    d_feature = d_rotations[:, 1:].reshape(rotations.shape[0], -1, 66)
    offsets = torch.einsum("gp,gpnc->gnc", feature, cache.selected_posedirs)
    d_offsets = torch.einsum("gpq,gpnc->gncq", d_feature, cache.selected_posedirs)
    return feature, d_feature, offsets, d_offsets, cache.selected_v_shaped + offsets


def stage2_pose_batched_fk_by_depth(
    rotations: torch.Tensor,
    d_rotations: torch.Tensor,
    joints_tensor: torch.Tensor,
    parents: torch.Tensor,
    fk_depth_indices: tuple[torch.Tensor, ...] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    group_count, joint_count = rotations.shape[:2]
    joints = joints_tensor[None].expand(group_count, -1, -1)
    relative_joints = joints.clone()
    relative_joints[:, 1:] -= joints.index_select(1, parents[1:])
    local = torch.zeros(group_count, joint_count, 4, 4, dtype=rotations.dtype, device=rotations.device)
    local[:, :, :3, :3] = rotations
    local[:, :, :3, 3] = relative_joints
    local[:, :, 3, 3] = 1
    d_local = torch.zeros(group_count, joint_count, 4, 4, 66, dtype=rotations.dtype, device=rotations.device)
    d_local[:, :, :3, :3] = d_rotations
    global_transforms = torch.zeros_like(local)
    d_global_transforms = torch.zeros_like(d_local)
    global_transforms[:, 0] = local[:, 0]
    d_global_transforms[:, 0] = d_local[:, 0]
    if fk_depth_indices is None:
        raise ValueError("MISSING_POSE_FK_DEPTH_INDICES")
    for indices in fk_depth_indices[1:]:
        parent_indices = parents.index_select(0, indices)
        parent = global_transforms.index_select(1, parent_indices)
        d_parent = d_global_transforms.index_select(1, parent_indices)
        child = local.index_select(1, indices)
        d_child = d_local.index_select(1, indices)
        global_transforms[:, indices] = parent @ child
        d_global_transforms[:, indices] = (
            torch.einsum("gjabq,gjbc->gjacq", d_parent, child)
            + torch.einsum("gjab,gjbcq->gjacq", parent, d_child)
        )
    rest = torch.eye(4, dtype=rotations.dtype, device=rotations.device)[None, None].repeat(
        group_count, joint_count, 1, 1,
    )
    rest[:, :, :3, 3] = -joints
    return (
        global_transforms @ rest,
        torch.einsum("gjabq,gjbc->gjacq", d_global_transforms, rest),
    )


def stage2_pose_batched_lbs_from_fk(
    cache: Stage2PoseBatchCache,
    selected_v_posed: torch.Tensor,
    d_pose_offsets: torch.Tensor,
    relative_transforms: torch.Tensor,
    d_relative_transforms: torch.Tensor,
    translations: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    homogeneous = torch.cat((selected_v_posed, torch.ones_like(selected_v_posed[..., :1])), dim=-1)
    d_homogeneous = torch.cat((d_pose_offsets, torch.zeros_like(d_pose_offsets[..., :1, :])), dim=-2)
    transforms_top = torch.einsum(
        "gnj,gjab->gnab", cache.selected_lbs_weights, relative_transforms[:, :, :3],
    )
    transforms = torch.zeros(
        transforms_top.shape[:2] + (4, 4), dtype=transforms_top.dtype,
        device=transforms_top.device,
    )
    transforms[:, :, :3] = transforms_top
    transforms[:, :, 3] = torch.einsum(
        "gnj,gjb->gnb", cache.selected_lbs_weights, relative_transforms[:, :, 3],
    )
    d_transforms = torch.einsum(
        "gnj,gjabq->gnabq", cache.selected_lbs_weights, d_relative_transforms,
    )
    skinned = torch.einsum("gnab,gnb->gna", transforms_top, homogeneous)
    d_skinned = (
        torch.einsum(
            "gnj,gjabq,gnb->gnaq", cache.selected_lbs_weights,
            d_relative_transforms[:, :, :3], homogeneous,
        )
        + torch.einsum("gnab,gnbq->gnaq", transforms_top, d_homogeneous)
    )
    scale = cache.scale.reshape(-1, 1, 1)
    mask = cache.layout.valid_mask[..., None]
    transform_mask = cache.layout.valid_mask[..., None, None]
    derivative_transform_mask = cache.layout.valid_mask[..., None, None, None]
    return (
        transforms * transform_mask,
        d_transforms * derivative_transform_mask,
        skinned * mask,
        d_skinned * mask[..., None],
        (skinned * scale + translations[:, None]) * mask,
        d_skinned * scale[..., None] * mask[..., None],
    )


def stage2_pose_batched_projection_from_vertices(
    cache: Stage2PoseBatchCache,
    vertices: torch.Tensor,
    d_vertices: torch.Tensor,
    image_width: int,
    image_height: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    group_count, nmax = vertices.shape[:2]
    uv, projection = project_unified_erp_with_jacobian(
        vertices.reshape(-1, 3), image_width, image_height,
    )
    uv = uv.reshape(group_count, nmax, 2)
    projection = projection.reshape(group_count, nmax, 2, 3)
    uv_pose = torch.einsum("gnab,gnbq->gnaq", projection, d_vertices)
    active = torch.cat((uv_pose[..., :3], projection, uv_pose[..., 3:]), dim=-1)
    mask = cache.layout.valid_mask
    return (
        uv * mask[..., None], projection * mask[..., None, None],
        uv_pose * mask[..., None, None],
        (uv - cache.layout.targets) * mask[..., None],
        active * mask[..., None, None],
    )


def stage2_pose_batched_group_analytic(
    fast_cache: Stage2PoseFastCache,
    batch_cache: Stage2PoseBatchCache,
    batch_parameters: Stage2PoseBatchParameters,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    rotations, d_rotations = stage2_pose_batched_rotations_and_jacobians(
        batch_parameters.group_orientations, batch_parameters.group_body_poses,
        batch_cache.left_hand_pose, batch_cache.right_hand_pose,
        derivative_mapping=batch_cache.derivative_mapping,
    )
    _, _, _, d_pose_offsets, selected_v_posed = stage2_pose_batched_feature_offsets_from_rotations(
        batch_cache, rotations, d_rotations,
    )
    relative, d_relative = stage2_pose_batched_fk_by_depth(
        rotations, d_rotations, batch_cache.joints, batch_cache.parents,
        batch_cache.fk_depth_indices,
    )
    _, _, _, _, vertices, d_vertices = stage2_pose_batched_lbs_from_fk(
        batch_cache, selected_v_posed, d_pose_offsets, relative, d_relative,
        batch_parameters.group_translations,
    )
    _, _, _, residual, active = stage2_pose_batched_projection_from_vertices(
        batch_cache, vertices, d_vertices,
        fast_cache.image_width, fast_cache.image_height,
    )
    return residual, active, batch_cache.layout.valid_mask


def pad_stage2_pose_row_scales(
    batch_cache: Stage2PoseBatchCache,
    row_scales_by_group: list[torch.Tensor],
) -> torch.Tensor:
    if len(row_scales_by_group) != batch_cache.layout.group_count:
        raise ValueError("Stage 2 pose row-scale group count mismatch")
    reference = row_scales_by_group[0]
    padded = torch.zeros(
        batch_cache.layout.group_count, batch_cache.layout.nmax, 2,
        dtype=reference.dtype, device=reference.device,
    )
    for group, (scales, count) in enumerate(zip(
        row_scales_by_group, batch_cache.layout.point_counts_host,
    )):
        padded[group, :count] = scales.reshape(count, 2)
    return padded.detach()


def stage2_pose_batched_normal_equations(
    fast_cache: Stage2PoseFastCache,
    batch_cache: Stage2PoseBatchCache,
    parameters: torch.Tensor,
    padded_row_scales: torch.Tensor,
    body_pose_anchors: list[torch.Tensor],
    pose_anchor_scale: float,
) -> NormalEquations:
    """Build the fixed FULL_BATCH Pose system with FP64 H/g accumulation."""
    if fast_cache.profile_timing and parameters.is_cuda:
        torch.cuda.synchronize(parameters.device)
    started = time.perf_counter()
    gathered = gather_pose_batch_parameters(parameters, batch_cache.layout)
    residual, active, _ = stage2_pose_batched_group_analytic(
        fast_cache, batch_cache, gathered,
    )
    result = stage2_pose_batched_accumulate_normal_equations(
        batch_cache, parameters, gathered, residual, active,
        padded_row_scales, body_pose_anchors, pose_anchor_scale,
    )
    if fast_cache.profile_timing and parameters.is_cuda:
        torch.cuda.synchronize(parameters.device)
    elapsed = (time.perf_counter() - started) * 1000.0
    fast_cache.normal_equations_ms += elapsed
    fast_cache.normal_equations_calls += 1
    fast_cache.batched_normal_equations_ms += elapsed
    fast_cache.batched_normal_equations_calls += 1
    fast_cache.batched_group_analytic_calls += batch_cache.layout.group_count
    return result


def stage2_pose_batched_accumulate_normal_equations(
    batch_cache: Stage2PoseBatchCache,
    parameters: torch.Tensor,
    gathered: Stage2PoseBatchParameters,
    residual: torch.Tensor,
    active: torch.Tensor,
    padded_row_scales: torch.Tensor,
    body_pose_anchors: list[torch.Tensor],
    pose_anchor_scale: float,
) -> NormalEquations:
    """Apply the fixed FP64 bmm and production-order frame reduction."""
    scales = padded_row_scales.to(parameters)
    scaled_residual = residual * scales
    scaled_jacobian = active * scales[..., None]
    group_count = batch_cache.layout.group_count
    residual_64 = scaled_residual.reshape(group_count, -1).to(torch.float64)
    jacobian_64 = scaled_jacobian.reshape(group_count, -1, 69).to(torch.float64)
    active_hessian = torch.bmm(jacobian_64.transpose(1, 2), jacobian_64)
    active_gradient = torch.bmm(
        jacobian_64.transpose(1, 2), residual_64[..., None],
    ).squeeze(-1)

    frame_hessian = torch.zeros(
        batch_cache.layout.frame_count, 75, 75,
        dtype=torch.float64, device=parameters.device,
    )
    frame_gradient = torch.zeros(
        batch_cache.layout.frame_count, 75,
        dtype=torch.float64, device=parameters.device,
    )
    for group, (frame, camera) in enumerate(zip(
        batch_cache.layout.frame_positions_host,
        batch_cache.layout.camera_indices_host,
    )):
        indices = (
            batch_cache.active_indices_cam2 if camera == 0
            else batch_cache.active_indices_cam4
        )
        frame_hessian[frame][indices[:, None], indices[None, :]] += active_hessian[group]
        frame_gradient[frame].index_add_(0, indices, active_gradient[group])

    anchor_parts: list[torch.Tensor] = []
    anchor_scale_squared = float(pose_anchor_scale) ** 2
    for frame in range(batch_cache.layout.frame_count):
        pose_delta = gathered.frame_body_poses[frame] - body_pose_anchors[frame].to(parameters)
        anchor_parts.append(pose_delta * pose_anchor_scale)
        frame_hessian[frame, 12:, 12:] += batch_cache.pose_anchor_identity_64
        frame_gradient[frame, 12:] += pose_delta.to(torch.float64) * anchor_scale_squared

    dimension = batch_cache.layout.frame_count * 75
    hessian = torch.zeros(dimension, dimension, dtype=torch.float64, device=parameters.device)
    for frame in range(batch_cache.layout.frame_count):
        block = slice(frame * 75, (frame + 1) * 75)
        hessian[block, block] = frame_hessian[frame]
    residual_parts = [
        scaled_residual[group, :count].reshape(-1)
        for group, count in enumerate(batch_cache.layout.point_counts_host)
    ]
    result = NormalEquations(
        residual=torch.cat((*residual_parts, *anchor_parts)),
        hessian=hessian,
        gradient=frame_gradient.reshape(-1),
    )
    return result


def build_stage2_pose_batched_linearization(
    fast_cache: Stage2PoseFastCache,
    batch_cache: Stage2PoseBatchCache,
    linearisation_point: torch.Tensor,
    body_pose_anchors: list[torch.Tensor],
    pose_anchor_scale: float,
    *,
    is_irls: bool,
    charbonnier_eps: float,
    aggregation_spec: dict | None,
    rep_scale: float,
) -> Linearization:
    """Bind serial weights/reference functions to the production batched NE."""
    common = build_stage2_pose_linearization_common(
        fast_cache, linearisation_point, body_pose_anchors, pose_anchor_scale,
        is_irls=is_irls, charbonnier_eps=charbonnier_eps,
        aggregation_spec=aggregation_spec, rep_scale=rep_scale,
    )
    padded_row_scales = pad_stage2_pose_row_scales(
        batch_cache, common.row_scales_by_group,
    )

    def normal_equations_fn(parameters: torch.Tensor) -> NormalEquations:
        return stage2_pose_batched_normal_equations(
            fast_cache, batch_cache, parameters, padded_row_scales,
            body_pose_anchors, pose_anchor_scale,
        )

    return Linearization(
        surrogate=common.surrogate_fn,
        jacobian=common.jacobian_fn,
        normal_equations=normal_equations_fn,
        weight_stats=common.weight_stats,
        aggregation_spec=common.spec,
        provenance=LinearizationProvenance(
            jacobian_backend="analytic_stage2_pose_batched",
            linearization_backend="direct_normal_equations_batched_full_group",
            linear_solver_backend="dense",
        ),
    )
