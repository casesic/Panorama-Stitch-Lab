"""Shared synthetic fixtures for staged-LM tests."""

from __future__ import annotations

from types import SimpleNamespace

import torch

from lib_pano.human_fitting.staged_lm.stage2_joint_fast import (
    Stage2JointFastCache,
    Stage2JointFrameCameraCache,
    Stage2JointModelCache,
)
from lib_pano.human_fitting.staged_lm.types import FrozenFrameParameters


DTYPE = torch.float64


def synthetic_joint_cache(frames: int = 1) -> Stage2JointFastCache:
    torch.manual_seed(51)
    joint_count, vertex_count, beta_dim = 55, 6, 2
    parents = torch.arange(joint_count) - 1
    parents[0] = -1
    base_vertices = torch.randn(vertex_count, 3, dtype=DTYPE) * 0.1
    beta_directions = torch.randn(vertex_count, 3, beta_dim, dtype=DTYPE) * 0.02
    regressor = torch.rand(joint_count, vertex_count, dtype=DTYPE)
    regressor /= regressor.sum(1, keepdim=True)
    base_joints = torch.einsum("jv,vc->jc", regressor, base_vertices)
    joint_directions = torch.einsum("jv,vcb->jcb", regressor, beta_directions)
    model = Stage2JointModelCache(
        base_vertices, beta_directions, base_joints, joint_directions,
        parents, joint_count, 486, vertex_count, beta_dim, 63,
        DTYPE, torch.device("cpu"),
    )
    groups = []
    for frame in range(frames):
        for camera_index, camera in enumerate(("cam2", "cam4")):
            ids = (
                torch.tensor([4, 1])
                if camera == "cam2" else torch.tensor([5, 2, 0])
            )
            weights = torch.rand(ids.numel(), joint_count, dtype=DTYPE)
            weights /= weights.sum(1, keepdim=True)
            groups.append(Stage2JointFrameCameraCache(
                frame, 513 + frame * 2, camera, ids,
                torch.randn(ids.numel(), 2, dtype=DTYPE) * 10 + 100,
                torch.randn(45, dtype=DTYPE) * 0.01,
                torch.randn(45, dtype=DTYPE) * 0.01,
                torch.tensor([1 + 0.2 * camera_index], dtype=DTYPE),
                base_vertices[ids], beta_directions[ids],
                torch.randn(486, ids.numel(), 3, dtype=DTYPE) * 1e-4,
                weights,
            ))
    return Stage2JointFastCache(model, groups, frames, 5760, 2880, 0.0)


def synthetic_joint_parameters(frames: int = 1) -> torch.Tensor:
    torch.manual_seed(52)
    parameters = torch.randn(frames * 75 + 2, dtype=DTYPE) * 0.02
    for frame in range(frames):
        parameters[frame * 75 + 3:frame * 75 + 6] += torch.tensor(
            [0.1, 0.2, 2.0], dtype=DTYPE,
        )
        parameters[frame * 75 + 9:frame * 75 + 12] += torch.tensor(
            [-0.1, 0.1, 2.2], dtype=DTYPE,
        )
    return parameters


# Short aliases keep the migrated test bodies mechanically unchanged.
_cache = synthetic_joint_cache
_parameters = synthetic_joint_parameters


def synthetic_stage2_context(frames: int = 1, dtype=torch.float64):
    """Small SMPL-X-like context suitable for shared/derived cache tests."""
    torch.manual_seed(61)
    joint_count, vertex_count, beta_dim, expression_dim = 55, 7, 2, 3
    shapedirs = torch.randn(vertex_count, 3, beta_dim, dtype=dtype) * 0.01
    expr_dirs = torch.randn(vertex_count, 3, expression_dim, dtype=dtype) * 0.01
    regressor = torch.rand(joint_count, vertex_count, dtype=dtype)
    regressor /= regressor.sum(1, keepdim=True)
    parents = torch.arange(joint_count, dtype=torch.long) - 1
    parents[0] = -1
    model = SimpleNamespace(
        v_template=torch.randn(vertex_count, 3, dtype=dtype) * 0.1,
        shapedirs=shapedirs,
        expr_dirs=expr_dirs,
        posedirs=torch.randn(486, vertex_count * 3, dtype=dtype) * 1e-4,
        J_regressor=regressor,
        parents=parents,
        lbs_weights=torch.rand(vertex_count, joint_count, dtype=dtype),
        NUM_BODY_JOINTS=21,
        NUM_HAND_JOINTS=15,
        num_expression_coeffs=expression_dim,
        use_pca=False,
    )
    model.lbs_weights /= model.lbs_weights.sum(1, keepdim=True)
    frame_indices = [513 + frame * 7 for frame in range(frames)]
    cam2_ids = [torch.tensor([4, 1], dtype=torch.long) for _ in frame_indices]
    cam4_ids = [torch.tensor([6, 2, 0], dtype=torch.long) for _ in frame_indices]
    cam2_targets = [torch.randn(2, 2, dtype=dtype) for _ in frame_indices]
    cam4_targets = [torch.randn(3, 2, dtype=dtype) for _ in frame_indices]
    frozen = []
    for frame in range(frames):
        frozen.append(FrozenFrameParameters(
            cam2_left_hand_pose=torch.randn(1, 45, dtype=dtype),
            cam2_right_hand_pose=torch.randn(1, 45, dtype=dtype),
            cam4_left_hand_pose=torch.randn(1, 45, dtype=dtype),
            cam4_right_hand_pose=torch.randn(1, 45, dtype=dtype),
            cam2_scale=torch.tensor([1.0 + frame * 0.01], dtype=dtype),
            cam4_scale=torch.tensor([1.2 + frame * 0.01], dtype=dtype),
        ))
    return SimpleNamespace(
        model=model, body_pose_dim=63, beta_dim=beta_dim,
        beta_anchor=torch.tensor([0.04, -0.03], dtype=dtype),
        frame_indices=frame_indices, frame_count=frames,
        image_width=5760, image_height=2880,
        frozen_params_by_frame=frozen,
        cam2_vertex_ids_by_frame=cam2_ids,
        cam2_targets_by_frame=cam2_targets,
        cam4_vertex_ids_by_frame=cam4_ids,
        cam4_targets_by_frame=cam4_targets,
        args=SimpleNamespace(profile_timing=False),
    )
