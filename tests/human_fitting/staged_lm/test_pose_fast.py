"""Pure logic tests for selected-vertex Stage 2 pose forward."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

try:
    from ._bootstrap import ensure_test_paths
except ImportError:
    from _bootstrap import ensure_test_paths
ensure_test_paths(__file__)

from lib_pano.human_fitting.staged_lm.stage2_pose_fast import (
    Stage2PoseFastCache,
    Stage2PoseFrameCameraCache,
    Stage2PoseModelCache,
    compute_selected_lbs,
    compute_selected_pose_offsets,
    fast_raw_groups,
    fast_selected_vertices,
)


def test_selected_rows_and_order() -> None:
    torch.manual_seed(1)
    vertex_count, pose_dim, joint_count = 6, 18, 3
    ids = torch.tensor([4, 1])
    v_shaped = torch.randn(vertex_count, 3)
    posedirs = torch.randn(pose_dim, vertex_count, 3)
    weights = torch.randn(vertex_count, joint_count).softmax(1)
    torch.testing.assert_close(v_shaped.index_select(0, ids), v_shaped[[4, 1]])
    torch.testing.assert_close(posedirs.index_select(1, ids), posedirs[:, [4, 1]])
    torch.testing.assert_close(weights.index_select(0, ids), weights[[4, 1]])
    print("test_selected_rows_and_order: PASS")


def test_selected_pose_offsets() -> None:
    torch.manual_seed(2)
    rotations = torch.linalg.qr(torch.randn(3, 3, 3)).Q
    full_posedirs = torch.randn(18, 6, 3)
    ids = torch.tensor([5, 2, 4])
    selected, feature = compute_selected_pose_offsets(rotations, full_posedirs[:, ids])
    full = torch.einsum("p,pvc->vc", feature, full_posedirs)
    torch.testing.assert_close(selected, full.index_select(0, ids))
    print("test_selected_pose_offsets: PASS")


def test_selected_lbs_matches_full() -> None:
    torch.manual_seed(3)
    vertices = torch.randn(6, 3)
    weights = torch.randn(6, 3).softmax(1)
    transforms = torch.eye(4).repeat(3, 1, 1)
    transforms[:, :3, 3] = torch.randn(3, 3)
    ids = torch.tensor([4, 1])
    full, _ = compute_selected_lbs(vertices, weights, transforms)
    selected, selected_transforms = compute_selected_lbs(vertices[ids], weights[ids], transforms)
    torch.testing.assert_close(selected, full[ids])
    assert selected_transforms.shape == (2, 4, 4)
    print("test_selected_lbs_matches_full: PASS")


def _artificial_cache() -> Stage2PoseFastCache:
    torch.manual_seed(4)
    joint_count, pose_dim = 55, 486
    parents = torch.zeros(joint_count, dtype=torch.long)
    model = Stage2PoseModelCache(
        v_shaped=torch.randn(6, 3), joints=torch.randn(joint_count, 3),
        parents=parents, joint_count=joint_count, pose_feature_dim=pose_dim,
        full_vertex_count=6, dtype=torch.float32, device=torch.device("cpu"),
    )
    groups = []
    for frame in range(2):
        for camera_index, camera in enumerate(("cam2", "cam4")):
            ids = torch.tensor([4, 1]) if camera == "cam2" else torch.tensor([5, 2, 0])
            groups.append(Stage2PoseFrameCameraCache(
                frame_position=frame, frame_index=10 + frame, camera=camera,
                vertex_ids=ids, targets=torch.zeros(ids.numel(), 2),
                left_hand_pose=torch.full((45,), camera_index * 0.01),
                right_hand_pose=torch.full((45,), camera_index * -0.02),
                scale=torch.tensor([1.0 + camera_index * 0.2]),
                selected_v_shaped=model.v_shaped[ids].clone(),
                selected_posedirs=torch.randn(pose_dim, ids.numel(), 3) * 1e-3,
                selected_lbs_weights=torch.randn(ids.numel(), joint_count).softmax(1),
            ))
    return Stage2PoseFastCache(model, groups, 2, 63, 5760, 2880, 0.0)


def test_camera_frame_pose_scale_and_translation_contract() -> None:
    cache = _artificial_cache()
    per_frame = 75
    parameters = torch.zeros(2 * per_frame)
    parameters[3:6] = torch.tensor([0.1, 0.2, 0.3])
    parameters[9:12] = torch.tensor([-0.2, 0.1, 0.4])
    parameters[12:75] = 0.01
    parameters[75 + 12:150] = -0.02
    groups = fast_raw_groups(cache, parameters)
    assert [(group.frame_index, group.camera) for group in groups] == [
        (10, "cam2"), (10, "cam4"), (11, "cam2"), (11, "cam4"),
    ]
    assert [group.valid_point_count for group in groups] == [2, 3, 2, 3]
    # Both cameras in a frame consume the shared body block, while different
    # hand poses, IDs, scale, orient and translation remain camera-specific.
    cam2 = cache.groups[0]
    result = fast_selected_vertices(
        cache, cam2, parameters[:3], parameters[3:6], parameters[12:75],
    )
    shifted = fast_selected_vertices(
        cache, cam2, parameters[:3], parameters[3:6] + 1.0, parameters[12:75],
    )
    torch.testing.assert_close(shifted - result, torch.ones_like(result))
    assert result.shape == (cam2.vertex_ids.numel(), 3)
    assert result.dtype == parameters.dtype and result.device == parameters.device
    assert result.shape[0] < cache.model.full_vertex_count
    assert not torch.allclose(groups[0].residual_2d, groups[2].residual_2d)
    print("test_camera_frame_pose_scale_and_translation_contract: PASS")


if __name__ == "__main__":
    test_selected_rows_and_order()
    test_selected_pose_offsets()
    test_selected_lbs_matches_full()
    test_camera_frame_pose_scale_and_translation_contract()
    print("\nAll Stage 2 selected-forward tests PASSED")
