"""Pure-logic tests for debug110 debug109-output visualization."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

_here = Path(__file__).resolve().parent
sys.path.insert(0, str(_here))
sys.path.insert(0, str(_here.parent / "src"))
import config_script  # noqa: F401

from debug109_staged_lm.types import FrozenFrameParameters
from debug110_vis_staged_lm import (
    build_visualization_params,
    parse_frame_indices,
    require_result_arrays,
    validate_parameter_dimension,
)


def _context(frame_indices: list[int], body_pose_dim: int = 6, beta_dim: int = 2):
    frozen = []
    for _ in frame_indices:
        frozen.append(FrozenFrameParameters(
            cam2_left_hand_pose=torch.arange(45).reshape(1, 45),
            cam2_right_hand_pose=torch.arange(45).reshape(1, 45) + 1,
            cam4_left_hand_pose=torch.arange(45).reshape(1, 45) + 2,
            cam4_right_hand_pose=torch.arange(45).reshape(1, 45) + 3,
            cam2_scale=torch.tensor([1.1]), cam4_scale=torch.tensor([1.2]),
        ))
    return SimpleNamespace(
        frame_count=len(frame_indices), frame_indices=frame_indices,
        body_pose_dim=body_pose_dim, beta_dim=beta_dim,
        body_pose_anchors=[torch.full((body_pose_dim,), float(i + 10)) for i in range(len(frame_indices))],
        beta_anchor=torch.tensor([0.1, -0.2]), frozen_params_by_frame=frozen,
    )


def test_stage1_dimensions_and_noncontinuous_frames() -> None:
    for frames in ([513], [513, 515]):
        context = _context(frames)
        vector = np.arange(len(frames) * 12, dtype=np.float32)
        validate_parameter_dimension(context, vector, "stage1")
        cam2, cam4 = build_visualization_params(context, vector, context.beta_anchor.numpy(), "stage1")
        assert len(cam2) == len(frames) == len(cam4)
        for position in range(len(frames)):
            np.testing.assert_array_equal(cam2[position].orient.numpy(), vector[position * 12:position * 12 + 3])
            np.testing.assert_array_equal(cam4[position].transl.numpy(), vector[position * 12 + 9:(position + 1) * 12])
            np.testing.assert_array_equal(cam2[position].body_pose.numpy(), context.body_pose_anchors[position].numpy())
    assert parse_frame_indices("513,515") == [513, 515]
    print("test_stage1_dimensions_and_noncontinuous_frames: PASS")


def test_pose_single_and_two_frame_slicing() -> None:
    for frames in ([513], [513, 515]):
        context = _context(frames)
        per_frame = 12 + context.body_pose_dim
        vector = np.arange(len(frames) * per_frame, dtype=np.float32)
        cam2, _ = build_visualization_params(context, vector, context.beta_anchor.numpy(), "pose_warmup")
        for position in range(len(frames)):
            expected_pose = vector[position * per_frame + 12:(position + 1) * per_frame]
            np.testing.assert_array_equal(cam2[position].body_pose.numpy(), expected_pose)
    print("test_pose_single_and_two_frame_slicing: PASS")


def test_joint_latent_excluded_and_physical_beta_unchanged() -> None:
    context = _context([513, 515])
    per_frame = 12 + context.body_pose_dim
    frame_region = np.arange(context.frame_count * per_frame, dtype=np.float32)
    latent = np.array([9.0, -9.0], dtype=np.float32)
    physical_beta = np.array([1.5, -1.25], dtype=np.float32)
    cam2, _ = build_visualization_params(
        context, np.concatenate([frame_region, latent]), physical_beta, "joint_refinement",
    )
    np.testing.assert_array_equal(cam2[-1].body_pose.numpy(), frame_region[-context.body_pose_dim:])
    np.testing.assert_array_equal(cam2[-1].beta.numpy(), physical_beta)
    assert not np.allclose(cam2[-1].beta.numpy(), np.tanh(physical_beta))
    print("test_joint_latent_excluded_and_physical_beta_unchanged: PASS")


def test_missing_lbfgs_error() -> None:
    try:
        require_result_arrays({"beta_anchor": np.zeros(2)}, "joint_refinement", "lbfgs")
    except KeyError as exc:
        message = str(exc)
        assert "x_stage2_joint_lbfgs is absent" in message
        assert "--run_lbfgs_oracle" in message
    else:
        raise AssertionError("missing L-BFGS result did not fail")
    print("test_missing_lbfgs_error: PASS")


if __name__ == "__main__":
    test_stage1_dimensions_and_noncontinuous_frames()
    test_pose_single_and_two_frame_slicing()
    test_joint_latent_excluded_and_physical_beta_unchanged()
    test_missing_lbfgs_error()
    print("\nAll debug110 tests PASSED")
