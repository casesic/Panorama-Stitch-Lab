"""Tests for debug029/debug077-compatible staged-LM exports."""

from __future__ import annotations

import json
import pickle
from types import SimpleNamespace

import numpy as np
import pytest
import torch

try:
    from ._bootstrap import ensure_test_paths
except ImportError:
    from _bootstrap import ensure_test_paths
ensure_test_paths(__file__)

from lib_pano.human_fitting.smplx_sequence import SMPLX_Seq_Forward_Module
from lib_pano.human_fitting.staged_lm.output import (
    build_compatible_smplx_sequences,
    export_debug029_compatible_result,
)
from lib_pano.human_fitting.staged_lm.types import FrozenFrameParameters
from lib_pano.human_vis.smplx_forward_vis import unpack_smplx_result


def _context(frames: list[int]):
    frozen = [
        FrozenFrameParameters(
            cam2_left_hand_pose=torch.full((1, 45), float(i + 1)),
            cam2_right_hand_pose=torch.full((1, 45), float(i + 2)),
            cam4_left_hand_pose=torch.full((1, 45), float(i + 3)),
            cam4_right_hand_pose=torch.full((1, 45), float(i + 4)),
            cam2_scale=torch.tensor([1.1]),
            cam4_scale=torch.tensor([1.2]),
        )
        for i in range(len(frames))
    ]
    return SimpleNamespace(
        frame_indices=frames,
        frame_count=len(frames),
        body_pose_dim=63,
        beta_dim=10,
        body_pose_anchors=[torch.zeros(63) for _ in frames],
        frozen_params_by_frame=frozen,
        cam2_vertex_ids_by_frame=[torch.tensor([1, 2]) for _ in frames],
        cam4_vertex_ids_by_frame=[torch.tensor([], dtype=torch.long) for _ in frames],
    )


def _joint_arrays(frame_count: int) -> dict[str, np.ndarray]:
    per_frame = 75
    frame_values = np.arange(frame_count * per_frame, dtype=np.float32)
    latent_beta = np.linspace(-0.5, 0.5, 10, dtype=np.float32)
    physical_beta = np.linspace(-1.0, 1.0, 10, dtype=np.float32)
    return {
        "x_stage2_pose_shape_lm": np.concatenate((frame_values, latent_beta)),
        "beta_lm": physical_beta,
    }


def test_build_compatible_sequences_uses_physical_beta() -> None:
    context = _context([513, 514])
    arrays = _joint_arrays(context.frame_count)
    cam2, cam4, beta = build_compatible_smplx_sequences(
        context, arrays, "pose_shape_refinement",
    )

    assert isinstance(cam2, SMPLX_Seq_Forward_Module.SMPLX_Param_Seq)
    assert isinstance(cam4, SMPLX_Seq_Forward_Module.SMPLX_Param_Seq)
    assert len(unpack_smplx_result(cam2)) == 2
    assert cam2.orient.shape == (2, 3)
    assert cam2.body_pose.shape == (2, 63)
    torch.testing.assert_close(cam2.beta, torch.from_numpy(arrays["beta_lm"]))
    torch.testing.assert_close(cam4.beta, beta)
    assert cam2.orient.device.type == "cpu"
    assert cam4.orient.device.type == "cpu"


def test_export_writes_debug077_layout(tmp_path) -> None:
    context = _context([513, 514])
    export_debug029_compatible_result(
        tmp_path, context, _joint_arrays(context.frame_count), "pose_shape_refinement",
    )

    for camera, valid_count in (("cam2", 2), ("cam4", 0)):
        seq_path = tmp_path / camera / "seg_513_514_seq.pkl"
        meta_path = tmp_path / camera / "seg_513_514_meta.json"
        with seq_path.open("rb") as file:
            sequence = pickle.load(file)
        assert len(unpack_smplx_result(sequence)) == 2
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        assert metadata["frame_idxs"] == [513, 514]
        assert metadata["valid_frame_count"] == valid_count
        assert metadata["export_source"] == "staged_lm"
        assert metadata["export_stage"] == "pose_shape_refinement"


def test_export_rejects_noncontiguous_frames(tmp_path) -> None:
    context = _context([513, 515])
    with pytest.raises(ValueError, match="contiguous ascending"):
        export_debug029_compatible_result(
            tmp_path, context, _joint_arrays(context.frame_count), "pose_shape_refinement",
        )
