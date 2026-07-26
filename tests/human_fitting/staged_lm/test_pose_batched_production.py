"""Production contract tests for the Stage 2 Pose FULL_BATCH normal equations."""

from __future__ import annotations

if __package__ in (None, ""):
    from _bootstrap import ensure_test_paths
    ensure_test_paths(__file__)

import inspect
from pathlib import Path

import torch

from helpers import synthetic_joint_parameters, synthetic_stage2_context
from lib_pano.human_fitting.staged_lm.config import PRODUCTION_POSE_LINEARIZATION
from lib_pano.human_fitting.staged_lm.irls import build_primary_aggregation_spec
from lib_pano.human_fitting.staged_lm.stage2_shared_cache import build_stage2_shared_geometry_cache
from lib_pano.human_fitting.staged_lm.pose.cache import build_stage2_pose_fast_cache
from lib_pano.human_fitting.staged_lm.pose.forward import fast_raw_groups
from lib_pano.human_fitting.staged_lm.pose.derivatives import (
    build_stage2_pose_analytic_linearization,
)
from lib_pano.human_fitting.staged_lm.pose.batched import (
    Stage2PoseBatchLayout,
    build_stage2_pose_batch_cache,
    build_stage2_pose_batched_linearization,
    gather_pose_batch_parameters,
)


def _fixture(frames: int = 2, *, irls: bool = False):
    context = synthetic_stage2_context(frames)
    shared = build_stage2_shared_geometry_cache(context)
    fast = build_stage2_pose_fast_cache(context, shared)
    parameters = synthetic_joint_parameters(frames)[:-2]
    anchors = [torch.zeros(63, dtype=parameters.dtype) for _ in range(frames)]
    scale = 0.3
    batch = build_stage2_pose_batch_cache(fast, pose_anchor_scale=scale)
    groups = fast_raw_groups(fast, parameters)
    spec = build_primary_aggregation_spec(groups)
    kwargs = dict(
        is_irls=irls, charbonnier_eps=1.0,
        aggregation_spec=spec, rep_scale=0.2,
    )
    return fast, batch, parameters, anchors, scale, kwargs


def test_layout_padding_mapping_and_device() -> None:
    fast, batch, parameters, _, _, _ = _fixture()
    assert isinstance(batch.layout, Stage2PoseBatchLayout)
    assert batch.layout.frame_positions_host == (0, 0, 1, 1)
    assert batch.layout.camera_indices_host == (0, 1, 0, 1)
    assert tuple(batch.layout.point_counts.tolist()) == batch.layout.point_counts_host
    assert batch.active_indices_cam2.tolist() == [*range(6), *range(12, 75)]
    assert batch.active_indices_cam4.tolist() == [*range(6, 12), *range(12, 75)]
    for group, count in enumerate(batch.layout.point_counts_host):
        assert not batch.layout.valid_mask[group, count:].any()
        assert torch.count_nonzero(batch.layout.targets[group, count:]) == 0
        assert torch.count_nonzero(batch.selected_v_shaped[group, count:]) == 0
        assert torch.count_nonzero(batch.selected_posedirs[group, :, count:]) == 0
        assert torch.count_nonzero(batch.selected_lbs_weights[group, count:]) == 0
    for value in (
        batch.selected_v_shaped, batch.selected_posedirs,
        batch.selected_lbs_weights, batch.left_hand_pose, batch.scale,
        batch.derivative_mapping, batch.pose_anchor_identity_64,
    ):
        assert value.device == parameters.device
    assert batch.pose_anchor_identity_64.dtype == torch.float64


def test_parameters_and_static_layout() -> None:
    _, batch, parameters, _, _, _ = _fixture()
    gathered = gather_pose_batch_parameters(parameters, batch.layout)
    assert gathered.frame_body_poses.shape == (2, 63)
    assert gathered.group_body_poses.shape == (4, 63)
    torch.testing.assert_close(gathered.group_body_poses[1], gathered.frame_body_poses[0])
    assert batch.derivative_mapping.shape == (22, 66, 3)
    assert batch.fk_depth_indices
    source = inspect.getsource(gather_pose_batch_parameters)
    assert "2 * frame" not in source and "MISSING_FRAME_BODY_POSES" not in source


def test_batched_linearization_keeps_serial_reference_functions() -> None:
    fast, batch, parameters, anchors, scale, kwargs = _fixture(irls=True)
    serial = build_stage2_pose_analytic_linearization(
        fast, parameters, anchors, scale, **kwargs,
    )
    production = build_stage2_pose_batched_linearization(
        fast, batch, parameters, anchors, scale, **kwargs,
    )
    assert production.provenance.jacobian_backend == "analytic_stage2_pose_batched"
    assert production.provenance.linearization_backend == "direct_normal_equations_batched_full_group"
    assert production.weight_stats == serial.weight_stats
    assert production.aggregation_spec == serial.aggregation_spec
    torch.testing.assert_close(
        production.surrogate(parameters), serial.surrogate(parameters),
        rtol=0, atol=0,
    )
    torch.testing.assert_close(
        production.jacobian(parameters), serial.jacobian(parameters),
        rtol=0, atol=0,
    )
    equations = production.normal_equations(parameters)
    assert equations.hessian.dtype == equations.gradient.dtype == torch.float64
    assert equations.hessian.shape == (150, 150)
    assert fast.batched_normal_equations_calls == 1
    assert fast.batched_group_analytic_calls == 4


def test_architecture_and_provenance() -> None:
    production = (
        Path(__file__).parents[3]
        / "src/lib_pano/human_fitting/staged_lm/pose/batched.py"
    )
    text = production.read_text()
    assert "script.benchmark" not in text
    assert "script.validation" not in text
    assert "from tests" not in text
    assert PRODUCTION_POSE_LINEARIZATION == "analytic_batched_full_group"


if __name__ == "__main__":
    test_layout_padding_mapping_and_device()
    test_parameters_and_static_layout()
    test_batched_linearization_keeps_serial_reference_functions()
    test_architecture_and_provenance()
    print("[PASS] production Pose FULL_BATCH")
