"""Tests for immutable geometry shared by Pose and Joint Stage 2 caches."""

from __future__ import annotations

import inspect
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
from smplx.lbs import blend_shapes, vertices2joints

try:
    from ._bootstrap import ensure_test_paths
except ImportError:
    from _bootstrap import ensure_test_paths
ensure_test_paths(__file__)

from tests.human_fitting.staged_lm.helpers import synthetic_stage2_context  # noqa: E402
from lib_pano.human_fitting.staged_lm.joint import (  # noqa: E402
    build_stage2_joint_fast_cache,
    joint_fast_raw_groups,
)
from lib_pano.human_fitting.staged_lm.pose.cache import build_stage2_pose_fast_cache  # noqa: E402
from lib_pano.human_fitting.staged_lm.pose.forward import fast_raw_groups  # noqa: E402
from lib_pano.human_fitting.staged_lm.stage2_shared_cache import (  # noqa: E402
    Stage2SharedGeometryCache,
    Stage2SharedModelLayout,
    build_stage2_shared_geometry_cache,
)
from lib_pano.human_fitting.staged_lm.pipeline import run_stage2  # noqa: E402
from lib_pano.human_fitting.staged_lm.types import StagePreparationTiming  # noqa: E402


def _pose_parameters(frames: int, dtype: torch.dtype) -> torch.Tensor:
    result = torch.zeros(frames * 75, dtype=dtype)
    for frame in range(frames):
        result[frame * 75 + 3:frame * 75 + 6] = torch.tensor(
            [0.1, 0.2, 2.0], dtype=dtype,
        )
        result[frame * 75 + 9:frame * 75 + 12] = torch.tensor(
            [-0.1, 0.1, 2.2], dtype=dtype,
        )
    return result


def _tensor_bytes(tensors) -> int:
    seen: set[int] = set()
    total = 0
    for tensor in tensors:
        pointer = tensor.data_ptr()
        if pointer not in seen:
            seen.add(pointer)
            total += tensor.numel() * tensor.element_size()
    return total


def _build(frames: int = 2):
    context = synthetic_stage2_context(frames)
    shared = build_stage2_shared_geometry_cache(context)
    pose = build_stage2_pose_fast_cache(context, shared)
    joint = build_stage2_joint_fast_cache(context, shared)
    return context, shared, pose, joint


def test_group_order_device_dtype_and_select_counts() -> None:
    context, shared, _pose, _joint = _build(2)
    assert [(group.frame_index, group.camera) for group in shared.groups] == [
        (513, "cam2"), (513, "cam4"),
        (520, "cam2"), (520, "cam4"),
    ]
    assert shared.posedirs_select_calls == 4
    assert shared.lbs_weights_select_calls == 4
    for group in shared.groups:
        assert group.vertex_ids.dtype == torch.long
        assert group.vertex_ids.device == context.model.v_template.device
        for tensor in (
            group.targets, group.left_hand_pose, group.right_hand_pose,
            group.scale, group.selected_posedirs, group.selected_lbs_weights,
        ):
            assert tensor.dtype == context.model.v_template.dtype
            assert tensor.device == context.model.v_template.device


def test_common_storage_is_shared_and_read_only() -> None:
    _context, shared, pose, joint = _build()
    tracked = []
    for shared_group, pose_group, joint_group in zip(
        shared.groups, pose.groups, joint.groups,
    ):
        for name in (
            "vertex_ids", "targets", "left_hand_pose", "right_hand_pose",
            "scale", "selected_posedirs", "selected_lbs_weights",
        ):
            common = getattr(shared_group, name)
            assert getattr(pose_group, name).data_ptr() == common.data_ptr()
            assert getattr(joint_group, name).data_ptr() == common.data_ptr()
            tracked.append(common)
    snapshots = [(tensor.clone(), tensor._version) for tensor in tracked]
    pose_parameters = _pose_parameters(pose.frame_count, pose.model.dtype)
    joint_parameters = torch.cat((
        pose_parameters, torch.zeros(joint.model.beta_dim, dtype=pose.model.dtype),
    ))
    fast_raw_groups(pose, pose_parameters)
    joint_fast_raw_groups(joint, joint_parameters, 3.0)
    for tensor, (value, version) in zip(tracked, snapshots):
        assert tensor._version == version
        assert torch.equal(tensor, value)


def test_derived_values_match_legacy_formulas() -> None:
    context, shared, pose, joint = _build()
    model = context.model
    beta = context.beta_anchor.reshape(1, -1)
    expression = torch.zeros(1, model.num_expression_coeffs, dtype=beta.dtype)
    shape_dirs = torch.cat((model.shapedirs, model.expr_dirs), dim=-1)
    pose_v = model.v_template + blend_shapes(
        torch.cat((beta, expression), dim=-1), shape_dirs,
    ).squeeze(0)
    pose_joints = vertices2joints(
        model.J_regressor, pose_v.unsqueeze(0),
    ).squeeze(0)
    zero_components = torch.zeros_like(torch.cat((beta, expression), dim=-1))
    joint_v = model.v_template + blend_shapes(
        zero_components, shape_dirs,
    ).squeeze(0)
    joint_joints = vertices2joints(
        model.J_regressor, joint_v.unsqueeze(0),
    ).squeeze(0)
    d_joints = torch.einsum("jv,vcb->jcb", model.J_regressor, model.shapedirs)
    assert torch.equal(pose.model.v_shaped, pose_v)
    assert torch.equal(pose.model.joints, pose_joints)
    assert torch.equal(joint.model.base_v_shaped, joint_v)
    assert torch.equal(joint.model.base_joints, joint_joints)
    assert torch.equal(joint.model.d_joints_d_beta, d_joints)
    posedirs = model.posedirs.reshape(486, model.v_template.shape[0], 3)
    for common, pose_group, joint_group in zip(
        shared.groups, pose.groups, joint.groups,
    ):
        ids = common.vertex_ids
        legacy_posedirs = posedirs.index_select(1, ids).detach().clone()
        legacy_weights = model.lbs_weights.index_select(0, ids).detach().clone()
        assert torch.equal(common.selected_posedirs, legacy_posedirs)
        assert torch.equal(common.selected_lbs_weights, legacy_weights)
        assert torch.equal(pose_group.selected_v_shaped, pose_v.index_select(0, ids))
        assert torch.equal(joint_group.selected_base_v_shaped, joint_v.index_select(0, ids))
        assert torch.equal(
            joint_group.selected_beta_shapedirs,
            model.shapedirs.index_select(0, ids),
        )


def test_persistent_bytes_are_deduplicated() -> None:
    _context, shared, pose, joint = _build()
    common = [shared.model.parents]
    pose_derived = [pose.model.v_shaped, pose.model.joints]
    joint_derived = [
        joint.model.base_v_shaped, joint.model.beta_shapedirs,
        joint.model.base_joints, joint.model.d_joints_d_beta,
    ]
    for shared_group, pose_group, joint_group in zip(
        shared.groups, pose.groups, joint.groups,
    ):
        common.extend((
            shared_group.vertex_ids, shared_group.targets,
            shared_group.left_hand_pose, shared_group.right_hand_pose,
            shared_group.scale, shared_group.selected_posedirs,
            shared_group.selected_lbs_weights,
        ))
        pose_derived.append(pose_group.selected_v_shaped)
        joint_derived.extend((
            joint_group.selected_base_v_shaped,
            joint_group.selected_beta_shapedirs,
        ))
    shared_bytes = _tensor_bytes(common)
    assert shared_bytes > 0
    assert _tensor_bytes(common + pose_derived + joint_derived) < (
        2 * shared_bytes
        + _tensor_bytes(pose_derived)
        + _tensor_bytes(joint_derived)
    )


def test_no_redundant_selected_row_clone_in_production() -> None:
    root = Path(__file__).resolve().parents[3] / "src/lib_pano/human_fitting/staged_lm"
    shared_source = (root / "stage2_shared_cache.py").read_text()
    pose_source = (root / "pose/cache.py").read_text()
    joint_source = (root / "joint/cache.py").read_text()
    assert "index_select(1, item[3]).detach().clone()" not in shared_source
    assert "index_select(0, item[3]).detach().clone()" not in shared_source
    assert "selected_posedirs=posedirs" not in pose_source + joint_source
    assert "selected_lbs_weights=model.lbs_weights" not in pose_source + joint_source
    assert "shared_cache=None" not in inspect.getsource(build_stage2_pose_fast_cache)
    assert "shared_cache=None" not in inspect.getsource(build_stage2_joint_fast_cache)


def test_stop_stage_lifecycle_build_counts() -> None:
    layout = Stage2SharedModelLayout(
        torch.tensor([-1]), 1, 0, 0, 0, 0,
        torch.float32, torch.device("cpu"),
    )
    shared = Stage2SharedGeometryCache(
        layout, [], 1, 1, 1, 2.0, 0.5, 1.0, 0.4, 2, 2,
    )
    pose_cache = SimpleNamespace(
        setup_ms=0.2, batch_cache_setup_ms=0.0,
        batch_padding_ratio=0.0, batch_persistent_bytes=0,
        batch_group_count=0, batch_nmax=0,
    )
    joint_cache = SimpleNamespace(
        setup_ms=0.3, batch_cache_setup_ms=0.0,
        batch_padding_ratio=0.0, batch_persistent_bytes=0,
        batch_group_count=0, batch_nmax=0,
    )
    pose_definition = SimpleNamespace(
        preparation_timing=StagePreparationTiming(0.1, 0.2, 0.3),
        analytic_cache=pose_cache,
    )
    joint_definition = SimpleNamespace(
        preparation_timing=StagePreparationTiming(0.2, 0.3, 0.4),
        analytic_cache=joint_cache,
    )
    pose_outcome = SimpleNamespace(
        lm_result=SimpleNamespace(wall_time_ms=1.0),
        definition=pose_definition,
    )
    joint_outcome = SimpleNamespace(
        lm_result=SimpleNamespace(wall_time_ms=1.0),
        definition=joint_definition,
    )
    pose_handoff = {"lm": {"ot_max": 0.0, "bp_max": 0.0, "ok": True}}
    joint_handoff = {"lm": {
        "state_max_diff": 0.0,
        "physical_beta_init_max_diff_to_anchor": 0.0,
        "raw_reprojection_max_diff": None,
        "shape_anchor_objective_at_init": 0.0,
        "relative_difference_joint_init_to_pose": None,
        "validation_scope": "state_only", "ok": True,
    }}
    for stop, expected_joint_builds in (("stage2_pose", 0), ("stage2", 1)):
        context = SimpleNamespace(
            device=torch.device("cpu"), args=SimpleNamespace(stop_after_stage=stop),
        )
        shared_builder = Mock(return_value=shared)
        pose_builder = Mock(return_value=(pose_definition, pose_handoff))
        joint_builder = Mock(return_value=(joint_definition, joint_handoff))
        outcomes = [pose_outcome] + ([joint_outcome] if expected_joint_builds else [])
        timing = {}
        with patch(
            "lib_pano.human_fitting.staged_lm.pipeline.build_stage2_shared_geometry_cache",
            shared_builder,
        ), patch(
            "lib_pano.human_fitting.staged_lm.pipeline.build_pose_stage_definition",
            pose_builder,
        ), patch(
            "lib_pano.human_fitting.staged_lm.pipeline.build_joint_stage_definition",
            joint_builder,
        ), patch(
            "lib_pano.human_fitting.staged_lm.pipeline.run_stage",
            side_effect=outcomes,
        ):
            run_stage2(context, SimpleNamespace(), timing)
        assert shared_builder.call_count == 1
        assert pose_builder.call_count == 1
        assert joint_builder.call_count == expected_joint_builds
        assert timing["stage2_shared_cache_setup_ms"] == 2.0
        assert timing["stage2_total_cache_setup_ms"] == (
            2.2 if not expected_joint_builds else 2.5
        )


if __name__ == "__main__":
    test_group_order_device_dtype_and_select_counts()
    test_common_storage_is_shared_and_read_only()
    test_derived_values_match_legacy_formulas()
    test_persistent_bytes_are_deduplicated()
    test_no_redundant_selected_row_clone_in_production()
    test_stop_stage_lifecycle_build_counts()
    print("test_stage2_shared_cache: PASS")
