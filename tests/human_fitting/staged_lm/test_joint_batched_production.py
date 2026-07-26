"""Production contracts for Stage-2 Joint FULL_BATCH normal equations."""
from __future__ import annotations

from pathlib import Path
import inspect
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch

from helpers import synthetic_joint_cache, synthetic_joint_parameters
from lib_pano.human_fitting.staged_lm import config
from lib_pano.human_fitting.staged_lm.stage_definitions.joint import (
    build_joint_stage_definition,
)
from lib_pano.human_fitting.staged_lm.joint.batched import (
    Stage2JointBatchLayout,
    build_stage2_joint_batch_cache,
    gather_stage2_joint_batch_parameters,
    pad_stage2_joint_row_scales,
    stage2_joint_batched_group_analytic,
    stage2_joint_batched_normal_equations,
)


def _fixture(frames: int = 1):
    cache = synthetic_joint_cache(frames)
    parameters = synthetic_joint_parameters(frames)
    batch = build_stage2_joint_batch_cache(cache, pose_anchor_scale=0.1)
    scales = [torch.ones(group.targets.numel()) for group in cache.groups]
    anchors = [parameters[f * 75 + 12:(f + 1) * 75].clone() for f in range(frames)]
    return cache, batch, parameters, scales, anchors, torch.tensor([0.04, -0.03])


def test_layout_padding_and_metadata() -> None:
    _, batch, _, _, _, _ = _fixture(2)
    layout = batch.layout
    assert isinstance(layout, Stage2JointBatchLayout)
    assert layout.frame_positions_host == (0, 0, 1, 1)
    assert layout.camera_indices_host == (0, 1, 0, 1)
    assert tuple(layout.frame_positions.tolist()) == layout.frame_positions_host
    for index, count in enumerate(layout.point_counts_host):
        assert not layout.valid_mask[index, count:].any()
        for tensor in (layout.targets, batch.selected_base_v_shaped,
                       batch.selected_beta_shapedirs, batch.selected_posedirs.transpose(1, 2),
                       batch.selected_lbs_weights):
            assert torch.count_nonzero(tensor[index, count:]) == 0


def test_static_layout_and_parameter_mapping() -> None:
    _, batch, parameters, _, _, _ = _fixture(2)
    packed = gather_stage2_joint_batch_parameters(parameters, batch.layout, 2)
    assert packed.frame_body_poses.shape == (2, 63)
    assert torch.equal(packed.group_orientations[0], parameters[:3])
    assert torch.equal(packed.group_orientations[1], parameters[6:9])
    assert tuple(batch.active_indices_cam2.tolist()) == (*range(6), *range(12, 75))
    assert tuple(batch.active_indices_cam4.tolist()) == (*range(6, 12), *range(12, 75))
    assert batch.derivative_mapping.shape == (22, 66, 3)
    assert batch.pose_anchor_identity_64.dtype == torch.float64


def test_full_batch_equation_contract() -> None:
    cache, batch, parameters, scales, anchors, beta_anchor = _fixture(2)
    padded = pad_stage2_joint_row_scales(batch, scales)
    cache.reset_timing()
    equations = stage2_joint_batched_normal_equations(
        cache, batch, parameters, 3.0, padded, anchors, beta_anchor, 0.1, 0.2,
    )
    analytic = stage2_joint_batched_group_analytic(cache, batch, parameters, 3.0)
    assert analytic.active_pose_jacobian.shape[-1] == 69
    assert analytic.beta_jacobian.shape[-1] == 2
    system = equations.structured_system
    assert system is not None and system.timing_sink is cache
    assert system.frame_hessians.shape == (2, 75, 75)
    assert system.frame_hessians.dtype == system.beta_hessian.dtype == torch.float64
    assert equations.gradient.dtype == torch.float64
    assert cache.batched_normal_equations_calls == 1
    # The explicit diagnostic call above adds one; the NE itself adds exactly one.
    assert cache.batched_shape_state_calls == 2
    assert cache.batched_full_analytic_calls == 2


def test_provenance_architecture_and_no_backend_cli() -> None:
    assert config.PRODUCTION_JOINT_LINEARIZATION == "analytic_batched_full_group"
    assert config.PRODUCTION_JOINT_FK == "combined_batched_depth"
    source = Path(inspect.getsourcefile(stage2_joint_batched_normal_equations)).read_text()
    assert "script.benchmark" not in source and "script.validation" not in source
    assert "build_stage2_joint_batched_linearization" in inspect.getsource(
        build_joint_stage_definition,
    )
    assert not any(
        "backend" in name
        for name in inspect.signature(build_joint_stage_definition).parameters
    )


def main() -> None:
    test_layout_padding_and_metadata()
    test_static_layout_and_parameter_mapping()
    test_full_batch_equation_contract()
    test_provenance_architecture_and_no_backend_cli()
    print("[PASS] Joint FULL_BATCH production contracts")


if __name__ == "__main__":
    main()
