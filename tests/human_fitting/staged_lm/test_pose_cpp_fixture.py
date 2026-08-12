"""Pose value dump, input-only fixture and parity-checker tests."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from lib_pano.human_fitting.staged_lm.pose.cache import (
    Stage2PoseFastCache,
    Stage2PoseFrameCameraCache,
    Stage2PoseModelCache,
)
from script.validation.staged_lm.check_pose_cpp_reference import (
    PIXEL_ABSOLUTE_TOLERANCE,
    check_pose_reference,
)
from script.validation.staged_lm.dump_pose_cpp_reference import write_pose_reference
from script.validation.staged_lm.export_pose_cpp_fixture import (
    FLOAT_INPUTS,
    INTEGER_INPUTS,
    export_pose_fixture,
)


def _cache() -> tuple[Stage2PoseFastCache, torch.Tensor]:
    torch.manual_seed(109)
    dtype = torch.float32
    joint_count = 55
    parents = torch.arange(joint_count, dtype=torch.long) - 1
    parents[0] = -1
    joints = torch.randn(joint_count, 3, dtype=dtype) * 0.02
    model = Stage2PoseModelCache(
        torch.empty(0, 3), joints, parents, joint_count, 486, 6,
        dtype, torch.device("cpu"),
    )
    groups = []
    for camera, count in (("cam2", 2), ("cam4", 3)):
        weights = torch.rand(count, joint_count, dtype=dtype)
        weights /= weights.sum(1, keepdim=True)
        groups.append(Stage2PoseFrameCameraCache(
            0, 513, camera, torch.arange(count),
            torch.randn(count, 2, dtype=dtype),
            torch.zeros(45, dtype=dtype), torch.zeros(45, dtype=dtype),
            torch.ones(1, dtype=dtype), torch.randn(count, 3, dtype=dtype),
            torch.randn(486, count, 3, dtype=dtype) * 1e-4, weights,
        ))
    cache = Stage2PoseFastCache(model, groups, 1, 63, 5760, 2880, 0.0)
    parameters = torch.randn(75, dtype=dtype) * 0.01
    parameters[3:6] += torch.tensor([0.1, 0.2, 2.0])
    parameters[9:12] += torch.tensor([-0.1, 0.1, 2.2])
    return cache, parameters


def test_pose_dump_and_fixture_are_deterministic_input_only(tmp_path):
    cache, parameters = _cache()
    reference = tmp_path / "reference"
    _, _, _, manifest = write_pose_reference(
        cache, parameters, reference, frame=513,
        image_width=5760, image_height=2880,
    )
    assert manifest["parameter_dim"] == 75
    assert manifest["parity"]["cam2_vertices_exact"]
    assert manifest["parity"]["cam4_residual_exact"]
    with np.load(reference / "pose_reference.npz", allow_pickle=False) as arrays:
        assert arrays["cam2_axis_angles"].shape == (55, 3)
        assert arrays["cam4_rotations"].shape == (55, 3, 3)
        assert arrays["cam2_pose_feature"].shape == (486,)
    first = export_pose_fixture(reference, tmp_path / "fixture_a")
    second = export_pose_fixture(reference, tmp_path / "fixture_b")
    expected = {f"{name}.f32" for name in FLOAT_INPUTS} | {
        f"{name}.i64" for name in INTEGER_INPUTS
    }
    assert {path.name for path in first.iterdir() if path.suffix in (".f32", ".i64")} == expected
    forbidden = ("rotations", "pose_feature", "offsets", "transforms", "vertices", "uv")
    assert not any(token in path.name for path in first.iterdir() for token in forbidden)
    for name in expected:
        assert (first / name).read_bytes() == (second / name).read_bytes()


def test_pose_checker_accepts_exact_outputs(tmp_path):
    cache, parameters = _cache()
    reference = tmp_path / "reference"
    write_pose_reference(cache, parameters, reference, frame=513, image_width=5760, image_height=2880)
    output = tmp_path / "output"
    output.mkdir()
    with np.load(reference / "pose_reference.npz", allow_pickle=False) as arrays:
        for camera in ("cam2", "cam4"):
            for field in (
                "axis_angles", "rotations", "pose_feature", "pose_offsets", "v_posed",
                "relative_joints", "local_transforms", "global_transforms",
                "relative_transforms", "selected_transforms", "skinned_vertices",
                "vertices", "uv", "raw_residual",
            ):
                arrays[f"{camera}_{field}"].astype("<f4").tofile(
                    output / f"{camera}_{field}.f32"
                )
    (output / "cpp_output_meta.txt").write_text(
        "schema_version=1\ncam2_count=2\ncam4_count=3\ncam2_total_value_forward_ms=1.0\n"
    )
    result = check_pose_reference(reference / "pose_reference.npz", output)
    assert result["pass"] and result["first_divergence"] == "NONE"
    assert (output / "pose_cpp_parity.json").is_file()
    assert (output / "pose_cpp_parity.md").is_file()


def test_residual_policy_uses_pixel_error_and_upstream_uv(tmp_path):
    cache, parameters = _cache()
    reference = tmp_path / "reference"
    write_pose_reference(cache, parameters, reference, frame=513, image_width=5760, image_height=2880)
    output = tmp_path / "output"
    output.mkdir()
    with np.load(reference / "pose_reference.npz", allow_pickle=False) as arrays:
        for camera in ("cam2", "cam4"):
            for field in (
                "axis_angles", "rotations", "pose_feature", "pose_offsets", "v_posed",
                "relative_joints", "local_transforms", "global_transforms",
                "relative_transforms", "selected_transforms", "skinned_vertices",
                "vertices", "uv", "raw_residual",
            ):
                value = arrays[f"{camera}_{field}"].copy()
                if field in ("uv", "raw_residual"):
                    value.flat[0] += PIXEL_ABSOLUTE_TOLERANCE * 0.49
                value.astype("<f4").tofile(output / f"{camera}_{field}.f32")
    (output / "cpp_output_meta.txt").write_text(
        "schema_version=1\ncam2_count=2\ncam4_count=3\n"
    )
    result = check_pose_reference(reference / "pose_reference.npz", output)
    assert result["pass"] and result["first_divergence"] == "NONE"
    assert result["comparisons"]["cam2_raw_residual"]["relative_is_diagnostic_only"]
    assert result["comparisons"]["cam2_raw_residual"]["upstream_uv_pass"]
