"""Contracts for the Pose analytic-Jacobian reference and checker."""

from __future__ import annotations

import numpy as np

from script.validation.staged_lm.check_pose_cpp_jacobian import FIELDS, check_pose_jacobian
from script.validation.staged_lm.dump_pose_cpp_reference import write_pose_reference
from script.validation.staged_lm.export_pose_cpp_fixture import export_pose_fixture
from tests.human_fitting.staged_lm.test_pose_cpp_fixture import _cache


VALUE_FIELDS = (
    "axis_angles", "rotations", "pose_feature", "pose_offsets", "v_posed",
    "relative_joints", "local_transforms", "global_transforms",
    "relative_transforms", "selected_transforms", "skinned_vertices",
    "vertices", "uv", "raw_residual",
)


def test_pose_jacobian_reference_layout_and_no_fixture_leakage(tmp_path):
    cache, parameters = _cache()
    reference = tmp_path / "reference"
    write_pose_reference(cache, parameters, reference, frame=513, image_width=5760, image_height=2880)
    with np.load(reference / "pose_reference.npz", allow_pickle=False) as arrays:
        assert arrays["cam2_variable_right_jacobians"].shape == (22, 3, 3)
        assert arrays["cam2_d_rotation"].shape == (55, 3, 3, 66)
        assert arrays["cam2_d_pose_feature"].shape == (486, 66)
        assert arrays["cam2_d_pose_offsets"].shape == (2, 3, 66)
        assert arrays["cam2_J_active"].shape == (2, 2, 69)
        assert arrays["cam2_J_frame"].shape == (4, 75)
        assert np.count_nonzero(arrays["cam2_d_pose_feature"][:, :3]) == 0
        assert np.count_nonzero(arrays["cam2_J_frame"][:, 6:12]) == 0
        assert np.count_nonzero(arrays["cam4_J_frame"][:, :6]) == 0
        np.testing.assert_array_equal(
            arrays["cam2_J_frame"][:, 12:75],
            arrays["cam2_J_active_flat"][:, 6:69],
        )
        np.testing.assert_array_equal(
            arrays["cam4_J_frame"][:, 12:75],
            arrays["cam4_J_active_flat"][:, 6:69],
        )
    fixture = export_pose_fixture(reference, tmp_path / "fixture")
    forbidden = {
        "d_rotation.f32", "d_pose_feature.f32", "d_pose_offsets.f32",
        "d_global_transforms.f32", "J_proj.f32", "J_active.f32", "J_frame.f32",
    }
    assert not any(
        path.name.endswith(name) for path in fixture.iterdir() for name in forbidden
    )


def test_pose_jacobian_checker_accepts_exact_serialization(tmp_path):
    cache, parameters = _cache()
    reference = tmp_path / "reference"
    write_pose_reference(cache, parameters, reference, frame=513, image_width=5760, image_height=2880)
    output = tmp_path / "output"
    output.mkdir()
    with np.load(reference / "pose_reference.npz", allow_pickle=False) as arrays:
        for camera in ("cam2", "cam4"):
            for field in VALUE_FIELDS:
                arrays[f"{camera}_{field}"].astype("<f4").tofile(
                    output / f"{camera}_{field}.f32"
                )
            for field, _ in FIELDS:
                arrays[f"{camera}_{field}"].astype("<f4").tofile(
                    output / f"{camera}_{field}.f32"
                )
    (output / "cpp_output_meta.txt").write_text(
        "schema_version=1\ncam2_count=2\ncam4_count=3\n"
        "shape.d_rotation=55,3,3,66\naxes.d_rotation=joint,row,col,param\n"
    )
    result = check_pose_jacobian(reference / "pose_reference.npz", output)
    assert result["pass"] and result["first_divergence"] == "NONE"
    assert result["comparisons"]["cam2_root_feature_zero"]["pass"]
    assert (output / "pose_jacobian_cpp_parity.json").is_file()
    assert (output / "pose_jacobian_cpp_parity.md").is_file()
