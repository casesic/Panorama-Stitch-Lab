"""Pose C++ normal-equation reference contract tests."""

from __future__ import annotations

import json

import numpy as np

from script.validation.staged_lm.check_pose_cpp_normal_equations import (
    CAMERA_FIELDS,
    FINAL_H_REL_TOL,
    FRAME_FIELDS,
    INTERMEDIATE_H_ACTIVE_REL_TOL,
    _metrics,
)
from script.validation.staged_lm.export_pose_cpp_fixture import FLOAT_INPUTS


def test_normal_equation_tolerance_classification():
    reference = np.ones(8, dtype=np.float64)
    strong = reference * (1.0 + 4e-7)
    passing = reference * (1.0 + 8e-7)
    failing = reference * (1.0 + 1.2e-6)
    assert _metrics(reference, strong, "hg")["classification"] == "STRONG_PASS"
    assert _metrics(reference, passing, "hg")["classification"] == "PASS"
    assert _metrics(reference, failing, "hg")["classification"] == "FAIL"
    assert FINAL_H_REL_TOL == 1e-6
    assert INTERMEDIATE_H_ACTIVE_REL_TOL == 1.1e-6
    intermediate = reference * (1.0 + 1.05e-6)
    assert _metrics(reference, intermediate, "intermediate_h_active")["pass"]
    assert not _metrics(reference, intermediate, "hg")["pass"]


def test_mixed_cross_backend_policy_and_upstream_gates():
    tiny = np.zeros(8, dtype=np.float32)
    weights = np.full(8, 0.9e-7, dtype=np.float32)
    assert _metrics(tiny, weights, "irls_weight", upstream_pass=True)["pass"]
    assert not _metrics(tiny, weights, "irls_weight", upstream_pass=False)["pass"]
    jacobian = np.ones(8, dtype=np.float32)
    tiny = jacobian * np.float32(1.0 - 9e-6)
    assert _metrics(tiny, jacobian, "weighted_jacobian")["pass"]
    gradient_reference = np.full(8, 10.0, dtype=np.float64)
    gradient = gradient_reference + 4e-5
    assert _metrics(gradient_reference, gradient, "gradient")["pass"]
    assert not _metrics(
        gradient_reference, gradient, "gradient", upstream_pass=False,
    )["pass"]


def test_fixture_contract_has_inputs_but_no_reference_leakage():
    assert "body_pose_anchor" in FLOAT_INPUTS
    forbidden = {
        "point_weights", "row_scales", "weighted_residual", "weighted_J_active",
        "H_active", "g_active", "H_frame", "g_frame",
    }
    assert forbidden.isdisjoint(FLOAT_INPUTS)


def test_checkpoint_order_and_layout_contract():
    assert [field for field, _suffix, _category in CAMERA_FIELDS] == [
        "point_weights", "row_scales", "weighted_residual", "weighted_J_active",
        "H_active", "g_active",
    ]
    assert [field for field, _suffix, _category in FRAME_FIELDS] == [
        "H_data_frame", "g_data_frame", "pose_anchor_delta",
        "pose_anchor_residual", "H_pose_anchor", "g_pose_anchor",
        "H_frame", "g_frame",
    ]
    cam2 = [*range(0, 6), *range(12, 75)]
    cam4 = [*range(6, 12), *range(12, 75)]
    assert len(cam2) == len(cam4) == 69
    assert cam2[6:] == cam4[6:] == list(range(12, 75))


def test_real_reference_manifest_records_production_contract():
    path = "data_process/Vis/cpp_reference_dump/pose/frame_000513/pose_reference_manifest.json"
    try:
        manifest = json.loads(open(path).read())
    except FileNotFoundError:
        return
    normal = manifest["normal_equations"]
    assert normal["group_order"] == ["cam2", "cam4"]
    assert normal["dtype_before_hg"] == "float32"
    assert normal["dtype_after_cast"] == "float64"
    assert "point0_u" in normal["residual_flatten_order"]
    assert manifest["parity"]["normal_equations_hessian_exact"]
    assert manifest["parity"]["normal_equations_gradient_exact"]
