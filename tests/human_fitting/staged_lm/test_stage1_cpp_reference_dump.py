"""Tests for the Stage 1 C++ intermediate reference exporter."""

from __future__ import annotations

import json

import numpy as np
import torch

from lib_pano.human_fitting.staged_lm.analytic_stage1 import (
    Stage1AnalyticCache,
    Stage1AnalyticGroupCache,
)
from lib_pano.human_fitting.staged_lm.irls import (
    build_primary_aggregation_spec_from_observations,
)
from script.validation.staged_lm.dump_stage1_cpp_reference import (
    SCHEMA_VERSION,
    write_reference_dump,
)


def _fixture():
    torch.manual_seed(109)
    dtype = torch.float32
    groups = []
    ids_by_camera = []
    for camera, count in (("cam2", 4), ("cam4", 5)):
        local = torch.randn(count, 3, dtype=dtype) * 0.15
        center = torch.tensor([0.02, -0.1, 2.5], dtype=dtype)
        targets = torch.randn(count, 2, dtype=dtype) * 15 + torch.tensor([2880.0, 1440.0])
        groups.append(Stage1AnalyticGroupCache(0, 513, camera, local, center, targets))
        ids_by_camera.append(torch.arange(count))
    cache = Stage1AnalyticCache(groups, 1, 5760, 2880, 0.0)
    parameters = torch.tensor([
        0.03, -0.02, 0.01, 0.1, -0.05, 0.2,
        -0.01, 0.04, -0.03, -0.1, 0.02, 0.3,
    ], dtype=dtype)
    spec = build_primary_aggregation_spec_from_observations(
        [513], [ids_by_camera[0]], [ids_by_camera[1]],
    )
    return cache, parameters, spec


def test_stage1_cpp_reference_dump(tmp_path):
    cache, parameters, spec = _fixture()
    npz_path, manifest_path, report_path, manifest = write_reference_dump(
        cache, parameters, tmp_path, frame=513, is_irls=True,
        charbonnier_eps=1.0, aggregation_spec=spec, rep_scale=0.01,
        objective_name="charbonnier_irls",
    )
    assert npz_path.is_file() and manifest_path.is_file() and report_path.is_file()
    loaded_manifest = json.loads(manifest_path.read_text())
    assert loaded_manifest["schema_version"] == SCHEMA_VERSION
    assert loaded_manifest["frame"] == 513
    assert loaded_manifest["parameter_dim"] == 12
    assert loaded_manifest["parity"]["H_exact"]
    assert loaded_manifest["parity"]["g_exact"]
    assert loaded_manifest["parity"]["residual_exact"]

    with np.load(npz_path, allow_pickle=False) as arrays:
        assert len(arrays.files) == len(set(arrays.files)) == len(manifest["arrays"])
        required = {
            "parameters", "cam2_R", "cam2_Jr", "cam2_points", "cam2_uv",
            "cam2_J_proj", "cam2_dpoint_dorient", "cam2_J_orientation",
            "cam2_J_translation", "cam2_J_frame", "cam2_raw_residual_2d",
            "cam2_row_scales", "cam2_weighted_residual", "cam2_weighted_J",
            "cam2_H", "cam2_g", "cam4_R", "cam4_Jr", "cam4_points",
            "cam4_J_proj", "cam4_J_frame", "cam4_H", "cam4_g",
            "H_frame", "g_frame", "H_global", "g_global",
        }
        assert required <= set(arrays.files)
        for name in arrays.files:
            assert np.isfinite(arrays[name]).all(), name
            assert list(arrays[name].shape) == manifest["arrays"][name]["shape"]
            assert str(arrays[name].dtype) == manifest["arrays"][name]["dtype"]
        assert arrays["cam2_points"].shape == (4, 3)
        assert arrays["cam4_points"].shape == (5, 3)
        assert arrays["cam2_J_frame"].shape == (8, 12)
        assert arrays["cam4_J_frame"].shape == (10, 12)
        assert np.count_nonzero(arrays["cam2_J_frame"][:, 6:]) == 0
        assert np.count_nonzero(arrays["cam4_J_frame"][:, :6]) == 0
        assert arrays["H_frame"].dtype == np.float64
        assert arrays["g_frame"].dtype == np.float64
        np.testing.assert_array_equal(arrays["H_frame"], arrays["H_frame"].T)
        np.testing.assert_array_equal(arrays["H_frame"], arrays["H_global"])
        np.testing.assert_array_equal(arrays["g_frame"], arrays["g_global"])
