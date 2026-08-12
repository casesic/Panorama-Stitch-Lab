"""Fixture serialization and C++ parity-checker contract tests."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from script.validation.staged_lm.check_stage1_cpp_reference import check_reference
from script.validation.staged_lm.check_stage1_cpp_reference import (
    NORMAL_EQUATION_PASS_RELATIVE,
    NORMAL_EQUATION_WARNING_RELATIVE,
)
from script.validation.staged_lm.export_stage1_cpp_fixture import INPUT_KEYS, export_fixture


def _reference(directory: Path) -> tuple[Path, dict[str, np.ndarray]]:
    rng = np.random.default_rng(109)
    counts = {"cam2": 2, "cam4": 3}
    arrays: dict[str, np.ndarray] = {
        "parameters": rng.normal(size=12).astype(np.float32),
        "H_frame": rng.normal(size=(12, 12)).astype(np.float64),
        "g_frame": rng.normal(size=12).astype(np.float64),
    }
    for camera, count in counts.items():
        arrays[f"{camera}_local_vertices"] = rng.normal(size=(count, 3)).astype(np.float32)
        arrays[f"{camera}_rotation_center"] = rng.normal(size=3).astype(np.float32)
        arrays[f"{camera}_targets"] = rng.normal(size=(count, 2)).astype(np.float32)
        shapes = {
            "R": (3, 3), "Jr": (3, 3), "points": (count, 3),
            "uv": (count, 2), "J_proj": (count, 2, 3),
            "dpoint_dorient": (count, 3, 3),
            "J_orientation": (count, 2, 3), "J_translation": (count, 2, 3),
            "J_frame": (2 * count, 12), "raw_residual_flat": (2 * count,),
            "point_weights": (count,), "row_scales": (2 * count,),
            "weighted_residual": (2 * count,), "weighted_J": (2 * count, 12),
        }
        for field, shape in shapes.items():
            arrays[f"{camera}_{field}"] = rng.normal(size=shape).astype(np.float32)
    npz = directory / "stage1_reference.npz"
    np.savez(npz, **arrays)
    manifest = {
        "stage": "STAGE1_RIGID", "frame": 513, "parameter_dim": 12,
        "image_width": 5760, "image_height": 2880,
        "cameras": {camera: {"point_count": count} for camera, count in counts.items()},
        "config": {
            "objective": "charbonnier_irls", "charbonnier_eps": 1.0,
            "aggregation_spec": {"terms": {
                "513|cam2": {"camera_weight": 1.0, "alpha_per_point": 0.5},
                "513|cam4": {"camera_weight": 1.0, "alpha_per_point": 1.0 / 3.0},
            }},
        },
    }
    (directory / "stage1_reference_manifest.json").write_text(json.dumps(manifest))
    return npz, arrays


def test_fixture_is_input_only_sized_and_deterministic(tmp_path):
    reference = tmp_path / "reference"
    reference.mkdir()
    _, arrays = _reference(reference)
    first = export_fixture(reference, tmp_path / "fixture_a")
    second = export_fixture(reference, tmp_path / "fixture_b")
    assert {path.stem for path in first.glob("*.f32")} == set(INPUT_KEYS)
    assert not any(token in path.name for path in first.iterdir()
                   for token in ("_R.", "_uv.", "Jacobian", "H_frame", "g_frame"))
    for name in INPUT_KEYS:
        first_path = first / f"{name}.f32"
        second_path = second / f"{name}.f32"
        assert first_path.read_bytes() == second_path.read_bytes()
        assert first_path.stat().st_size == arrays[name].size * 4
    metadata = dict(line.split("=", 1) for line in
                    (first / "fixture_meta.txt").read_text().splitlines())
    assert metadata["schema_version"] == "1"
    assert metadata["binary_endianness"] == "little"
    assert metadata["cam2_count"] == "2" and metadata["cam4_count"] == "3"


def test_checker_reads_binary_outputs_and_writes_reports(tmp_path):
    reference = tmp_path / "reference"
    reference.mkdir()
    npz, arrays = _reference(reference)
    output = tmp_path / "cpp_output"
    output.mkdir()
    for camera in ("cam2", "cam4"):
        for output_name, reference_name in (
            ("R", "R"), ("Jr", "Jr"), ("points", "points"), ("uv", "uv"),
            ("J_proj", "J_proj"), ("dpoint_dorient", "dpoint_dorient"),
            ("J_orientation", "J_orientation"),
            ("J_translation", "J_translation"), ("J_frame", "J_frame"),
            ("raw_residual", "raw_residual_flat"),
            ("point_weights", "point_weights"), ("row_scales", "row_scales"),
            ("weighted_residual", "weighted_residual"), ("weighted_J", "weighted_J"),
        ):
            arrays[f"{camera}_{reference_name}"].astype("<f4").tofile(
                output / f"{camera}_{output_name}.f32"
            )
    arrays["H_frame"].astype("<f8").tofile(output / "H_frame.f64")
    arrays["g_frame"].astype("<f8").tofile(output / "g_frame.f64")
    (output / "cpp_output_meta.txt").write_text(
        "schema_version=1\ncam2_count=2\ncam4_count=3\ntotal_ms=1.0\n"
    )
    result = check_reference(npz, output)
    assert result["pass"] and result["first_divergence"] == "NONE"
    assert result["policy"]["normal_equation_pass_relative"] == 1e-6
    assert result["policy"]["normal_equation_warning_relative"] == 5e-7
    assert (output / "stage1_cpp_parity.json").is_file()
    assert (output / "stage1_cpp_parity.md").is_file()


def test_cross_backend_normal_equation_policy_has_warning_headroom():
    assert NORMAL_EQUATION_WARNING_RELATIVE < NORMAL_EQUATION_PASS_RELATIVE
