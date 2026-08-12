import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
REFERENCE = ROOT / "data_process/Vis/cpp_reference_dump/staged_lm_pipeline/frame_000513"


def test_pipeline_reference_and_stage_order():
    summary = json.loads((REFERENCE / "stage_summary.json").read_text())
    assert summary["actual_initial_lm_order"] == [
        "initial_linearization", "lambda_initialization", "initial_objective",
    ]
    assert summary["stage_order"] == ["stage1", "pose", "pose_shape"]


def test_pipeline_fixture_has_no_stage_output_leakage():
    fixture = REFERENCE / "cpp_fixture"
    forbidden = ("final", "trajectory", "objective", "rho", "candidate", "delta")
    assert not [path.name for path in fixture.iterdir()
                if any(token in path.name for token in forbidden)]
    assert not (fixture / "model_parameters.f32").exists()


def test_pipeline_handoff_internal_invariants():
    output = REFERENCE / "cpp_output"
    stage1 = np.fromfile(output / "stage1_final_parameters.f32", "<f4")
    pose_initial = np.fromfile(output / "pose_initial_parameters.f32", "<f4")
    pose_final = np.fromfile(output / "pose_final_parameters.f32", "<f4")
    shape_initial = np.fromfile(output / "pose_shape_initial_parameters.f32", "<f4")
    assert np.array_equal(stage1, pose_initial[:12])
    assert np.array_equal(pose_final, shape_initial[:75])


def test_pipeline_parity_ready():
    report = json.loads((
        REFERENCE / "cpp_output/staged_lm_pipeline_cpp_parity.json"
    ).read_text())
    assert report["PIPELINE_FIRST_DIVERGENCE"] == "NONE"
    assert report["STAGED_LM_PIPELINE_CPP_CPU_REFERENCE_READY"] == "PASS"
