import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
OUTPUT = ROOT / "data_process/Vis/cpp_reference_dump/pose_offset_gemm/frame_000513/output"


def test_gemm_candidate_passes_existing_feature_policy():
    report = json.loads((OUTPUT / "pose_offset_gemm_parity.json").read_text())
    assert report["default_scalar_path"] == "UNCHANGED"
    assert not report["formal_tolerance_changed"]
    assert all(value["formal_feature_gate"] for value in report["contexts"].values())
    assert report["POSE_OFFSET_DERIVATIVE_CPU_GEMM_BAKEOFF_READY"] == "PASS"


def test_gemm_benchmark_is_interleaved_and_repeated():
    lines = (OUTPUT / "benchmark_samples.csv").read_text().splitlines()[1:]
    assert len(lines) == 4 * 20 * 2
    for context in ("pose_cam2", "pose_cam4", "pose_shape_cam2", "pose_shape_cam4"):
        selected = [line for line in lines if line.startswith(context + ",")]
        assert len(selected) == 40
        assert {line.split(",")[2] for line in selected} == {"scalar", "eigen_gemm"}


def test_gemm_has_no_explicit_pack_or_unpack():
    report = json.loads((OUTPUT / "pose_offset_gemm_parity.json").read_text())
    assert not report["input_pack_required"]
    assert not report["output_unpack_required"]
    assert report["explicit_pack_ms"] == report["explicit_unpack_ms"] == 0.0
