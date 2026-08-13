import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / "data_process/Vis/cpp_reference_dump/lbs_selected_transform_vectorization/frame_000513"


def report():
    return json.loads((BASE / "benchmark_summary.json").read_text())


def test_variant_flag_isolation_and_metadata():
    data = report(); variants = data["compile_variants"]
    assert "EIGEN_DONT_VECTORIZE" in variants["a"]["compile_flags"]
    assert "EIGEN_DONT_VECTORIZE" not in variants["b"]["compile_flags"]
    assert "march=native" not in variants["b"]["compile_flags"]
    assert "march=native" in variants["c"]["compile_flags"]
    assert variants["a"]["environment"]["eigen_vectorize"] == "OFF"
    assert variants["b"]["environment"]["eigen_vectorize_sse"] == "ON"
    assert variants["c"]["environment"]["eigen_vectorize_avx512"] == "ON"
    assert data["eigen_threads"] == 1


def test_six_context_parity_timing_and_formula():
    data = report()
    for variant in data["compile_variants"].values():
        assert len(variant["contexts"]) == 6
        for context in variant["contexts"].values():
            assert context["formal_gate"]
            assert context["dt_parity_vs_scalar"]["rel_frob"] <= 2e-6
            assert set(context["dt_timing"]) == {
                "median_ms", "mean_ms", "min_ms", "max_ms", "p25_ms", "p75_ms"}
            assert context["input_pack_required"] == "NO"
    for key in "ABC":
        expected = sum(x["calls_per_frame"] * x["dt_timing"]["median_ms"]
                       for x in data["compile_variants"][key.lower()]["contexts"].values())
        assert abs(expected - data[f"frame_selected_transform_{key}_ms"]) < 1e-9


def test_projection_compatibility_and_next_rule():
    data = report()
    assert data["best_variant"] == "GEMM_VECTORIZE_NATIVE"
    assert data["WINNING_KERNEL_BUILD_POLICIES_COMPATIBLE"] == "YES"
    assert data["best_projected_pipeline_ms"] < 1000
    assert data["one_fps_headroom_percent"] >= 10
    assert data["NEXT_OPTIMIZATION_EXPERIMENT"] == \
        "STAGED_LM_CPP_OPTIMIZED_KERNEL_INTEGRATION_BAKEOFF"
    assert data["formal_tolerance_changed"] is False
    assert data["gemm_source_changed"] is False
    assert data["LBS_DERIVATIVE_SELECTED_TRANSFORM_VECTORISATION_BAKEOFF_READY"] == "PASS"
