import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
REPORT = ROOT / "data_process/Vis/cpp_reference_dump/pose_offset_vectorization/frame_000513/benchmark_summary.json"


def load():
    return json.loads(REPORT.read_text())


def test_variant_build_policy_isolated():
    report = load()
    assert "EIGEN_DONT_VECTORIZE" in report["variants"]["a"]["compile_flags"]
    assert "EIGEN_DONT_VECTORIZE" not in report["variants"]["b"]["compile_flags"]
    assert "march=native" not in report["variants"]["b"]["compile_flags"]
    assert "march=native" in report["variants"]["c"]["compile_flags"]
    assert all("ffp-contract=off" in report["variants"][key]["compile_flags"] for key in "abc")
    assert all(report["variants"][key]["environment"]["eigen_nb_threads"] == "1" for key in "abc")


def test_schema_parity_and_statistics():
    report = load()
    assert report["POSE_OFFSET_DERIVATIVE_CPU_VECTORISATION_BAKEOFF_READY"] == "PASS"
    for variant in report["variants"].values():
        assert len(variant["contexts"]) == 4
        for context in variant["contexts"].values():
            assert context["formal_feature_gate"]
            assert set(context["timing"]) == {
                "median_ms", "mean_ms", "min_ms", "max_ms", "p25_ms", "p75_ms"
            }


def test_speedup_and_fixed_trajectory_projection():
    report = load()
    assert report["speedup_a_to_b"] == (
        report["frame_pose_offset_ms_a"] / report["frame_pose_offset_ms_b"]
    )
    assert report["speedup_b_to_c"] == (
        report["frame_pose_offset_ms_b"] / report["frame_pose_offset_ms_c"]
    )
    for key in "abc":
        expected = 4617.56 - report["speedup_scalar_to_best"] * report[
            "frame_pose_offset_ms_c"] + report[f"frame_pose_offset_ms_{key}"]
        assert abs(report[f"projected_pipeline_{key}"] - expected) < 1e-9


def test_best_variant_and_next_experiment():
    report = load()
    assert report["best_variant"] == "GEMM_VECTORIZE_NATIVE"
    assert report["projected_top_hotspots"][0]["kernel"] == "jac.lbs_derivative"
    assert report["NEXT_OPTIMIZATION_EXPERIMENT"] == "LBS_DERIVATIVE_DIRECT_CONTRACTION_BAKEOFF"
    assert report["VECTORISATION_WORTH_INTEGRATING"] == "YES"
    assert report["NATIVE_BUILD_PERFORMANCE_VALUE"] == "MEANINGFUL"
