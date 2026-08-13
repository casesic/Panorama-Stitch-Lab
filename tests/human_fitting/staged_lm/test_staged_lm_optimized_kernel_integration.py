import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / "data_process/Vis/cpp_reference_dump/staged_lm_optimized_integration/frame_000513"


def report():
    return json.loads((BASE / "benchmark_summary.json").read_text())


def test_build_policy_and_dispatch_are_isolated():
    data = report(); reference = data["reference_build"]; optimized = data["optimized_build"]
    assert reference["eigen_vectorize"] == reference["march_native"] == "OFF"
    assert optimized["eigen_vectorize"] == optimized["march_native"] == "ON"
    assert optimized["pose_offset_kernel"] == "EIGEN_GEMM_NATIVE"
    assert optimized["lbs_selected_transform_kernel"] == "EIGEN_GEMM_NATIVE"
    assert data["pose_offset_optimized_calls"] == 16
    assert data["lbs_selected_transform_optimized_calls"] == 20
    assert data["optimized_fallback_count"] == 0


def test_trajectory_and_existing_parity_policy():
    data = report()
    assert data["TRAJECTORY_EXACT"] == "YES"
    assert data["OPTIMIZED_PIPELINE_FIRST_DIVERGENCE"] == "NONE"
    assert all(data["tensor_gates"].values())
    assert all(value["sequence_exact"] and value["counters_exact"]
               for value in data["stage_parity"].values())
    assert data["formal_tolerance_changed"] is False
    rows = list(csv.DictReader((BASE / "trajectory_comparison.csv").open()))
    assert len(rows) == 32 and all(row["accepted_exact"] == "True" for row in rows)


def test_measured_formulas_reconciliation_and_next_experiment():
    data = report(); median = data["optimized_compute_median_ms"]
    assert abs(data["measured_compute_fps"] - 1000 / median) < 1e-12
    assert abs(data["projection_error_percent"] -
               100 * (median-data["projected_ms"])/data["projected_ms"]) < 1e-12
    assert abs(data["speedup_reconciliation_error_percent"]) <= 5
    assert data["optimized_top_hotspots"][0]["kernel"] == "value.pose_offsets"
    assert data["category_breakdown"]["objective"]["percent"] >= 30
    assert data["NEXT_OPTIMIZATION_EXPERIMENT"] == \
        "POSE_OBJECTIVE_POSE_OFFSET_GEMV_BAKEOFF"
    assert data["INTEGRATION_CLASSIFICATION"] == "APPROVED"
    assert data["STAGED_LM_CPP_OPTIMIZED_KERNEL_INTEGRATION_BAKEOFF_READY"] == "PASS"
