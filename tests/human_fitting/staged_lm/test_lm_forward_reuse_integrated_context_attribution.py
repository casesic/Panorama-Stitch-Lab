import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / "data_process/Vis/cpp_reference_dump/lm_forward_reuse_integrated_context_attribution/frame_000513"


def summary():
    return json.loads((BASE / "benchmark_summary.json").read_text())


def test_modes_directions_and_exact_trajectory():
    value = summary()
    assert value["repeat_counts"] == {"D0": 30, "D1": 30, "D2": 30, "D3": 30, "D4": 20, "D5": 20}
    assert value["direction_counts"]["D2"] == {"LINEARIZATION_TO_OBJECTIVE": 2}
    assert value["direction_counts"]["D3"] == {"OBJECTIVE_TO_LINEARIZATION": 6}
    assert value["direction_counts"]["D4"] == {"OBJECTIVE_TO_LINEARIZATION": 6}
    assert value["direction_counts"]["D5"] == {"DROP": 6}
    assert value["trajectory_exact"] and value["first_divergence"] == "NONE"
    assert value["tolerance_changes"] == 0


def test_derivative_formula_hypotheses_and_reconciliation():
    value = summary()
    assert value["metrics"]["D0"]["derivative_only_ms"] > 0
    assert value["metrics"]["D4"]["shadow_forward_ms"] > 0
    assert value["restoration_ratio"] < 0.4
    assert value["retain_only_penalty_ms"] <= 0
    assert value["hypotheses"]["lost_forward_warmup"] == "WEAK_OR_NOT_SUPPORTED"
    assert value["hypotheses"]["retained_payload"] == "WEAK_OR_NOT_SUPPORTED"
    assert value["directional_explained_fraction"] >= 0.75
    assert value["reconciliation"]["error_percent"] <= 5
    assert value["next_optimization_experiment"] == "LM_FORWARD_REUSE_CACHED_VALUE_ACCESS_PATH_BAKEOFF"
    assert value["ready"] is True


def test_required_artifact_schemas():
    required = {
        "linearization_context_breakdown.csv", "reuse_direction_benchmark.csv",
        "derivative_penalty_top10.csv", "integrated_context_hypothesis_matrix.csv",
        "forward_derivative_buffer_overlap.csv", "cached_payload_field_audit.csv",
        "allocation_lifetime_audit.csv", "benchmark_samples.csv",
        "speedup_reconciliation.csv", "benchmark_summary.json", "benchmark_summary.md",
    }
    assert not [name for name in required if not (BASE / name).is_file()]
    calls = list(csv.DictReader((BASE / "linearization_context_breakdown.csv").open()))
    assert {row["context"] for row in calls} == {"initial", "accepted_state"}
    assert {row["diagnostic"] for row in calls} == {f"D{index}" for index in range(6)}
    payload = list(csv.DictReader((BASE / "cached_payload_field_audit.csv").open()))
    assert {row["stage"] for row in payload} == {"pose", "pose_shape"}
    assert all(float(row["derivative_required_bytes"]) < float(row["full_payload_bytes"]) for row in payload)
