import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
OUTPUT = ROOT / "data_process/Vis/cpp_reference_dump/staged_lm_performance/frame_000513"


def test_performance_summary_is_reconciled_and_correctness_gated():
    summary = json.loads((OUTPUT / "benchmark_summary.json").read_text())
    assert summary["correctness_baseline_pass"]
    assert summary["unattributed_percent"] <= 2.0
    assert summary["model_error_percent"] <= 5.0
    assert summary["STAGED_LM_CPP_PERFORMANCE_ATTRIBUTION_READY"] == "PASS"


def test_performance_benchmark_reset_is_deterministic():
    lines = (OUTPUT / "pipeline_runs.csv").read_text().splitlines()
    checksums = [line.rsplit(",", 1)[-1] for line in lines[1:]]
    assert len(checksums) >= 3
    assert len(set(checksums)) == 1


def test_performance_artifacts_exist():
    for name in (
        "kernel_breakdown.csv", "call_multiplicity.csv",
        "hotspot_microbench.csv", "materialization_memory.csv",
        "amdahl_analysis.csv", "benchmark_summary.md",
        "vectorization_report.txt", "perf_report.txt",
    ):
        assert (OUTPUT / name).is_file()
