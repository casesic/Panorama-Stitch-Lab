import json
from pathlib import Path

ROOT=Path("data_process/Vis/cpp_reference_dump/pose_offset_derivative_gemm_blocking_bakeoff/frame_000513")
def test_blocking_bakeoff_contract():
    d=json.loads((ROOT/"benchmark_summary.json").read_text())
    assert d["ready"]=="PASS" and d["preferred_baseline_policy"]=="optimized_native_stable_p"
    assert d["corpus_sha256"]=="5f4e2a9fcafd52fbd83ee19c983ec24030977ece0474e0ca60077f50124f037e"
    assert d["dimensions"]=={"M_cam2":3069,"M_cam4":2781,"K":486,"N":66,"Q":66}
    assert len([x for x in d["screening_candidates"] if x.startswith("B_")])<=4
    assert len([x for x in d["screening_candidates"] if x.startswith("C_")])<=4
    assert d["d_not_run"] and d["BEST_BLOCKING_TRAJECTORY_EXACT"]
    assert d["first_divergence"]=="NONE" and d["tolerance_changes"]==0
    assert d["allocations"]["extra_full_matrix_copy"] is False
    assert d["stable_p"]["deep_copies"]==0
    assert d["integration_worthiness"]=="REJECTED_PERFORMANCE"
    assert d["next_experiment"]=="POSE_LBS_FUSED_DOWNSTREAM_BAKEOFF"
    assert len(list(__import__('csv').DictReader((ROOT/"finalist_samples.csv").open())))>=90
