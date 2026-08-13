import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/"data_process/Vis/cpp_reference_dump/pose_offset_derivative_transposed_gemm_integration/frame_000513"

def test_transposed_integration_contract():
    s=json.loads((OUT/"benchmark_summary.json").read_text())
    assert s["build_policy"]["same_binary"]
    assert s["dispatch"]["O4"]=={"original":16,"transposed":0}
    assert s["dispatch"]["O5"]=={"original":0,"transposed":16}
    assert s["dispatch"]["deep_copy"]==0
    assert not any((s["zero_copy"][k] for k in ("P_transpose_materialized","D_transpose_materialized","output_transpose_materialized")))
    assert s["zero_copy"]["user_pack_bytes"]==s["zero_copy"]["output_conversion_bytes"]==0
    assert s["TRAJECTORY_EXACT"]=="YES" and s["first_divergence"]=="NONE"
    assert s["milestones"]["MEASURED_GT_5FPS"]
    assert s["integration_classification"]=="CORRECT_BUT_NO_SYSTEM_WIN"
    assert s["retention_recommendation"]=="NO"
    assert len(s["top_hotspots"])==20 and s["reconciliation_error_percent"]<=5
    assert s["next_experiment"]=="POSE_OFFSET_DERIVATIVE_GEMM_BLOCKING_BAKEOFF"

def test_transposed_numeric_policy_unchanged():
    s=json.loads((OUT/"benchmark_summary.json").read_text())
    assert s["local_parity"]["max_abs"]==s["local_parity"]["rel_frob"]==0
    assert all(x["max_abs"]==0 for x in s["final_parity"].values())
    assert max(x["max_abs"] for x in s["canonical_parity"].values())<=1e-5
    assert s["tolerance_change_count"]==0
