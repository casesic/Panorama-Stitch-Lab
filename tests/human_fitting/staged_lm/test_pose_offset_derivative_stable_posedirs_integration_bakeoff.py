import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/"data_process/Vis/cpp_reference_dump/pose_offset_derivative_stable_posedirs_integration/frame_000513"

def test_stable_posedirs_integration_contract():
    s=json.loads((OUT/"benchmark_summary.json").read_text())
    assert s["o3_kernel_set"]=="optimized_native"
    assert s["o4_kernel_set"]=="optimized_native + stable_reference"
    assert s["stable_owner"]["type"]=="shared_ptr<const vector<float>>"
    assert s["copy_counts"]=={"O3":16,"O4":0}
    assert s["copy_bytes"]["O3"]==90_979_200 and s["copy_bytes"]["O4"]==0
    assert s["pointer_identity"]["cam2"] and s["pointer_identity"]["cam4"]
    assert s["TRAJECTORY_EXACT"]=="YES" and s["first_divergence"]=="NONE"
    assert s["milestones"]["MEASURED_GT_4FPS"]
    assert s["reconciliation_error_percent"]<=5
    assert len(s["top_hotspots"])==20
    assert s["integration_classification"]=="APPROVED_GT_4FPS"
    assert s["next_experiment"]=="POSE_OFFSET_DERIVATIVE_TRANSPOSED_GEMM_INTEGRATION_BAKEOFF"

def test_stable_posedirs_numeric_policy_unchanged():
    s=json.loads((OUT/"benchmark_summary.json").read_text())
    assert s["tolerance_change_count"]==0
    assert all(v["max_abs"]==0 and v["rel_frob"]==0 for v in s["final_parity"].values())
    assert max(v["max_abs"] for v in s["canonical_parity"].values())<=1e-5
