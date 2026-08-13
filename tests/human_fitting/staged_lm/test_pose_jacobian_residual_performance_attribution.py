import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/"data_process/Vis/cpp_reference_dump/pose_jacobian_residual_performance_attribution/frame_000513"

def test_pose_jacobian_residual_attribution_contract():
    s=json.loads((OUT/"benchmark_summary.json").read_text())
    assert s["preferred_policy"]=="optimized_native_stable_p"
    assert s["normal_equations_independent"]
    assert s["call_counts"]=={"pose_linearizations":6,"pose_camera_calls":12,"pose_shape_linearizations":2,"pose_shape_camera_calls":4}
    assert s["dimensions"]=={"Pose":"Q66","PoseShape":"base Q66 plus combined Q76 and beta Q10"}
    assert s["TRAJECTORY_EXACT"]=="YES" and s["first_divergence"]=="NONE"
    assert s["jacobian_reconciliation_error_percent"]<=5
    assert len(s["residual_top10"])==10 and len(s["full_pipeline_top20"])==20
    assert s["JACOBIAN_TOP1"]=="pose_offset_derivative_gemm"
    assert s["next_experiment"]=="POSE_OFFSET_DERIVATIVE_GEMM_BLOCKING_BAKEOFF"
    assert s["ready"]=="PASS"

def test_opportunity_and_amdahl_formulas():
    s=json.loads((OUT/"benchmark_summary.json").read_text())
    assert len(s["optimization_opportunities"])>=3
    for x in s["optimization_opportunities"]:
        assert x["saving_high_ms"]>=x["saving_low_ms"]>=0
    projections={str(x["speedup"]):x for x in s["amdahl"] if x["scope"]=="whole_jacobian"}
    assert projections["2"]["projected_ms"]>projections["3"]["projected_ms"]>projections["infinite"]["projected_ms"]
    assert s["timer_overhead"]["median_empty_pair_ns"]>0
