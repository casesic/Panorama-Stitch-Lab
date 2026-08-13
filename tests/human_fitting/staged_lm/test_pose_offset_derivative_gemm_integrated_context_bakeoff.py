import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/"data_process/Vis/cpp_reference_dump/pose_offset_derivative_gemm_integrated_context_bakeoff/frame_000513"

def test_integrated_context_bakeoff_contract():
    summary=json.loads((OUT/"benchmark_summary.json").read_text())
    assert summary["corpus_sha256"]=="5f4e2a9fcafd52fbd83ee19c983ec24030977ece0474e0ca60077f50124f037e"
    assert summary["ready"]=="PASS"
    assert summary["trajectory_shadow"]=={"scope":"all 16 derivative outputs plus deterministic downstream implication","TRAJECTORY_EXACT":True,"FIRST_DIVERGENCE":"NONE","pose_downstream":"PASS","pose_shape_downstream":"PASS"}
    assert summary["candidates"]["B_STABLE_P"]["parity"]["pass"]
    assert summary["candidates"]["C_STABLE_P_TRANSPOSED"]["parity"]["pass"]
    assert summary["selected_posedirs_copy_audit"]["count_per_frame"]==16
    assert summary["best_approved_variant"]=="STABLE_P"
    assert summary["next_experiment"]=="POSE_OFFSET_DERIVATIVE_STABLE_POSEDIRS_INTEGRATION_BAKEOFF"
