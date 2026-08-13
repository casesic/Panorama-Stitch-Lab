import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[3]
BASE=ROOT/"data_process/Vis/cpp_reference_dump/lbs_direct_contraction/frame_000513"
def report():return json.loads((BASE/"output/benchmark_summary.json").read_text())

def test_fixture_is_input_only_and_has_six_contexts():
 m=json.loads((BASE/"fixture/manifest.json").read_text())
 assert m["REFERENCE_LEAKAGE"]=="NONE" and not m["contains_expected_outputs"]
 assert len(m["contexts"])==6 and {v["parameter_count"] for v in m["contexts"].values()}=={66,76}
 assert not list((BASE/"fixture").rglob("*expected*"))

def test_parity_and_output_contract():
 r=report();assert r["LBS_DERIVATIVE_DIRECT_CONTRACTION_BAKEOFF_READY"]=="PASS"
 for v in r["contexts"].values():
  assert v["formal_gate"] and v["current_vs_direct"]["rel_frob"]<=2e-6
  assert v["transform_term_parity"]["rel_frob"]<=2e-6
  assert v["vertex_term_parity"]["rel_frob"]<=2e-6

def test_memory_and_operation_accounting():
 for v in report()["contexts"].values():
  n,q=v["point_count"],v["parameter_count"]
  assert v["current_materialized_mib"]==(n*25*q*4)/2**20
  assert v["direct_materialized_mib"]==(n*3*q*4)/2**20
  assert v["current_flops_estimate"]==n*q*1811
  assert v["direct_flops_estimate"]==n*q*1671

def test_weighted_cost_projection_and_decision():
 r=report();c=sum(v["calls_per_frame"]*v["current_median_ms"] for v in r["contexts"].values());d=sum(v["calls_per_frame"]*v["direct_median_ms"] for v in r["contexts"].values())
 assert abs(r["current_lbs_frame_ms"]-c)<1e-9 and abs(r["direct_lbs_frame_ms"]-d)<1e-9
 assert abs(r["direct_only_projected_pipeline_ms"]-(4617.56-(c-d)))<1e-9
 assert abs(r["composed_projected_pipeline_ms"]-(r["direct_only_projected_pipeline_ms"]-(2074.991-30.17469)))<1e-9
 assert r["candidate_classification"]=="MARGINAL"
 assert r["next_experiment"]=="LBS_DERIVATIVE_SELECTED_TRANSFORM_GEMM_BAKEOFF"
