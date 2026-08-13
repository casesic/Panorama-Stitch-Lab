import csv,hashlib,json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[3]
BASE=ROOT/'data_process/Vis/cpp_reference_dump/pose_offset_derivative_integrated_context_attribution/frame_000513'
def report():return json.loads((BASE/'benchmark_summary.json').read_text())

def test_manifest_input_only_and_hash():
 d=report();m=list(csv.DictReader((BASE/'call_manifest.csv').open()));assert len(m)==16;assert {int(x['Q']) for x in m}=={66};assert sum(x['stage']=='pose_jacobian' for x in m)==12;assert sum(x['stage']=='pose_shape_jacobian' for x in m)==4
 corpus=BASE/'integrated/derivative_input_corpus';assert not [x for x in corpus.iterdir() if any(k in x.name for k in ('output','expected','jacobian'))];h=hashlib.sha256();
 for p in sorted(corpus.iterdir()):h.update(p.name.encode());h.update(p.read_bytes())
 assert h.hexdigest()==d['corpus_sha256']

def test_exclusive_accounting_and_same_policy():
 d=report();s=d['subcomponent_totals'];assert abs(d['integrated_family_ms']-sum(x['ms'] for x in s.values())-d['frame_reconciliation_error_ms'])<1e-9;assert d['frame_reconciliation_error_percent']<=5;assert d['same_source'] and d['same_gemm_implementation'] and d['same_build_policy'] and d['same_input_layout'] and d['same_output_layout'] and d['same_input_corpus'];assert len(list(csv.DictReader((BASE/'integrated_breakdown.csv').open())))==16*9

def test_cache_duplicate_allocation_amdahl_and_next():
 d=report();assert set(d['cache_sensitivity'])=={'warm_core_ms','trajectory_order_core_ms','cache_disturbed_core_ms','cache_sensitivity_ratio'};assert d['allocation_counts']['resize_count']==d['allocation_counts']['actual_reallocation_count']==16;assert d['d_pose_feature_camera_independent'];assert d['duplicated_pose_feature_calls']==8;assert len(d['amdahl_projection'])==4
 for x in d['amdahl_projection']:assert abs(x['fps']-1000/x['pipeline_ms'])<1e-12
 assert d['DERIVATIVE_SUBCOMPONENT_TOP1']=='GEMM_CORE_MS';assert d['TRAJECTORY_EXACT']=='YES';assert d['FIRST_DIVERGENCE']=='NONE';assert d['NEXT_OPTIMIZATION_EXPERIMENT']=='POSE_OFFSET_DERIVATIVE_GEMM_INTEGRATED_CONTEXT_BAKEOFF';assert d['POSE_OFFSET_DERIVATIVE_INTEGRATED_CONTEXT_ATTRIBUTION_READY']=='PASS'
