import csv,json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[3]
BASE=ROOT/'data_process/Vis/cpp_reference_dump/pose_objective_pose_offset_vectorization/frame_000513'

def report():return json.loads((BASE/'benchmark_summary.json').read_text())

def test_build_isolation_packet_and_timer_policy():
 d=report();v=d['variants'];assert 'EIGEN_DONT_VECTORIZE' in v['a']['compile_flags'];assert 'EIGEN_DONT_VECTORIZE' not in v['b']['compile_flags'];assert 'march=native' not in v['b']['compile_flags'];assert 'march=native' in v['c']['compile_flags']
 assert v['a']['environment']['eigen_vectorize']=='OFF';assert v['b']['environment']['eigen_vectorize_sse']=='ON';assert v['c']['environment']['eigen_vectorize_avx512']=='ON';assert all(x['timer_overhead_percent']<2 for x in v.values());assert d['eigen_threads']==1

def test_32_state_parity_shadow_and_frame_formula():
 d=report();assert d['state_count']==32;assert d['corpus_identical_across_variants'];assert len(list(csv.DictReader((BASE/'benchmark_samples.csv').open())))==32*2*100*3
 for v in d['variants'].values():
  assert len(v['contexts'])==32;assert all(x['formal_gate'] for x in v['contexts'].values());assert abs(v['shared_frame_ms']-v['objective_frame_ms']-v['linearization_frame_ms'])<1e-9
 assert all(x['PIPELINE_FIRST_DIVERGENCE']=='NONE' for x in d['shadow_parity'].values());assert d['formal_tolerance_changed'] is False

def test_selection_projection_reuse_and_next_rule():
 d=report();assert d['BEST_APPROVED_VARIANT']=='C';assert d['a_to_b_shared_speedup']>=1.5;assert d['b_to_c_shared_speedup']>=1.05;assert d['current_measured_pipeline_ms']==420.919;assert d['duplicate_pair_count']==20;assert d['updated_reuse_upper_bound_ms']<60.032
 assert d['WINNING_BUILD_POLICY_COMPATIBLE_WITH_CURRENT_OPTIMIZED_NATIVE']=='YES';assert d['NEXT_OPTIMIZATION_EXPERIMENT']=='VALUE_POSE_OFFSET_GEMV_OPTIMIZED_INTEGRATION_BAKEOFF';assert d['POSE_OBJECTIVE_POSE_OFFSET_VECTORISATION_BAKEOFF_READY']=='PASS'
