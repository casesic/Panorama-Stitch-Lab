import csv,json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[3]
BASE=ROOT/'data_process/Vis/cpp_reference_dump/value_pose_offset_gemv_optimized_integration/frame_000513'
def report():return json.loads((BASE/'benchmark_summary.json').read_text())

def test_dispatch_build_and_scope():
 d=report();x=d['dispatch'];assert x['o2']['value_pose_offset_scalar_calls']=='32';assert x['o2']['value_pose_offset_gemv_calls']=='0';assert x['o3']['value_pose_offset_scalar_calls']=='0';assert x['o3']['value_pose_offset_gemv_calls']=='32';assert x['o3']['value_pose_offset_gemv_fallback_count']=='0';assert x['o3']['pose_offset_optimized_calls']=='16';assert x['o3']['lbs_selected_transform_optimized_calls']=='20';assert x['o3_caller_distribution']=={'pose_objective':12,'pose_jacobian':12,'pose_shape_objective':4,'pose_shape_jacobian':4};assert d['build_policy']['o2_o3_identical'];assert d['production_python_modifications']==0

def test_formal_parity_trajectory_and_handoffs():
 d=report();assert all(d['tensor_gates'].values());assert all(x['sequence_exact'] and x['counters_and_termination_exact'] for x in d['stage_parity'].values());assert all(d['handoff'].values());assert d['TRAJECTORY_EXACT']=='YES';assert d['VALUE_GEMV_INTEGRATION_FIRST_DIVERGENCE']=='NONE';assert d['formal_tolerance_changed'] is False

def test_measured_performance_attribution_and_next():
 d=report();assert d['O3_compute_median_ms']<d['O2_compute_median_ms'];assert d['O3_compute_fps']>3;assert d['O2_to_O3_speedup']>1.2;assert d['dispatch']['o3']['value_pose_offset_gemv_calls']=='32';assert d['unattributed_percent']<=2;assert d['integrated_reuse_upper_bound_ms']>0;assert d['REUSE_PRIORITY_DOWNGRADED']=='NO';assert d['INTEGRATION_CLASSIFICATION']=='APPROVED';assert d['NEXT_OPTIMIZATION_EXPERIMENT']=='POSE_OFFSET_DERIVATIVE_INTEGRATED_CONTEXT_ATTRIBUTION';assert d['VALUE_POSE_OFFSET_GEMV_OPTIMIZED_INTEGRATION_BAKEOFF_READY']=='PASS'

def test_output_tables_exist_and_nonempty():
 for name in ('trajectory_comparison.csv','benchmark_samples.csv','kernel_breakdown.csv','objective_breakdown.csv','linearization_breakdown.csv','speedup_reconciliation.csv','updated_reuse_audit.csv'):
  assert list(csv.reader((BASE/name).open()))
