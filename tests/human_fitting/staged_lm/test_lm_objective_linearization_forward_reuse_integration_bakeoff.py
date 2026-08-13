import csv,json
from pathlib import Path
R=Path('data_process/Vis/cpp_reference_dump/lm_objective_linearization_forward_reuse_integration/frame_000513')
def test_forward_reuse_integration_contract():
 d=json.loads((R/'benchmark_summary.json').read_text())
 assert d['ready']=='PASS' and d['baseline_policy']=='optimized_native_stable_p'
 assert d['reuse_policy']=='optimized_native_stable_p_forward_reuse'
 assert d['generic_run_lm_modified'] is False
 assert d['hit_counts']=={'pose_lin_to_obj':1,'pose_obj_to_lin':5,'pose_shape_lin_to_obj':1,'pose_shape_obj_to_lin':1,'total':8}
 assert d['camera_forward_execution_counts']=={'O4':32,'O6':16,'reduction_percent':50}
 assert d['false_hit_counts']=={'exact_key':0,'wrong_role':0,'wrong_context':0,'stale_rejected':0}
 assert d['payload_copy_stats']['large_payload_deep_copy_count']==0
 assert d['trajectory_exact'] and d['first_divergence']=='NONE' and d['tolerance_changes']==0
 assert d['baseline_compute_stats']['median_ms']>0 and d['reuse_compute_stats']['median_ms']>0
 assert abs(d['full_speedup']-d['baseline_compute_stats']['median_ms']/d['reuse_compute_stats']['median_ms'])<1e-12
 assert d['reconciliation_error_percent']<=5
 assert d['preferred_recommendation']=='NO'
 assert d['next_experiment']=='LM_FORWARD_REUSE_INTEGRATED_CONTEXT_ATTRIBUTION'
 assert len([r for r in csv.DictReader((R/'benchmark_samples.csv').open()) if r['mode']=='o4'])>=30
 assert len([r for r in csv.DictReader((R/'benchmark_samples.csv').open()) if r['mode']=='o6'])>=30
