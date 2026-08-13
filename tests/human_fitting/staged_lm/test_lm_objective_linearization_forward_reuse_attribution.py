import csv,json
from pathlib import Path
R=Path('data_process/Vis/cpp_reference_dump/lm_objective_linearization_forward_reuse_attribution/frame_000513')
def test_forward_reuse_attribution_contract():
 d=json.loads((R/'benchmark_summary.json').read_text());rows=list(csv.DictReader((R/'lm_state_timeline.csv').open()))
 assert d['ready']=='PASS' and d['preferred_policy']=='optimized_native_stable_p'
 assert d['pose_exact_state_pairs']==6 and d['pose_shape_exact_state_pairs']==2
 assert d['duplicate_camera_forward_count']==16 and d['total_value_forward_executions']==32
 assert d['necessary_unique_state_forward_executions']==16
 assert d['initial_direction_pairs']==2 and d['accepted_direction_pairs']==6
 assert d['pose_common_exact'] and d['pose_shape_common_exact']
 assert abs(d['total_reuse_upper_bound_ms']-(d['pose_reuse_upper_bound_ms']+d['pose_shape_reuse_upper_bound_ms']))<1e-9
 assert abs(d['conservative_75_percent_saving_ms']-.75*d['total_reuse_upper_bound_ms'])<1e-9
 assert d['erp_unused_jacobian_audit']['independent_of_exact_reuse']
 assert d['next_experiment']=='LM_OBJECTIVE_LINEARIZATION_FORWARD_REUSE_INTEGRATION_BAKEOFF'
 assert sum(r['reusable']=='YES' for r in rows)==16
