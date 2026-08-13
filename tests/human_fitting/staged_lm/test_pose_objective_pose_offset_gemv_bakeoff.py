import csv,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];BASE=ROOT/'data_process/Vis/cpp_reference_dump/pose_objective_pose_offset_gemv/frame_000513'
def load():return json.loads((BASE/'benchmark_summary.json').read_text())
def test_fixture_input_only_and_manifest():
 files=[x.name for x in (BASE/'fixture').iterdir()];assert not [x for x in files if any(y in x for y in ('expected','output','uv','residual','objective','jacobian'))]
 rows=list(csv.DictReader((BASE/'fixture/call_manifest.csv').open()));assert len(rows)==32;assert {x['stage'] for x in rows}=={'pose_objective','pose_jacobian','pose_shape_objective','pose_shape_jacobian'}
def test_layout_parity_shadow_and_schema():
 d=load();assert d['IS_VALUE_POSE_OFFSET_KERNEL_SHARED']=='YES';assert d['POSE_DIRS_ZERO_COPY_MATRIX']=='YES';assert d['INPUT_PACK_REQUIRED']==d['OUTPUT_UNPACK_REQUIRED']=='NO';assert d['objective_call_count']==d['linearization_call_count']==16
 assert all(x['formal_gate'] for x in d['contexts'].values());assert all(v.startswith('PASS') or v=='NONE' for v in d['shadow_parity'].values());assert d['formal_tolerance_changed'] is False
def test_cost_duplicate_projection_and_next_rule():
 d=load();assert abs(d['total_family_scalar_frame_ms']-(d['objective_scalar_frame_ms']+d['linearization_scalar_frame_ms']))<1e-9;assert abs(d['shared_family_speedup']-d['total_family_scalar_frame_ms']/d['total_family_gemv_frame_ms'])<1e-12
 assert d['duplicate_state_pairs']>0;assert d['shared_family_hypothetical_pipeline_ms']<1000/3;assert d['CANDIDATE_CLASSIFICATION']=='STRONG_WIN';assert d['NEXT_OPTIMIZATION_EXPERIMENT']=='POSE_OBJECTIVE_POSE_OFFSET_VECTORISATION_BAKEOFF';assert d['POSE_OBJECTIVE_POSE_OFFSET_GEMV_BAKEOFF_READY']=='PASS'
