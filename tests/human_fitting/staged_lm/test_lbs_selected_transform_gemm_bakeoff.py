import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];BASE=ROOT/'data_process/Vis/cpp_reference_dump/lbs_selected_transform_gemm/frame_000513'
def load():return json.loads((BASE/'output/benchmark_summary.json').read_text())
def test_fixture_and_layout():
 m=json.loads((BASE/'fixture/manifest.json').read_text());r=load();assert m['REFERENCE_LEAKAGE']=='NONE' and len(m['contexts'])==6
 assert r['D_A_ZERO_COPY_FLATTEN']=='YES' and r['INPUT_PACK_REQUIRED']=='NO' and r['OUTPUT_UNPACK_REQUIRED']=='NO'
 assert r['MATERIALIZATION_REDUCTION']=='NONE' and r['explicit_temporary_bytes']==0
def test_shapes_and_parity():
 for v in load()['contexts'].values():
  assert v['matrix_shapes']=={'weights':[v['point_count'],55],'dA_flat':[55,16*v['Q']],'dT_flat':[v['point_count'],16*v['Q']]}
  assert v['formal_gate'] and v['dT_parity']['rel_frob']<=2e-6 and v['transform_term_parity']['rel_frob']<=2e-6
  assert v['vertex_term_parity']['max_abs']==0 and v['dskinned_parity']['rel_frob']<=2e-6
def test_timing_schema_and_weighted_formulas():
 r=load();required={'scalar_dt_median_ms','gemm_pack_median_ms','gemm_core_median_ms','gemm_unpack_median_ms','gemm_dt_total_median_ms','current_full_lbs_median_ms','gemm_full_lbs_median_ms'}
 assert all(required<=set(v) for v in r['contexts'].values())
 sc=sum(v['calls_per_frame']*v['scalar_dt_median_ms'] for v in r['contexts'].values());gm=sum(v['calls_per_frame']*v['gemm_dt_total_median_ms'] for v in r['contexts'].values());cf=sum(v['calls_per_frame']*v['current_full_lbs_median_ms'] for v in r['contexts'].values());gf=sum(v['calls_per_frame']*v['gemm_full_lbs_median_ms'] for v in r['contexts'].values())
 assert abs(sc-r['selected_transform_current_frame_ms'])<1e-9 and abs(gm-r['selected_transform_gemm_frame_ms'])<1e-9 and abs(cf-r['full_lbs_current_frame_ms'])<1e-9 and abs(gf-r['full_lbs_gemm_frame_ms'])<1e-9
 assert abs(r['gemm_lbs_only_projected_ms']-(4617.56-(cf-gf)))<1e-9
def test_classification_and_next_experiment():
 r=load();assert r['LBS_DERIVATIVE_SELECTED_TRANSFORM_GEMM_BAKEOFF_READY']=='PASS';assert r['SELECTED_TRANSFORM_SPEEDUP_CLASS']=='STRONG_WIN';assert r['FULL_LBS_IMPACT_CLASS']=='STRONG';assert r['residual_lbs_breakdown']['selected_transform_ms']>r['residual_lbs_breakdown']['downstream_ms'];assert r['next_experiment']=='LBS_DERIVATIVE_SELECTED_TRANSFORM_VECTORISATION_BAKEOFF'
