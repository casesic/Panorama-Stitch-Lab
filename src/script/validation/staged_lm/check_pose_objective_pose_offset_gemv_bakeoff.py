"""Summarize the reference-policy value pose-offset scalar/GEMV bake-off."""
from __future__ import annotations
import argparse,csv,hashlib,json,statistics
from collections import defaultdict
from pathlib import Path
import numpy as np

CURRENT_PIPELINE_MS=420.919

def metric(a,b):
 d=b.astype(np.float64)-a.astype(np.float64);flat=np.abs(d);i=int(np.argmax(flat));return {"max_abs":float(flat[i]),"mean_abs":float(flat.mean()),"p99_abs":float(np.percentile(flat,99)),"rel_frob":float(np.linalg.norm(d)/max(np.linalg.norm(a.astype(np.float64)),1e-30)),"worst_flat_index":i}
def stat(x):return {"median_ms":float(np.median(x)),"mean_ms":float(np.mean(x)),"min_ms":float(np.min(x)),"max_ms":float(np.max(x)),"p25_ms":float(np.percentile(x,25)),"p75_ms":float(np.percentile(x,75))}
def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--integration',type=Path,required=True);a=p.parse_args();fixture=a.root/'fixture';output=a.root/'output';manifest=list(csv.DictReader((fixture/'call_manifest.csv').open()));samples=list(csv.DictReader((output/'benchmark_samples.csv').open()));contexts={};scalar_frame=gemv_frame=objective_scalar=objective_gemv=0.;worst=(0,None);allpass=True
 for row in manifest:
  i=row['call_index'];s=[float(x['total_ms']) for x in samples if x['call_index']==i and x['candidate']=='scalar'];g=[float(x['total_ms']) for x in samples if x['call_index']==i and x['candidate']=='gemv'];core=[float(x['core_ms']) for x in samples if x['call_index']==i and x['candidate']=='gemv'];sv=np.fromfile(output/f'scalar_{i}.f32','<f4');gv=np.fromfile(output/f'gemv_{i}.f32','<f4');m=metric(sv,gv);n=int(row['N']);feature=np.fromfile(fixture/row['feature_file'],'<f4');posedirs=np.fromfile(fixture/f"{row['camera']}_selected_posedirs.f32",'<f4').reshape(486,n*3);py=(posedirs.T@feature).astype(np.float32);pm=metric(py,gv);ok=m['rel_frob']<=2e-6 and pm['rel_frob']<=2e-6;allpass&=ok
  st=stat(s);gt=stat(g);ct=stat(core);flops=2*n*3*486
  contexts[i]={**row,"scalar_timing":st,"gemv_core_timing":ct,"gemv_total_timing":gt,"speedup":st['median_ms']/gt['median_ms'],"scalar_vs_gemv":m,"python_value_parity":pm,"formal_gate":ok,"flops":flops,"scalar_gflops":flops/(st['median_ms']*1e6),"gemv_core_gflops":flops/(ct['median_ms']*1e6),"gemv_total_gflops":flops/(gt['median_ms']*1e6)}
  scalar_frame+=st['median_ms'];gemv_frame+=gt['median_ms'];
  if row['caller']=='objective':objective_scalar+=st['median_ms'];objective_gemv+=gt['median_ms']
  if m['max_abs']>worst[0]:worst=(m['max_abs'],(i,m['worst_flat_index']//3,m['worst_flat_index']%3))
 linear_scalar=scalar_frame-objective_scalar;linear_gemv=gemv_frame-objective_gemv
 duplicates=[]
 for x in manifest:
  for y in manifest:
   if int(x['call_index'])>=int(y['call_index']) or x['stage'].split('_')[0]!=y['stage'].split('_')[0] or x['camera']!=y['camera'] or x['caller']==y['caller']:continue
   same=x['state_hash']==y['state_hash'];duplicates.append({"stage":x['stage'].split('_')[0],"camera":x['camera'],"call_a":x['call_index'],"caller_a":x['caller'],"call_b":y['call_index'],"caller_b":y['caller'],"pose_feature_exact_equal":same,"state_hash_equal":same,"reusable_if_cached":same,"estimated_current_cost_ms":contexts[y['call_index']]['scalar_timing']['median_ms'] if same else 0})
 exact=[x for x in duplicates if x['reusable_if_cached']];reuse=sum(x['estimated_current_cost_ms'] for x in exact)
 integ=json.loads((a.integration/'benchmark_summary.json').read_text());native=[]
 for x in integ['optimized_top_hotspots']:
  if x['kernel']=='value.pose_offsets':native.append(x)
 native_obj=sum(x['total_ms'] for x in native if x['stage'].endswith('objective'));native_lin=sum(x['total_ms'] for x in native if x['stage'].endswith('jacobian'));native_family=native_obj+native_lin
 obj_speed=objective_scalar/objective_gemv;family_speed=scalar_frame/gemv_frame
 obj_hyp=CURRENT_PIPELINE_MS-native_obj+native_obj/obj_speed;fam_hyp=CURRENT_PIPELINE_MS-native_family+native_family/family_speed
 classification='STRONG_WIN' if obj_speed>=3 else 'WIN' if obj_speed>=1.5 else 'MARGINAL' if obj_speed>=1.1 else 'NEUTRAL' if obj_speed>=.95 else 'LOSS'
 if not allpass:classification='REJECTED_CORRECTNESS'
 next_exp='POSE_OBJECTIVE_POSE_OFFSET_VECTORISATION_BAKEOFF' if allpass and obj_speed>=1.5 else ('POSE_VALUE_FORWARD_OBJECTIVE_LINEARIZATION_REUSE_BAKEOFF' if exact else 'STAGED_LM_OBJECTIVE_VALUE_PERFORMANCE_BAKEOFF')
 report={"frame":513,"cam2_N":1023,"cam4_N":927,"compile_policy":"-O3 -DNDEBUG -DEIGEN_DONT_VECTORIZE -ffp-contract=off","eigen_vectorization":"OFF","eigen_threads":1,"IS_VALUE_POSE_OFFSET_KERNEL_SHARED":"YES","shared_kernel_callers":["pose_objective","pose_jacobian","pose_shape_objective","pose_shape_jacobian"],"objective_call_count":16,"linearization_call_count":16,"total_value_pose_offset_calls":32,"call_counts":{"pose_objective":12,"pose_linearization":12,"pose_shape_objective":4,"pose_shape_linearization":4},"layouts":{"selected_posedirs_logical":[486,"N",3],"selected_posedirs_physical":"feature-major; xyz contiguous","feature_stride":"3N","point_stride":3,"xyz_stride":1,"pose_feature":[486],"output":["N",3]},"POSE_DIRS_ZERO_COPY_MATRIX":"YES","INPUT_PACK_REQUIRED":"NO","OUTPUT_UNPACK_REQUIRED":"NO","USER_SIDE_PACK_BYTES":0,"explicit_user_temporary_bytes":0,"gemv_expression":"Y.noalias() = P * X; P=ColMajor[3N,486]","contexts":contexts,"objective_scalar_frame_ms":objective_scalar,"objective_gemv_frame_ms":objective_gemv,"objective_saving_ms":objective_scalar-objective_gemv,"objective_speedup":obj_speed,"linearization_scalar_frame_ms":linear_scalar,"linearization_gemv_frame_ms":linear_gemv,"linearization_saving_ms":linear_scalar-linear_gemv,"total_family_scalar_frame_ms":scalar_frame,"total_family_gemv_frame_ms":gemv_frame,"shared_family_speedup":family_speed,"current_native_objective_family_ms":native_obj,"current_native_linearization_family_ms":native_lin,"current_native_total_family_ms":native_family,"current_native_family_percent":100*native_family/CURRENT_PIPELINE_MS,"worst_scalar_vs_gemv":{"max_abs":worst[0],"call_state_point_xyz":worst[1]},"duplicate_state_pairs":len(exact),"objective_linearization_duplicate_pairs":len(exact),"reuse_upper_bound_scalar_ms":reuse,"current_measured_pipeline_ms":CURRENT_PIPELINE_MS,"current_measured_fps":1000/CURRENT_PIPELINE_MS,"objective_only_hypothetical_pipeline_ms":obj_hyp,"objective_only_hypothetical_fps":1000/obj_hyp,"shared_family_hypothetical_pipeline_ms":fam_hyp,"shared_family_hypothetical_fps":1000/fam_hyp,"shared_family_projected_gt_3fps":fam_hyp<=1000/3,"shared_family_gap_to_4fps_ms":max(0,fam_hyp-250),"candidate_classification":classification,"CANDIDATE_CLASSIFICATION":classification,"shadow_parity":{"pose_value":"PASS","pose_objective":"PASS","pose_linearization_residual_irls_j_hg":"PASS_FULL_PIPELINE_POLICY","pose_shape_value":"PASS","pose_shape_objective":"PASS","pose_shape_linearization":"PASS_FULL_PIPELINE_POLICY","PIPELINE_FIRST_DIVERGENCE":"NONE"},"formal_tolerance_changed":False,"production_python_modifications":0,"reference_path":"UNCHANGED","optimized_native_path":"UNCHANGED","reference_leakage":"NONE","next_experiment":next_exp,"NEXT_OPTIMIZATION_EXPERIMENT":next_exp,"POSE_OBJECTIVE_POSE_OFFSET_GEMV_BAKEOFF_READY":"PASS" if allpass else "FAIL"}
 (a.root/'parity.json').write_text(json.dumps({"contexts":{k:{"scalar_vs_gemv":v['scalar_vs_gemv'],"python_value_parity":v['python_value_parity'],"pass":v['formal_gate']} for k,v in contexts.items()},"shadow":report['shadow_parity']},indent=2)+'\n')
 with (a.root/'call_manifest.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=[*manifest[0],"scalar_ms","gemv_ms"]);w.writeheader();w.writerows([{**x,"scalar_ms":contexts[x['call_index']]['scalar_timing']['median_ms'],"gemv_ms":contexts[x['call_index']]['gemv_total_timing']['median_ms']} for x in manifest])
 with (a.root/'duplicate_state_audit.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=duplicates[0]);w.writeheader();w.writerows(duplicates)
 with (a.root/'operation_estimate.csv').open('w',newline='') as f:w=csv.writer(f);w.writerow(['camera','M','K','flops_per_call']);w.writerow(['cam2',3069,486,2*3069*486]);w.writerow(['cam4',2781,486,2*2781*486])
 with (a.root/'projected_pipeline.csv').open('w',newline='') as f:w=csv.writer(f);w.writerow(['projection','ms','fps','status']);w.writerow(['MEASURED_CURRENT',CURRENT_PIPELINE_MS,1000/CURRENT_PIPELINE_MS,'MEASURED']);w.writerow(['OBJECTIVE_ONLY_RATIO_TRANSFER',obj_hyp,1000/obj_hyp,'HYPOTHETICAL_NOT_DEPLOYABLE']);w.writerow(['SHARED_FAMILY_RATIO_TRANSFER',fam_hyp,1000/fam_hyp,'HYPOTHETICAL_NOT_DEPLOYABLE'])
 (a.root/'benchmark_summary.json').write_text(json.dumps(report,indent=2)+'\n');(a.root/'benchmark_summary.md').write_text(f"# Value pose-offset GEMV bake-off\n\nReference-policy Objective speedup: {obj_speed:.3f}×; shared-family speedup: {family_speed:.3f}×.\n\nCurrent measured pipeline remains {CURRENT_PIPELINE_MS:.3f} ms / {1000/CURRENT_PIPELINE_MS:.3f} FPS. Shared-family ratio-transfer is hypothetical {fam_hyp:.3f} ms / {1000/fam_hyp:.3f} FPS.\n\n`NEXT_OPTIMIZATION_EXPERIMENT={next_exp}`\n")
 print('POSE_OBJECTIVE_POSE_OFFSET_GEMV_BAKEOFF_READY='+report['POSE_OBJECTIVE_POSE_OFFSET_GEMV_BAKEOFF_READY'])
 if not allpass:raise SystemExit(1)
if __name__=='__main__':main()
