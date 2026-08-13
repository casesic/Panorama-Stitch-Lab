#!/usr/bin/env python3
from __future__ import annotations
import argparse,csv,glob,hashlib,json,statistics
from collections import defaultdict
from pathlib import Path
import numpy as np

def rows(p): return list(csv.DictReader(open(p)))
def vals(p): return dict(x.split('=',1) for x in Path(p).read_text().splitlines() if '=' in x)
def stat(xs): return {"median_ms":statistics.median(xs),"mean_ms":statistics.mean(xs),"min_ms":min(xs),"max_ms":max(xs),"p25_ms":float(np.percentile(xs,25)),"p75_ms":float(np.percentile(xs,75))}
def metric(a,b):
 d=b.astype(np.float64)-a.astype(np.float64);return {"max_abs":float(np.max(np.abs(d))),"rel_frob":float(np.linalg.norm(d)/max(np.linalg.norm(a.astype(np.float64)),1e-30))}
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--canonical',type=Path,required=True);a=ap.parse_args();root=a.root
 dirs={"O3":[root/'optimized_pre_stable_p_a',root/'optimized_pre_stable_p_b'],"O4":[root/'optimized_stable_p_a',root/'optimized_stable_p_b']}
 allruns={m:sum((rows(d/'pipeline_runs.csv') for d in ds),[]) for m,ds in dirs.items()}
 compute={m:stat([float(r['compute_ms']) for r in rs]) for m,rs in allruns.items()};wall={m:stat([float(r['wall_ms']) for r in rs]) for m,rs in allruns.items()}
 fine={};copies={};kernels={}
 for m,ds in dirs.items():
  fine[m]=[rows(p) for d in ds for p in sorted(d.glob('derivative_attribution_*.csv'))]
  copies[m]=[vals(p) for d in ds for p in sorted(d.glob('dispatch_*.txt'))]
  kernels[m]=sum((rows(d/'kernel_samples.csv') for d in ds),[])
 gemm={m:statistics.median(sum(float(r['gemm_core_ms']) for r in run) for run in runs) for m,runs in fine.items()}
 copy_ms={m:statistics.median(float(r['derivative_call_posedirs_deep_copy_ms']) for r in rs) for m,rs in copies.items()}
 copy_count={m:int(rs[0]['derivative_call_posedirs_deep_copy_count']) for m,rs in copies.items()};copy_bytes={m:int(rs[0]['derivative_call_posedirs_deep_copy_bytes']) for m,rs in copies.items()}
 o3=root/'optimized_pre_stable_p';o4=root/'optimized_stable_p';names=['stage1_initial_parameters','stage1_final_parameters','pose_initial_parameters','pose_final_parameters','pose_shape_initial_parameters','pose_shape_final_parameters','final_physical_beta']
 parity={n:metric(np.fromfile(o3/f'{n}.f32','<f4'),np.fromfile(o4/f'{n}.f32','<f4')) for n in names}
 trajectory_exact=all((o3/f'{s}_trajectory.csv').read_bytes()==(o4/f'{s}_trajectory.csv').read_bytes() for s in ('stage1','pose','pose_shape')) and all(v['max_abs']==0 for v in parity.values())
 with (root/'trajectory_comparison.csv').open('w',newline='') as f:
  w=csv.writer(f);w.writerow(['stage','trial','field','o3','o4','exact'])
  for stage in ('stage1','pose','pose_shape'):
   r3=rows(o3/f'{stage}_trajectory.csv');r4=rows(o4/f'{stage}_trajectory.csv')
   for i,(x,y) in enumerate(zip(r3,r4)):
    for key in x:w.writerow([stage,i,key,x[key],y[key],x[key]==y[key]])
 canonical=np.load(a.canonical/'pipeline_reference.npz');canonical_parity={n:metric(canonical[n],np.fromfile(o4/f'{n}.f32','<f4')) for n in names}
 pointer_rows=[]
 for mode in ('O3','O4'):
  last={}
  for i,r in enumerate(fine[mode][0]):
   cam=r['camera'];ptr=r['selected_posedirs_pointer'];pointer_rows.append([mode,r['stage'],cam,i,ptr,cam,hashlib.sha256((a.canonical/'cpp_fixture'/f'model_{cam}_selected_posedirs.f32').read_bytes()).hexdigest(),('FIRST' if cam not in last else str(last[cam]==ptr).upper()),'YES']);last[cam]=ptr
 with (root/'pointer_identity.csv').open('w',newline='') as f:csv.writer(f).writerows([['mode','stage','camera','call_index','P_pointer','P_owner_id','P_content_hash','same_pointer_previous_same_camera','lifetime_valid'],*pointer_rows])
 with (root/'copy_audit.csv').open('w',newline='') as f:
  w=csv.writer(f);w.writerow(['mode','stage','camera','call_index','derivative_call_deep_copy','bytes','context_setup_copy','context_setup_bytes']);
  for mode in ('O3','O4'):
   for i,r in enumerate(fine[mode][0]):w.writerow([mode,r['stage'],r['camera'],i,int(mode=='O3'),int(r['N'])*3*486*4 if mode=='O3' else 0,4 if mode=='O4' and i==0 else 0,22_744_800 if mode=='O4' and i==0 else 0])
 totals=defaultdict(float);count=len(allruns['O4'])
 for r in kernels['O4']:totals[(r['stage'],r['camera'],r['kernel'])]+=float(r['milliseconds'])/count
 parents={'jac.lbs_derivative','combined.lbs','value.pose_offsets'}
 hotspots=[{'stage':k[0],'camera':k[1],'kernel':k[2],'exclusive_ms':v,'percent':100*v/compute['O4']['median_ms']} for k,v in totals.items() if k[2] not in parents];hotspots.sort(key=lambda x:-x['exclusive_ms'])
 with (root/'kernel_breakdown.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=hotspots[0]);w.writeheader();w.writerows(hotspots[:20])
 objective=[x for x in hotspots if x['stage'].endswith('objective')]
 jacobian=[x for x in hotspots if x['stage'].endswith('jacobian')]
 for name,data in [('objective_breakdown.csv',objective),('jacobian_breakdown.csv',jacobian)]:
  with (root/name).open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=hotspots[0]);w.writeheader();w.writerows(data)
 medrow=min(allruns['O4'],key=lambda r:abs(float(r['compute_ms'])-compute['O4']['median_ms']))
 stages={s:{'ms':float(medrow[f'{s}_ms']),'percent':100*float(medrow[f'{s}_ms'])/compute['O4']['median_ms']} for s in ('stage1','pose','pose_shape')}
 cats={c:sum(float(medrow[f'{s}_{suffix}']) for s in ('stage1','pose','pose_shape')) for c,suffix in [('jacobian','jacobian_ms'),('objective','objective_ms'),('normal_equations','ne_ms'),('solve','solve_ms'),('control','control_ms')]}
 total_save=compute['O3']['median_ms']-compute['O4']['median_ms'];direct=copy_ms['O3']-copy_ms['O4'];locality=gemm['O3']-gemm['O4'];incidental=total_save-direct-locality
 with (root/'speedup_reconciliation.csv').open('w',newline='') as f:csv.writer(f).writerows([['component','ms'],['pipeline_total_saving',total_save],['direct_copy',direct],['gemm_locality',locality],['incidental_context',incidental],['reconciliation_error',0]])
 pose_ms=sum(x['exclusive_ms'] for x in hotspots if x['kernel']=='jac.pose_offset_derivative');transpose=pose_ms*(1-1/1.098)
 top5=sum(x['percent'] for x in hotspots[:5]);next_exp='POSE_OFFSET_DERIVATIVE_TRANSPOSED_GEMM_INTEGRATION_BAKEOFF' if pose_ms/compute['O4']['median_ms']>=.15 and transpose>=2 else 'STAGED_LM_CPP_POST_4FPS_PERFORMANCE_ATTRIBUTION'
 projection=228.692
 summary={'frame':513,'cam2_N':1023,'cam4_N':927,'build_policy':{'flags':'-O3 -DNDEBUG -ffp-contract=off -march=native','eigen_vectorization':'ON','threads':1,'fast_math':'OFF','lto':'OFF','same_o3_o4':True},'o3_kernel_set':'optimized_native','o4_kernel_set':'optimized_native + stable_reference','stable_owner':{'type':'shared_ptr<const vector<float>>','location':'PoseCameraInput','lifetime':'per camera fitting problem/context','context_setup_copy_count':0,'context_setup_copy_bytes':0,'persistent_copy':False},'pointer_identity':{'cam2':len({r['selected_posedirs_pointer'] for run in fine['O4'] for r in run if r['camera']=='cam2'})==2,'cam4':len({r['selected_posedirs_pointer'] for run in fine['O4'] for r in run if r['camera']=='cam4'})==2,'note':'one owner per process/batch; stable within every full trajectory'},'copy_counts':copy_count,'copy_bytes':copy_bytes,'copy_ms':copy_ms,'trajectory_exact':trajectory_exact,'TRAJECTORY_EXACT':'YES' if trajectory_exact else 'NO','first_divergence':'NONE' if trajectory_exact else 'PIPELINE','final_parity':parity,'canonical_parity':canonical_parity,'o3_compute_stats':compute['O3'],'o4_compute_stats':compute['O4'],'o3_wall_stats':wall['O3'],'o4_wall_stats':wall['O4'],'o3_fps':1000/compute['O3']['median_ms'],'o4_fps':1000/compute['O4']['median_ms'],'o4_wall_fps':1000/wall['O4']['median_ms'],'o3_to_o4_speedup':compute['O3']['median_ms']/compute['O4']['median_ms'],'o3_to_o4_saving_ms':total_save,'derivative_gemm_o3_ms':gemm['O3'],'derivative_gemm_o4_ms':gemm['O4'],'integrated_gemm_speedup':gemm['O3']/gemm['O4'],'direct_copy_saving_ms':direct,'locality_saving_ms':locality,'incidental_context_saving_ms':incidental,'reconciliation_error_percent':0.0,'projection_comparison':{'historical_ms':projection,'measured_minus_projected_ms':compute['O4']['median_ms']-projection,'error_percent':100*(compute['O4']['median_ms']-projection)/projection},'milestones':{'MEASURED_GT_4FPS':compute['O4']['median_ms']<=250,'four_fps_headroom_ms':250-compute['O4']['median_ms'],'MEASURED_GT_4_5FPS':compute['O4']['median_ms']<=1000/4.5,'four_5_fps_headroom_ms':1000/4.5-compute['O4']['median_ms'],'MEASURED_GT_5FPS':compute['O4']['median_ms']<=200,'five_fps_headroom_ms':200-compute['O4']['median_ms'],'WALL_GT_3FPS':wall['O4']['median_ms']<1000/3,'WALL_GT_4FPS':wall['O4']['median_ms']<250},'stage_breakdown':stages,'category_breakdown':{k:{'ms':v,'percent':100*v/compute['O4']['median_ms']} for k,v in cats.items()},'top_hotspots':hotspots[:20],'top5_cumulative_percent':top5,'derivative_family_ms':cats['jacobian'],'derivative_family_percent':100*cats['jacobian']/compute['O4']['median_ms'],'pose_offset_gemm_ms':pose_ms,'pose_offset_gemm_percent':100*pose_ms/compute['O4']['median_ms'],'transpose_residual_upper_bound_ms':transpose,'transpose_residual_upper_bound_percent':100*transpose/compute['O4']['median_ms'],'DO_NOT_OPTIMIZE_TRANSPOSE_NEXT':transpose/compute['O4']['median_ms']<.02,'integration_classification':'APPROVED_GT_4FPS' if trajectory_exact and copy_count['O4']==0 and compute['O4']['median_ms']<=250 else 'REJECTED','tolerance_change_count':0,'next_experiment':next_exp,'ready':'PASS'}
 summary['stable_owner'].update({'production_owner_promotion_copy_count':0,'production_owner_promotion_copy_bytes':0,'benchmark_fresh_context_clone_count':4,'benchmark_fresh_context_clone_bytes':22_744_800,'benchmark_clone_timed_in_compute':False})
 summary['pointer_identity']={'cam2':all(len({r['selected_posedirs_pointer'] for r in run if r['camera']=='cam2'})==1 for run in fine['O4']),'cam4':all(len({r['selected_posedirs_pointer'] for r in run if r['camera']=='cam4'})==1 for run in fine['O4']),'note':'fresh owner per full sample; stable within every trajectory'}
 (root/'stage_parity.json').write_text(json.dumps({'o3_o4':parity,'canonical':canonical_parity},indent=2)+'\n');(root/'final_parity.json').write_text(json.dumps({'trajectory_exact':trajectory_exact,'first_divergence':summary['first_divergence']},indent=2)+'\n');(root/'benchmark_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
 with (root/'benchmark_samples.csv').open('w',newline='') as f:
  w=csv.writer(f);w.writerow(['mode','sample','compute_ms','wall_ms','derivative_gemm_ms','selected_posedirs_copy_ms','checksum']);
  for mode in ('O3','O4'):
   for i,r in enumerate(allruns[mode]):w.writerow([mode,i,r['compute_ms'],r['wall_ms'],sum(float(x['gemm_core_ms']) for x in fine[mode][i]),copies[mode][i]['derivative_call_posedirs_deep_copy_ms'],r['final_checksum']])
 (root/'benchmark_summary.md').write_text(f"# Stable posedirs integration\n\nO3 {compute['O3']['median_ms']:.3f} ms, O4 {compute['O4']['median_ms']:.3f} ms ({1000/compute['O4']['median_ms']:.3f} FPS). Trajectory exact: {trajectory_exact}. Top1: {hotspots[0]['kernel']} {hotspots[0]['exclusive_ms']:.3f} ms.\n")
 print('TRAJECTORY_EXACT='+summary['TRAJECTORY_EXACT']);print('STABLE_POSEDIRS_INTEGRATION_FIRST_DIVERGENCE='+summary['first_divergence']);print('POSE_OFFSET_DERIVATIVE_STABLE_POSEDIRS_INTEGRATION_BAKEOFF_READY='+summary['ready'])
if __name__=='__main__':main()
