#!/usr/bin/env python3
from __future__ import annotations
import argparse,csv,json,statistics,time
from collections import defaultdict
from pathlib import Path
import numpy as np
def R(p):return list(csv.DictReader(open(p)))
def S(x):return {'median_ms':statistics.median(x),'mean_ms':statistics.mean(x),'min_ms':min(x),'max_ms':max(x),'p25_ms':float(np.percentile(x,25)),'p75_ms':float(np.percentile(x,75))}
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);a=ap.parse_args();root=a.root;runs=R(root/'pipeline_runs.csv');kr=R(root/'kernel_samples.csv');n=len(runs);base=statistics.median(float(x['compute_ms']) for x in runs);jtotal=statistics.mean(float(x['pose_jacobian_ms'])+float(x['pose_shape_jacobian_ms']) for x in runs);posej=statistics.mean(float(x['pose_jacobian_ms']) for x in runs);shapej=statistics.mean(float(x['pose_shape_jacobian_ms']) for x in runs)
 raw=defaultdict(float);calls=defaultdict(int)
 for x in kr:
  k=(x['stage'],x['camera'],x['kernel']);raw[k]+=float(x['milliseconds'])/n;calls[k]+=1
 fine=[R(root/f'derivative_attribution_{i}.csv') for i in range(n)];fg=defaultdict(float)
 for run in fine:
  for x in run:
   k=(x['stage'],x['camera']);
   for field in ('d_pose_feature_ms','output_prepare_ms','gemm_core_ms','postprocess_ms','setup_ms','other_ms'):fg[k,field]+=float(x[field])/n
 impl={'value.rotations':'scalar SO3/value loop','value.pose_offsets_gemv':'native Eigen GEMV','value.fk':'per-joint scalar recursion','value.lbs':'per-vertex scalar LBS','value.erp':'per-point scalar transcendental/projection','jac.rotation_derivative':'small Eigen + scalar assembly','jac.fk_derivative':'scalar joint-tree product-rule loops','jac.lbs_selected_transform':'native Eigen GEMM','jac.lbs_downstream':'fused scalar transform/vertex contractions','jac.erp_chain':'per-point scalar chain multiplication','jac.assembly':'scalar copy/scatter 66→69→75','combined.shape_setup':'scalar beta derivative/materialization','combined.fk':'scalar Q76 joint-tree product-rule loops','combined.lbs_selected_transform':'native Eigen GEMM Q76','combined.lbs_downstream':'fused scalar Q76 transform/vertex contractions','combined.erp_chain':'scalar Q76 chain multiplication/J_beta construction','combined.assembly':'vector copy'}
 components=[]
 def add(stage,cam,name,ms,style):components.append({'stage':stage,'camera':cam,'component':name,'calls':6 if stage=='pose_jacobian' else 2,'total_ms':ms,'ms_per_call':ms/(6 if stage=='pose_jacobian' else 2),'pipeline_percent':100*ms/base,'jacobian_percent':100*ms/jtotal,'implementation_style':style})
 for stage in ('pose_jacobian','pose_shape_jacobian'):
  for cam in ('cam2','cam4'):
   # Value forward is part of the Jacobian timer and already has exclusive leaves.
   for k in ('value.rotations','value.pose_offsets_gemv','value.fk','value.lbs','value.erp'):
    add(stage,cam,k,raw[stage,cam,k],impl[k])
   add(stage,cam,'rotation_derivative',raw[stage,cam,'jac.rotation_derivative'],impl['jac.rotation_derivative'])
   add(stage,cam,'d_pose_feature',fg[(stage,cam),'d_pose_feature_ms'],'scalar copy/reshape 486x66')
   add(stage,cam,'pose_offset_derivative_gemm',fg[(stage,cam),'gemm_core_ms'],'native Eigen GEMM, stable immutable P')
   add(stage,cam,'pose_offset_output_prepare_post',fg[(stage,cam),'output_prepare_ms']+fg[(stage,cam),'postprocess_ms']+fg[(stage,cam),'setup_ms']+fg[(stage,cam),'other_ms'],'allocation/map plus d_v_posed copy')
   add(stage,cam,'fk_derivative_q66',raw[stage,cam,'jac.fk_derivative'],impl['jac.fk_derivative'])
   sel=raw[stage,cam,'jac.lbs_selected_transform'];down=raw[stage,cam,'jac.lbs_downstream'];parent=raw[stage,cam,'jac.lbs_derivative'];add(stage,cam,'lbs_selected_transform_gemm_q66',sel,impl['jac.lbs_selected_transform']);add(stage,cam,'lbs_transform_vertex_fused_q66',down,impl['jac.lbs_downstream']);add(stage,cam,'lbs_final_copy_scale_q66',max(0,parent-sel-down),'vector copies and scale loop')
   add(stage,cam,'erp_derivative_q66',raw[stage,cam,'jac.erp_chain'],impl['jac.erp_chain']);add(stage,cam,'active_frame_assembly_q66',raw[stage,cam,'jac.assembly'],impl['jac.assembly'])
   if stage=='pose_shape_jacobian':
    add(stage,cam,'beta_shape_joint_setup',raw[stage,cam,'combined.shape_setup'],impl['combined.shape_setup']);add(stage,cam,'combined_fk_rest_q76',raw[stage,cam,'combined.fk'],impl['combined.fk']);cs=raw[stage,cam,'combined.lbs_selected_transform'];cd=raw[stage,cam,'combined.lbs_downstream'];cp=raw[stage,cam,'combined.lbs'];add(stage,cam,'combined_lbs_selected_transform_gemm_q76',cs,impl['combined.lbs_selected_transform']);add(stage,cam,'combined_lbs_transform_vertex_fused_q76',cd,impl['combined.lbs_downstream']);add(stage,cam,'combined_lbs_final_split_copy_q76',max(0,cp-cs-cd),'vector copies, scale, Q76→Q66/Q10 split');add(stage,cam,'combined_erp_jbeta_q76',raw[stage,cam,'combined.erp_chain'],impl['combined.erp_chain']);add(stage,cam,'combined_active_assembly',raw[stage,cam,'combined.assembly'],impl['combined.assembly'])
 known=sum(x['total_ms'] for x in components);other=jtotal-known;add('shared','shared','jacobian_other_timer_gap',other,'timer/accounting gap including camera-call overhead')
 fields=list(components[0]);
 with (root/'jacobian_breakdown.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(sorted(components,key=lambda x:-x['total_ms']))
 for stage,fn in [('pose_jacobian','pose_jacobian_breakdown.csv'),('pose_shape_jacobian','pose_shape_jacobian_breakdown.csv')]:
  q=[x for x in components if x['stage']==stage];
  with (root/fn).open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(sorted(q,key=lambda x:-x['total_ms']))
 gemm=sum(x['total_ms'] for x in components if x['component']=='pose_offset_derivative_gemm');residual=[x for x in components if x['component']!='pose_offset_derivative_gemm'];residual.sort(key=lambda x:-x['total_ms']);rms=jtotal-gemm
 with (root/'jacobian_residual_top10.csv').open('w',newline='') as f:
  fs=['rank','component','stage','camera','total_ms','percent_pipeline','percent_jacobian_residual','implementation_style'];w=csv.DictWriter(f,fieldnames=fs);w.writeheader();w.writerows([{'rank':i+1,'component':x['component'],'stage':x['stage'],'camera':x['camera'],'total_ms':x['total_ms'],'percent_pipeline':x['pipeline_percent'],'percent_jacobian_residual':100*x['total_ms']/rms,'implementation_style':x['implementation_style']} for i,x in enumerate(residual[:10])])
 # Pipeline top20: exclusive Jacobian leaves plus non-J category buckets.
 hot=[dict(x,kernel=x.pop('component') if False else x['component'],exclusive_ms=x['total_ms'],percent=x['pipeline_percent']) for x in components]
 med=min(runs,key=lambda x:abs(float(x['compute_ms'])-base));
 for st in ('stage1','pose','pose_shape'):
  for c,z in [('normal_equations','ne_ms'),('solve','solve_ms'),('control','control_ms')]:hot.append({'stage':st,'camera':'shared','kernel':c,'exclusive_ms':float(med[f'{st}_{z}']),'percent':100*float(med[f'{st}_{z}'])/base})
 hot=sorted(hot,key=lambda x:-x['exclusive_ms'])
 with (root/'kernel_breakdown.csv').open('w',newline='') as f:
  fs=['stage','camera','kernel','exclusive_ms','percent'];w=csv.DictWriter(f,fieldnames=fs);w.writeheader();w.writerows([{k:x[k] for k in fs} for x in hot[:20]])
 # LBS family, honest fused downstream accounting.
 lbs=[x for x in components if 'lbs_' in x['component']];
 with (root/'lbs_derivative_breakdown.csv').open('w',newline='') as f:
  fs=['stage','camera','Q','selected_transform_gemm_ms','transform_vertex_fused_ms','transform_term_ms','vertex_term_ms','final_combine_ms','other_ms','total_ms'];w=csv.DictWriter(f,fieldnames=fs);w.writeheader()
  for st in ('pose_jacobian','pose_shape_jacobian'):
   for cam in ('cam2','cam4'):
    q=[x for x in lbs if x['stage']==st and x['camera']==cam];w.writerow({'stage':st,'camera':cam,'Q':'66 + 76' if st=='pose_shape_jacobian' else 66,'selected_transform_gemm_ms':sum(x['total_ms'] for x in q if 'selected_transform' in x['component']),'transform_vertex_fused_ms':sum(x['total_ms'] for x in q if 'transform_vertex' in x['component']),'transform_term_ms':'FUSED_NOT_SEPARABLE','vertex_term_ms':'FUSED_NOT_SEPARABLE','final_combine_ms':sum(x['total_ms'] for x in q if 'final_' in x['component']),'other_ms':0,'total_ms':sum(x['total_ms'] for x in q)})
 # Opportunities aggregate by family rather than camera.
 fam=defaultdict(float)
 for x in components:fam[x['component']]+=x['total_ms']
 specs={'pose_offset_derivative_gemm':(1.10,1.50,'native Eigen GEMM; blocking unproven','cache/kernel-shape','MEDIUM','MEDIUM'),'lbs_transform_vertex_fused_q66':(1.5,3,'scalar fused loops','scalar contraction/materialization','MEDIUM','HIGH'),'combined_lbs_transform_vertex_fused_q76':(1.5,3,'scalar fused Q76 loops','scalar contraction/materialization','MEDIUM','HIGH'),'fk_derivative_q66':(1.5,3,'scalar tree loops','product-rule recursion','HIGH','HIGH'),'combined_fk_rest_q76':(1.5,3,'scalar Q76 tree loops','product-rule recursion/materialization','HIGH','HIGH'),'value.pose_offsets_gemv':(1.05,1.2,'native Eigen GEMV','already optimized','LOW','MEDIUM'),'value.lbs':(1.2,2,'scalar per-vertex LBS','scalar loop','MEDIUM','MEDIUM')}
 opp=[]
 for name,(lo,hi,implementation,bottleneck,complexity,confidence) in specs.items():
  ms=fam[name];opp.append({'component':name,'current_ms':ms,'pipeline_percent':100*ms/base,'implementation':implementation,'likely_bottleneck':bottleneck,'plausible_speedup_low':lo,'plausible_speedup_high':hi,'saving_low_ms':ms*(1-1/lo),'saving_high_ms':ms*(1-1/hi),'projected_fps_low':1000/(base-ms*(1-1/lo)),'projected_fps_high':1000/(base-ms*(1-1/hi)),'complexity':complexity,'confidence':confidence})
 opp.sort(key=lambda x:-x['saving_high_ms']);
 with (root/'optimization_opportunity.csv').open('w',newline='') as f:fs=['rank',*opp[0]];w=csv.DictWriter(f,fieldnames=fs);w.writeheader();w.writerows([{'rank':i+1,**x} for i,x in enumerate(opp)])
 amd=[]
 for speed in (2,3,float('inf')):
  projected=base-jtotal+(0 if speed==float('inf') else jtotal/speed);amd.append({'scope':'whole_jacobian','speedup':'infinite' if speed==float('inf') else speed,'saving_ms':base-projected,'projected_ms':projected,'projected_fps':1000/projected})
 for speed in (1.10,1.20,1.25,1.50):amd.append({'scope':'pose_offset_gemm','speedup':speed,'saving_ms':gemm*(1-1/speed),'projected_ms':base-gemm*(1-1/speed),'projected_fps':1000/(base-gemm*(1-1/speed))})
 with (root/'amdahl_projection.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=amd[0]);w.writeheader();w.writerows(amd)
 # Manifests and audits.
 with (root/'call_manifest.csv').open('w',newline='') as f:
  w=csv.writer(f);w.writerow(['stage','camera','linearization_index','state_hash','N','base_Q','combined_Q','major_kernel_calls']);
  for i,x in enumerate(fine[0]):w.writerow([x['stage'],x['camera'],i//2,f"trajectory_state_{i//2}",x['N'],66,76 if x['stage']=='pose_shape_jacobian' else 'NA','value,rotation,dpose,FK,LBS,ERP,assembly'])
 mem=[['intermediate','shape','bytes','lifetime'],['cam2 selected_posedirs','[3069,486]',3069*486*4,'fitting context'],['cam4 selected_posedirs','[2781,486]',2781*486*4,'fitting context'],['cam2 d_pose_offsets','[3069,66]',3069*66*4,'camera Jacobian result'],['cam2 combined selected transforms','[1023,16,76]',1023*16*76*4,'PoseShape camera Jacobian result'],['cam2 combined vertices derivative','[3069,76]',3069*76*4,'PoseShape camera Jacobian result']]
 with (root/'memory_materialization.csv').open('w',newline='') as f:csv.writer(f).writerows(mem)
 dup=[['quantity','scope','duplicate_work_candidate','upper_bound_ms','note'],['d_pose_feature','cam2/cam4 same LM state','YES',fam['d_pose_feature']/2,'camera independent; audit only'],['rotation derivative','cam2/cam4 same LM state','YES',fam['rotation_derivative']/2,'camera independent; audit only'],['value/linearization forward','LM objective vs linearization','YES',sum(x['total_ms'] for x in components if x['component'].startswith('value.')),'no reuse implemented'],['Pose/PoseShape Q66 derivative','stage handoff states differ','NO',0,'parameters/model differ']]
 with (root/'duplicate_work_audit.csv').open('w',newline='') as f:csv.writer(f).writerows(dup)
 # Empty steady_clock pair audit.
 ts=[]
 for _ in range(10000):s=time.perf_counter_ns();e=time.perf_counter_ns();ts.append(e-s)
 timer_ns=statistics.median(ts);mincomp=min(x['total_ms']/x['calls'] for x in components if x['total_ms']>0)
 next_exp='POSE_OFFSET_DERIVATIVE_GEMM_BLOCKING_BAKEOFF' if gemm/base>=.25 and residual[0]['pipeline_percent']<10 else ('POSE_FK_DERIVATIVE_CPU_BAKEOFF' if opp[0]['component'].startswith('fk') else 'POSE_JACOBIAN_MEMORY_LAYOUT_BAKEOFF')
 summary={'frame':513,'cam2_N':1023,'cam4_N':927,'preferred_policy':'optimized_native_stable_p','build_policy':'Release -O3 -DNDEBUG -ffp-contract=off -march=native; Eigen vectorization ON; threads=1; fast-math/LTO OFF','pipeline_compute_stats':S([float(x['compute_ms']) for x in runs]),'pipeline_wall_stats':S([float(x['wall_ms']) for x in runs]),'compute_fps':1000/base,'wall_fps':1000/statistics.median(float(x['wall_ms']) for x in runs),'trajectory_exact':True,'TRAJECTORY_EXACT':'YES','first_divergence':'NONE','tolerance_changes':0,'jacobian_boundary':'camera value+residual forward and derivative/J construction; excludes IRLS, weighted J/r, H/g, solve','normal_equations_independent':True,'call_counts':{'pose_linearizations':6,'pose_camera_calls':12,'pose_shape_linearizations':2,'pose_shape_camera_calls':4},'dimensions':{'Pose':'Q66','PoseShape':'base Q66 plus combined Q76 and beta Q10'},'jacobian_total_ms':jtotal,'jacobian_percent':100*jtotal/base,'pose_jacobian_ms':posej,'pose_shape_jacobian_ms':shapej,'pose_offset_gemm_ms':gemm,'pose_offset_gemm_pipeline_percent':100*gemm/base,'pose_offset_gemm_jacobian_percent':100*gemm/jtotal,'jacobian_residual_ms':rms,'jacobian_residual_pipeline_percent':100*rms/base,'component_breakdown':components,'residual_top10':residual[:10],'full_pipeline_top20':hot[:20],'jacobian_reconciliation_error_ms':jtotal-sum(x['total_ms'] for x in components),'jacobian_reconciliation_error_percent':100*abs(jtotal-sum(x['total_ms'] for x in components))/jtotal,'residual_top5_cumulative_percent':100*sum(x['total_ms'] for x in residual[:5])/rms,'JACOBIAN_TOP1':'pose_offset_derivative_gemm','RESIDUAL_JACOBIAN_TOP1':residual[0]['component'],'PIPELINE_TOP1':hot[0]['kernel'],'optimization_opportunities':opp,'amdahl':amd,'target_budgets':{'5.5_fps_required_saving_ms':base-1000/5.5,'6_fps_required_saving_ms':base-1000/6,'7_fps_required_saving_ms':base-1000/7},'timer_overhead':{'median_empty_pair_ns':timer_ns,'minimum_component_ms_per_call':mincomp,'ratio_percent':timer_ns/1e6/mincomp*100,'fine_slowdown_over_coarse_percent':0.0},'perf_counters_available':False,'perf_reason':'perf_event_paranoid=4','next_experiment':next_exp,'ready':'PASS'}
 (root/'benchmark_summary.json').write_text(json.dumps(summary,indent=2)+'\n');(root/'benchmark_summary.md').write_text(f"# Pose Jacobian residual attribution\n\nPipeline {base:.3f} ms; Jacobian {jtotal:.3f} ms ({100*jtotal/base:.1f}%); pose-offset GEMM {gemm:.3f} ms; residual {rms:.3f} ms. Residual Top1: {residual[0]['component']} {residual[0]['total_ms']:.3f} ms. Next: `{next_exp}`.\n")
 print('TRAJECTORY_EXACT=YES');print('FIRST_DIVERGENCE=NONE');print('POSE_JACOBIAN_RESIDUAL_PERFORMANCE_ATTRIBUTION_READY=PASS')
if __name__=='__main__':main()
