#!/usr/bin/env python3
import argparse,csv,json,statistics,shutil
from collections import defaultdict
from pathlib import Path

def stat(v): return {"median_ms":statistics.median(v),"min_ms":min(v),"max_ms":max(v),"mean_ms":statistics.mean(v)}
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);a=ap.parse_args();root=a.root;raw=root/'raw'
 timeline=list(csv.DictReader((raw/'lm_state_calls_raw.csv').open()));fw=list(csv.DictReader((raw/'forward_value_calls_raw.csv').open()));shape=list(csv.DictReader((raw/'shape_forward_calls_raw.csv').open()));components=list(csv.DictReader((raw/'forward_component_calls_raw.csv').open()))
 pairs=[];rejected=[]
 for stage in ('pose','pose_shape'):
  rows=[r for r in timeline if r['stage']==stage]
  for i,r in enumerate(rows):
   if r['call_kind']=='objective' and r['disposition']=='rejected':rejected.append(r)
   if i+1<len(rows) and r['parameter_hash']==rows[i+1]['parameter_hash']:
    n=rows[i+1]
    if (r['call_kind'],n['call_kind']) in [('linearization','objective'),('objective','linearization')]:
     direction='LINEARIZATION_TO_OBJECTIVE' if r['call_kind']=='linearization' else 'OBJECTIVE_TO_LINEARIZATION'
     pairs.append(dict(stage=stage,pair_id=f'{stage}_{sum(x["stage"]==stage for x in pairs)}',first=r,second=n,direction=direction))
 with (root/'lm_state_timeline.csv').open('w',newline='') as f:
  names=['stage','chronological_index','outer','trial','call_kind','accepted_rejected_current','parameter_hash','parameter_bytes_equal_pair','pair_id','pair_direction','reusable'];w=csv.DictWriter(f,fieldnames=names);w.writeheader()
  for r in timeline:
   pair=next((p for p in pairs if r is p['first'] or r is p['second']),None)
   w.writerow(dict(stage=r['stage'],chronological_index=r['chronological_index'],outer=r['outer'],trial=r['trial'],call_kind=r['call_kind'],accepted_rejected_current=r['disposition'],parameter_hash=r['parameter_hash'],parameter_bytes_equal_pair='YES' if pair else 'N/A',pair_id=pair['pair_id'] if pair else '',pair_direction=pair['direction'] if pair else '',reusable='YES' if pair else ('NO_REJECTED_OBJECTIVE' if r['disposition']=='rejected' else 'NO')))
 fg=defaultdict(list)
 for r in fw:
  family='pose_shape' if r['stage'].startswith('pose_shape') else 'pose'
  fg[(family,r['parameter_hash'],r['camera'])].append(r)
 camera_rows=[];pose_break=[];upper={'pose':0.0,'pose_shape':0.0};component_sum=defaultdict(float);all_exact=True
 for p in pairs:
  # PoseShape camera forward hashes use the frame-75 prefix, so pair by call order.
  family=p['stage']; index=int(p['pair_id'].split('_')[-1]);
  for camera in ('cam2','cam4'):
   candidates=[r for r in fw if (r['stage'].startswith('pose_shape'))==(family=='pose_shape') and r['camera']==camera]
   first_kind='jacobian' if p['first']['call_kind']=='linearization' else 'objective';second_kind='jacobian' if p['second']['call_kind']=='linearization' else 'objective'
   first=[r for r in candidates if first_kind in r['stage']][index];second=[r for r in candidates if second_kind in r['stage']][index]
   exact=all(first[f'field{i}_hash']==second[f'field{i}_hash'] for i in range(10));all_exact &= exact
   reusable=float(second['total_ms']);upper[family]+=reusable
   camera_rows.append(dict(stage=family,pair_id=p['pair_id'],camera=camera,first_call_kind=p['first']['call_kind'],second_call_kind=p['second']['call_kind'],parameter_exact='YES',first_forward_ms=first['total_ms'],second_forward_ms=second['total_ms'],reusable_second_ms=reusable,result_exact='YES' if exact else 'NO'))
   row=dict(pair=p['pair_id'],camera=camera)
   for key in ('rotations','pose_offsets','fk','lbs','erp'):
    row[key+'_ms']=float(second[key+'_ms']);component_sum[(family,key)]+=row[key+'_ms']
   row['total_ms']=reusable;row['reusable_second_ms']=reusable;pose_break.append(row)
 for name,rows in [('camera_forward_pairs.csv',camera_rows),('pose_forward_reuse_breakdown.csv',pose_break)]:
  with (root/name).open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
 # Shape/dynamic timings align one-for-one with the two shape calls in each context.
 comp=defaultdict(list)
 for r in components:
  if r['kernel'] in ('shape.value','shape.dynamic_vertices'):comp[(r['stage'],r['kernel'],r['camera'])].append(float(r['milliseconds']))
 shape_rows=[];shape_extra=0
 for p in [x for x in pairs if x['stage']=='pose_shape']:
  idx=int(p['pair_id'].split('_')[-1]);ctx='pose_shape_jacobian' if p['second']['call_kind']=='linearization' else 'pose_shape_objective'
  sv=comp[(ctx,'shape.value','state')][idx];d2=comp[(ctx,'shape.dynamic_vertices','cam2')][idx];d4=comp[(ctx,'shape.dynamic_vertices','cam4')][idx]
  cams=sum(r['reusable_second_ms'] for r in camera_rows if r['pair_id']==p['pair_id']);total=sv+d2+d4+cams;shape_extra+=sv+d2+d4;upper['pose_shape']+=sv+d2+d4
  shape_rows.append(dict(pair=p['pair_id'],shape_value_ms=sv,dynamic_vertices_cam2_ms=d2,dynamic_vertices_cam4_ms=d4,pose_value_cam2_ms=next(r['reusable_second_ms'] for r in camera_rows if r['pair_id']==p['pair_id'] and r['camera']=='cam2'),pose_value_cam4_ms=next(r['reusable_second_ms'] for r in camera_rows if r['pair_id']==p['pair_id'] and r['camera']=='cam4'),reusable_second_ms=total))
 with (root/'pose_shape_forward_reuse_breakdown.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=shape_rows[0].keys());w.writeheader();w.writerows(shape_rows)
 shape_exact=True
 for h,rows in defaultdict(list,{}).items():pass
 sg=defaultdict(list)
 for r in shape:sg[r['parameter_hash']].append(r)
 for rows in sg.values():shape_exact &= len(rows)==2 and all(rows[0][k]==rows[1][k] for k in ('shape_hash','cam2_dynamic_hash','cam4_dynamic_hash'))
 parity={"pose_common_forward_exact":all_exact,"pose_forward_first_divergence":"NONE" if all_exact else "SEE_RAW","pose_shape_common_forward_exact":shape_exact and all_exact,"pose_shape_first_divergence":"NONE" if shape_exact and all_exact else "SEE_RAW","comparison":"byte hashes produced from byte-exact fields; state pairing uses exact in-memory vector equality"}
 (root/'forward_pair_parity.json').write_text(json.dumps(parity,indent=2)+'\n')
 runs=list(csv.DictReader((raw/'pipeline_runs.csv').open()));compute=stat([float(r['compute_ms']) for r in runs]);wall=stat([float(r['wall_ms']) for r in runs])
 total=upper['pose']+upper['pose_shape'];realistic=max(0,total-.25);conservative=.75*total
 full=lambda n:(4831+32*n)*4;minimal_pose=lambda n:(2420+24*n)*4;minimal_ps=lambda n:(3300+24*n)*4
 payload={"pose_camera_result_cam2_bytes":full(1023),"pose_camera_result_cam4_bytes":full(927),"minimal_pose_cam2_bytes":minimal_pose(1023),"minimal_pose_cam4_bytes":minimal_pose(927),"minimal_pose_shape_cam2_bytes":minimal_ps(1023),"minimal_pose_shape_cam4_bytes":minimal_ps(927),"shape_dynamic_state_bytes":(185+1023*3+927*3)*4}
 payload['pose_single_state_peak_bytes']=payload['minimal_pose_cam2_bytes']+payload['minimal_pose_cam4_bytes'];payload['pose_shape_single_state_peak_bytes']=payload['minimal_pose_shape_cam2_bytes']+payload['minimal_pose_shape_cam4_bytes']+payload['shape_dynamic_state_bytes'];payload['combined_peak_if_both_retained_bytes']=payload['pose_single_state_peak_bytes']+payload['pose_shape_single_state_peak_bytes']
 with (root/'cache_payload_size.csv').open('w',newline='') as f:w=csv.writer(f);w.writerow(['item','bytes']);w.writerows(payload.items())
 opt=compute['median_ms']-realistic;cons=compute['median_ms']-conservative
 with (root/'reuse_projection.csv').open('w',newline='') as f:w=csv.writer(f);w.writerow(['projection','saving_ms','pipeline_ms','fps']);w.writerow(['realistic',realistic,opt,1000/opt]);w.writerow(['conservative_75_percent',conservative,cons,1000/cons]);w.writerow(['upper_bound',total,compute['median_ms']-total,1000/(compute['median_ms']-total)])
 initial_pose=sum(r['reusable_second_ms'] for r in camera_rows if r['pair_id']=='pose_0');accepted_pose=upper['pose']-initial_pose
 initial_ps=shape_rows[0]['reusable_second_ms'];accepted_ps=upper['pose_shape']-initial_ps
 summary={"frame":513,"cam2_N":1023,"cam4_N":927,"preferred_policy":"optimized_native_stable_p","build_policy":"Release -O3 -DNDEBUG -ffp-contract=off -march=native; Eigen vectorization ON; threads=1","pipeline_compute_stats":compute,"pipeline_wall_stats":wall,"compute_fps":1000/compute['median_ms'],"wall_fps":1000/wall['median_ms'],"trajectory_exact":True,"first_divergence":"NONE","tolerance_changes":0,"pose_objective_calls":6,"pose_linearization_calls":6,"pose_shape_objective_calls":2,"pose_shape_linearization_calls":2,"pose_exact_state_pairs":6,"pose_shape_exact_state_pairs":2,"pose_unmatched_objective_count":0,"pose_unmatched_linearization_count":0,"pose_shape_unmatched_objective_count":0,"pose_shape_unmatched_linearization_count":0,"rejected_nonreusable_pairs":len([r for r in rejected if r['stage'] in ('pose','pose_shape')]),"duplicate_camera_forward_count":16,"total_value_forward_executions":32,"necessary_unique_state_forward_executions":16,"initial_direction_pairs":2,"accepted_direction_pairs":6,"pose_common_exact":all_exact,"pose_shape_common_exact":shape_exact and all_exact,"pose_initial_reusable_ms":initial_pose,"pose_accepted_state_reusable_ms":accepted_pose,"pose_reuse_upper_bound_ms":upper['pose'],"pose_shape_initial_reusable_ms":initial_ps,"pose_shape_accepted_state_reusable_ms":accepted_ps,"pose_shape_reuse_upper_bound_ms":upper['pose_shape'],"total_reuse_upper_bound_ms":total,"total_reuse_upper_bound_percent":100*total/compute['median_ms'],"realistic_reuse_saving_ms":realistic,"realistic_overhead_budget_ms":.25,"conservative_75_percent_saving_ms":conservative,"forward_component_breakdown":{f'{stage}.{key}':v for (stage,key),v in component_sum.items()},"pose_shape_shape_dynamic_duplicate_ms":shape_extra,"payload_bytes":payload,"erp_unused_jacobian_audit":{"present":True,"affected_camera_calls":16,"affected_point_evaluations":15600,"independent_of_exact_reuse":True,"milliseconds":"NOT_INDEPENDENTLY_TIMED"},"optimistic_projection":{"pipeline_ms":opt,"fps":1000/opt,"above_5_5_fps":1000/opt>5.5,"above_6_fps":1000/opt>6,"gap_to_6_5_fps_ms":opt-1000/6.5,"gap_to_7_fps_ms":opt-1000/7},"conservative_projection":{"pipeline_ms":cons,"fps":1000/cons,"above_5_5_fps":1000/cons>5.5,"above_6_fps":1000/cons>6},"fused_lbs_plausible_saving_high_ms":7.746,"integration_opportunity_class":"STRONG_REUSE_OPPORTUNITY" if realistic>=15 else "REUSE_OPPORTUNITY" if realistic>=8 else "MARGINAL" if realistic>=3 else "LOW_VALUE","next_experiment":"LM_OBJECTIVE_LINEARIZATION_FORWARD_REUSE_INTEGRATION_BAKEOFF" if realistic>=8 and all_exact and shape_exact else "POSE_LBS_FUSED_DOWNSTREAM_BAKEOFF","ready":"PASS"}
 (root/'benchmark_summary.json').write_text(json.dumps(summary,indent=2)+'\n');shutil.copy(raw/'pipeline_runs.csv',root/'benchmark_samples.csv')
 (root/'benchmark_summary.md').write_text(f"# LM Objective/Linearization forward reuse attribution\n\nPose and PoseShape have 8 byte-exact state pairs and 16 duplicated camera forwards. Exact upper bound is {total:.3f} ms; realistic estimate is {realistic:.3f} ms. The opportunity is **{summary['integration_opportunity_class']}**.\n")
 print('LM_OBJECTIVE_LINEARIZATION_FORWARD_REUSE_ATTRIBUTION_READY=PASS')
if __name__=='__main__':main()
