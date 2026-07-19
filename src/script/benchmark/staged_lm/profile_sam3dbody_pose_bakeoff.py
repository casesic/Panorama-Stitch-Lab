#!/usr/bin/env python3
"""Benchmark-only pose initialization bake-off using official SMPL-X caches."""
from __future__ import annotations
import argparse,contextlib,io,json,sys,time
from pathlib import Path
from typing import Any
import numpy as np
import torch
if __package__ in (None,""):sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
from lib_pano.human_fitting.staged_lm.stages import build_stage1_definition,build_pose_stage_definition,build_joint_stage_definition,run_stage
from lib_pano.human_fitting.staged_lm.stage2_shared_cache import build_stage2_shared_geometry_cache
from script.human_fitting.staged_lm.app import build_context
from script.human_fitting.staged_lm.cli import build_argument_parser

CANDIDATES=("CURRENT_HYBRID_BASELINE","SAM_CONVERTED_POSE_CAM2","SAM_CONVERTED_POSE_CAM4")

def parser():
 p=build_argument_parser()
 for action in p._actions:
  if action.dest=="output_dir":action.required=False;action.default="/tmp/sam_pose_bakeoff"
 p.add_argument('--official_init_dir',required=True);p.add_argument('--frame_sets',default='513;514;513,514');p.add_argument('--output_json',required=True)
 return p

def _stage(outcome):
 r=outcome.lm_result;c=r.counters;t=r.profile_timing or {}
 return {"status":r.status,"wall_ms":r.wall_time_ms,"accepted_steps":c.accepted_steps,"rejected_trials":c.rejected_trials,
  "linearization_count":c.linearization_count,"normal_equation_count":c.normal_equation_count,"solve_attempt_count":c.solve_attempt_count,
  "objective_calls":next((int(t[k]) for k in t if k.endswith('objective_calls')),None),"final_lambda":r.final_lambda,
  "objective":r.objective,"metrics":r.rms_metrics,"schur_dense_fallback_calls":t.get('schur_dense_fallback_calls',0),
  "structured_to_dense_calls":t.get('structured_to_dense_calls',0),"profile_timing":t}

def _load_pose(root:Path,cam:str,frames:list[int],device,dtype):
 values=[]
 for frame in frames:
  with np.load(root/cam/f'{frame:05d}.npz',allow_pickle=False) as z:
   pose=z['body_pose'];
   if pose.shape!=(63,) or pose.dtype.kind not in 'fc' or not np.isfinite(pose).all():raise ValueError(f'INVALID_CONVERTED_POSE {frame} {cam} {pose.shape}')
   values.append(torch.from_numpy(pose.copy()).to(device=device,dtype=dtype))
 return values

def _run(args,frames,candidate):
 local=argparse.Namespace(**vars(args));local.frames=','.join(map(str,frames));local.profile_timing=True;local.progress_every_accepted=0
 context=build_context(local)
 if candidate!="CURRENT_HYBRID_BASELINE":
  cam='cam2' if candidate.endswith('CAM2') else 'cam4';context.body_pose_anchors=_load_pose(Path(args.official_init_dir),cam,frames,context.device,context.dtype)
 if not all(torch.isfinite(x).all() and x.shape==(63,) for x in context.body_pose_anchors):raise RuntimeError('INVALID_BODY_POSE_ANCHOR')
 stage1=build_stage1_definition(context);initial_objective=float(stage1.lm_true_objective_fn(stage1.initial_lm_parameters));initial_metrics=stage1.raw_metrics_fn(stage1.initial_lm_parameters)
 started=time.perf_counter();s1=run_stage(stage1,context);shared=build_stage2_shared_geometry_cache(context)
 pose_def,_=build_pose_stage_definition(context,s1,shared);pose=run_stage(pose_def,context)
 joint_def,_=build_joint_stage_definition(context,pose,shared);joint=run_stage(joint_def,context)
 if context.device.type=='cuda':torch.cuda.synchronize(context.device)
 wall=(time.perf_counter()-started)*1000
 stages={"stage1":_stage(s1),"pose":_stage(pose),"joint":_stage(joint)}
 total_ne=sum(x['normal_equation_count'] for x in stages.values());total_obj=sum(x['objective_calls'] or 0 for x in stages.values())
 return {"candidate":candidate,"initial":{"objective":initial_objective,"metrics":initial_metrics,"finite":True},"stages":stages,
  "fitting_segment_ms":wall,"ms_per_frame":wall/len(frames),"fps":1000*len(frames)/wall,"total_ne_calls":total_ne,"total_objective_calls":total_obj}

def _quality(candidate,baseline):
 c=candidate['stages']['joint'];b=baseline['stages']['joint'];od=abs(c['objective']-b['objective']);orel=od/max(abs(b['objective']),1e-16)
 keys=('rms','median','p95');md=max(abs(float(c['metrics'][k])-float(b['metrics'][k])) for k in keys)
 return {"objective_relative":orel,"metrics_max_abs":md,"pass":orel<=1e-5 and md<=1e-3 and c['status']==b['status']}

def main():
 a=parser().parse_args();result={"baseline_definition":"CURRENT_HYBRID_BASELINE: HMR orient, bbox-ray translation, averaged HMR body pose/beta, HMR hands/scale","frame_sets":[]}
 for spec in a.frame_sets.split(';'):
  frames=[int(x) for x in spec.split(',')];item={"frames":frames,"candidates":{}}
  for candidate in CANDIDATES:
   print(f'[SAM_POSE_BAKEOFF] frames={frames} candidate={candidate}')
   with contextlib.redirect_stdout(io.StringIO()):run=_run(a,frames,candidate)
   item['candidates'][candidate]=run
  baseline=item['candidates'][CANDIDATES[0]]
  for candidate in CANDIDATES[1:]:
   run=item['candidates'][candidate];quality=_quality(run,baseline)
   stage1_obj=(baseline['stages']['stage1']['objective_calls'] or 0);pose_obj=(baseline['stages']['pose']['objective_calls'] or 0)
   gates={"quality":quality,"stage1_objective_reduction":(stage1_obj-(run['stages']['stage1']['objective_calls'] or 0))/max(stage1_obj,1),
    "pose_objective_reduction":(pose_obj-(run['stages']['pose']['objective_calls'] or 0))/max(pose_obj,1),
    "total_objective_reduction":(baseline['total_objective_calls']-run['total_objective_calls'])/max(baseline['total_objective_calls'],1),
    "wall_reduction":(baseline['fitting_segment_ms']-run['fitting_segment_ms'])/baseline['fitting_segment_ms']}
   gates['effective']=quality['pass'] and (gates['stage1_objective_reduction']>=.25 or gates['pose_objective_reduction']>=.25 or gates['total_objective_reduction']>=.20 or gates['wall_reduction']>=.10)
   run['gates']=gates
  item['recommendation']=next((c for c in CANDIDATES[1:] if item['candidates'][c]['gates']['effective']),'KEEP_CURRENT_HYBRID_BASELINE');result['frame_sets'].append(item)
 path=Path(a.output_json);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(result,indent=2));print(f'[SAVE] {path}')
if __name__=='__main__':main()
