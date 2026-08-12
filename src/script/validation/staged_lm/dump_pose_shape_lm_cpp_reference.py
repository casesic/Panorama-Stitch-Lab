"""Dump canonical frame-513 Pose/Shape structured LM trajectory."""
from __future__ import annotations
import argparse,json,sys,time
from pathlib import Path
import numpy as np,torch
if __package__ in (None,""):sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
from lib_pano.human_fitting.staged_lm.lm.convergence import *
from lib_pano.human_fitting.staged_lm.lm.equations import _evaluate_linearization,compute_scaled_gradient,normal_equations_max_diagonal,solve_normal_equations
from lib_pano.human_fitting.staged_lm.lm.trial import evaluate_trial_reduction
from lib_pano.human_fitting.staged_lm.pose_shape.shape import physical_beta_from_latent
from lib_pano.human_fitting.staged_lm.stage2_shared_cache import build_stage2_shared_geometry_cache
from lib_pano.human_fitting.staged_lm.stage_definitions.rigid import build_stage1_definition
from lib_pano.human_fitting.staged_lm.stage_definitions.pose import build_pose_stage_definition
from lib_pano.human_fitting.staged_lm.stage_definitions.pose_shape import build_pose_shape_stage_definition
from lib_pano.human_fitting.staged_lm.stage_runtime import run_stage
from script.human_fitting.staged_lm.app import build_context
def n(x):return x.detach().cpu().contiguous().numpy()
def trace(stage,ctx):
 x=stage.initial_lm_parameters.detach().clone();arrays={"initial_parameters":n(x)};lins=[];trials=[];oc=lc=sc=rej=acc=0;obj_ms=lin_ms=sol_ms=0.
 def objective(q):
  nonlocal oc,obj_ms;s=time.perf_counter();v=float(stage.lm_true_objective_fn(q));obj_ms+=(time.perf_counter()-s)*1000;oc+=1;return v
 def linearize(q,outer,obj,lam):
  nonlocal lc,lin_ms;s=time.perf_counter();model=stage.linearization_builder(q,stage.aggregation_spec);eq,_=_evaluate_linearization(model,q);lin_ms+=(time.perf_counter()-s)*1000;sys=eq.structured_system;i=len(lins);arrays.update({f"linearization_{i}_parameters":n(q),f"linearization_{i}_A":n(sys.frame_hessians[0]),f"linearization_{i}_B":n(sys.frame_beta_cross[0]),f"linearization_{i}_C":n(sys.beta_hessian),f"linearization_{i}_g_frame":n(sys.frame_gradients[0]),f"linearization_{i}_g_beta":n(sys.beta_gradient)});lins.append({"index":i,"outer":outer,"objective":obj,"lambda_before":lam,"max_diagonal":normal_equations_max_diagonal(eq),"scaled_gradient":compute_scaled_gradient(eq)});lc+=1;return eq
 eq=linearize(x,-1,float('nan'),None);damping=initialise_damping_from_max_diagonal(normal_equations_max_diagonal(eq));cur=objective(x);initial_obj=cur;initial_lambda=damping;lins[0].update(objective=cur,lambda_before=damping);status="RUNNING";conv="NONE"
 for outer in range(stage.lm_max_accepted_steps):
  before=cur;accepted=False;last=0.
  for trial in range(ctx.solver_config.lm_max_trials):
   lam=damping;s=time.perf_counter();sr=solve_normal_equations(eq,damping);sol_ms+=(time.perf_counter()-s)*1000;sc+=1;record={"index":len(trials),"outer":outer,"trial":trial,"lambda_before":lam,"solve_success":sr.success,"solve_backend":sr.backend,"schur_primary_failed":sr.schur_failed}
   if not sr.success:damping=increase_damping_after_rejection(damping);rej+=1;record.update(accepted=False,lambda_after=damping,lambda_branch="REJECT_X10");trials.append(record);continue
   delta=sr.delta.double();df=delta.float();candidate=(x+df).detach();co=objective(candidate);pred,actual,rho,ok=evaluate_trial_reduction(eq,eq.gradient,delta,before,co);idx=len(trials);arrays.update({f"trial_{idx}_delta_double":n(delta),f"trial_{idx}_delta_float":n(df),f"trial_{idx}_candidate":n(candidate)});last=float(delta.norm());branch="ACCEPT_DIV3" if ok and rho>.75 else "ACCEPT_MUL3" if ok and rho<.25 else "ACCEPT_KEEP" if ok else "REJECT_X10";record.update(candidate_objective=co,predicted_reduction=pred,actual_reduction=actual,rho=rho,accepted=ok,delta_norm=last,lambda_branch=branch)
   if ok:x=candidate;cur=co;acc+=1;accepted=True;damping=update_damping_after_acceptance(damping,rho)
   else:damping=increase_damping_after_rejection(damping);rej+=1
   record["lambda_after"]=damping;trials.append(record)
   if ok:break
  if not accepted:status=classify_no_accepted_trial(acc,compute_scaled_gradient(eq));conv="NO_ACCEPTED_TRIAL";break
  nxt=linearize(x,outer,cur,damping);rel=(before-cur)/max(abs(before),1e-16);c=check_convergence(compute_scaled_gradient(nxt),last,rel,ctx.solver_config);lins[-1].update(relative_objective_change=rel,step_norm=last,convergence=c or "NONE")
  if c:status=c;conv=c;break
  eq=nxt
 if status=="RUNNING":status="MAX_ACCEPTED_STEPS";conv=status
 beta,_=physical_beta_from_latent(x[-10:],ctx.objective_config.beta_abs_limit);arrays.update(final_parameters=n(x),final_frame_parameters=n(x[:75]),final_beta_latent=n(x[-10:]),final_physical_beta=n(beta),final_objective=np.asarray(cur,dtype=np.float64));trajectory={"schema_version":"pose_shape_lm_cpp_reference_v1","initial_objective":initial_obj,"initial_lambda":initial_lambda,"linearizations":lins,"trials":trials,"final_objective":cur,"final_lambda":damping,"status":status,"convergence_branch":conv,"accepted_steps":acc,"rejected_trials":rej,"outer_iterations":len({q['outer'] for q in trials}),"objective_call_count":oc,"linearization_call_count":lc,"structured_solve_calls":sc,"dense_fallback_calls":sum(q['schur_primary_failed'] for q in trials),"total_trials":len(trials),"timing_ms":{"objective":obj_ms,"linearization":lin_ms,"solve":sol_ms}};return trajectory,arrays
def main():
 p=argparse.ArgumentParser();p.add_argument('--frame',type=int,default=513);p.add_argument('--hmr_dir',required=True);p.add_argument('--sam3d_dense_dir',required=True);p.add_argument('--bbox_dir',required=True);p.add_argument('--output_dir',type=Path,required=True);p.add_argument('--device',default='cpu');p.add_argument('--image_width',type=int,default=5760);p.add_argument('--image_height',type=int,default=2880);p.add_argument('--reproj_objective',default='charbonnier_irls');a=p.parse_args();a.frames=str(a.frame);a.profile_timing=False;a.validate_device_contract=False;a.stop_after_stage='stage2_pose';ctx=build_context(a);s1=run_stage(build_stage1_definition(ctx),ctx);shared=build_stage2_shared_geometry_cache(ctx);ps,_=build_pose_stage_definition(ctx,s1,shared);po=run_stage(ps,ctx);stage,_=build_pose_shape_stage_definition(ctx,po,shared);traj,arrays=trace(stage,ctx);prod=run_stage(stage,ctx).lm_result;parity={"parameters_exact":bool(torch.equal(torch.from_numpy(arrays['final_parameters']),prod.parameters.cpu())),"objective_exact":traj['final_objective']==prod.objective,"lambda_exact":traj['final_lambda']==prod.final_lambda,"status_exact":traj['status']==prod.status,"accepted_exact":traj['accepted_steps']==prod.counters.accepted_steps,"rejected_exact":traj['rejected_trials']==prod.counters.rejected_trials};assert all(parity.values()),parity;a.output_dir.mkdir(parents=True,exist_ok=True);np.savez(a.output_dir/'pose_shape_lm_reference.npz',**arrays);(a.output_dir/'pose_shape_lm_reference_trajectory.json').write_text(json.dumps(traj,indent=2,allow_nan=True)+'\n');manifest={"schema_version":"pose_shape_lm_cpp_reference_v1","frame":a.frame,"total_dim":85,"frame_dim":75,"beta_dim":10,"max_accepted_steps":stage.lm_max_accepted_steps,"max_trials":ctx.solver_config.lm_max_trials,"solver_config":vars(ctx.solver_config),"reference_leakage":"NONE","fixture_source":"../../pose_shape/frame_000513/cpp_fixture","production_parity":parity};(a.output_dir/'pose_shape_lm_reference_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');(a.output_dir/'pose_shape_lm_reference_report.md').write_text('# Pose/Shape LM trajectory reference\n\nValidation transcription matches production.\n');print('POSE_SHAPE_LM_REFERENCE',traj['status'],len(traj['trials']))
if __name__=='__main__':main()
