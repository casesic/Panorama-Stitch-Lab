"""Real-data Joint FULL_BATCH Arrowhead and complete-LM shadow runner."""

from __future__ import annotations

import argparse,contextlib,io,json,statistics,sys,time
from pathlib import Path
from typing import Any
import torch

if __package__ in (None,""):sys.path.insert(0,str(Path(__file__).resolve().parents[3]))

from lib_pano.human_fitting.staged_lm.device import resolve_execution_device
from lib_pano.human_fitting.staged_lm.lm_solver import run_lm_optimization
from lib_pano.human_fitting.staged_lm.stage2_shared_cache import build_stage2_shared_geometry_cache
from lib_pano.human_fitting.staged_lm.stages import build_stage1_definition,build_pose_stage_definition,build_joint_stage_definition,run_stage
from script.benchmark.staged_lm.batch_common import environment_record
from script.benchmark.staged_lm.joint_batch_candidates import JointNEMode,build_joint_batch_cache
from script.benchmark.staged_lm.joint_shadow_common import arrowhead_errors,build_joint_shadow_linearization_common,diagnostic_gate,evaluate_joint_equations,joint_microkernel_errors,schur_errors
from script.benchmark.staged_lm.joint_shadow_lm import compare_joint_lm_results,run_joint_lm_independent_batch,run_joint_lm_lockstep
from script.human_fitting.staged_lm.app import build_context
from script.human_fitting.staged_lm.cli import build_argument_parser


def parser():
    p=build_argument_parser()
    for action in p._actions:
        if action.dest=="output_dir":action.required=False;action.default="/tmp/staged_lm_joint_shadow"
    p.add_argument("--frame_sweep",default="1,2,4,8")
    p.add_argument("--candidate_modes",default="batch_forward_serial_accumulation,serial_forward_batch_gemm,full_batch_production_reduction,full_batch_tensor_beta_reduction")
    p.add_argument("--pose_perturbation_scales",default="0")
    p.add_argument("--beta_latent_perturbation_scales",default="0")
    p.add_argument("--warmup",type=int,default=1);p.add_argument("--repeats",type=int,default=5)
    p.add_argument("--profile_isolated_ne",action="store_true");p.add_argument("--trace_dir")
    p.add_argument("--output_json",required=True);return p


def _csv(value):return [x.strip() for x in value.split(",") if x.strip()]
def _mode(value):return JointNEMode(value)


def _result(result):
    c=result.counters
    return {"status":result.status,"objective":result.objective,"parameters":result.parameters.detach().cpu().tolist(),
            "accepted_steps":c.accepted_steps,"rejected_trials":c.rejected_trials,"normal_equation_count":c.normal_equation_count,
            "solve_attempt_count":c.solve_attempt_count,"schur_success_count":c.schur_success_count,
            "schur_dense_fallback_count":c.schur_dense_fallback_count,"structured_to_dense_count":c.structured_to_dense_count,
            "final_lambda":result.final_lambda,"wall_time_ms":result.wall_time_ms,"metrics":result.rms_metrics,
            "trial_history":result.trial_history,"weight_history":result.weight_history}


def _serial(stage,context,initial):
    from lib_pano.human_fitting.staged_lm.stage2_joint_jacobian import build_stage2_joint_analytic_linearization
    def linearize(x):
        return build_stage2_joint_analytic_linearization(
            stage.analytic_cache, x, context.objective_config.beta_abs_limit,
            context.beta_anchor, context.body_pose_anchors,
            context.scaling.pose_anchor_scale, context.scaling.shape_anchor_scale,
            is_irls=context.is_irls, charbonnier_eps=context.charbonnier_eps,
            aggregation_spec=stage.aggregation_spec,
            rep_scale=context.scaling.rep_scale,
        )
    return run_lm_optimization("STAGE2_JOINT_SERIAL_REFERENCE",linearize,
        stage.raw_metrics_fn,initial,stage.parameter_dimension,stage.lm_max_accepted_steps,context.solver_config.lm_max_trials,
        context.solver_config,True,0,stage.lm_true_objective_fn,validate_irls=False)


def _timed(fn,warmup,repeats,reference):
    for _ in range(warmup):fn()
    values=[];last=None
    for _ in range(repeats):
        if reference.is_cuda:torch.cuda.synchronize(reference.device)
        started=time.perf_counter();last=fn()
        if reference.is_cuda:torch.cuda.synchronize(reference.device)
        values.append((time.perf_counter()-started)*1000)
    ordered=sorted(values);return last,{"p50":statistics.median(values),"p90":ordered[min(len(ordered)-1,int(.9*(len(ordered)-1)))],"mean":statistics.mean(values),"min":min(values),"max":max(values)}


def _perturb(initial,pose_scale,beta_scale,beta_dim):
    value=initial.detach().clone();generator=torch.Generator(device=value.device);generator.manual_seed(20260715)
    if pose_scale:value[:-beta_dim]+=torch.randn(value[:-beta_dim].shape,generator=generator,device=value.device,dtype=value.dtype)*pose_scale
    if beta_scale:value[-beta_dim:]+=torch.randn(value[-beta_dim:].shape,generator=generator,device=value.device,dtype=value.dtype)*beta_scale
    return value


def _run_count(args,frames):
    local=argparse.Namespace(**vars(args));local.frames=",".join(frames);local.output_dir="/tmp/staged_lm_joint_shadow";local.stop_after_stage="stage2";local.profile_timing=True;local.progress_every_accepted=0
    context=build_context(local)
    with contextlib.redirect_stdout(io.StringIO()):
        s1=run_stage(build_stage1_definition(context),context);shared=build_stage2_shared_geometry_cache(context)
        pose,_=build_pose_stage_definition(context,s1,shared);pose_out=run_stage(pose,context)
        joint,_=build_joint_stage_definition(context,pose_out,shared)
    cache=joint.analytic_cache
    if joint.initial_lm_parameters.is_cuda:torch.cuda.synchronize(joint.initial_lm_parameters.device)
    started=time.perf_counter();batch=cache.batch_cache or build_joint_batch_cache(cache,context.scaling.pose_anchor_scale)
    if joint.initial_lm_parameters.is_cuda:torch.cuda.synchronize(joint.initial_lm_parameters.device)
    preparation=batch.setup_ms
    output={"T":len(frames),"G":2*len(frames),"batch_preparation":{"setup_ms":preparation,"persistent_bytes":batch.persistent_bytes,
        "padding_ratio":batch.layout.padding_ratio,"sections":{"layout":batch.padding_layout_ms,"base":batch.padding_base_vertices_ms,"beta_dirs":batch.padding_beta_dirs_ms,"posedirs":batch.padding_posedirs_ms,"weights":batch.padding_weights_ms,"hands_scale":batch.stack_hands_scale_ms,"static":batch.static_layout_ms}},"perturbations":{}}
    modes=[_mode(x) for x in _csv(args.candidate_modes)]
    for ps in map(float,_csv(args.pose_perturbation_scales)):
      for bs in map(float,_csv(args.beta_latent_perturbation_scales)):
        key=f"pose={ps}|beta={bs}";initial=_perturb(joint.initial_lm_parameters,ps,bs,batch.layout.beta_dim)
        with contextlib.redirect_stdout(io.StringIO()):serial,serial_t=_timed(lambda:_serial(joint,context,initial),args.warmup,args.repeats,initial)
        item={"serial":_result(serial),"serial_timing":serial_t,"modes":{}}
        for mode in modes:
            common=build_joint_shadow_linearization_common(joint,batch,context,initial);scales=common["row_scales_by_group"]
            ref=evaluate_joint_equations(JointNEMode.SERIAL_REFERENCE,joint,batch,initial,scales,context)
            cache.batched_shape_state_calls=cache.batched_full_analytic_calls=0
            cand=evaluate_joint_equations(mode,joint,batch,initial,scales,context)
            per_ne_shape_calls=cache.batched_shape_state_calls
            per_ne_analytic_calls=cache.batched_full_analytic_calls
            errors=arrowhead_errors(cand,ref);damping=float(ref.structured_system.max_diagonal())*1e-3
            schur=schur_errors(cand,ref,damping);micro=diagnostic_gate(errors,schur)
            micro_j=joint_microkernel_errors(joint,batch,initial,context.objective_config.beta_abs_limit)
            _,serial_ne_t=_timed(lambda:evaluate_joint_equations(JointNEMode.SERIAL_REFERENCE,joint,batch,initial,scales,context),args.warmup,max(1,args.repeats),initial)
            _,batch_ne_t=_timed(lambda:evaluate_joint_equations(mode,joint,batch,initial,scales,context),args.warmup,max(1,args.repeats),initial)
            with contextlib.redirect_stdout(io.StringIO()):
                lock=run_joint_lm_lockstep(joint,batch,context,mode=mode,initial_parameters=initial)
                if initial.is_cuda:torch.cuda.reset_peak_memory_stats(initial.device)
                candidate,timing=_timed(lambda:run_joint_lm_independent_batch(joint,batch,context,mode=mode,initial_parameters=initial),args.warmup,args.repeats,initial)
            comparison=compare_joint_lm_results(serial,candidate,context.objective_config.beta_abs_limit,batch.layout.beta_dim)
            gate=(lock.acceptance_mismatch_count==0 and comparison["status_match"] and comparison["accepted_match"] and comparison["rejected_match"] and comparison["trial_count_match"] and comparison["schur_match"] and comparison["final_parameter_max_abs"]<=(5e-5 if initial.is_cuda else 1e-5) and comparison["physical_beta_max_abs"]<=(5e-6 if initial.is_cuda else 2e-6) and comparison["objective_relative"]<=1e-6 and comparison["metrics_max_abs"]<=1e-3)
            memory={"persistent_bytes":batch.persistent_bytes}
            if initial.is_cuda:memory.update(max_memory_allocated=torch.cuda.max_memory_allocated(initial.device),max_memory_reserved=torch.cuda.max_memory_reserved(initial.device))
            item["modes"][mode.value]={"microkernel":{**errors,**micro_j},"schur":schur,"diagnostic_gate":micro,"solver_gate":gate,
                "lockstep":lock.to_json_dict(),"independent_lm":_result(candidate),"comparison":comparison,"timing":timing,
                "speedup":serial_t["p50"]/timing["p50"],"equation_timing":{"serial":serial_ne_t,"candidate":batch_ne_t,
                    "speedup":serial_ne_t["p50"]/batch_ne_t["p50"],"break_even_ne_calls":preparation/max(serial_ne_t["p50"]-batch_ne_t["p50"],1e-30)},
                "shape_state_calls_per_ne":per_ne_shape_calls,
                "full_analytic_calls_per_ne":per_ne_analytic_calls,"memory":memory}
        output["perturbations"][key]=item
    return output


def main(argv=None):
    args=parser().parse_args(argv);frames=_csv(args.frames);device=resolve_execution_device(args.device)
    output={"environment":environment_record(device,torch.float32),"input":{"frames":frames,"objective":args.reproj_objective},"frame_sets":{}}
    for count in map(int,_csv(args.frame_sweep)):
        if count>len(frames):output["frame_sets"][str(count)]={"status":"SKIPPED_MISSING_REAL_DATA"};continue
        print(f"[JOINT_SHADOW] device={args.device} T={count}");output["frame_sets"][str(count)]=_run_count(args,frames[:count])
    full=[]
    for fs in output["frame_sets"].values():
      for perturb in fs.get("perturbations",{}).values():
       if "full_batch_production_reduction" in perturb.get("modes",{}):full.append(perturb["modes"]["full_batch_production_reduction"])
    solver_ok=bool(full) and all(x["solver_gate"] for x in full)
    perf_ok=bool(full) and all(x["speedup"]>=1.3 for x in full)
    output["classification"]="EXPECTED_FP32_OPERATION_ORDER_DIFFERENCE" if solver_ok else "UNRESOLVED_NUMERICAL_DIFFERENCE"
    output["integration_readiness"]="READY_TO_INTEGRATE_FULL_BATCH_NE_SERIAL_WEIGHTS" if solver_ok and perf_ok else "DO_NOT_INTEGRATE"
    output["recommendation"]=output["integration_readiness"]
    target=Path(args.output_json);target.parent.mkdir(parents=True,exist_ok=True);target.write_text(json.dumps(output,indent=2),encoding="utf-8");print(f"[SAVE] {target}");return 0

if __name__=="__main__":raise SystemExit(main())
