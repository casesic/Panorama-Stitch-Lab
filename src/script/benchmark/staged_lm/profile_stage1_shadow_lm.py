"""Real-data Stage-1 FULL_BATCH normal-equation and LM shadow runner."""
from __future__ import annotations

import argparse, contextlib, io, json, statistics, sys, time
from pathlib import Path
from typing import Any
import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from lib_pano.human_fitting.staged_lm.lm.runner import run_lm_optimization
from lib_pano.human_fitting.staged_lm.stage_definitions.pose_shape import build_pose_shape_stage_definition
from lib_pano.human_fitting.staged_lm.stage_definitions.pose import build_pose_stage_definition
from lib_pano.human_fitting.staged_lm.stage_definitions.rigid import build_stage1_definition
from lib_pano.human_fitting.staged_lm.stage_runtime import run_stage
from lib_pano.human_fitting.staged_lm.stage2_shared_cache import build_stage2_shared_geometry_cache
from lib_pano.human_fitting.staged_lm.pose_shape import physical_beta_from_latent
from lib_pano.human_fitting.staged_lm.types import StageOutcome
from script.benchmark.staged_lm.batch_common import environment_record
from script.benchmark.staged_lm.stage1_batch_candidates import (
    build_stage1_batch_cache, gather_stage1_batch_parameters,
    batched_stage1_objective,
)
from script.benchmark.staged_lm.stage1_shadow_common import (
    Stage1NEMode, build_stage1_shadow_linearization_common, equation_errors,
    evaluate_stage1_equations, analytic_errors, step_errors,
)
from script.benchmark.staged_lm.stage1_shadow_lm import (
    compare_stage1_lm_results, run_stage1_lm_independent_batch,
    run_stage1_lm_lockstep,
)
from script.human_fitting.staged_lm.app import build_context
from script.human_fitting.staged_lm.cli import build_argument_parser


def parser():
    p=build_argument_parser()
    for action in p._actions:
        if action.dest=="output_dir":action.required=False;action.default="/tmp/staged_lm_stage1_shadow"
    p.add_argument("--frame_sweep",default="1,2,4,8")
    p.add_argument("--candidate_modes",default="batch_forward_serial_frame_gemm,serial_forward_batch_group_gemm,full_batch_group_gemm")
    p.add_argument("--rotation_perturbation_scales",default="0")
    p.add_argument("--translation_perturbation_scales",default="0")
    p.add_argument("--warmup",type=int,default=1);p.add_argument("--repeats",type=int,default=5)
    p.add_argument("--run_downstream_regression",action="store_true")
    p.add_argument("--profile_isolated_ne",action="store_true");p.add_argument("--trace_dir")
    p.add_argument("--output_json",required=True);return p


def _csv(value):return [x.strip() for x in value.split(",") if x.strip()]
def _mode(value):return Stage1NEMode(value)


def _result(result):
    c=result.counters
    return {"status":result.status,"objective":result.objective,"wall_time_ms":result.wall_time_ms,
            "accepted_steps":c.accepted_steps,"rejected_trials":c.rejected_trials,
            "linearization_count":c.linearization_count,
            "solve_attempt_count":c.solve_attempt_count,"solve_success_count":c.solve_success_count,
            "final_lambda":result.final_lambda,
            "metrics":result.rms_metrics,"trial_history":result.trial_history,"weight_history":result.weight_history,
            "profile_timing":result.profile_timing or {}}


def _serial(stage,context,initial):
    return run_lm_optimization(
        "STAGE1_SERIAL_REFERENCE",lambda x:stage.linearization_builder(x,stage.aggregation_spec),
        stage.raw_metrics_fn,initial,stage.parameter_dimension,stage.lm_max_accepted_steps,
        context.solver_config.lm_max_trials,context.solver_config,True,
        stage.lm_true_objective_fn)


def _timed(fn,warmup,repeats,reference):
    for _ in range(warmup):fn()
    values=[];last=None
    for _ in range(repeats):
        if reference.is_cuda:torch.cuda.synchronize(reference.device)
        started=time.perf_counter();last=fn()
        if reference.is_cuda:torch.cuda.synchronize(reference.device)
        values.append((time.perf_counter()-started)*1000)
    ordered=sorted(values)
    return last,{"p50":statistics.median(values),"p90":ordered[min(len(ordered)-1,int(.9*(len(ordered)-1)))],
                 "mean":statistics.mean(values),"min":min(values),"max":max(values)}


def _perturb(initial,rotation,translation):
    value=initial.detach().clone();generator=torch.Generator(device=value.device);generator.manual_seed(20260715)
    framed=value.reshape(-1,12)
    if rotation:
        noise=torch.randn((framed.shape[0],6),generator=generator,device=value.device,dtype=value.dtype)*rotation
        framed[:,0:3]+=noise[:,0:3];framed[:,6:9]+=noise[:,3:6]
    if translation:
        noise=torch.randn((framed.shape[0],6),generator=generator,device=value.device,dtype=value.dtype)*translation
        framed[:,3:6]+=noise[:,0:3];framed[:,9:12]+=noise[:,3:6]
    return value


def _profile_equation(fn,reference,trace_path):
    activities=[torch.profiler.ProfilerActivity.CPU]
    if reference.is_cuda:activities.append(torch.profiler.ProfilerActivity.CUDA)
    with torch.profiler.profile(activities=activities,
        schedule=torch.profiler.schedule(wait=1,warmup=1,active=1,repeat=1),
        record_shapes=True,profile_memory=True,with_stack=False) as profiler:
        for _ in range(3):fn();profiler.step()
    averages=profiler.key_averages();counts={item.key:int(item.count) for item in averages}
    path=Path(trace_path);path.parent.mkdir(parents=True,exist_ok=True)
    try:profiler.export_chrome_trace(str(path))
    except Exception:pass
    selected=("aten::einsum","aten::bmm","aten::matmul","aten::index_select","aten::gather",
              "aten::cat","aten::zeros","aten::copy_","aten::item","aten::arange")
    return {"operator_counts":{name:counts.get(name,0) for name in selected},
            "kernel_launch_count":sum(item.count for item in averages if "cuda" in item.key.lower()),
            "top_cpu":averages.table(sort_by="self_cpu_time_total",row_limit=30)}


def _downstream(context,stage,serial,candidate):
    output={}
    shared=build_stage2_shared_geometry_cache(context)
    outcomes=[]
    for label,result in (("serial",serial),("candidate",candidate)):
        s1=StageOutcome(result,stage.lm_true_objective_fn,stage)
        pose_def,_=build_pose_stage_definition(context,s1,shared);pose=run_stage(pose_def,context)
        joint_def,_=build_pose_shape_stage_definition(context,pose,shared);joint=run_stage(joint_def,context)
        outcomes.append((pose,joint))
        output[label]={"pose":_result(pose.lm_result),"joint":_result(joint.lm_result)}
    pc=compare_stage1_lm_results(outcomes[0][0].lm_result,outcomes[1][0].lm_result)
    jc=compare_stage1_lm_results(outcomes[0][1].lm_result,outcomes[1][1].lm_result)
    beta_dim=int(context.beta_dim);beta_abs_limit=context.objective_config.beta_abs_limit
    serial_beta,_=physical_beta_from_latent(outcomes[0][1].lm_result.parameters[-beta_dim:],beta_abs_limit)
    candidate_beta,_=physical_beta_from_latent(outcomes[1][1].lm_result.parameters[-beta_dim:],beta_abs_limit)
    beta_error=float((candidate_beta-serial_beta).abs().max())
    gate=(pc["first_divergent_trial"] is None and jc["first_divergent_trial"] is None
          and pc["accepted_match"] and jc["accepted_match"]
          and pc["objective_relative"]<=1e-6 and jc["objective_relative"]<=1e-6
          and pc["metrics_max_abs"]<=1e-4 and jc["metrics_max_abs"]<=1e-4
          and beta_error<=5e-6 and pc["solve_path_match"] and jc["solve_path_match"])
    output["comparison"]={"pose":pc,"joint":jc,"physical_beta_max_abs":beta_error,
        "downstream_first_divergence":pc["first_divergent_trial"] or jc["first_divergent_trial"]}
    output["gate_pass"]=gate
    return output


def _run_count(args,frames):
    local=argparse.Namespace(**vars(args));local.frames=",".join(frames);local.profile_timing=True
    context=build_context(local);stage=build_stage1_definition(context);cache=stage.analytic_cache
    batch=build_stage1_batch_cache(cache);modes=[_mode(x) for x in _csv(args.candidate_modes)]
    rotations=list(map(float,_csv(args.rotation_perturbation_scales)));translations=list(map(float,_csv(args.translation_perturbation_scales)))
    if len(rotations)!=len(translations):raise ValueError("PERTURBATION_SCALE_LENGTH_MISMATCH")
    cache.reset_timing()
    with contextlib.redirect_stdout(io.StringIO()): baseline=_serial(stage,context,stage.initial_lm_parameters)
    attribution={**cache.timing_dict(),"accepted_steps":baseline.counters.accepted_steps,
        "rejected_trials":baseline.counters.rejected_trials,"solve_attempt_count":baseline.counters.solve_attempt_count,
        "linearization_count":baseline.counters.linearization_count,
        "wall_time_ms":baseline.wall_time_ms,"lm_profile_timing":baseline.profile_timing or {}}
    output={"T":len(frames),"G":batch.layout.group_count,"production_baseline":attribution,
        "batch_preparation":{"setup_ms":batch.setup_ms,
        "persistent_bytes":batch.persistent_bytes,"padding_ratio":batch.layout.padding_ratio,
        "group_count":batch.layout.group_count,"nmax":batch.layout.nmax,
        "sections":{"layout":batch.layout_setup_ms,"vertices":batch.vertex_padding_ms,"static":batch.static_indices_ms}},"perturbations":{}}
    packed=gather_stage1_batch_parameters(stage.initial_lm_parameters,batch.layout)
    _,serial_objective_t=_timed(lambda:stage.lm_true_objective_fn(stage.initial_lm_parameters),args.warmup,args.repeats,stage.initial_lm_parameters)
    _,batch_objective_t=_timed(lambda:batched_stage1_objective(
        batch,packed.group_orientations,packed.group_translations,cache.image_width,cache.image_height,
        objective_mode=context.args.reproj_objective,charbonnier_eps=context.charbonnier_eps,
        rep_scale=context.scaling.rep_scale,aggregation_spec=stage.aggregation_spec),args.warmup,args.repeats,stage.initial_lm_parameters)
    output["diagnostic_only_objective_candidate"]={"serial":serial_objective_t,"batch":batch_objective_t,
        "speedup":serial_objective_t["p50"]/batch_objective_t["p50"]}
    for rs,ts in zip(rotations,translations):
        initial=_perturb(stage.initial_lm_parameters,rs,ts);key=f"rotation={rs}|translation={ts}"
        with contextlib.redirect_stdout(io.StringIO()):serial,serial_t=_timed(lambda:_serial(stage,context,initial),args.warmup,args.repeats,initial)
        item={"serial":_result(serial),"serial_timing":serial_t,"modes":{}}
        for mode in modes:
            common=build_stage1_shadow_linearization_common(cache,batch,initial,is_irls=context.is_irls,
                charbonnier_eps=context.charbonnier_eps,aggregation_spec=stage.aggregation_spec,rep_scale=context.scaling.rep_scale)
            ref=evaluate_stage1_equations(Stage1NEMode.SERIAL_REFERENCE,cache,batch,initial,common.padded_row_scales,common.flat_row_scales)
            cand=evaluate_stage1_equations(mode,cache,batch,initial,common.padded_row_scales,common.flat_row_scales)
            _,st=_timed(lambda:evaluate_stage1_equations(Stage1NEMode.SERIAL_REFERENCE,cache,batch,initial,common.padded_row_scales,common.flat_row_scales),args.warmup,args.repeats,initial)
            _,ct=_timed(lambda:evaluate_stage1_equations(mode,cache,batch,initial,common.padded_row_scales,common.flat_row_scales),args.warmup,args.repeats,initial)
            with contextlib.redirect_stdout(io.StringIO()):
                lock=run_stage1_lm_lockstep(stage,batch,context,candidate_mode=mode,initial_parameters=initial)
                if initial.is_cuda:torch.cuda.reset_peak_memory_stats(initial.device)
                candidate,timing=_timed(lambda:run_stage1_lm_independent_batch(stage,batch,context,mode=mode,initial_parameters=initial),args.warmup,args.repeats,initial)
                memory=({"peak_allocated":torch.cuda.max_memory_allocated(initial.device),
                         "peak_reserved":torch.cuda.max_memory_reserved(initial.device)} if initial.is_cuda else {})
            comparison=compare_stage1_lm_results(serial,candidate)
            errors=equation_errors(cand,ref);errors.update(analytic_errors(cache,batch,initial,cand));errors["step"]=step_errors(cand,ref,stage.parameter_dimension)
            diagnostic=all(errors[name]["relative_frobenius"]<=limit for name,limit in (("residual",1e-6),("active_jacobian",5e-5),("scaled_residual",1e-6),("H",2e-5),("g",2e-5)))
            trajectory=(comparison["first_divergent_trial"] is None and comparison["status_match"] and comparison["accepted_match"] and comparison["rejected_match"] and comparison["trial_count_match"] and comparison["solve_path_match"] and comparison["final_lambda_relative"]<=1e-5)
            raw_parameter=comparison["final_parameter_max_abs"]<=(5e-5 if initial.is_cuda else 1e-5)
            final_quality=comparison["objective_relative"]<=1e-6 and comparison["metrics_max_abs"]<=(1e-4 if initial.is_cuda else 1e-3)
            solver=trajectory and raw_parameter and final_quality
            item["modes"][mode.value]={"microkernel":errors,"diagnostic_gate":diagnostic,
                "solver_gate":solver,"solver_trajectory_gate":trajectory,
                "raw_parameter_strict_gate":raw_parameter,"final_objective_metric_gate":final_quality,
                "semantic_audit_status":("PENDING_SEMANTIC_AUDIT" if trajectory and final_quality and not raw_parameter else "NOT_REQUIRED"),
                "lockstep":lock.to_json_dict(),"independent_lm":{"candidate":_result(candidate),
                "comparison":comparison,"timing":timing,"speedup":serial_t["p50"]/timing["p50"]},
                "equation_timing":{"serial":st,"candidate":ct,"speedup":st["p50"]/ct["p50"],
                    "break_even_ne_calls":batch.setup_ms/max(st["p50"]-ct["p50"],1e-30)},
                "full_analytic_calls":cand.full_analytic_calls,"memory":memory}
            if args.run_downstream_regression and mode is Stage1NEMode.FULL_BATCH_GROUP_GEMM:
                with contextlib.redirect_stdout(io.StringIO()):item["modes"][mode.value]["downstream"]=_downstream(context,stage,serial,candidate)
        output["perturbations"][key]=item
    if args.profile_isolated_ne and len(frames) in (1,8):
        initial=stage.initial_lm_parameters
        common=build_stage1_shadow_linearization_common(cache,batch,initial,is_irls=context.is_irls,
            charbonnier_eps=context.charbonnier_eps,aggregation_spec=stage.aggregation_spec,rep_scale=context.scaling.rep_scale)
        trace_dir=args.trace_dir or "/tmp/staged_lm_stage1_shadow_traces";output["profiler"]={}
        for mode in (Stage1NEMode.SERIAL_REFERENCE,*modes):
            output["profiler"][mode.value]=_profile_equation(
                lambda mode=mode:evaluate_stage1_equations(mode,cache,batch,initial,common.padded_row_scales,common.flat_row_scales),
                initial,Path(trace_dir)/f"stage1_t{len(frames)}_{mode.value}_ne.json")
    first=next(iter(output["perturbations"].values()),None)
    if first and Stage1NEMode.FULL_BATCH_GROUP_GEMM.value in first["modes"]:
        mode=first["modes"][Stage1NEMode.FULL_BATCH_GROUP_GEMM.value]
        saved=max(0.0,(mode["equation_timing"]["serial"]["p50"]-mode["equation_timing"]["candidate"]["p50"])*attribution["linearization_count"])
        output["amdahl_projection"]={"measured_stage1_wall_ms":attribution["wall_time_ms"],
            "predicted_ne_only_stage1_ms":max(0.0,attribution["wall_time_ms"]-saved),
            "ideal_ne_zero_stage1_ms":max(0.0,attribution["wall_time_ms"]-mode["equation_timing"]["serial"]["p50"]*attribution["linearization_count"]),
            "objective_zero_upper_bound_stage1_ms":max(0.0,attribution["wall_time_ms"]-attribution["analytic_objective_ms"]),
            "measured_ne_calls":attribution["linearization_count"],
            "measured_objective_calls":attribution["analytic_objective_calls"]}
    return output


def main():
    args=parser().parse_args();frames=_csv(args.frames);counts=list(map(int,_csv(args.frame_sweep)))
    result={"environment":environment_record(torch.device(args.device),torch.float32),"input":vars(args),"frame_sets":[]}
    for count in counts:
        if count>len(frames):result["frame_sets"].append({"T":count,"status":"SKIPPED_INSUFFICIENT_REAL_FRAMES"});continue
        print(f"[STAGE1_SHADOW] device={args.device} T={count}");result["frame_sets"].append(_run_count(args,frames[:count]))
    gates=[mode for frame in result["frame_sets"] if "perturbations" in frame for perturb in frame["perturbations"].values() for mode in perturb["modes"].values()]
    ready=bool(gates) and all(x["solver_gate"] and x["lockstep"]["acceptance_mismatch_count"]==0 for x in gates)
    pending=bool(gates) and all(x.get("solver_trajectory_gate",False) and x.get("final_objective_metric_gate",False) for x in gates) and any(not x.get("raw_parameter_strict_gate",True) for x in gates)
    result["classification"]="EXACT_OR_NEAR_EXACT_EQUIVALENCE" if all(x["diagnostic_gate"] for x in gates) else "EXPECTED_FP32_OPERATION_ORDER_DIFFERENCE"
    result["integration_readiness"]=("READY_TO_INTEGRATE_STAGE1_FULL_BATCH_NE_SERIAL_WEIGHTS" if ready
        else "PENDING_SEMANTIC_AUDIT" if pending else "DO_NOT_INTEGRATE")
    path=Path(args.output_json);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(result,indent=2,ensure_ascii=False));print(f"[SAVE] {path}");return 0


if __name__=="__main__":raise SystemExit(main())
