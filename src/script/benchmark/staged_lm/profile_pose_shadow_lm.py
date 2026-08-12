"""Run real-data Stage 2 Pose batched-NE lockstep and independent LM shadows."""

from __future__ import annotations

import argparse
import contextlib
import io
import json
from pathlib import Path
import statistics
import sys
import time
from typing import Any

import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from lib_pano.human_fitting.staged_lm.lm.runner import run_lm_optimization
from lib_pano.human_fitting.staged_lm.device import resolve_execution_device
from lib_pano.human_fitting.staged_lm.stage2_shared_cache import build_stage2_shared_geometry_cache
from lib_pano.human_fitting.staged_lm.stage_definitions.pose import build_pose_stage_definition
from lib_pano.human_fitting.staged_lm.stage_definitions.rigid import build_stage1_definition
from lib_pano.human_fitting.staged_lm.stage_runtime import run_stage
from script.benchmark.staged_lm.batch_common import environment_record
from script.benchmark.staged_lm.pose_batch_candidates import build_pose_group_batch
from script.benchmark.staged_lm.pose_shadow_common import PoseNEMode
from script.benchmark.staged_lm.pose_shadow_common import (
    evaluate_pose_equations, serial_pose_row_scales,
)
from script.benchmark.staged_lm.pose_shadow_lm import (
    compare_lm_results, run_pose_lm_independent_batch, run_pose_lm_lockstep,
)
from script.human_fitting.staged_lm.app import build_context
from script.human_fitting.staged_lm.cli import build_argument_parser


def parser() -> argparse.ArgumentParser:
    result = build_argument_parser()
    for action in result._actions:
        if action.dest == "output_dir":
            action.required = False; action.default = "/tmp/staged_lm_pose_shadow"
    result.add_argument("--frame_sweep", default="1,2,4,8")
    result.add_argument("--candidate_modes", default="batch_forward_serial_hg,serial_forward_batch_hg,full_batch")
    result.add_argument("--weight_sources", default="serial,batch")
    result.add_argument("--hg_reduction_modes", default="serial_group,batched_bmm_serial_frame")
    result.add_argument("--pose_perturbation_scales", default="0,0.001,0.01,0.05")
    result.add_argument("--warmup", type=int, default=1)
    result.add_argument("--repeats", type=int, default=5)
    result.add_argument("--profile_isolated_ne", action="store_true")
    result.add_argument("--trace_dir")
    result.add_argument("--output_json", required=True)
    return result


def _csv(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def _result_json(result) -> dict[str, Any]:
    return {
        "status": result.status, "objective": result.objective,
        "accepted_steps": result.counters.accepted_steps,
        "rejected_trials": result.counters.rejected_trials,
        "solve_count": result.counters.solve_attempt_count,
        "linearization_count": result.counters.linearization_count,
        "linearization_count": result.counters.linearization_count,
        "final_lambda": result.final_lambda, "wall_time_ms": result.wall_time_ms,
        "metrics": result.rms_metrics, "trial_history": result.trial_history,
        "weight_history": result.weight_history,
    }


def _run_serial(stage, context, initial, *, profile=True):
    return run_lm_optimization(
        "STAGE2_POSE_SERIAL_SHADOW",
        lambda x: stage.linearization_builder(x, stage.aggregation_spec),
        stage.raw_metrics_fn, initial, stage.parameter_dimension,
        stage.lm_max_accepted_steps, context.solver_config.lm_max_trials,
        context.solver_config, profile, stage.lm_true_objective_fn,
    )


def _perturb(initial: torch.Tensor, scale: float, seed: int = 20260715) -> torch.Tensor:
    if scale == 0: return initial.detach().clone()
    generator = torch.Generator(device=initial.device); generator.manual_seed(seed)
    return initial + torch.randn(initial.shape, dtype=initial.dtype, device=initial.device, generator=generator) * scale


def _timed_runs(fn, warmup: int, repeats: int, reference: torch.Tensor):
    for _ in range(warmup): fn()
    samples=[]; last=None
    for _ in range(repeats):
        if reference.is_cuda: torch.cuda.synchronize(reference.device)
        started=time.perf_counter(); last=fn()
        if reference.is_cuda: torch.cuda.synchronize(reference.device)
        samples.append((time.perf_counter()-started)*1000)
    values=sorted(samples)
    return last, {"p50":statistics.median(samples),"p90":values[min(len(values)-1,max(0,int(.9*len(values))-1))],
                  "mean":statistics.mean(samples),"min":min(samples),"max":max(samples)}


def _profile_isolated(path: Path, callables: dict[str, Any], reference: torch.Tensor) -> dict[str, Any]:
    path.mkdir(parents=True, exist_ok=True); output={}
    activities=[torch.profiler.ProfilerActivity.CPU]
    if reference.is_cuda: activities.append(torch.profiler.ProfilerActivity.CUDA)
    for name, fn in callables.items():
        try:
            with torch.profiler.profile(activities=activities,record_shapes=True,profile_memory=True,with_stack=False) as prof:
                fn()
            trace=path/f"{name}.json";prof.export_chrome_trace(str(trace))
            averages=prof.key_averages()
            output[name]={"trace":str(trace),"operator_count":sum(event.count for event in averages),
                          "top":str(averages.table(sort_by="self_cuda_time_total" if reference.is_cuda else "self_cpu_time_total",row_limit=30))}
        except Exception as error:
            output[name]={"error":repr(error)}
    return output


def _run_frame_count(args, frames: list[str]) -> dict[str, Any]:
    local=argparse.Namespace(**vars(args)); local.frames=",".join(frames);local.output_dir="/tmp/staged_lm_pose_shadow"
    local.stop_after_stage="stage2_pose";local.profile_timing=True
    context=build_context(local)
    with contextlib.redirect_stdout(io.StringIO()):
        stage1=run_stage(build_stage1_definition(context),context)
        shared=build_stage2_shared_geometry_cache(context)
        pose_stage,_=build_pose_stage_definition(context,stage1,shared)
    if pose_stage.initial_lm_parameters.is_cuda:torch.cuda.synchronize(pose_stage.initial_lm_parameters.device)
    started=time.perf_counter();batch=build_pose_group_batch(pose_stage.analytic_cache)
    if pose_stage.initial_lm_parameters.is_cuda:torch.cuda.synchronize(pose_stage.initial_lm_parameters.device)
    preparation_ms=(time.perf_counter()-started)*1000
    output={"T":len(frames),"G":2*len(frames),"batch_preparation":{"measured_ms":preparation_ms,**batch.preparation_timing,
            "persistent_bytes":batch.persistent_bytes},"perturbations":{}}
    modes=[PoseNEMode(value) for value in _csv(args.candidate_modes)]
    for perturb_text in _csv(args.pose_perturbation_scales):
        perturb=float(perturb_text);initial=_perturb(pose_stage.initial_lm_parameters,perturb)
        with contextlib.redirect_stdout(io.StringIO()):
            serial,serial_timing=_timed_runs(lambda:_run_serial(pose_stage,context,initial),args.warmup,args.repeats,initial)
        perturbation={"initial_objective":float(pose_stage.lm_true_objective_fn(initial).detach()),
            "serial":_result_json(serial),"serial_timing":serial_timing,"modes":{}}
        for mode in modes:
            weights=_csv(args.weight_sources)
            reductions=_csv(args.hg_reduction_modes) if mode is PoseNEMode.FULL_BATCH else ["serial_group" if mode is PoseNEMode.BATCH_FORWARD_SERIAL_HG else "batched_bmm_serial_frame"]
            for weight in weights:
                for reduction in reductions:
                    key=f"{mode.value}|weights={weight}|reduction={reduction}"
                    with contextlib.redirect_stdout(io.StringIO()):
                        lockstep=run_pose_lm_lockstep(pose_stage,batch,context,candidate_mode=mode,
                            weight_source=weight,hg_reduction_mode=reduction,initial_parameters=initial)
                        if initial.is_cuda: torch.cuda.reset_peak_memory_stats(initial.device)
                        independent,timing=_timed_runs(lambda:run_pose_lm_independent_batch(
                            pose_stage,batch,context,candidate_mode=mode,weight_source=weight,
                            hg_reduction_mode=reduction,initial_parameters=initial),args.warmup,args.repeats,initial)
                        memory={"pose_group_batch_persistent_bytes":batch.persistent_bytes}
                        if initial.is_cuda:
                            memory.update({"max_memory_allocated":torch.cuda.max_memory_allocated(initial.device),
                                           "max_memory_reserved":torch.cuda.max_memory_reserved(initial.device)})
                    comparison=compare_lm_results(serial,independent)
                    speedup=serial_timing["p50"]/timing["p50"]
                    serial_ne=statistics.median([r.serial_equation_ms for r in lockstep.trial_records])
                    batch_ne=statistics.median([r.batch_equation_ms for r in lockstep.trial_records])
                    is_cuda=initial.is_cuda
                    trajectory_gate=(lockstep.acceptance_mismatch_count==0 and comparison["status_match"]
                        and comparison["accepted_match"] and comparison["rejected_match"] and comparison["trial_count_match"])
                    final_gate=(comparison["final_parameter_max_abs"] <= (5e-5 if is_cuda else 1e-5)
                        and comparison["final_parameter_relative"] <= (5e-6 if is_cuda else 1e-6)
                        and comparison["final_objective_relative"] <= (1e-6 if is_cuda else 1e-7)
                        and comparison["metrics_max_abs"] <= (1e-3 if is_cuda else 1e-5))
                    microkernel={
                        "residual_relative_max":max(r.residual_relative for r in lockstep.trial_records),
                        "residual_max_abs":max(r.residual_max_abs for r in lockstep.trial_records),
                        "jacobian_relative_max":max(r.jacobian_relative for r in lockstep.trial_records),
                        "H_relative_max":max(r.h_relative for r in lockstep.trial_records),
                        "g_relative_max":max(r.g_relative for r in lockstep.trial_records),
                    }
                    numerical_gate=(microkernel["residual_max_abs"]<=1e-3
                        and microkernel["residual_relative_max"]<=1e-6
                        and microkernel["jacobian_relative_max"]<=(5e-5 if is_cuda else 2e-5)
                        and microkernel["H_relative_max"]<=2e-5 and microkernel["g_relative_max"]<=2e-5)
                    perturbation["modes"][key]={"lockstep":lockstep.to_json_dict(),
                        "independent_lm":_result_json(independent),"comparison":comparison,
                        "timing":timing,"speedup":speedup,
                        "equation_timing":{"serial_p50_ms":serial_ne,"batch_p50_ms":batch_ne},
                        "break_even_ne_calls":preparation_ms/max(serial_ne-batch_ne,1e-30),
                        "microkernel_errors":microkernel,"numerical_gate":numerical_gate,
                        "trajectory_gate":trajectory_gate,"final_result_gate":final_gate,
                        "performance_gate":speedup >= 1.5,"memory":memory}
        output["perturbations"][perturb_text]=perturbation
    if args.profile_isolated_ne:
        parameters=pose_stage.initial_lm_parameters
        scales,_=serial_pose_row_scales(pose_stage,parameters,batch,
            objective_mode=context.args.reproj_objective,charbonnier_eps=context.charbonnier_eps,
            rep_scale=context.scaling.rep_scale)
        callables={}
        for name,mode,reduction in (
            (f"pose_t{len(frames)}_serial_ne",PoseNEMode.SERIAL_PRODUCTION,"production_order"),
            (f"pose_t{len(frames)}_full_batch_ne",PoseNEMode.FULL_BATCH,"batched_bmm_serial_frame"),
            (f"pose_t{len(frames)}_batch_forward_serial_hg_ne",PoseNEMode.BATCH_FORWARD_SERIAL_HG,"serial_group"),
            (f"pose_t{len(frames)}_serial_forward_batch_hg_ne",PoseNEMode.SERIAL_FORWARD_BATCH_HG,"batched_bmm_serial_frame"),
        ):
            callables[name]=lambda m=mode,r=reduction:evaluate_pose_equations(m,pose_stage,batch,parameters,scales,
                context.body_pose_anchors,context.scaling.pose_anchor_scale,context.image_width,context.image_height,
                hg_reduction_mode=r)
        output["profiler"]=_profile_isolated(Path(args.trace_dir or "/tmp/staged_lm_pose_shadow_traces"),callables,parameters)
    return output


def main(argv=None) -> int:
    args=parser().parse_args(argv);frames=_csv(args.frames);sweep=[int(x) for x in _csv(args.frame_sweep)]
    device=resolve_execution_device(args.device)
    output={"environment":environment_record(device,torch.float32),"input":{"frames":frames,"objective":args.reproj_objective},"frame_sets":{}}
    for count in sweep:
        if count>len(frames):output["frame_sets"][str(count)]={"status":"SKIPPED_MISSING_REAL_DATA"};continue
        print(f"[POSE_SHADOW] device={args.device} T={count}")
        output["frame_sets"][str(count)]=_run_frame_count(args,frames[:count])
    full_results=[]
    for frame_set in output["frame_sets"].values():
        for perturbation in frame_set.get("perturbations",{}).values():
            for key,value in perturbation["modes"].items():
                if key.startswith("full_batch|weights=serial|"):
                    full_results.append(value)
    all_gates=bool(full_results) and all(
        value["numerical_gate"] and value["trajectory_gate"]
        and value["final_result_gate"] and value["performance_gate"]
        for value in full_results
    )
    output["recommendation"]="INTEGRATE_FULL_BATCH_NE" if all_gates else "DO_NOT_INTEGRATE"
    target=Path(args.output_json);target.parent.mkdir(parents=True,exist_ok=True)
    target.write_text(json.dumps(output,indent=2),encoding="utf-8")
    print(f"[SAVE] {target}");return 0


if __name__=="__main__":raise SystemExit(main())
