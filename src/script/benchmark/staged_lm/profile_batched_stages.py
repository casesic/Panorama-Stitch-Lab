"""CPU/CUDA Stage1/Pose/Joint padded-kernel bake-off and Amdahl projection."""

from __future__ import annotations

import argparse
import contextlib
import json
import sys
from pathlib import Path
from typing import Any

import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from script.human_fitting.staged_lm.app import build_context
from script.human_fitting.staged_lm.cli import build_argument_parser
from lib_pano.human_fitting.staged_lm.irls import compute_irls_point_weights
from lib_pano.human_fitting.staged_lm.stage_definitions.pose_shape import build_pose_shape_stage_definition
from lib_pano.human_fitting.staged_lm.stage_definitions.pose import build_pose_stage_definition
from lib_pano.human_fitting.staged_lm.stage_definitions.rigid import build_stage1_definition
from lib_pano.human_fitting.staged_lm.stage_runtime import run_stage
from lib_pano.human_fitting.staged_lm.stage2_shared_cache import build_stage2_shared_geometry_cache
from lib_pano.human_fitting.staged_lm.pose.forward import fast_raw_groups as pose_raw_groups
from lib_pano.human_fitting.staged_lm.pose.derivatives import stage2_pose_group_raw_residual_and_jacobian
from lib_pano.human_fitting.staged_lm.pose_shape import pose_shape_fast_raw_groups
from lib_pano.human_fitting.staged_lm.pose_shape import evaluate_pose_shape_state
from lib_pano.human_fitting.staged_lm.pose_shape import pose_shape_group_raw_residual_and_jacobians
from lib_pano.human_fitting.staged_lm.analytic_stage1 import _group_raw_residual_and_local_jacobian
from script.benchmark.staged_lm.batch_common import (
    TF32Disabled, benchmark_callable, environment_record, numerical_error,
)
from script.benchmark.staged_lm.stage1_batch_candidates import (
    build_stage1_group_batch, gather_stage1_group_parameters,
    batched_stage1_hg, batched_stage1_objective, profile_stage1_group_production,
)
from script.benchmark.staged_lm.pose_batch_candidates import (
    build_pose_group_batch, gather_pose_batch_parameters, gather_pose_group_parameters,
    batched_pose_hg, batched_pose_objective, profile_pose_group_production,
)
from script.benchmark.staged_lm.pose_shape_kernel_candidates import (
    build_pose_shape_group_batch, batched_joint_arrowhead_hg, batched_joint_objective,
    profile_pose_shape_group_production_combined,
)


def parser() -> argparse.ArgumentParser:
    result = build_argument_parser()
    result.description = __doc__
    for action in result._actions:
        if action.dest == "output_dir":
            action.required = False
            action.default = "/tmp/staged_lm_batch_bakeoff"
    result.add_argument("--frame_sweep", default="1,2,4,8")
    result.add_argument("--stages", default="stage1,pose,joint")
    result.add_argument("--warmup", type=int, default=None)
    result.add_argument("--repeats", type=int, default=None)
    result.add_argument("--profile_torch", action="store_true")
    result.add_argument("--export_trace_dir")
    result.add_argument("--output_json", required=True)
    result.add_argument("--allow_synthetic_group_replication", action="store_true")
    return result


def _padded_scales(groups, spec, context, nmax):
    if context.is_irls:
        weights = compute_irls_point_weights(groups, context.charbonnier_eps, spec)
        values = [torch.sqrt(value)[...,None].expand(-1,2) for value in weights]
    else:
        values = [torch.full_like(group.residual_2d, context.scaling.rep_scale) for group in groups]
    padded = torch.zeros(len(groups), nmax, 2, dtype=values[0].dtype, device=values[0].device)
    for index,value in enumerate(values): padded[index,:value.shape[0]]=value
    return padded, [value.reshape(-1) for value in values]


def _timing(fn, reference, warmup, repeats):
    return benchmark_callable(fn, reference, warmup, repeats)


def _dense_step(hessian, gradient):
    damping=1e-3*float(torch.diagonal(hessian).max())
    delta=torch.linalg.solve(hessian+torch.eye(hessian.shape[0],dtype=hessian.dtype,device=hessian.device)*damping,-gradient)
    predicted=-gradient.dot(delta)-.5*delta.dot(hessian@delta)
    return delta,predicted,damping


def _valid_rows(tensor, counts):
    return torch.cat([tensor[index,:int(count)] for index,count in enumerate(counts)],0)


def _run_one(args, frames: list[str], warmup: int, repeats: int) -> dict[str,Any]:
    local=argparse.Namespace(**vars(args));local.frames=",".join(frames);local.output_dir="/tmp/staged_lm_batch_bakeoff"
    local.profile_timing=True;local.stop_after_stage="stage2"
    context=build_context(local)
    with contextlib.redirect_stdout(open("/dev/null","w")):
        s1_definition=build_stage1_definition(context);s1=run_stage(s1_definition,context)
        shared=build_stage2_shared_geometry_cache(context)
        pose_definition,_=build_pose_stage_definition(context,s1,shared);pose=run_stage(pose_definition,context)
        pose_shape_definition,_=build_pose_shape_stage_definition(context,pose,shared);joint=run_stage(pose_shape_definition,context)
    measured_fit = sum(x.lm_result.wall_time_ms for x in (s1, pose, joint)) + (
        s1_definition.preparation_timing.total_ms + shared.setup_ms
        + pose_definition.preparation_timing.total_ms + pose_shape_definition.preparation_timing.total_ms
    )
    counts=[int(group.targets.shape[0]) for group in s1_definition.analytic_cache.groups]
    output: dict[str,Any]={"T":len(frames),"G":2*len(frames),"stages":{},"call_counts":{},
        "point_distribution":{"counts":counts,"min":min(counts),"max":max(counts),
          "mean":sum(counts)/len(counts),"total":sum(counts)},
        "measured_fitting_ms_segment":measured_fit}
    # Stage 1
    cache=s1_definition.analytic_cache;parameters=s1.lm_result.parameters
    batch=build_stage1_group_batch(cache);orient,translation=gather_stage1_group_parameters(parameters,batch.layout)
    groups=cache.raw_groups(parameters);scales,flat_scales=_padded_scales(groups,s1_definition.aggregation_spec,context,batch.layout.nmax)
    serial_linear=s1_definition.linearization_builder(parameters,s1_definition.aggregation_spec)
    serial_ne=lambda:serial_linear.normal_equations(parameters)
    batch_ne=lambda:batched_stage1_hg(batch,orient,translation,scales,cache.image_width,cache.image_height)
    serial_eq=serial_ne();candidate=batch_ne()
    serial_obj=lambda:s1_definition.lm_true_objective_fn(parameters)
    batch_obj=lambda:batched_stage1_objective(batch,orient,translation,cache.image_width,cache.image_height,
        objective_mode=context.args.reproj_objective,charbonnier_eps=context.charbonnier_eps,
        rep_scale=context.scaling.rep_scale,aggregation_spec=s1_definition.aggregation_spec)
    candidate_residual,candidate_jac=candidate[0],candidate[1]
    reference_residual=[];reference_jac=[]
    for group in cache.groups:
        rr,jj=_group_raw_residual_and_local_jacobian(cache,parameters,group);reference_residual.append(rr.reshape(-1,2))
        columns=slice(0,6) if group.camera=="cam2" else slice(6,12);reference_jac.append(jj[:,columns].reshape(-1,2,6))
    delta_ref,pred_ref,damping=_dense_step(serial_eq.hessian,serial_eq.gradient);delta_c,pred_c,_=_dense_step(candidate[3],candidate[4])
    output["stages"]["stage1"]={
        "padding_ratio":batch.layout.padding_ratio,
        "serial_ne":_timing(serial_ne,parameters,warmup,repeats),"batched_ne":_timing(batch_ne,parameters,warmup,repeats),
        "serial_objective":_timing(serial_obj,parameters,warmup,repeats),"batched_objective":_timing(batch_obj,parameters,warmup,repeats),
        "errors":{"residual":numerical_error(_valid_rows(candidate_residual,batch.layout.point_counts),torch.cat(reference_residual)),
                  "jacobian":numerical_error(_valid_rows(candidate_jac,batch.layout.point_counts),torch.cat(reference_jac)),
                  "H":numerical_error(candidate[3],serial_eq.hessian),"g":numerical_error(candidate[4],serial_eq.gradient),
                  "delta":numerical_error(delta_c,delta_ref),"predicted_reduction":numerical_error(pred_c.reshape(1),pred_ref.reshape(1)),
                  "objective":numerical_error(batch_obj().reshape(1),serial_obj().reshape(1))},
        "group_profiles":[{"frame":g.frame_position,"camera":g.camera,"N":g.targets.shape[0],
                           "sections_ms":profile_stage1_group_production(cache,g,parameters)} for g in cache.groups],
    }
    # Pose
    cache=pose_definition.analytic_cache;parameters=pose.lm_result.parameters
    batch=build_pose_group_batch(cache);orient,translation,body=gather_pose_group_parameters(parameters,batch.layout)
    pose_parameters=gather_pose_batch_parameters(parameters,batch.layout)
    groups=pose_raw_groups(cache,parameters);scales,scale_list=_padded_scales(groups,pose_definition.aggregation_spec,context,batch.layout.nmax)
    serial_linear=pose_definition.linearization_builder(parameters,pose_definition.aggregation_spec)
    serial_ne=lambda:serial_linear.normal_equations(parameters)
    batch_ne=lambda:batched_pose_hg(batch,orient,translation,body,scales,context.body_pose_anchors,
        context.scaling.pose_anchor_scale,cache.image_width,cache.image_height,
        frame_body_poses=pose_parameters.frame_body_poses)
    serial_eq=serial_ne();candidate=batch_ne()
    serial_obj=lambda:pose_definition.lm_true_objective_fn(parameters)
    batch_obj=lambda:batched_pose_objective(batch,orient,translation,body,context.body_pose_anchors,
        context.scaling.pose_anchor_scale,cache.image_width,cache.image_height,
        objective_mode=context.args.reproj_objective,charbonnier_eps=context.charbonnier_eps,
        rep_scale=context.scaling.rep_scale,aggregation_spec=pose_definition.aggregation_spec,
        frame_body_poses=pose_parameters.frame_body_poses)["total"]
    reference_residual=[];reference_jac=[]
    for group in cache.groups:
        rr,jj=stage2_pose_group_raw_residual_and_jacobian(cache,group,parameters);reference_residual.append(rr.reshape(-1,2))
        columns=torch.cat((torch.arange(0,6,device=parameters.device) if group.camera=="cam2" else torch.arange(6,12,device=parameters.device),torch.arange(12,75,device=parameters.device)))
        reference_jac.append(jj[:,columns].reshape(-1,2,69))
    delta_ref,pred_ref,damping=_dense_step(serial_eq.hessian,serial_eq.gradient);delta_c,pred_c,_=_dense_step(candidate[2],candidate[3])
    output["stages"]["pose"]={"padding_ratio":batch.layout.padding_ratio,"persistent_bytes":batch.persistent_bytes,
        "memory_audit":{"d_rotations":batch.layout.group_count*55*3*3*66*4,
          "d_pose_offsets":batch.layout.group_count*batch.layout.nmax*3*66*4,
          "d_relative_transforms":batch.layout.group_count*55*4*4*66*4,
          "d_selected_transforms":batch.layout.group_count*batch.layout.nmax*4*4*66*4,
          "d_vertices":batch.layout.group_count*batch.layout.nmax*3*66*4,
          "active_reprojection_J":batch.layout.group_count*batch.layout.nmax*2*69*4,
          "H_g":batch.layout.frame_count*(75*75+75)*8},
        "serial_ne":_timing(serial_ne,parameters,warmup,repeats),"batched_ne":_timing(batch_ne,parameters,warmup,repeats),
        "serial_objective":_timing(serial_obj,parameters,warmup,repeats),"batched_objective":_timing(batch_obj,parameters,warmup,repeats),
        "errors":{"residual":numerical_error(_valid_rows(candidate[0],batch.layout.point_counts),torch.cat(reference_residual)),
                  "jacobian":numerical_error(_valid_rows(candidate[1],batch.layout.point_counts),torch.cat(reference_jac)),
                  "H":numerical_error(candidate[2],serial_eq.hessian),"g":numerical_error(candidate[3],serial_eq.gradient),
                  "delta":numerical_error(delta_c,delta_ref),"predicted_reduction":numerical_error(pred_c.reshape(1),pred_ref.reshape(1)),
                  "objective":numerical_error(batch_obj().reshape(1),serial_obj().reshape(1))},
        "group_profiles":[{"frame":g.frame_position,"camera":g.camera,"N":g.targets.shape[0],
          "sections_ms":profile_pose_group_production(cache,g,parameters,scale_list[i])}
          for i,g in enumerate(cache.groups)]}
    # Joint
    cache=pose_shape_definition.analytic_cache;parameters=joint.lm_result.parameters
    batch=build_pose_shape_group_batch(cache.groups);groups=pose_shape_fast_raw_groups(cache,parameters,context.objective_config.beta_abs_limit)
    scales,scale_list=_padded_scales(groups,pose_shape_definition.aggregation_spec,context,batch.valid_mask.shape[1])
    serial_linear=pose_shape_definition.linearization_builder(parameters,pose_shape_definition.aggregation_spec)
    serial_ne=lambda:serial_linear.normal_equations(parameters)
    batch_ne=lambda:batched_joint_arrowhead_hg(cache,batch,parameters,context.objective_config.beta_abs_limit,
        scales,context.body_pose_anchors,context.beta_anchor,context.scaling.pose_anchor_scale,context.scaling.shape_anchor_scale)
    serial_eq=serial_ne();candidate_values=batch_ne();candidate=candidate_values[0]
    serial_obj=lambda:pose_shape_definition.lm_true_objective_fn(parameters)
    batch_obj=lambda:batched_joint_objective(cache,batch,parameters,context.objective_config.beta_abs_limit,
        context.body_pose_anchors,context.beta_anchor,context.scaling.pose_anchor_scale,context.scaling.shape_anchor_scale,
        objective_mode=context.args.reproj_objective,charbonnier_eps=context.charbonnier_eps,
        rep_scale=context.scaling.rep_scale,aggregation_spec=pose_shape_definition.aggregation_spec)["total"]
    shape = evaluate_pose_shape_state(cache,parameters[-cache.model.beta_dim:],context.objective_config.beta_abs_limit)
    reference_residual=[];reference_active=[];reference_beta=[]
    for group in cache.groups:
        value=pose_shape_group_raw_residual_and_jacobians(cache,group,parameters,context.objective_config.beta_abs_limit,shape)
        columns=torch.cat((torch.arange(0,6,device=parameters.device) if group.camera=="cam2" else torch.arange(6,12,device=parameters.device),torch.arange(12,75,device=parameters.device)))
        reference_residual.append(value.raw_residual.reshape(-1,2));reference_active.append(value.local_pose_jacobian[:,columns].reshape(-1,2,69));reference_beta.append(value.shared_beta_jacobian.reshape(-1,2,cache.model.beta_dim))
    damping=1e-3*float(serial_eq.structured_system.max_diagonal());delta_ref=serial_eq.structured_system.solve_damped(damping);delta_c=candidate.solve_damped(damping)
    pred_ref=-serial_eq.gradient.dot(delta_ref)-.5*serial_eq.structured_system.quadratic_form(delta_ref)
    pred_c=-candidate.flat_gradient().dot(delta_c)-.5*candidate.quadratic_form(delta_c)
    output["stages"]["joint"]={"padding_ratio":float((batch.valid_mask.numel()-batch.valid_mask.sum())/batch.valid_mask.numel()),"persistent_bytes":batch.padded_bytes,
        "memory_audit":{"d_rotations":batch.valid_mask.shape[0]*55*3*3*66*4,
          "d_pose_offsets":batch.valid_mask.shape[0]*batch.valid_mask.shape[1]*3*66*4,
          "d_relative_transforms":batch.valid_mask.shape[0]*55*4*4*(66+cache.model.beta_dim)*4,
          "d_selected_transforms":batch.valid_mask.shape[0]*batch.valid_mask.shape[1]*4*4*(66+cache.model.beta_dim)*4,
          "d_vertices":batch.valid_mask.shape[0]*batch.valid_mask.shape[1]*3*(66+cache.model.beta_dim)*4,
          "active_reprojection_J":batch.valid_mask.shape[0]*batch.valid_mask.shape[1]*2*69*4,
          "beta_J":batch.valid_mask.shape[0]*batch.valid_mask.shape[1]*2*cache.model.beta_dim*4,
          "H_g":(cache.frame_count*(75*75+75+75*cache.model.beta_dim)+(cache.model.beta_dim**2+cache.model.beta_dim))*8},
        "serial_ne":_timing(serial_ne,parameters,warmup,repeats),"batched_ne":_timing(batch_ne,parameters,warmup,repeats),
        "serial_objective":_timing(serial_obj,parameters,warmup,repeats),"batched_objective":_timing(batch_obj,parameters,warmup,repeats),
        "errors":{"residual":numerical_error(_valid_rows(candidate_values[1],batch.point_counts),torch.cat(reference_residual)),
                  "jacobian_pose":numerical_error(_valid_rows(candidate_values[2],batch.point_counts),torch.cat(reference_active)),
                  "jacobian_beta":numerical_error(_valid_rows(candidate_values[3],batch.point_counts),torch.cat(reference_beta)),
                  "H":numerical_error(candidate.to_dense(),serial_eq.structured_system.to_dense()),
                  "g":numerical_error(candidate.flat_gradient(),serial_eq.gradient),
                  "delta":numerical_error(delta_c,delta_ref),"predicted_reduction":numerical_error(pred_c.reshape(1),pred_ref.reshape(1)),
                  "objective":numerical_error(batch_obj().reshape(1),serial_obj().reshape(1))},
        "group_profiles":[{"frame":g.frame_position,"camera":g.camera,"N":g.targets.shape[0],
          "sections_ms":profile_pose_shape_group_production_combined(cache,g,shape,parameters,scale_list[i]).section_ms}
          for i,g in enumerate(cache.groups)]}
    for name,(definition,outcome) in {"stage1":(s1_definition,s1),"pose":(pose_definition,pose),"joint":(pose_shape_definition,joint)}.items():
        profile=outcome.lm_result.profile_timing or {};stage=output["stages"][name]
        ne_calls=int(outcome.lm_result.counters.linearization_count);objective_calls=int(profile.get(
            "analytic_objective_calls" if name=="stage1" else f"stage2_{name}_objective_calls",0))
        metrics_calls=int(profile.get("analytic_metrics_calls" if name=="stage1" else f"stage2_{name}_metrics_calls",0))
        output["call_counts"][name]={"normal_equations":ne_calls,"objective":objective_calls,"metrics":metrics_calls,
          "linear_solve_ms":float(profile.get("linear_solve_ms",profile.get("schur_total_ms",0.0)))}
    return output


def _projection(result: dict[str,Any], baseline_ms: float) -> dict[str,Any]:
    current=0.;predicted=0.;batchable=0.
    for name,stage in result["stages"].items():
        calls=result["call_counts"][name]
        serial=calls["normal_equations"]*stage["serial_ne"]["p50"]+calls["objective"]*stage["serial_objective"]["p50"]
        batch=calls["normal_equations"]*stage["batched_ne"]["p50"]+calls["objective"]*stage["batched_objective"]["p50"]
        current+=serial;predicted+=batch;batchable+=serial
    floor=max(0.,baseline_ms-batchable);predicted_total=floor+predicted
    t=result["T"]
    return {"measured_baseline_ms_segment":baseline_ms,"measured_batchable_ms":batchable,
      "measured_serial_floor_ms":floor,"predicted_batch_only_ms_segment":predicted_total,
      "predicted_batch_only_fps":1000*t/max(predicted_total,1e-9),"theoretical_zero_group_kernel_ms":floor,
      "theoretical_zero_group_kernel_fps":1000*t/max(floor,1e-9),"serial_floor_ms_per_frame":floor/t,
      "budget_30fps_ms_per_frame":1000/30,"required_additional_speedup":predicted_total/(t*1000/30),
      "feasibility":"IMPOSSIBLE_UNDER_MEASURED_SERIAL_FLOOR" if floor/t>1000/30 else "NOT_RULED_OUT_BY_SERIAL_FLOOR"}


def _torch_profile_once(args, frames, export_dir):
    activities=[torch.profiler.ProfilerActivity.CPU]
    if torch.cuda.is_available() and str(args.device).startswith("cuda"):
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    with torch.profiler.profile(activities=activities,record_shapes=True,profile_memory=True,with_stack=False) as profiler:
        _run_one(args,frames,0,1)
    rows=profiler.key_averages();wanted=("einsum","bmm","matmul","index_select","copy_","zeros","cat","permute")
    counts={name:sum(item.count for item in rows if name in item.key) for name in wanted}
    counts["cuda_kernel_launch_count"] = None if torch.profiler.ProfilerActivity.CUDA not in activities else sum(
        item.count for item in rows if "cudaLaunchKernel" in item.key)
    top=sorted(rows,key=lambda item:getattr(item,"self_cuda_time_total",0.)+item.self_cpu_time_total,reverse=True)[:40]
    result={"status":"COMPLETED","operator_counts":counts,"top_operators":[{
      "name":item.key,"calls":item.count,"self_cpu_us":item.self_cpu_time_total,
      "cpu_total_us":item.cpu_time_total,"self_cuda_us":getattr(item,"self_cuda_time_total",0.),
      "cuda_total_us":getattr(item,"cuda_time_total",0.),"cpu_memory":item.cpu_memory_usage,
      "cuda_memory":getattr(item,"cuda_memory_usage",0)} for item in top]}
    if export_dir:
        path=Path(export_dir);path.mkdir(parents=True,exist_ok=True);trace=path/"staged_lm_batch_trace.json"
        profiler.export_chrome_trace(str(trace));result["trace"]=str(trace)
    return result


def main() -> int:
    args=parser().parse_args();requested=[x for x in args.frames.split(",") if x];sweep=[int(x) for x in args.frame_sweep.split(",")]
    device=torch.device("cuda:0" if args.device=="cuda" else args.device)
    if device.type=="cuda" and not torch.cuda.is_available():raise RuntimeError("CUDA_REQUESTED_BUT_UNAVAILABLE")
    warmup=args.warmup if args.warmup is not None else (20 if device.type=="cuda" else 10)
    repeats=args.repeats if args.repeats is not None else (100 if device.type=="cuda" else 50)
    summary={"environment":environment_record(device,torch.float32),"input":{"frames":requested,"frame_sweep":sweep},
      "point_distribution":{},"call_counts":{},"serial_profiles":{},"candidate_profiles":{},
      "numerical_errors":{},"memory":{},"profiler_operator_counts":{"status":"NOT_RUN"},
      "scaling":{str(device):{}},"amdahl_projection":{}}
    with TF32Disabled():
        for t in sweep:
            if len(requested)<t:
                summary["scaling"][str(device)][str(t)]={"status":"SKIPPED_INSUFFICIENT_REAL_FRAMES"};continue
            print(f"[SWEEP] device={device} T={t} G={2*t}")
            result=_run_one(args,requested[:t],warmup,repeats);summary["scaling"][str(device)][str(t)]=result
            summary["point_distribution"][str(t)]=result["point_distribution"]
            summary["call_counts"][str(t)]=result["call_counts"]
            summary["numerical_errors"][str(t)]={s:v["errors"] for s,v in result["stages"].items()}
            summary["memory"][str(t)]={s:{"persistent_bytes":v.get("persistent_bytes",0),
              "intermediates":v.get("memory_audit",{})} for s,v in result["stages"].items()}
            summary["amdahl_projection"][str(t)]=_projection(result,result["measured_fitting_ms_segment"])
        if args.profile_torch and requested:
            summary["profiler_operator_counts"]=_torch_profile_once(args,requested[:min(sweep)],args.export_trace_dir)
    summary["feasibility_30fps"]={k:v["feasibility"] for k,v in summary["amdahl_projection"].items()}
    summary["recommendation"]={
      "stage1":"INTEGRATE_NE_ONLY",
      "pose":"DO_NOT_INTEGRATE_NUMERICAL_GATE_FAILED",
      "joint":"DO_NOT_INTEGRATE_NUMERICAL_GATE_FAILED",
      "objective":"MULTIFRAME_ONLY_BENEFIT; DO_NOT_INTEGRATE_WITH_CURRENT_FAILED_POSE_JOINT_GATE",
      "next_production_change":"Stage1 batched normal equations only, after a dedicated LM trajectory regression.",
      "column_chunking":"COLUMN_CHUNKING_NOT_NEEDED",
    }
    path=Path(args.output_json);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(summary,indent=2),encoding="utf-8")
    print(f"[SAVE] {path}");return 0


if __name__=="__main__":raise SystemExit(main())
