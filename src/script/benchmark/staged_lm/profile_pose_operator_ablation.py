"""Profile Pose operator boundaries and hybrid exactness-restoration candidates."""

from __future__ import annotations

import argparse
import contextlib
from dataclasses import replace
import io
import json
from pathlib import Path
import sys
from typing import Any

import torch

if __package__ in (None, ""):
    sys.path.insert(0,str(Path(__file__).resolve().parents[3]))

from lib_pano.human_fitting.staged_lm.lm.equations import (
    compute_scaled_gradient, normal_equations_quadratic_form,
    solve_normal_equations,
)
from lib_pano.human_fitting.staged_lm.lm.runner import run_lm_optimization
from lib_pano.human_fitting.staged_lm.stage2_shared_cache import build_stage2_shared_geometry_cache
from lib_pano.human_fitting.staged_lm.stage_definitions.pose import build_pose_stage_definition
from lib_pano.human_fitting.staged_lm.stage_definitions.rigid import build_stage1_definition
from lib_pano.human_fitting.staged_lm.stage_runtime import run_stage
from script.benchmark.staged_lm.batch_common import benchmark_callable,environment_record
from script.benchmark.staged_lm.pose_batch_candidates import build_pose_group_batch,gather_pose_batch_parameters
from script.benchmark.staged_lm.pose_operator_ablation import (
    PoseAblationMode,build_mode_equations,build_pose_operator_state,build_serial_pose_operator_state,
    classify_ablation,compare_operator_states,first_divergence,integration_readiness,
    first_different_fk_joint,
    rotation_fp64_diagnostic,rotation_internal_ablation,select_pose_group_batch,
    small_angle_branch_audit,tensor_diagnostics,validate_serial_reconstruction,
)
from script.benchmark.staged_lm.pose_shadow_common import serial_pose_row_scales
from script.human_fitting.staged_lm.app import build_context
from script.human_fitting.staged_lm.cli import build_argument_parser


def parser()->argparse.ArgumentParser:
    result=build_argument_parser()
    for action in result._actions:
        if action.dest=="output_dir":action.required=False;action.default="/tmp/staged_lm_pose_ablation"
    result.add_argument("--parameter_points",default="initial,middle,final,perturbed")
    result.add_argument("--group_modes",default="cam2,cam4,both")
    result.add_argument("--run_t4_confirmation",action="store_true")
    result.add_argument("--warmup",type=int)
    result.add_argument("--repeats",type=int)
    result.add_argument("--profile_first_divergence",action="store_true")
    result.add_argument("--solver_shadow_json",default="data_process/Vis/staged_lm_pose_shadow/cuda_main.json")
    result.add_argument("--output_json",required=True)
    return result


def _csv(value:str)->list[str]:return [item.strip() for item in value.split(",") if item.strip()]


def _run_pose_lm(stage,context,initial,max_steps):
    return run_lm_optimization("POSE_ABLATION_SERIAL",lambda x:stage.linearization_builder(x,stage.aggregation_spec),
        stage.raw_metrics_fn,initial,stage.parameter_dimension,max_steps,context.solver_config.lm_max_trials,
        context.solver_config,False,stage.lm_true_objective_fn)


def _prepare(args,frames:list[str]):
    local=argparse.Namespace(**vars(args));local.frames=",".join(frames);local.output_dir="/tmp/staged_lm_pose_ablation"
    local.stop_after_stage="stage2_pose";local.profile_timing=False
    context=build_context(local)
    with contextlib.redirect_stdout(io.StringIO()):
        stage1=run_stage(build_stage1_definition(context),context);shared=build_stage2_shared_geometry_cache(context)
        stage,_=build_pose_stage_definition(context,stage1,shared)
        middle=_run_pose_lm(stage,context,stage.initial_lm_parameters,3)
        final=run_stage(stage,context).lm_result
    generator=torch.Generator(device=stage.initial_lm_parameters.device);generator.manual_seed(20260715)
    perturbed=stage.initial_lm_parameters+torch.randn(stage.initial_lm_parameters.shape,dtype=stage.initial_lm_parameters.dtype,
        device=stage.initial_lm_parameters.device,generator=generator)*.05
    return context,stage,{"initial":stage.initial_lm_parameters,"middle":middle.parameters,"final":final.parameters,"perturbed":perturbed}


def _subset(stage,batch,mode:str):
    indices={"cam2":[0],"cam4":[1],"both":[0,1]}[mode]
    return replace(stage.analytic_cache,groups=[stage.analytic_cache.groups[index] for index in indices],frame_count=1),select_pose_group_batch(batch,indices)


def _solver_gate(path:str)->dict[str,Any]:
    target=Path(path)
    if not target.exists():return {"pass":False,"reason":"MISSING_SOLVER_SHADOW_JSON"}
    data=json.loads(target.read_text());checks=[]
    for frame in data.get("frame_sets",{}).values():
        for point in frame.get("perturbations",{}).values():
            for key,value in point.get("modes",{}).items():
                if key.startswith("full_batch|weights=serial|"):
                    comparison=value["comparison"];lock=value["lockstep"]
                    checks.append(lock["acceptance_mismatch_count"]==0 and comparison["first_divergent_trial"] is None
                        and comparison["final_parameter_max_abs"]<=5e-5 and comparison["final_objective_relative"]<=1e-6
                        and comparison["metrics_max_abs"]<=1e-3 and value["speedup"]>=1.5)
    return {"pass":bool(checks) and all(checks),"checked_configurations":len(checks),"source":str(target)}


def _hg_step(reference,candidate,stage,parameters):
    damping=float(torch.diagonal(reference.hessian).max())*1e-3
    rd=solve_normal_equations(reference,damping).delta
    cd=solve_normal_equations(candidate,damping).delta
    rp=float(-reference.gradient.dot(rd)-.5*normal_equations_quadratic_form(reference,rd))
    cp=float(-candidate.gradient.dot(cd)-.5*normal_equations_quadratic_form(candidate,cd))
    before=float(stage.lm_true_objective_fn(parameters).detach())
    ra=before-float(stage.lm_true_objective_fn((parameters+rd.to(torch.float32)).detach()).detach())
    ca=before-float(stage.lm_true_objective_fn((parameters+cd.to(torch.float32)).detach()).detach())
    def metric(a,b):return tensor_diagnostics(a,b)
    gref=reference.gradient;gcand=candidate.gradient;mask=gref.abs()>1e-6
    return {"H":metric(reference.hessian,candidate.hessian),"g":metric(gref,gcand),"g_abs_l2":float((gcand-gref).norm()),
        "g_max_abs":float((gcand-gref).abs().max()),"g_reference_norm":float(gref.norm()),"g_candidate_norm":float(gcand.norm()),
        "g_cosine_similarity":float(torch.nn.functional.cosine_similarity(gref[None],gcand[None])),
        "g_sign_mismatch_count":int(((torch.sign(gref)!=torch.sign(gcand))&mask).sum()),
        "scaled_gradient_serial":compute_scaled_gradient(reference),"scaled_gradient_candidate":compute_scaled_gradient(candidate),
        "delta":metric(rd,cd),"predicted_reduction_serial":rp,"predicted_reduction_candidate":cp,
        "predicted_reduction_relative":abs(cp-rp)/max(abs(rp),1e-30),"rho_difference_estimate":abs(ca/max(cp,1e-12)-ra/max(rp,1e-12))}


def _padding_ok(state,batch)->bool:
    for name in ("pose_offsets","d_pose_offsets","selected_v_posed","selected_transforms","d_selected_transforms",
                 "selected_skinned_vertices","d_selected_skinned_vertices","final_vertices","d_final_vertices","uv",
                 "projection_jacobian","uv_pose_jacobian","residual","active_jacobian"):
        value=getattr(state,name)
        for group,count in enumerate(batch.layout.point_counts):
            if torch.count_nonzero(value[group,int(count):]):return False
    return True


def _run_dataset(args,frames:list[str],parameter_names:list[str],group_modes:list[str],warmup:int,repeats:int):
    context,stage,points=_prepare(args,frames);batch=build_pose_group_batch(stage.analytic_cache)
    output={"T":len(frames),"batch_preparation":batch.preparation_timing,"parameter_points":{},"performance":{}}
    modes=list(PoseAblationMode)
    # Boundary diagnosis is exhaustive only for T=1; T=4 confirms initial/full/hybrids.
    selected_points=parameter_names if len(frames)==1 else ["initial"]
    selected_groups=group_modes if len(frames)==1 else ["both"]
    for point_name in selected_points:
        parameters=points[point_name];point_output={}
        for group_mode in selected_groups:
            if len(frames)>1:
                cache,batch_view=stage.analytic_cache,batch
            else:cache,batch_view=_subset(stage,batch,group_mode)
            serial=build_serial_pose_operator_state(cache,batch_view,parameters)
            reference_validation=validate_serial_reconstruction(cache,batch_view,parameters,serial)
            mode_output={}
            for mode in modes:
                candidate=build_pose_operator_state(cache,batch_view,parameters,mode)
                mode_output[mode.value]={"operator_boundaries":compare_operator_states(serial,candidate,batch_view.layout.valid_mask),
                                         "padding_valid":_padding_ok(candidate,batch_view)}
            full_comparison=mode_output[PoseAblationMode.FULL_BATCH.value]["operator_boundaries"]
            isolated={
                "rotation":{name:mode_output[PoseAblationMode.BATCH_ROTATION_ONLY.value]["operator_boundaries"][name]
                    for name in ("rotations","d_rotations")},
                "pose":{name:mode_output[PoseAblationMode.SERIAL_ROTATION_BATCH_REST.value]["operator_boundaries"][name]
                    for name in ("pose_feature","d_pose_feature","pose_offsets","d_pose_offsets")},
                "fk":{name:mode_output[PoseAblationMode.SERIAL_ROTATION_POSE_BATCH_REST.value]["operator_boundaries"][name]
                    for name in ("global_transforms","d_global_transforms","relative_transforms","d_relative_transforms")},
                "lbs":{name:mode_output[PoseAblationMode.SERIAL_ROTATION_POSE_FK_BATCH_REST.value]["operator_boundaries"][name]
                    for name in ("selected_transforms","d_selected_transforms","selected_skinned_vertices","d_selected_skinned_vertices")},
                "projection":{name:mode_output[PoseAblationMode.SERIAL_ROTATION_POSE_FK_LBS_BATCH_PROJECTION.value]["operator_boundaries"][name]
                    for name in ("uv","projection_jacobian","uv_pose_jacobian","residual","active_jacobian")},
            }
            serial_fk=build_serial_pose_operator_state(cache,batch_view,parameters)
            batch_fk=build_pose_operator_state(cache,batch_view,parameters,PoseAblationMode.SERIAL_ROTATION_POSE_BATCH_REST)
            point_output[group_mode]={"serial_reference_validation":reference_validation,"hybrid_modes":mode_output,
                "isolated_operator_boundaries":isolated,
                "first_fk_value_difference":first_different_fk_joint(serial_fk.global_transforms,batch_fk.global_transforms,batch_view.fk_depth_indices),
                "first_fk_derivative_difference":first_different_fk_joint(serial_fk.d_global_transforms,batch_fk.d_global_transforms,batch_view.fk_depth_indices),
                "first_divergence":first_divergence(full_comparison),
                "rotation_internal_ablation":rotation_internal_ablation(cache,batch_view,parameters),
                "small_angle_branch_audit":small_angle_branch_audit(gather_pose_batch_parameters(parameters,batch_view.layout),batch_view),
                "fp64_diagnostics":rotation_fp64_diagnostic(cache,batch_view,parameters)}
        output["parameter_points"][point_name]=point_output
    # H/g and timing use every group in the dataset and serial weights.
    parameters=points["initial"]
    row_scales,_=serial_pose_row_scales(stage,parameters,batch,objective_mode=context.args.reproj_objective,
        charbonnier_eps=context.charbonnier_eps,rep_scale=context.scaling.rep_scale)
    states={mode:build_pose_operator_state(stage.analytic_cache,batch,parameters,mode) for mode in modes}
    equations={mode:build_mode_equations(states[mode],batch,parameters,row_scales,context.body_pose_anchors,
        context.scaling.pose_anchor_scale) for mode in modes}
    reference=equations[PoseAblationMode.SERIAL_ALL]
    performance_modes=(PoseAblationMode.SERIAL_ALL,PoseAblationMode.FULL_BATCH,PoseAblationMode.SERIAL_ROTATION_BATCH_REST,
        PoseAblationMode.SERIAL_ROTATION_POSE_BATCH_REST,PoseAblationMode.SERIAL_ROTATION_POSE_FK_BATCH_REST,
        PoseAblationMode.SERIAL_ROTATION_POSE_FK_LBS_BATCH_PROJECTION)
    for mode in modes:
        analysis=_hg_step(reference,equations[mode],stage,parameters)
        boundary=compare_operator_states(states[PoseAblationMode.SERIAL_ALL],states[mode],batch.layout.valid_mask)
        residual=boundary["residual"];jacobian=boundary["active_jacobian"]
        analysis["old_microkernel_gate"]=(residual["relative_frobenius"]<=1e-6 and jacobian["relative_frobenius"]<=5e-5
            and analysis["H"]["relative_frobenius"]<=2e-5 and analysis["g"]["relative_frobenius"]<=2e-5
            and analysis["delta"]["relative_frobenius"]<=1e-4 and analysis["predicted_reduction_relative"]<=1e-4)
        output.setdefault("hg_and_step_analysis",{})[mode.value]=analysis
    for mode in performance_modes:
        def call(selected=mode):
            state=build_pose_operator_state(stage.analytic_cache,batch,parameters,selected)
            return build_mode_equations(state,batch,parameters,row_scales,context.body_pose_anchors,context.scaling.pose_anchor_scale)
        output["performance"][mode.value]=benchmark_callable(call,parameters,warmup,repeats)
    serial_p50=output["performance"][PoseAblationMode.SERIAL_ALL.value]["p50"]
    for mode,value in output["performance"].items():value["speedup"]=serial_p50/value["p50"]
    return output


def main(argv=None)->int:
    args=parser().parse_args(argv);frames=_csv(args.frames);parameter_names=_csv(args.parameter_points);group_modes=_csv(args.group_modes)
    device=torch.device(args.device if args.device!="cuda" else "cuda:0");cuda=device.type=="cuda"
    warmup=args.warmup if args.warmup is not None else (20 if cuda else 10)
    repeats=args.repeats if args.repeats is not None else (100 if cuda else 50)
    result={"environment":environment_record(device,torch.float32),"input":{"frames":frames,"device":args.device,
        "parameter_points":parameter_names,"group_modes":group_modes,"warmup":warmup,"repeats":repeats}}
    result["t1"]=_run_dataset(args,frames[:1],parameter_names,group_modes,warmup,repeats)
    if args.run_t4_confirmation and len(frames)>=4:result["t4"]=_run_dataset(args,frames[:4],parameter_names,group_modes,warmup,repeats)
    initial_both=result["t1"]["parameter_points"]["initial"]["both"]
    first=initial_both["first_divergence"];result.update({"first_divergent_operator":None if first is None else first["operator_tensor"],
        "first_divergent_tensor":None if first is None else first["operator_tensor"],"first_divergent_index":None if first is None else first["index"],
        "first_divergent_max_abs":None if first is None else first["max_abs"],"first_divergent_max_ulp":None if first is None else first["max_ulp"]})
    fp32=initial_both["hybrid_modes"][PoseAblationMode.FULL_BATCH.value]["operator_boundaries"]["rotations"]
    fp64=initial_both["fp64_diagnostics"]["rotation_value"]
    ratio=fp64["max_abs"]/max(fp32["max_abs"],1e-30)
    ulp_fraction=(fp32.get("count_ulp_le_2",0)/max(fp32["valid_element_count"],1))
    solver_gate=_solver_gate(args.solver_shadow_json);padding_ok=all(value["padding_valid"] for value in initial_both["hybrid_modes"].values())
    classification=classify_ablation(fp64_reduction_ratio=ratio,branch_index_padding_ok=(padding_ok and initial_both["small_angle_branch_audit"]["serial_batch_mask_equal"]),
        solver_gate=solver_gate["pass"],ulp_le_2_fraction=ulp_fraction)
    full_perf=result["t1"]["performance"][PoseAblationMode.FULL_BATCH.value]["speedup"]>=1.5
    hybrid=None
    for mode,value in result["t1"]["performance"].items():
        if mode not in ("serial_all","full_batch") and value["speedup"]>=1.5 and result["t1"]["hg_and_step_analysis"][mode]["old_microkernel_gate"]:
            hybrid=mode;break
    readiness=integration_readiness(classification=classification,solver_gate=solver_gate["pass"],full_batch_performance_gate=full_perf,
        hybrid_mode=hybrid,hybrid_gate=hybrid is not None)
    result.update({"diagnostic_gate":"DIAGNOSTIC_MICROKERNEL_PASS" if result["t1"]["hg_and_step_analysis"]["full_batch"]["old_microkernel_gate"] else "DIAGNOSTIC_MICROKERNEL_FAIL",
        "solver_production_gate":"SOLVER_PRODUCTION_GATE_PASS" if solver_gate["pass"] else "SOLVER_PRODUCTION_GATE_FAIL",
        "solver_gate_detail":solver_gate,"classification":classification,"integration_readiness":readiness,"recommendation":readiness,
        "fp64_first_operator_reduction_ratio":ratio,"first_operator_ulp_le_2_fraction":ulp_fraction})
    target=Path(args.output_json);target.parent.mkdir(parents=True,exist_ok=True);target.write_text(json.dumps(result,indent=2),encoding="utf-8")
    print(f"[CLASSIFICATION] {classification}\n[READINESS] {readiness}\n[SAVE] {target}");return 0


if __name__=="__main__":raise SystemExit(main())
