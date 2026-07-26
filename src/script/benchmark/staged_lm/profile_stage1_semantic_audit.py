"""Run the single Stage-1 T=4 semantic/Hessian audit configuration."""
from __future__ import annotations

import argparse, contextlib, io, json, sys
from dataclasses import asdict
from pathlib import Path
import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from lib_pano.human_fitting.staged_lm.stage_definitions.rigid import build_stage1_definition
from script.benchmark.staged_lm.batch_common import environment_record
from script.benchmark.staged_lm.profile_stage1_shadow_lm import _downstream, _result
from script.benchmark.staged_lm.stage1_batch_candidates import build_stage1_batch_cache
from script.benchmark.staged_lm.stage1_semantic_audit import (
    evaluate_gates, geometry_audit, hessian_audit, objective_audit,
    parameter_labels, raw_parameter_audit,
)
from script.benchmark.staged_lm.stage1_shadow_common import (
    Stage1NEMode, build_stage1_shadow_linearization_common,
    evaluate_stage1_equations,
)
from script.benchmark.staged_lm.stage1_shadow_lm import (
    compare_stage1_lm_results, run_stage1_full_batch_candidate,
    run_stage1_serial_reference,
)
from script.human_fitting.staged_lm.app import build_context
from script.human_fitting.staged_lm.cli import build_argument_parser


def parser():
    p=build_argument_parser()
    for action in p._actions:
        if action.dest=="output_dir": action.required=False;action.default="/tmp/stage1_semantic_audit"
    p.add_argument("--rotation_perturbation_scale",type=float,default=.01)
    p.add_argument("--translation_perturbation_scale",type=float,default=.01)
    p.add_argument("--seed",type=int,default=20260715)
    p.add_argument("--candidate_mode",default="full_batch_group_gemm")
    p.add_argument("--source_shadow_json")
    p.add_argument("--run_downstream_regression",action="store_true")
    p.add_argument("--audit_repeats",type=int,default=1)
    p.add_argument("--output_json",required=True)
    return p


def perturb(initial,rotation,translation,seed):
    value=initial.detach().clone();generator=torch.Generator(device=value.device);generator.manual_seed(seed)
    framed=value.reshape(-1,12)
    noise=torch.randn((framed.shape[0],6),generator=generator,device=value.device,dtype=value.dtype)*rotation
    framed[:,0:3]+=noise[:,0:3];framed[:,6:9]+=noise[:,3:6]
    noise=torch.randn((framed.shape[0],6),generator=generator,device=value.device,dtype=value.dtype)*translation
    framed[:,3:6]+=noise[:,0:3];framed[:,9:12]+=noise[:,3:6]
    return value


def source_check(path):
    if not path:return {"provided":False}
    data=json.loads(Path(path).read_text());frame=next((x for x in data.get("frame_sets",[]) if x.get("T")==4),None)
    key="rotation=0.01|translation=0.01";mode=None if frame is None else frame.get("perturbations",{}).get(key,{}).get("modes",{}).get("full_batch_group_gemm")
    return {"provided":True,"path":path,"found_configuration":mode is not None,
            "expected_final_parameter_max_abs":None if mode is None else mode["independent_lm"]["comparison"]["final_parameter_max_abs"],
            "expected_acceptance_mismatch":None if mode is None else mode["lockstep"]["acceptance_mismatch_count"],
            "expected_first_divergent_trial":None if mode is None else mode["independent_lm"]["comparison"]["first_divergent_trial"]}


def run_audit(args):
    frames=[x.strip() for x in args.frames.split(",") if x.strip()]
    if len(frames)!=4:raise ValueError("SEMANTIC_AUDIT_REQUIRES_EXACTLY_T4")
    if args.candidate_mode!="full_batch_group_gemm":raise ValueError("SEMANTIC_AUDIT_REQUIRES_FULL_BATCH_GROUP_GEMM")
    if args.reproj_objective!="charbonnier_irls":raise ValueError("SEMANTIC_AUDIT_REQUIRES_CHARBONNIER_IRLS")
    args.profile_timing=True
    context=build_context(args);stage=build_stage1_definition(context);cache=stage.analytic_cache
    batch=build_stage1_batch_cache(cache);labels=parameter_labels(context.frame_indices)
    initial=perturb(stage.initial_lm_parameters,args.rotation_perturbation_scale,args.translation_perturbation_scale,args.seed)
    runs=[];serial=candidate=None
    for repeat in range(args.audit_repeats):
        with contextlib.redirect_stdout(io.StringIO()):
            serial=run_stage1_serial_reference(stage,context,initial.clone())
            candidate=run_stage1_full_batch_candidate(stage,batch,context,initial.clone())
        comparison=compare_stage1_lm_results(serial,candidate);raw=raw_parameter_audit(serial.parameters,candidate.parameters,labels)
        runs.append({"repeat":repeat,"raw_parameter_max_abs":raw.max_abs,"worst_label":raw.worst,
                     "comparison":comparison})
    raw=raw_parameter_audit(serial.parameters,candidate.parameters,labels)
    comparison=compare_stage1_lm_results(serial,candidate)
    geometry=geometry_audit(cache,serial.parameters,candidate.parameters)
    objective=objective_audit(stage,serial.parameters,candidate.parameters)
    common=build_stage1_shadow_linearization_common(cache,batch,serial.parameters,
        is_irls=context.is_irls,charbonnier_eps=context.charbonnier_eps,
        aggregation_spec=stage.aggregation_spec,rep_scale=context.scaling.rep_scale)
    evaluation=evaluate_stage1_equations(Stage1NEMode.SERIAL_REFERENCE,cache,batch,serial.parameters,
        common.padded_row_scales,common.flat_row_scales)
    hessian=hessian_audit(evaluation.equations.hessian,evaluation.equations.gradient,
                          candidate.parameters-serial.parameters,labels)
    hessian["raw_coordinates"]["actual_objective_delta"]=objective["candidate"]-objective["serial"]
    downstream={"gate_pass":False,"status":"NOT_RUN"}
    if args.run_downstream_regression:
        with contextlib.redirect_stdout(io.StringIO()): downstream=_downstream(context,stage,serial,candidate)
    gates=evaluate_gates(raw,geometry,objective,comparison,downstream,hessian)
    rigid=geometry["rigid_transforms"]
    output={"environment":environment_record(torch.device(args.device),torch.float32),
      "exact_configuration":{"frames":[int(x) for x in frames],"T":4,"G":len(cache.groups),"device":args.device,
        "reproj_objective":args.reproj_objective,"rotation_perturbation":args.rotation_perturbation_scale,
        "translation_perturbation":args.translation_perturbation_scale,"seed":args.seed,
        "candidate":args.candidate_mode},"source_shadow_json_check":source_check(args.source_shadow_json),
      "reproduction":{"serial_result":_result(serial),"candidate_result":_result(candidate),
        "comparison":comparison,"audit_repeats":runs},"parameter_labels":[asdict(x) for x in labels],
      "raw_parameter_audit":asdict(raw),"rigid_transform_audit":rigid,
      "geometry_3d":geometry["geometry_3d"],"uv_projection":geometry["uv_projection"],
      "raw_residual":geometry["raw_residual"],"objective":{k:v for k,v in objective.items() if k!="path"},
      "objective_path":objective["path"],"hessian":hessian,"downstream":downstream,
      "gates":{k:v for k,v in gates.items() if k not in ("classification","integration_readiness")},
      "classification":gates["classification"],"integration_readiness":gates["integration_readiness"],
      "recommendation":"Integrate FULL_BATCH only if all semantic gates and flatness audit pass; no hybrid was run."}
    return output


def main():
    args=parser().parse_args();result=run_audit(args);path=Path(args.output_json);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(result,indent=2,ensure_ascii=False));raw=result["raw_parameter_audit"];h=result["hessian"]
    print(f"[RAW_WORST] {raw['worst']}")
    print(f"[GEOMETRY] uv_max={result['uv_projection']['global']['max']:.3e}px vertex_norm_max={result['geometry_3d']['global']['normalized_max_by_rms_radius']:.3e}")
    print(f"[HESSIAN] flat={h['flat_direction_classification']} normalized_rayleigh={h['jacobi_scaled']['normalized_scaled_rayleigh']:.3e}")
    print(f"[CLASSIFICATION] {result['classification']}");print(f"[READINESS] {result['integration_readiness']}");print(f"[SAVE] {path}")
    return 0


if __name__=="__main__":raise SystemExit(main())
