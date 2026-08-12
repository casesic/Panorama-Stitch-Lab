"""Cold-start and steady-state benchmark for the complete staged-LM pipeline."""

from __future__ import annotations

import contextlib
import copy
import gc
import io
import json
import math
import statistics
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any
from unittest.mock import patch

import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from script.human_fitting.staged_lm.app import build_context
from lib_pano.human_fitting.staged_lm.output import stage_outcome_to_json
from script.human_fitting.staged_lm.cli import build_argument_parser
from lib_pano.human_fitting.staged_lm.output import (
    build_output_directory, collect_stage2_arrays,
    save_experiment_outputs,
)
from script.validation.staged_lm.objective_breakdown import (
    compute_objective_breakdown,
)
from lib_pano.human_fitting.staged_lm.pipeline import run_stage2
from lib_pano.human_fitting.staged_lm.residuals import (
    flatten_reprojection_groups,
    make_raw_reproj_groups,
)
from lib_pano.human_fitting.staged_lm.stage_definitions import (
    joint as pose_shape_definitions,
    pose as pose_definitions,
    rigid as rigid_definitions,
)
from lib_pano.human_fitting.staged_lm.stage_definitions.pose_shape import build_pose_shape_stage_definition
from lib_pano.human_fitting.staged_lm.stage_definitions.pose import build_pose_stage_definition
from lib_pano.human_fitting.staged_lm.stage_definitions.rigid import build_stage1_definition
from lib_pano.human_fitting.staged_lm.stage_runtime import run_stage
from lib_pano.human_fitting.staged_lm.stage2_shared_cache import (
    build_stage2_shared_geometry_cache,
)


def parse_args():
    parser = build_argument_parser()
    parser.add_argument("--benchmark_warmup_runs", type=int, default=1)
    parser.add_argument("--benchmark_repeats", type=int, default=5)
    parser.add_argument("--benchmark_profile_breakdown", action="store_true")
    parser.add_argument("--benchmark_compare_cpu_cuda", action="store_true")
    parser.add_argument("--benchmark_include_output_save", action="store_true")
    parser.add_argument("--benchmark_output_json")
    parser.add_argument("--benchmark_show_stage_logs", action="store_true")
    parser.add_argument("--profile_torch", action="store_true")
    parser.add_argument("--export_trace")
    parser.add_argument("--audit_full_forward_calls", action="store_true")
    return parser.parse_args()


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _range(name: str, enabled: bool):
    return torch.profiler.record_function(name) if enabled else contextlib.nullcontext()


def run_fitting_once(context, *, record_ranges: bool = False) -> dict[str, Any]:
    """Run all production stage builders/solvers from unchanged context initial state."""
    timing: dict[str, float] = {"timing_schema_version": 5}
    _sync(context.device)
    fitting_start = time.perf_counter()
    with _range("stage1_prepare", record_ranges):
        stage1_definition = build_stage1_definition(context)
    preparation = stage1_definition.preparation_timing
    timing["stage1_prepare_total_ms"] = preparation.total_ms
    timing["stage1_cache_setup_ms"] = preparation.cache_setup_ms
    timing["stage1_definition_bind_ms"] = preparation.definition_bind_ms
    with _range("stage1_solver", record_ranges):
        start=time.perf_counter(); stage1=run_stage(stage1_definition,context);_sync(context.device)
        timing["stage1_run_ms"]=(time.perf_counter()-start)*1000.0
        timing["stage1_run_overhead_ms"] = max(
            0.0, timing["stage1_run_ms"] - stage1.lm_result.wall_time_ms,
        )
    with _range("stage2_pipeline", record_ranges):
        stage2=run_stage2(context,stage1,timing,profiler_ranges=record_ranges)
    pose=stage2.pose_warmup;joint=stage2.pose_shape_refinement
    if joint is None:raise RuntimeError("DEBUG121_REQUIRES_FULL_STAGE2")
    with _range("gate_c", record_ranges):
        start=time.perf_counter()
        beta=context.objective_config.beta_abs_limit*torch.tanh(joint.lm_result.parameters[-context.beta_dim:])
        checks={
            "beta_sat_lt_0.9":float(beta.abs().max())/context.objective_config.beta_abs_limit<.9,
            "s1_status_ok":stage1.lm_result.status not in ("LM_NO_ACCEPTED_TRIAL","NUMERICAL_FAILURE"),
            "pose_status_ok":pose.lm_result.status not in ("LM_NO_ACCEPTED_TRIAL","NUMERICAL_FAILURE"),
            "joint_status_ok":joint.lm_result.status not in ("LM_NO_ACCEPTED_TRIAL","NUMERICAL_FAILURE"),
        }
        gate={"verdict":"GATE_C_NOT_EVALUATED","oracle_enabled":False,
              "checks":checks}
        timing["gate_c_ms"]=(time.perf_counter()-start)*1000.0
    _sync(context.device)
    timing["fitting_total_ms"]=(time.perf_counter()-fitting_start)*1000.0
    terms_fn = joint.definition.objective_terms_fn
    if terms_fn is None:
        raise RuntimeError("MISSING_JOINT_OBJECTIVE_TERMS_FN")
    _sync(context.device)
    breakdown_start = time.perf_counter()
    compute_objective_breakdown(
        terms_fn,
        joint.lm_result.parameters,
        expected_true_objective=joint.lm_result.objective,
    )
    _sync(context.device)
    timing["objective_breakdown_ms"] = (
        time.perf_counter() - breakdown_start
    ) * 1000.0
    return {"stage1":stage1,"pose":pose,"joint":joint,"timing":timing,
            "gate":gate,"physical_beta":beta.detach(),
            "definitions":(stage1_definition,pose.definition,joint.definition)}


class FullForwardCallAudit:
    """Debug-only phase counter for production full reprojection forwards."""

    def __init__(self) -> None:
        self.counts = {
            "stage_preparation": 0,
            "solver_run": 0,
            "stage_json_build": 0,
            "objective_breakdown": 0,
            "output_array_build": 0,
        }
        self.current_phase = "stage_preparation"
        self._original = make_raw_reproj_groups

    @contextmanager
    def phase(self, name: str):
        previous = self.current_phase
        self.current_phase = name
        try:
            yield
        finally:
            self.current_phase = previous

    def counted_forward(self, *args, **kwargs):
        self.counts[self.current_phase] += 1
        return self._original(*args, **kwargs)

    @contextmanager
    def installed(self):
        with contextlib.ExitStack() as stack:
            for module in (
                rigid_definitions,
                pose_definitions,
                pose_shape_definitions,
            ):
                stack.enter_context(patch.object(
                    module, "make_raw_reproj_groups", self.counted_forward,
                ))
            yield self


def audit_full_forward_calls(context) -> dict[str, int]:
    """Run production APIs once and attribute every full raw-group call."""
    audit = FullForwardCallAudit()
    with audit.installed():
        with audit.phase("stage_preparation"):
            stage1_definition = build_stage1_definition(context)
        with audit.phase("solver_run"):
            stage1 = run_stage(stage1_definition, context)
        with audit.phase("stage_preparation"):
            shared_cache = build_stage2_shared_geometry_cache(context)
            pose_definition, _ = build_pose_stage_definition(
                context, stage1, shared_cache,
            )
        with audit.phase("solver_run"):
            pose = run_stage(pose_definition, context)
        with audit.phase("stage_preparation"):
            pose_shape_definition, _ = build_pose_shape_stage_definition(
                context, pose, shared_cache,
            )
        with audit.phase("solver_run"):
            joint = run_stage(pose_shape_definition, context)
        with audit.phase("stage_json_build"):
            stage_outcome_to_json(stage1)
            stage_outcome_to_json(pose)
            stage_outcome_to_json(joint)
        with audit.phase("objective_breakdown"):
            if pose_shape_definition.objective_terms_fn is None:
                raise RuntimeError("MISSING_JOINT_OBJECTIVE_TERMS_FN")
            compute_objective_breakdown(
                pose_shape_definition.objective_terms_fn,
                joint.lm_result.parameters,
                expected_true_objective=joint.lm_result.objective,
            )
        beta = context.objective_config.beta_abs_limit * torch.tanh(
            joint.lm_result.parameters[-context.beta_dim:]
        )
        with audit.phase("output_array_build"):
            collect_stage2_arrays(
                stage1_definition.initial_lm_parameters,
                pose_definition.initial_lm_parameters,
                pose_shape_definition.initial_lm_parameters,
                stage1, pose, joint, context.beta_anchor, context.beta_latent0,
                beta, stage1_definition.raw_groups_fn,
                pose_definition.raw_groups_fn, pose_shape_definition.raw_groups_fn,
            )
    return audit.counts


def _summary(values: list[float]) -> dict[str,float]:
    ordered=sorted(values)
    position=.9*(len(ordered)-1);lower=math.floor(position);upper=math.ceil(position)
    p90=(ordered[lower] if lower==upper else
         ordered[lower]+(ordered[upper]-ordered[lower])*(position-lower))
    return {"p50":statistics.median(values),
            "p90":p90,
            "mean":statistics.mean(values),"min":min(values),"max":max(values)}


def _summarize_runs(runs: list[dict[str,Any]], frame_count: int) -> dict[str,Any]:
    keys=sorted({key for run in runs for key in run["timing"]})
    breakdown={key:_summary([run["timing"].get(key,0.) for run in runs]) for key in keys}
    fit=breakdown["fitting_total_ms"]
    return {"runs":len(runs),"timing":breakdown,"fit_ms_segment":fit["p50"],
            "fit_ms_per_frame":fit["p50"]/frame_count,
            "effective_fps":1000.0*frame_count/fit["p50"],
            "effective_person_frames_per_second":1000.0*frame_count/fit["p50"]}


def _stage_comparison(cpu, cuda, name: str) -> dict[str,float]:
    left=cpu[name].lm_result;right=cuda[name].lm_result
    parameter_diff=float((left.parameters.cpu()-right.parameters.cpu()).abs().max())
    objective_rel=abs(left.objective-right.objective)/max(abs(left.objective),1e-16)
    for metric in ("rms","median","p95"):
        absolute=abs(left.rms_metrics[metric]-right.rms_metrics[metric])
        relative=absolute/max(abs(left.rms_metrics[metric]),1e-16)
        if absolute>1e-2 and relative>1e-4:raise RuntimeError(f"CPU_CUDA_METRIC_MISMATCH {name} {metric}")
    torch.testing.assert_close(right.parameters.cpu(),left.parameters.cpu(),rtol=1e-4,atol=1e-3)
    if objective_rel>1e-4:raise RuntimeError(f"CPU_CUDA_OBJECTIVE_MISMATCH {name}")
    left_def=cpu["definitions"][("stage1","pose","joint").index(name)]
    right_def=cuda["definitions"][("stage1","pose","joint").index(name)]
    left_raw=flatten_reprojection_groups(left_def.raw_groups_fn(left.parameters)).detach().cpu()
    right_raw=flatten_reprojection_groups(right_def.raw_groups_fn(right.parameters)).detach().cpu()
    torch.testing.assert_close(right_raw,left_raw,rtol=1e-5,atol=2e-3)
    return {"parameter_max_abs":parameter_diff,"objective_relative":objective_rel,
            "raw_max_abs":float((left_raw-right_raw).abs().max())}


def compare_cpu_cuda(args) -> dict[str,Any]:
    if not torch.cuda.is_available():raise RuntimeError("CUDA_REQUESTED_BUT_UNAVAILABLE")
    cpu_args=copy.deepcopy(args);cpu_args.device="cpu";cpu_args.profile_timing=False
    cuda_args=copy.deepcopy(args);cuda_args.device=args.device if str(args.device).startswith("cuda") else "cuda:0";cuda_args.profile_timing=False
    cpu_context=build_context(cpu_args);cuda_context=build_context(cuda_args)
    cpu_s1=build_stage1_definition(cpu_context);cuda_s1=build_stage1_definition(cuda_context)
    torch.testing.assert_close(cuda_context.beta_anchor.cpu(),cpu_context.beta_anchor.cpu(),rtol=0,atol=0)
    torch.testing.assert_close(cuda_context.beta_latent0.cpu(),cpu_context.beta_latent0.cpu(),rtol=0,atol=0)
    torch.testing.assert_close(cuda_s1.initial_lm_parameters.cpu(),cpu_s1.initial_lm_parameters.cpu(),rtol=0,atol=0)
    for left,right in zip(cpu_context.body_pose_anchors,cuda_context.body_pose_anchors):
        torch.testing.assert_close(right.cpu(),left.cpu(),rtol=0,atol=0)
    for left,right in zip(cpu_context.cam2_vertex_ids_by_frame,cuda_context.cam2_vertex_ids_by_frame):
        torch.testing.assert_close(right.cpu(),left.cpu(),rtol=0,atol=0)
    for left,right in zip(cpu_context.cam2_targets_by_frame,cuda_context.cam2_targets_by_frame):
        torch.testing.assert_close(right.cpu(),left.cpu(),rtol=0,atol=0)
    for left,right in zip(cpu_context.cam4_vertex_ids_by_frame,cuda_context.cam4_vertex_ids_by_frame):
        torch.testing.assert_close(right.cpu(),left.cpu(),rtol=0,atol=0)
    for left,right in zip(cpu_context.cam4_targets_by_frame,cuda_context.cam4_targets_by_frame):
        torch.testing.assert_close(right.cpu(),left.cpu(),rtol=0,atol=0)
    for left,right in zip(cpu_context.cam4_vertex_ids_by_frame,cuda_context.cam4_vertex_ids_by_frame):
        torch.testing.assert_close(right.cpu(),left.cpu(),rtol=0,atol=0)
    for left,right in zip(cpu_context.cam4_targets_by_frame,cuda_context.cam4_targets_by_frame):
        torch.testing.assert_close(right.cpu(),left.cpu(),rtol=0,atol=0)
    cpu=run_fitting_once(cpu_context);cuda=run_fitting_once(cuda_context)
    comparison={name:_stage_comparison(cpu,cuda,name) for name in ("stage1","pose","joint")}
    torch.testing.assert_close(cuda["physical_beta"].cpu(),cpu["physical_beta"].cpu(),rtol=1e-4,atol=1e-4)
    comparison["physical_beta_max_abs"]=float((cuda["physical_beta"].cpu()-cpu["physical_beta"].cpu()).abs().max())
    return comparison


def _final_result_summary(run: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name in ("stage1", "pose", "joint"):
        lm = run[name].lm_result
        result[name] = {
            "objective": lm.objective,
            "status": lm.status,
            "accepted": lm.counters.accepted_steps,
            "rejected": lm.counters.rejected_trials,
            "metrics": lm.rms_metrics,
        }
    result["physical_beta"] = run["physical_beta"].detach().cpu().tolist()
    return result


def _unique_tensor_bytes(tensors: list[torch.Tensor]) -> int:
    seen: set[int] = set()
    total = 0
    for tensor in tensors:
        pointer = tensor.data_ptr()
        if pointer not in seen:
            seen.add(pointer)
            total += tensor.numel() * tensor.element_size()
    return total


def _cache_memory_audit(run: dict[str, Any]) -> dict[str, int]:
    """Report persistent cache bytes, deduplicating shared tensor pointers."""
    _stage1, pose_definition, pose_shape_definition = run["definitions"]
    pose = pose_definition.analytic_cache
    joint = pose_shape_definition.analytic_cache
    shared: list[torch.Tensor] = [pose.model.parents]
    pose_derived = [pose.model.v_shaped, pose.model.joints]
    joint_derived = [
        joint.model.base_v_shaped, joint.model.beta_shapedirs,
        joint.model.base_joints, joint.model.d_joints_d_beta,
    ]
    for pose_group, pose_shape_group in zip(pose.groups, joint.groups):
        shared.extend((
            pose_group.vertex_ids, pose_group.targets,
            pose_group.left_hand_pose, pose_group.right_hand_pose,
            pose_group.scale, pose_group.selected_posedirs,
            pose_group.selected_lbs_weights,
        ))
        pose_derived.append(pose_group.selected_v_shaped)
        joint_derived.extend((
            pose_shape_group.selected_base_v_shaped,
            pose_shape_group.selected_beta_shapedirs,
        ))
    shared_bytes = _unique_tensor_bytes(shared)
    pose_bytes = _unique_tensor_bytes(pose_derived)
    joint_bytes = _unique_tensor_bytes(joint_derived)
    return {
        "shared_persistent_tensor_bytes": shared_bytes,
        "pose_derived_tensor_bytes": pose_bytes,
        "joint_derived_tensor_bytes": joint_bytes,
        "unique_stage2_cache_tensor_bytes": _unique_tensor_bytes(
            shared + pose_derived + joint_derived,
        ),
        "legacy_duplicated_common_tensor_bytes": shared_bytes,
        "estimated_legacy_stage2_cache_tensor_bytes": (
            2 * shared_bytes + pose_bytes + joint_bytes
        ),
        "pose_batch_persistent_tensor_bytes": int(pose.batch_persistent_bytes),
    }


def _save_benchmark_output(args, context, run, summary) -> dict[str, float]:
    """Exercise the production output conversion/save boundary outside FPS timing."""
    s1, pose, joint = run["stage1"], run["pose"], run["joint"]
    s1_def, pose_def, joint_def = run["definitions"]
    build_start = time.perf_counter()
    with _range("output_build", True):
        arrays = collect_stage2_arrays(
            s1_def.initial_lm_parameters, pose_def.initial_lm_parameters,
            joint_def.initial_lm_parameters, s1, pose, joint,
            context.beta_anchor, context.beta_latent0, run["physical_beta"],
            s1_def.raw_groups_fn, pose_def.raw_groups_fn, joint_def.raw_groups_fn,
        )
    build_ms = (time.perf_counter()-build_start)*1000.0
    directory = build_output_directory(Path(args.output_dir), context.frame_indices)
    start = time.perf_counter()
    with _range("output_save", True):
        save_experiment_outputs(directory, {"benchmark": summary}, arrays)
    return {"output_array_build_ms": build_ms,
            "output_save_ms": (time.perf_counter()-start)*1000.0}


def main() -> int:
    args=parse_args();args.stop_after_stage="stage2"
    # Fine-grained stage profiling synchronizes CUDA work.  It is enabled only
    # for the separate breakdown run below, never for cold/steady wall timing.
    args.profile_timing=False
    command_start=time.perf_counter()
    context=build_context(args)
    resident_memory=(torch.cuda.memory_allocated(context.device)
                     if context.device.type=="cuda" else 0)
    cold_start=time.perf_counter();cold=run_fitting_once(context);cold_fit_ms=(time.perf_counter()-cold_start)*1000.0
    cold_final=_final_result_summary(cold)
    # A StageOutcome retains stage definitions and their analytic caches.  It
    # is not part of the persistent production context and must not leak into
    # subsequent steady-state memory measurements.
    del cold
    gc.collect()
    for _ in range(args.benchmark_warmup_runs):
        stream=contextlib.nullcontext() if args.benchmark_show_stage_logs else contextlib.redirect_stdout(io.StringIO())
        with stream:run_fitting_once(context)
    gc.collect()
    runs=[];last_run=None
    for _ in range(args.benchmark_repeats):
        if last_run is not None:
            del last_run
            gc.collect()
        if context.device.type=="cuda":torch.cuda.reset_peak_memory_stats(context.device)
        before=torch.cuda.memory_allocated(context.device) if context.device.type=="cuda" else 0
        stream=contextlib.nullcontext() if args.benchmark_show_stage_logs else contextlib.redirect_stdout(io.StringIO())
        with stream:r=run_fitting_once(context)
        after=torch.cuda.memory_allocated(context.device) if context.device.type=="cuda" else 0
        record={"timing":dict(r["timing"])}
        if context.device.type=="cuda":
            record["memory"]={"memory_allocated_before":before,
                "max_memory_allocated":torch.cuda.max_memory_allocated(context.device),
                "max_memory_reserved":torch.cuda.max_memory_reserved(context.device),
                "memory_allocated_after":after}
        runs.append(record);last_run=r
        del r
    steady=_summarize_runs(runs,context.frame_count)
    summary={"execution":{"requested_device":args.device,"resolved_device":str(context.device),
                "dtype":"float32","environment":context.device_environment},
             "context_timing":context.context_timing,"cold_first_fit_ms":cold_fit_ms,
             "cold_command_ms":context.context_timing["context_total_ms"]+cold_fit_ms,
             "steady_state":steady,"cold_final":cold_final,
             "command_elapsed_without_output_ms":(time.perf_counter()-command_start)*1000.0}
    if last_run is not None:
        summary["cache_memory_audit"] = _cache_memory_audit(last_run)
    if context.device.type=="cuda":
        summary["context_model_resident_bytes"]=resident_memory
        summary["memory"]=[run["memory"] for run in runs]
    if args.benchmark_profile_breakdown:
        old=context.args.profile_timing;context.args.profile_timing=True
        stream=contextlib.nullcontext() if args.benchmark_show_stage_logs else contextlib.redirect_stdout(io.StringIO())
        with stream:profiled=run_fitting_once(context)
        context.args.profile_timing=old
        summary["profile_breakdown"]={name:(profiled[name].lm_result.profile_timing or {})
                                      for name in ("stage1","pose","joint")}
    if args.benchmark_compare_cpu_cuda:summary["cpu_cuda_comparison"]=compare_cpu_cuda(args)
    if args.audit_full_forward_calls:
        stream = (
            contextlib.nullcontext()
            if args.benchmark_show_stage_logs
            else contextlib.redirect_stdout(io.StringIO())
        )
        with stream:
            summary["full_forward_call_audit"] = audit_full_forward_calls(context)
    # Preserve the expensive benchmark even if profiler aggregation or trace
    # export is interrupted later.
    if args.benchmark_output_json:
        path=Path(args.benchmark_output_json);path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(summary,ensure_ascii=False,indent=2))
    if args.profile_torch:
        activities=[torch.profiler.ProfilerActivity.CPU]
        if context.device.type=="cuda":activities.append(torch.profiler.ProfilerActivity.CUDA)
        with torch.profiler.profile(activities=activities,record_shapes=True,profile_memory=True,with_stack=False) as profiler:
            with _range("context_build", True):
                profiled_context=build_context(copy.deepcopy(args))
            run_fitting_once(profiled_context,record_ranges=True)
        sort="self_cuda_time_total" if context.device.type=="cuda" else "self_cpu_time_total"
        key_averages=profiler.key_averages()
        print(key_averages.table(sort_by=sort,row_limit=30))
        selected=("aten::einsum","aten::bmm","aten::matmul","aten::select","aten::stack",
                  "aten::cat","aten::_to_copy","aten::cholesky","aten::cholesky_solve")
        averages={item.key:item.count for item in key_averages}
        summary["profiler_operator_call_counts"]={name:int(averages.get(name,0)) for name in selected}
        if args.export_trace:
            path=Path(args.export_trace);path.parent.mkdir(parents=True,exist_ok=True);profiler.export_chrome_trace(str(path))
    if args.benchmark_include_output_save:
        if last_run is None:
            raise RuntimeError("BENCHMARK_OUTPUT_REQUIRES_AT_LEAST_ONE_REPEAT")
        summary["output"]=_save_benchmark_output(args,context,last_run,summary)
        summary["command_with_output_ms"]=(time.perf_counter()-command_start)*1000.0
    print(f"[COLD] context={context.context_timing['context_total_ms']:.3f} first_fit={cold_fit_ms:.3f}")
    print(f"[STEADY] {steady}")
    if args.benchmark_output_json:
        path=Path(args.benchmark_output_json);path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(summary,ensure_ascii=False,indent=2))
    return 0


if __name__=="__main__":raise SystemExit(main())
