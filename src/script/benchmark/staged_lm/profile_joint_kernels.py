"""Profile and bake off isolated Stage 2 joint analytic H/g kernels."""

from __future__ import annotations

import argparse
import math
import sys
from collections import defaultdict
from dataclasses import replace
from pathlib import Path
import statistics
import time

import numpy as np
import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from script.human_fitting.staged_lm.app import build_context
from lib_pano.human_fitting.staged_lm.irls import (
    build_primary_aggregation_spec, compute_irls_point_weights,
)
from lib_pano.human_fitting.staged_lm.stage2_joint_fast import (
    build_stage2_joint_fast_cache, evaluate_joint_shape_state,
    joint_fast_raw_groups,
)
from lib_pano.human_fitting.staged_lm.stage2_shared_cache import build_stage2_shared_geometry_cache
from lib_pano.human_fitting.staged_lm.stage2_joint_jacobian import (
    accumulate_stage2_joint_arrowhead_system,
    compute_joint_fk_with_combined_jacobian,
)
from script.validation.staged_lm.references.joint_backends import (
    compute_joint_fk_with_pose_and_beta_jacobians_separate_reference,
)
from script.benchmark.staged_lm.joint_kernel_candidates import (
    accumulate_active_group_hg_bench, accumulate_full_group_hg_bench,
    accumulate_stage2_joint_arrowhead_fp32_bench,
    benchmark_callable, build_joint_group_batch,
    batched_joint_group_analytic_bench, batched_group_hg_bench,
    compute_joint_fk_with_combined_jacobian_bench, hg_error,
    joint_depth_layout, lbs_weight_sparsity, profile_joint_group_baseline,
    selected_lbs_direct_b2, selected_lbs_low_memory_b1, tensor_audit,
    pack_exact_lbs_weights, selected_lbs_exact_sparse_b3,
)
from lib_pano.human_fitting.staged_lm.stage2_pose_jacobian import compute_selected_lbs_with_jacobian
from lib_pano.human_fitting.staged_lm.stages import (
    build_stage1_definition, build_pose_stage_definition,
    build_joint_stage_definition, run_stage,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("frames", "hmr_dir", "sam3d_dense_dir", "bbox_dir"):
        parser.add_argument(f"--{name}", required=True)
    parser.add_argument("--image_width", type=int, default=5760)
    parser.add_argument("--image_height", type=int, default=2880)
    parser.add_argument("--joint_npz")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--profile_torch", action="store_true")
    parser.add_argument("--export_trace")
    parser.add_argument("--run_candidates", action="store_true")
    return parser.parse_args()


def experiment_args(cli: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(
        frames=cli.frames, hmr_dir=cli.hmr_dir, sam3d_dense_dir=cli.sam3d_dense_dir,
        bbox_dir=cli.bbox_dir, image_width=cli.image_width, image_height=cli.image_height,
        output_dir="/tmp/debug119", reproj_objective="charbonnier_irls",
        device=cli.device, validate_device_contract=False,
        stop_after_stage="stage2_pose", run_lbfgs_oracle=False,
        jacobian_backend="analytic_stage1", stage1_eval_backend="analytic",
        validate_stage1_fast_eval=False,
        stage2_pose_linearization_backend="analytic", stage2_pose_eval_backend="fast",
        validate_stage2_pose_fast_eval=False,
        stage2_joint_linearization_backend="analytic", stage2_joint_eval_backend="fast",
        stage2_joint_linear_solver="schur", validate_stage2_joint_fast_eval=False,
        stage2_joint_fk_jacobian_backend="combined",
        validate_irls_first_order=False, validate_irls_each_accepted=False,
        profile_timing=False, progress_every_accepted=0,
    )


def _move_cache(cache, device: torch.device) -> None:
    for object_ in (cache.model, *cache.groups):
        for name, value in vars(object_).items():
            if isinstance(value, torch.Tensor):
                setattr(object_, name, value.to(device))
    cache.model.device = device


def _load_parameters(cli, context):
    stage1 = run_stage(build_stage1_definition(context), context)
    shared = build_stage2_shared_geometry_cache(context)
    pose_definition, _ = build_pose_stage_definition(context, stage1, shared)
    pose = run_stage(pose_definition, context)
    joint_definition, _ = build_joint_stage_definition(context, pose, shared)
    if cli.joint_npz:
        archive = np.load(cli.joint_npz)
        if "x_stage2_joint_lm" in archive:
            return torch.as_tensor(
                archive["x_stage2_joint_lm"], dtype=torch.float32,
                device=context.device,
            ), joint_definition
    return joint_definition.initial_lm_parameters.detach().clone(),joint_definition


def _summary(samples):
    ordered = sorted(samples)
    return {
        "p50": statistics.median(samples),
        "p90": ordered[min(len(ordered)-1, math.ceil(.9*len(ordered))-1)],
        "mean": statistics.mean(samples), "min": min(samples),
    }


def _rel(a, b):
    return float((a-b).norm()/max(float(b.norm()), 1e-30))


def _sync(tensor: torch.Tensor) -> None:
    if tensor.is_cuda:
        torch.cuda.synchronize(tensor.device)


def main() -> int:
    cli = parse_args()
    if cli.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA_REQUESTED_BUT_UNAVAILABLE")
    device = torch.device(cli.device)
    context = build_context(experiment_args(cli))
    parameters,joint_definition = _load_parameters(cli, context)
    cache = joint_definition.analytic_cache
    _move_cache(cache, device); parameters = parameters.to(device)
    beta_limit = context.objective_config.beta_abs_limit
    beta_anchor = context.beta_anchor.to(device)
    anchors = [value.to(device) for value in context.body_pose_anchors]
    groups = joint_fast_raw_groups(cache, parameters, beta_limit)
    spec = build_primary_aggregation_spec(groups)
    weights = compute_irls_point_weights(groups, context.charbonnier_eps, spec)
    scales = [torch.sqrt(value).repeat_interleave(2).detach() for value in weights]
    shape = evaluate_joint_shape_state(cache, parameters[-cache.model.beta_dim:], beta_limit)
    counts = [int(group.vertex_ids.numel()) for group in cache.groups]
    nmax, total = max(counts), sum(counts)
    padding = len(counts)*nmax-total
    print(f"[ENV] device={device} torch={torch.__version__} threads={torch.get_num_threads()} "
          f"gpu={torch.cuda.get_device_name(device) if device.type=='cuda' else 'none'}")
    for item in cache.groups:
        print(f"[GROUP] frame={item.frame_index} camera={item.camera} N={item.vertex_ids.numel()} "
              f"rows={item.targets.numel()} posedirs={tuple(item.selected_posedirs.shape)} "
              f"weights={tuple(item.selected_lbs_weights.shape)}")
    print(f"[DISTRIBUTION] groups={len(counts)} min={min(counts)} max={nmax} "
          f"mean={statistics.mean(counts):.2f} median={statistics.median(counts):.2f} total={total} "
          f"padding={padding} padding_ratio={padding/(len(counts)*nmax):.6f}")
    batch = build_joint_group_batch(cache.groups, device=device)
    print(f"[BATCH_LAYOUT] padded_bytes={batch.padded_bytes} shape_base={tuple(batch.selected_base_v.shape)}")
    depths, by_depth = joint_depth_layout(cache.model.parents)
    print(f"[TREE] depth_count={len(by_depth)} joints_by_depth={[int(v.numel()) for v in by_depth]}")
    print(f"[WEIGHT_SPARSITY] {lbs_weight_sparsity(cache.groups)}")

    # Fine-grained section timing. Profiling itself is intentionally isolated
    # from production counters and normal-equation timings.
    section_samples = defaultdict(list)
    per_group_samples = [defaultdict(list) for _ in cache.groups]
    serial_sum_samples = []
    first_profile = None
    for _ in range(cli.warmup):
        for group, scale in zip(cache.groups, scales):
            profile_joint_group_baseline(cache, group, shape, parameters, scale)
    for _ in range(cli.repeats):
        serial_total = 0.0
        for group_index, (group, scale) in enumerate(zip(cache.groups, scales)):
            profile = profile_joint_group_baseline(cache, group, shape, parameters, scale)
            first_profile = first_profile or profile
            group_total = sum(profile.section_ms.values()); serial_total += group_total
            for name, elapsed in profile.section_ms.items():
                section_samples[name].append(elapsed)
                per_group_samples[group_index][name].append(elapsed)
        serial_sum_samples.append(serial_total)
    total_p50 = statistics.median(serial_sum_samples)
    for name, samples in section_samples.items():
        stats = _summary(samples)
        print(f"[JOINT_PRODUCTION_COMBINED SECTION {name}] group_p50={stats['p50']:.4f} "
              f"group_p90={stats['p90']:.4f} mean={stats['mean']:.4f} min={stats['min']:.4f}")
    group_p50_values = []
    for index, group in enumerate(cache.groups):
        totals = [sum(per_group_samples[index][name][repeat] for name in per_group_samples[index])
                  for repeat in range(cli.repeats)]
        group_p50_values.append(statistics.median(totals))
        print(f"[PER_GROUP] frame={group.frame_index} camera={group.camera} N={group.targets.shape[0]} "
              f"p50={statistics.median(totals):.4f} p90={_summary(totals)['p90']:.4f}")
    weighted_mean = sum(value * count for value, count in zip(group_p50_values, counts)) / sum(counts)
    print(f"[GROUP_AGGREGATE] p50_across_groups={statistics.median(group_p50_values):.4f} "
          f"weighted_mean_by_N={weighted_mean:.4f} min_N={min(counts)} max_N={max(counts)}")
    print(f"[JOINT_PRODUCTION_COMBINED SERIAL_GROUP_SUM] p50={total_p50:.4f}ms")
    audit = tensor_audit(first_profile.tensors)
    largest = max(audit, key=lambda value: value["bytes"])
    print(f"[TENSOR_LARGEST] {largest}")
    print(f"[TENSOR_LISTED_SUM] bytes={sum(item['bytes'] for item in audit)} note=not_peak_lifetime")
    for item in audit:
        print(f"[TENSOR] {item}")
    if device.type == "cuda":
        print(f"[CUDA_MEMORY] allocated={torch.cuda.max_memory_allocated(device)} "
              f"reserved={torch.cuda.max_memory_reserved(device)}")

    if cli.run_candidates:
        t = first_profile.tensors
        parents = cache.model.parents
        baseline_fk = lambda: compute_joint_fk_with_pose_and_beta_jacobians_separate_reference(
            t["rotations"], t["d_rotations"], shape.joints, shape.d_joints_d_latent, parents,
        )
        combined_fk = lambda: compute_joint_fk_with_combined_jacobian(
            t["rotations"], t["d_rotations"], shape.joints, shape.d_joints_d_latent, parents,
        )
        base_fk_value = baseline_fk(); candidate_fk = combined_fk()
        fk_error = max(_rel(candidate_fk.d_relative_transforms_d_pose, base_fk_value.d_relative_transforms_d_pose),
                       _rel(candidate_fk.d_relative_transforms_d_beta, base_fk_value.d_relative_transforms_d_beta))
        print(f"[JOINT_SEPARATE_REFERENCE] reference_only=True "
              f"reference_backend=separate correctness_rel={fk_error:.3e} "
              f"separate={benchmark_callable(baseline_fk,parameters,cli.warmup,cli.repeats)} "
              f"combined={benchmark_callable(combined_fk,parameters,cli.warmup,cli.repeats)}")

        lbs_args = (t["selected_v_posed"], t["combined_v"], cache.groups[0].selected_lbs_weights,
                    t["relative_transforms"], t["combined_a"])
        baseline_lbs = lambda: compute_selected_lbs_with_jacobian(*lbs_args)
        b1 = lambda: selected_lbs_low_memory_b1(*lbs_args)
        b2 = lambda: selected_lbs_direct_b2(*lbs_args)
        base_lbs_value = baseline_lbs()
        b1_value, b2_value = b1(), b2()
        print(f"[CANDIDATE_B1] rel={_rel(b1_value[1],base_lbs_value[1]):.3e} "
              f"timing={benchmark_callable(b1,parameters,cli.warmup,cli.repeats)}")
        print(f"[CANDIDATE_B2] rel={_rel(b2_value[1],base_lbs_value[1]):.3e} "
              f"timing={benchmark_callable(b2,parameters,cli.warmup,cli.repeats)}")
        print(f"[BASELINE_LBS] timing={benchmark_callable(baseline_lbs,parameters,cli.warmup,cli.repeats)}")
        exact_indices,exact_weights=pack_exact_lbs_weights(cache.groups[0].selected_lbs_weights)
        b3=lambda:selected_lbs_exact_sparse_b3(
            t["selected_v_posed"],t["combined_v"],exact_indices,exact_weights,
            t["relative_transforms"],t["combined_a"])
        b3_value=b3()
        print(f"[CANDIDATE_B3] rel={_rel(b3_value[1],base_lbs_value[1]):.3e} "
              f"packed_width={exact_indices.shape[1]} timing={benchmark_callable(b3,parameters,cli.warmup,cli.repeats)}")

        result = first_profile.result; scale = scales[0]
        d0 = accumulate_full_group_hg_bench(result, scale, gemm_dtype=torch.float64)
        active = accumulate_active_group_hg_bench(result, cache.groups[0].camera, scale)
        print(f"[CANDIDATE_C] error={hg_error(active,d0)} "
              f"baseline={benchmark_callable(lambda:accumulate_full_group_hg_bench(result,scale,gemm_dtype=torch.float64),parameters,cli.warmup,cli.repeats)} "
              f"active={benchmark_callable(lambda:accumulate_active_group_hg_bench(result,cache.groups[0].camera,scale),parameters,cli.warmup,cli.repeats)}")
        old_matmul = torch.backends.cuda.matmul.allow_tf32
        old_cudnn = torch.backends.cudnn.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
        try:
            d1 = accumulate_full_group_hg_bench(result, scale, gemm_dtype=torch.float32)
            error = hg_error(d1, d0)
            p = result.local_pose_jacobian.shape[1]; b = result.shared_beta_jacobian.shape[1]
            h0 = torch.cat((torch.cat((d0.frame_hessian,d0.frame_beta_cross),1),
                            torch.cat((d0.frame_beta_cross.T,d0.beta_hessian),1)),0)
            h1 = torch.cat((torch.cat((d1.frame_hessian,d1.frame_beta_cross),1),
                            torch.cat((d1.frame_beta_cross.T,d1.beta_hessian),1)),0)
            g0 = torch.cat((d0.frame_gradient,d0.beta_gradient)); g1 = torch.cat((d1.frame_gradient,d1.beta_gradient))
            damping = float(torch.diagonal(h0).max())*1e-3
            eye = torch.eye(p+b,dtype=torch.float64,device=device)
            delta0 = torch.linalg.solve(h0+damping*eye,-g0); delta1 = torch.linalg.solve(h1+damping*eye,-g1)
            pred0 = -g0@delta0-.5*(delta0@h0@delta0); pred1=-g1@delta1-.5*(delta1@h1@delta1)
            print(f"[CANDIDATE_D1] error={error} delta_rel={_rel(delta1,delta0):.3e} "
                  f"pred_rel={float(abs(pred1-pred0)/max(abs(pred0),1e-30)):.3e} "
                  f"fp64={benchmark_callable(lambda:accumulate_full_group_hg_bench(result,scale,gemm_dtype=torch.float64),parameters,cli.warmup,cli.repeats)} "
                  f"fp32={benchmark_callable(lambda:accumulate_full_group_hg_bench(result,scale,gemm_dtype=torch.float32),parameters,cli.warmup,cli.repeats)}")
        finally:
            torch.backends.cuda.matmul.allow_tf32=old_matmul;torch.backends.cudnn.allow_tf32=old_cudnn

        for batch_size in (2,4,8,16):
            if batch_size>len(cache.groups):continue
            subset=cache.groups[:batch_size];padded=build_joint_group_batch(subset,device=device)
            orientations=[];translations=[];body=[]
            per_frame_dim=12+cache.model.body_pose_dim
            for item in subset:
                start=item.frame_position*per_frame_dim;frame=parameters[start:start+per_frame_dim]
                orientations.append(frame[:3] if item.camera=="cam2" else frame[6:9])
                translations.append(frame[3:6] if item.camera=="cam2" else frame[9:12]);body.append(frame[12:per_frame_dim])
            kernel_args=(padded.valid_mask,padded.selected_base_v,padded.selected_beta_dirs,padded.selected_posedirs,
                padded.selected_weights,padded.targets,padded.left_hand_pose,padded.right_hand_pose,padded.scale,
                torch.stack(orientations),torch.stack(translations),torch.stack(body),shape.beta,
                shape.d_beta_d_latent,shape.joints,shape.d_joints_d_latent,cache.model.parents,
                cache.image_width,cache.image_height)
            scale_padded=torch.zeros(batch_size,padded.valid_mask.shape[1],2,dtype=parameters.dtype,device=device)
            for index,item in enumerate(subset):
                count=int(item.vertex_ids.numel());scale_padded[index,:count]=scales[index].reshape(count,2)
            candidate_value=batched_joint_group_analytic_bench(*kernel_args)
            worst_residual=worst_pose=worst_beta=0.
            for index,item in enumerate(subset):
                reference=profile_joint_group_baseline(cache,item,shape,parameters,scales[index]).result
                count=int(item.vertex_ids.numel());camera_slice=slice(0,6) if item.camera=="cam2" else slice(6,12)
                reference_active=torch.cat((reference.local_pose_jacobian[:,camera_slice],reference.local_pose_jacobian[:,12:]),1)
                worst_residual=max(worst_residual,float((candidate_value[0][index,:count].reshape(-1)-reference.raw_residual).abs().max()))
                worst_pose=max(worst_pose,_rel(candidate_value[1][index,:count].reshape(-1,69),reference_active))
                worst_beta=max(worst_beta,_rel(candidate_value[2][index,:count].reshape(-1,cache.model.beta_dim),reference.shared_beta_jacobian))
            kernel_timing=benchmark_callable(lambda:batched_joint_group_analytic_bench(*kernel_args),parameters,cli.warmup,cli.repeats)
            hg_timing=benchmark_callable(lambda:batched_group_hg_bench(kernel_args,scale_padded),parameters,cli.warmup,cli.repeats)
            padding_ratio=1-sum(int(item.vertex_ids.numel()) for item in subset)/(batch_size*padded.valid_mask.shape[1])
            print(f"[BATCH_{batch_size}] residual_max={worst_residual:.3e} pose_rel={worst_pose:.3e} beta_rel={worst_beta:.3e} "
                  f"padding_ratio={padding_ratio:.6f} padded_bytes={padded.padded_bytes} "
                  f"kernel={kernel_timing} hg_total={hg_timing}")

        if joint_definition is not None and device.type=="cpu":
            d0_definition=replace(joint_definition,fast_eval_validator=None)
            d0_outcome=run_stage(d0_definition,context)
            original_builder=joint_definition.linearization_builder
            def d1_builder(point,specification):
                linear=original_builder(point,specification)
                current_groups=joint_fast_raw_groups(cache,point,beta_limit)
                current_weights=compute_irls_point_weights(current_groups,context.charbonnier_eps,specification)
                frozen=[torch.sqrt(value).repeat_interleave(2).detach() for value in current_weights]
                linear["normal_equations_fn"]=lambda value:accumulate_stage2_joint_arrowhead_fp32_bench(
                    cache,value,beta_limit,frozen,anchors,beta_anchor,
                    context.scaling.pose_anchor_scale,context.scaling.shape_anchor_scale)
                linear["linear_solver_backend"]="arrowhead_schur"
                return linear
            d1_definition=replace(joint_definition,linearization_builder=d1_builder,
                                  fast_eval_validator=None,linear_solver_backend="arrowhead_schur")
            d1_outcome=run_stage(d1_definition,context)
            left,right=d0_outcome.lm_result,d1_outcome.lm_result
            objective_rel=abs(left.objective-right.objective)/max(abs(left.objective),1e-30)
            parameter_max=float((left.parameters-right.parameters).abs().max())
            history_equal=left.trial_history==right.trial_history
            print(f"[D1_LM_REGRESSION] objective_rel={objective_rel:.3e} parameter_max={parameter_max:.3e} "
                  f"acc_rej_d0={left.counters.accepted_steps}/{left.counters.rejected_trials} "
                  f"acc_rej_d1={right.counters.accepted_steps}/{right.counters.rejected_trials} "
                  f"lambda_d0={left.final_lambda:.9e} lambda_d1={right.final_lambda:.9e} "
                  f"status_d0={left.status} status_d1={right.status} trial_history_exact={history_equal}")

        # torch.compile is tested only on the pure B1 kernel. PyTorch 2.0 can
        # reject fullgraph compilation on some CPU installations; report that
        # outcome instead of changing production code.
        try:
            try:
                import torch._dynamo as dynamo
                explanation=dynamo.explain(selected_lbs_low_memory_b1,*lbs_args)
                if isinstance(explanation, tuple):
                    summary=explanation[0]
                else:
                    summary=(f"graph_count={getattr(explanation,'graph_count','unknown')} "
                             f"graph_break_count={getattr(explanation,'graph_break_count','unknown')}")
                print(f"[DYNAMO_EXPLAIN_B1] {summary}")
            except Exception as explain_error:
                print(f"[DYNAMO_EXPLAIN_B1] unavailable={type(explain_error).__name__}:{explain_error}")
            compile_start=time.perf_counter();compiled=torch.compile(selected_lbs_low_memory_b1,fullgraph=True,dynamic=False)
            compiled(*lbs_args); _sync(parameters); compile_ms=(time.perf_counter()-compile_start)*1000
            compiled_timing=benchmark_callable(lambda:compiled(*lbs_args),parameters,cli.warmup,cli.repeats)
            print(f"[COMPILE_B1] status=PASS compile_ms={compile_ms:.3f} timing={compiled_timing}")
        except Exception as exc:
            print(f"[COMPILE_B1] status=SKIPPED_OR_FAILED reason={type(exc).__name__}:{exc}")

    complete = benchmark_callable(
        lambda: accumulate_stage2_joint_arrowhead_system(
            cache,parameters,beta_limit,scales,anchors,beta_anchor,
            context.scaling.pose_anchor_scale,context.scaling.shape_anchor_scale,
        ), parameters, cli.warmup, cli.repeats,
    )
    print(f"[COMPLETE_NORMAL_EQUATIONS] {complete}")

    if cli.profile_torch:
        activities=[torch.profiler.ProfilerActivity.CPU]
        if device.type=="cuda":activities.append(torch.profiler.ProfilerActivity.CUDA)
        with torch.profiler.profile(activities=activities,record_shapes=True,profile_memory=True,with_stack=False) as profiler:
            accumulate_stage2_joint_arrowhead_system(
                cache,parameters,beta_limit,scales,anchors,beta_anchor,
                context.scaling.pose_anchor_scale,context.scaling.shape_anchor_scale,
            )
        print(profiler.key_averages().table(
            sort_by="self_cuda_time_total" if device.type=="cuda" else "self_cpu_time_total",
            row_limit=30,
        ))
        if cli.export_trace:
            path=Path(cli.export_trace);path.parent.mkdir(parents=True,exist_ok=True)
            profiler.export_chrome_trace(str(path));print(f"[TRACE] {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
