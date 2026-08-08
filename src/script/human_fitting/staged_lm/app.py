"""Application orchestration for production staged-LM fitting.

The numerical implementation lives in ``lib_pano.human_fitting.staged_lm``;
this module owns data/context construction, stage orchestration, result
assembly, and output serialization.

Stages
------
Stage 1:
    Optimise per-frame dual-camera orient/transl (12 dims/frame).
    Body pose and beta are frozen to the HMR initialisation.

Stage 2:
    Perform a pose warmup with beta fixed, then pose/shape joint refinement
    with shared bounded beta active.

The production solver is the fixed analytic/fast/Schur/combined-FK
Levenberg-Marquardt path. Reference and oracle modes are provided by the
validation scripts rather than this CLI.

Usage example
-------------
Run all stages with the Charbonnier IRLS objective::

python src/script/human_fitting/staged_lm/run.py \
    --frames 513 \
    --hmr_dir data_process/Output/20260311_riding/hmr_cache \
    --sam3d_dense_dir data_process/Output/20260311_riding/sam3dbody/smplx_dense \
    --bbox_dir data_process/Output/20260311_riding/image/equirect_crop_rotate \
    --output_dir data_process/Vis/staged_lm \
    --reproj_objective charbonnier_irls \
    --stop_after_stage stage2


Use ``--stop_after_stage stage1`` to save after rigid alignment, or
``--stop_after_stage stage2_pose`` to save after the pose warmup.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from lib_pano.human3d.crop_upright_coord import resolve_dataset_rotate_rules
from lib_pano.human_fitting import DualCamERPFitter

from lib_pano.human_fitting.staged_lm.config import (
    BETA_ATANH_EPS,
    R_CAM4,
    ScalingConstants,
    SolverConfig,
    resolve_objective_config,
)
from lib_pano.human_fitting.staged_lm.data_loading import (
    build_frozen_frame_parameters,
    build_segment_initializations,
    load_frame_observations,
    load_hmr_initializations,
    precompute_valid_points,
    to_tensor_vertex_ids_and_targets,
)
from lib_pano.human_fitting.staged_lm.device import (
    deterministic_beta_latent_from_anchor,
    describe_execution_environment,
    move_smplx_param_sequence,
    print_execution_environment,
    resolve_execution_device,
    validate_context_device,
)
from lib_pano.human_fitting.staged_lm.output import (
    build_beta_metrics,
    build_output_directory,
    build_initial_results,
    collect_stage1_arrays,
    collect_stage2_pose_arrays,
    collect_stage2_arrays,
    export_debug029_compatible_result,
    save_experiment_outputs,
    stage_outcome_to_json,
)
from lib_pano.human_fitting.staged_lm.pipeline import run_stage2
from lib_pano.human_fitting.staged_lm.stage_definitions.rigid import (
    build_stage1_definition,
)
from lib_pano.human_fitting.staged_lm.stage_runtime import run_stage
from lib_pano.human_fitting.staged_lm.stage_timing import build_pipeline_timing
from lib_pano.human_fitting.staged_lm.types import StageContext


REPOSITORY_ROOT = Path(__file__).resolve().parents[4]


@dataclass
class ProductionRunResult:
    """Structured result returned by :func:`run_production_pipeline`."""

    context: StageContext
    stage1_outcome: Any
    pose_outcome: Any | None
    joint_outcome: Any | None
    results: dict[str, Any]
    arrays: dict[str, Any]
    output_directory: Path
    pipeline_timing: dict[str, Any]


def build_context(args) -> StageContext:
    """Load all data and build the shared :class:`StageContext`."""
    context_start = time.perf_counter()
    context_timing: dict[str, float] = {}
    requested_device = getattr(args, "device", "cpu")
    start = time.perf_counter()
    device = resolve_execution_device(requested_device)
    dtype = torch.float32
    # environment = describe_execution_environment(device)
    context_timing["resolve_device_ms"] = (time.perf_counter() - start) * 1000.0
    # print_execution_environment(requested_device, device, environment)
    frame_indices = [int(s.strip()) for s in args.frames.split(",")]
    frame_count = len(frame_indices)
    image_width, image_height = args.image_width, args.image_height
    is_irls = args.reproj_objective == "charbonnier_irls"
    mode = "charbonnier_irls" if is_irls else "raw_l2"
    print(f"[MODE] {mode}")
    objective_config = resolve_objective_config(args)

    rotate_rules_config = REPOSITORY_ROOT / "configs" / "rotate_upright_rules.json"
    rotate_rules = resolve_dataset_rotate_rules(rotate_rules_config, Path(args.bbox_dir))

    # Load per-frame observations and HMR initialisations for both cameras.
    start = time.perf_counter()
    frame_observations = []
    for frame_index in frame_indices:
        frame_observations.append(load_frame_observations(
            frame_index, Path(args.sam3d_dense_dir), Path(args.bbox_dir),
            image_width, image_height, rotate_rules,
        ))
    context_timing["load_observations_ms"] = (time.perf_counter() - start) * 1000.0

    start = time.perf_counter()
    _, cam2_hmr_inits, cam4_hmr_inits = load_hmr_initializations(
        frame_indices, Path(args.hmr_dir),
    )
    context_timing["load_hmr_ms"] = (time.perf_counter() - start) * 1000.0
    start = time.perf_counter()
    cam2_segment_init, cam4_segment_init = build_segment_initializations(
        frame_indices, cam2_hmr_inits, cam4_hmr_inits,
    )
    context_timing["build_segment_init_ms"] = (time.perf_counter() - start) * 1000.0

    start = time.perf_counter()
    move_smplx_param_sequence(cam2_segment_init, device, dtype)
    move_smplx_param_sequence(cam4_segment_init, device, dtype)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    context_timing["move_hmr_to_device_ms"] = (time.perf_counter() - start) * 1000.0

    # Fit SMPL-X initial parameters and extract shared beta / body-pose.
    start = time.perf_counter()
    fitter = DualCamERPFitter(primary_observation_source="sam3d_dense", gender="neutral", R_cam4=R_CAM4)
    context_timing["fitter_create_ms"] = (time.perf_counter() - start) * 1000.0
    start = time.perf_counter()
    model = fitter._get_model()
    context_timing["smplx_model_load_ms"] = (time.perf_counter() - start) * 1000.0
    start = time.perf_counter()
    model = model.to(device=device, dtype=dtype)
    model.eval()
    fitter._fitter._model = model
    if fitter._fitter._model is not model:
        raise RuntimeError("DEVICE_CONTRACT_VIOLATION: fitter model reference was not retained")
    fitter.R_cam4 = fitter.R_cam4.to(device=device, dtype=dtype)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    context_timing["smplx_model_to_device_ms"] = (time.perf_counter() - start) * 1000.0
    start = time.perf_counter()
    cam2_smplx, cam4_smplx, shared_beta, shared_body_pose = fitter._init_params(
        cam2_segment_init, cam4_segment_init,
        [obs[0] for obs in frame_observations],
        [obs[1] for obs in frame_observations],
        (image_width, image_height),
    )
    shared_beta = shared_beta.detach().to(device=device, dtype=dtype)
    shared_body_pose = shared_body_pose.detach().to(device=device, dtype=dtype)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    context_timing["init_params_ms"] = (time.perf_counter() - start) * 1000.0

    body_pose_dim = int(shared_body_pose.shape[-1])
    beta_dim = int(shared_beta.numel())
    body_pose_anchors = [shared_body_pose.detach().clone()[t].reshape(-1) for t in range(frame_count)]
    beta_anchor = shared_beta.detach().clone().reshape(-1)

    # Frozen per-frame parameters (hand poses, scale).
    frozen_params_by_frame = [build_frozen_frame_parameters(cam2_smplx, cam4_smplx, t) for t in range(frame_count)]

    # Pre-compute valid SAM3D dense correspondences.
    sample_ids = fitter._sam3d_dense_sample_ids
    start = time.perf_counter()
    valid_points_by_frame = [
        precompute_valid_points(obs[0], obs[1], sample_ids) for obs in frame_observations
    ]
    context_timing["valid_point_selection_ms"] = (time.perf_counter() - start) * 1000.0
    start = time.perf_counter()
    cam2_vid, cam2_tgt, cam4_vid, cam4_tgt = to_tensor_vertex_ids_and_targets(
        valid_points_by_frame, device=device, target_dtype=dtype,
    )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    context_timing["observation_tensor_create_ms"] = (time.perf_counter() - start) * 1000.0

    # Beta feasibility check: the anchor must stay inside the tanh domain.
    safe_bound = objective_config.beta_abs_limit * (1.0 - BETA_ATANH_EPS)
    if float(beta_anchor.abs().max()) >= safe_bound:
        raise RuntimeError("INFEASIBLE_BETA_ANCHOR")
    beta_latent0 = deterministic_beta_latent_from_anchor(
        beta_anchor, objective_config.beta_abs_limit,
    )

    # Scaling constants shared across stages.
    total_valid_points = sum(vp.cam2.count + vp.cam4.count for vp in valid_points_by_frame)
    rep_scale = 1.0 / (objective_config.reproj_sigma_px * math.sqrt(max(total_valid_points, 1)))
    pose_anchor_scale = math.sqrt(2.0 * objective_config.pose_anchor_weight / (frame_count * body_pose_dim))
    shape_anchor_scale = math.sqrt(2.0 * objective_config.shape_anchor_weight / beta_dim)
    scaling = ScalingConstants(
        rep_scale=rep_scale,
        pose_anchor_scale=pose_anchor_scale,
        shape_anchor_scale=shape_anchor_scale,
    )

    context = StageContext(
        args=args,
        device=device,
        dtype=dtype,
        device_environment=None,
        context_timing=context_timing,
        frame_indices=frame_indices,
        image_width=image_width,
        image_height=image_height,
        frame_count=frame_count,
        is_irls=is_irls,
        charbonnier_eps=objective_config.charbonnier_eps,
        body_pose_dim=body_pose_dim,
        beta_dim=beta_dim,
        body_pose_anchors=body_pose_anchors,
        beta_anchor=beta_anchor,
        beta_latent0=beta_latent0,
        scaling=scaling,
        solver_config=SolverConfig(),
        objective_config=objective_config,
        frozen_params_by_frame=frozen_params_by_frame,
        cam2_vertex_ids_by_frame=cam2_vid,
        cam2_targets_by_frame=cam2_tgt,
        cam4_vertex_ids_by_frame=cam4_vid,
        cam4_targets_by_frame=cam4_tgt,
        model=model,
        fitter=fitter,
        cam2_smplx_parameters=cam2_smplx,
        cam4_smplx_parameters=cam4_smplx,
    )
    start = time.perf_counter()
    validate_context_device(context)
    context_timing["device_validation_ms"] = (time.perf_counter() - start) * 1000.0
    context_timing["context_total_ms"] = (time.perf_counter() - context_start) * 1000.0
    return context


def run_production_pipeline(
    args: Any,
    *,
    save_outputs: bool = True,
) -> ProductionRunResult:
    """Run the fixed production path and optionally serialize its outputs."""
    command_start = time.perf_counter()
    context = build_context(args)
    pipeline_timing = build_pipeline_timing(context)
    if context.device.type == "cuda":
        torch.cuda.synchronize(context.device)
    fitting_start = time.perf_counter()
    stage1_definition = build_stage1_definition(context)
    stage1_preparation = stage1_definition.preparation_timing
    pipeline_timing["stage1_prepare"] = stage1_preparation.total_ms
    profile = pipeline_timing.get("profile")
    if isinstance(profile, dict):
        profile["stage1_prepare_breakdown"] = {
            "total_ms": stage1_preparation.total_ms,
            "cache_ms": stage1_preparation.cache_setup_ms,
            "bind_ms": stage1_preparation.definition_bind_ms,
        }
    start = time.perf_counter()
    stage1_outcome = run_stage(stage1_definition, context)
    if context.device.type == "cuda":
        torch.cuda.synchronize(context.device)
    stage1_run_ms = (time.perf_counter() - start) * 1000.0
    pipeline_timing["stage1_run_ms"] = stage1_run_ms
    results = build_initial_results(
        args, context, pipeline_timing, stage1_outcome,
    )
    output_directory = build_output_directory(Path(args.output_dir), context.frame_indices)

    if args.stop_after_stage == "stage1":
        pipeline_timing["fitting_total_ms"] = (time.perf_counter() - fitting_start) * 1000.0
        arrays: dict[str, Any] = {}
        if getattr(args, "export_debug029_format", False):
            arrays = collect_stage1_arrays(
                stage1_definition.initial_lm_parameters, stage1_outcome,
                context.beta_anchor, context.beta_latent0, stage1_definition.raw_groups_fn,
            )
        if save_outputs:
            save_experiment_outputs(
                output_directory, results,
                pipeline_timing=pipeline_timing, command_start=command_start,
            )
            if getattr(args, "export_debug029_format", False):
                export_debug029_compatible_result(
                    Path(args.output_dir), context, arrays, "stage1",
                )
        return ProductionRunResult(
            context, stage1_outcome, None, None, results, arrays,
            output_directory, pipeline_timing,
        )

    stage2_outcome = run_stage2(context, stage1_outcome, pipeline_timing)
    pose_outcome = stage2_outcome.pose_warmup
    results["handoffs"] = {key: value for key, value in stage2_outcome.handoffs.items() if value is not None}
    results["stage2"] = {"pose_warmup": stage_outcome_to_json(pose_outcome)}

    if args.stop_after_stage == "stage2_pose":
        pipeline_timing["fitting_total_ms"] = (time.perf_counter() - fitting_start) * 1000.0
        pose_definition = pose_outcome.definition
        arrays = {}
        if getattr(args, "export_debug029_format", False):
            arrays = collect_stage2_pose_arrays(
                stage1_definition.initial_lm_parameters,
                pose_definition.initial_lm_parameters,
                stage1_outcome, pose_outcome, context.beta_anchor, context.beta_latent0,
                stage1_definition.raw_groups_fn, pose_definition.raw_groups_fn,
            )
        if save_outputs:
            save_experiment_outputs(
                output_directory, results,
                pipeline_timing=pipeline_timing, command_start=command_start,
            )
            if getattr(args, "export_debug029_format", False):
                export_debug029_compatible_result(
                    Path(args.output_dir), context, arrays, "pose_warmup",
                )
        return ProductionRunResult(
            context, stage1_outcome, pose_outcome, None, results, arrays,
            output_directory, pipeline_timing,
        )

    joint_outcome = stage2_outcome.joint_refinement
    if joint_outcome is None:
        raise RuntimeError(
            "Stage 2 joint refinement result is missing in full Stage 2 mode"
        )
    joint_definition = joint_outcome.definition
    beta_abs_limit = context.objective_config.beta_abs_limit
    beta_lm = beta_abs_limit * torch.tanh(joint_outcome.lm_result.parameters[-context.beta_dim:])
    beta_metrics = build_beta_metrics(
        beta_lm, context.beta_anchor, beta_abs_limit,
    )
    pipeline_timing["fitting_total_ms"] = (
        time.perf_counter() - fitting_start
    ) * 1000.0
    results["stage2"]["joint_refinement"] = stage_outcome_to_json(joint_outcome)
    results["stage2"]["beta"] = beta_metrics

    pose_definition = pose_outcome.definition
    arrays = {}
    if getattr(args, "export_debug029_format", False):
        arrays = collect_stage2_arrays(
            stage1_definition.initial_lm_parameters, pose_definition.initial_lm_parameters,
            joint_definition.initial_lm_parameters, stage1_outcome, pose_outcome, joint_outcome,
            context.beta_anchor, context.beta_latent0, beta_lm,
            stage1_definition.raw_groups_fn, pose_definition.raw_groups_fn,
            joint_definition.raw_groups_fn,
        )
    if save_outputs:
        save_experiment_outputs(
            output_directory, results,
            pipeline_timing=pipeline_timing, command_start=command_start,
        )
        if getattr(args, "export_debug029_format", False):
            export_debug029_compatible_result(
                Path(args.output_dir), context, arrays, "joint_refinement",
            )
    return ProductionRunResult(
        context, stage1_outcome, pose_outcome, joint_outcome, results, arrays,
        output_directory, pipeline_timing,
    )
