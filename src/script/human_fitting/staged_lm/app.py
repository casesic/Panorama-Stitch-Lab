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
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

from lib_pano.human3d.crop_upright_coord import resolve_dataset_rotate_rules
from lib_pano.human_fitting import DualCamERPFitter

from lib_pano.human_fitting.staged_lm.config import (
    BETA_ATANH_EPS,
    PRODUCTION_JOINT_EVALUATION,
    PRODUCTION_JOINT_FK,
    PRODUCTION_JOINT_LINEARIZATION,
    PRODUCTION_POSE_EVALUATION,
    PRODUCTION_POSE_LINEARIZATION,
    PRODUCTION_STAGE1_EVALUATION,
    PRODUCTION_STAGE1_LINEARIZATION,
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
    build_output_directory,
    collect_stage1_arrays,
    collect_stage2_pose_arrays,
    collect_stage2_arrays,
    compute_objective_breakdown,
    lbfgs_result_to_json,
    lm_result_to_json,
    save_experiment_outputs,
)
from lib_pano.human_fitting.staged_lm.stages import (
    build_stage1_definition,
    run_stage,
    run_stage2,
)
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
    pipeline_timing: dict[str, float]


def build_context(args) -> StageContext:
    """Load all data and build the shared :class:`StageContext`."""
    context_start = time.perf_counter()
    context_timing: dict[str, float] = {}
    requested_device = getattr(args, "device", "auto")
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
    move_smplx_param_sequence(cam2_smplx, device, dtype)
    move_smplx_param_sequence(cam4_smplx, device, dtype)
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


def _stage_json(outcome) -> dict:
    definition = outcome.definition
    summary = definition.input_summary if definition is not None else None
    if summary is None:
        raise RuntimeError("MISSING_STAGE_INPUT_SUMMARY_FOR_OUTPUT")
    return {
        "lm": lm_result_to_json(outcome.lm_result),
        "lbfgs": lbfgs_result_to_json(outcome.lbfgs_result),
        "input_summary": {
            "T": summary.frame_count,
            "param_dim": summary.parameter_dimension,
            "n_groups": summary.group_count,
            "total_valid_points": summary.total_valid_points,
        },
        "verdict": outcome.verdict,
    }


def run_production_pipeline(
    args: Any,
    *,
    save_outputs: bool = True,
) -> ProductionRunResult:
    """Run the fixed production path and optionally serialize its outputs."""
    command_start = time.perf_counter()
    context = build_context(args)
    pipeline_timing = dict(context.context_timing)
    pipeline_timing["timing_schema_version"] = 5
    if context.device.type == "cuda":
        torch.cuda.synchronize(context.device)
    fitting_start = time.perf_counter()
    stage1_definition = build_stage1_definition(context)
    stage1_preparation = stage1_definition.preparation_timing
    pipeline_timing["stage1_prepare_total_ms"] = stage1_preparation.total_ms
    pipeline_timing["stage1_cache_setup_ms"] = stage1_preparation.cache_setup_ms
    pipeline_timing["stage1_definition_bind_ms"] = stage1_preparation.definition_bind_ms
    start = time.perf_counter()
    stage1_outcome = run_stage(stage1_definition, context)
    if context.device.type == "cuda":
        torch.cuda.synchronize(context.device)
    stage1_run_ms = (time.perf_counter() - start) * 1000.0
    pipeline_timing["stage1_run_ms"] = stage1_run_ms
    pipeline_timing["stage1_run_overhead_ms"] = max(
        0.0, stage1_run_ms - stage1_outcome.lm_result.wall_time_ms,
    )
    saved_config = {
        **vars(args),
        "jacobian_backend": PRODUCTION_STAGE1_LINEARIZATION,
        "stage1_eval_backend": PRODUCTION_STAGE1_EVALUATION,
        "stage2_pose_linearization_backend": PRODUCTION_POSE_LINEARIZATION,
        "stage2_pose_eval_backend": PRODUCTION_POSE_EVALUATION,
        "stage2_joint_linearization_backend": PRODUCTION_JOINT_LINEARIZATION,
        "stage2_joint_eval_backend": PRODUCTION_JOINT_EVALUATION,
        "stage2_joint_linear_solver": "schur",
        "stage2_joint_fk_jacobian_backend": PRODUCTION_JOINT_FK,
        "run_lbfgs_oracle": False,
        "validate_stage1_fast_eval": False,
        "validate_stage2_pose_fast_eval": False,
        "validate_stage2_joint_fast_eval": False,
        "validate_irls_first_order": False,
        "validate_irls_each_accepted": False,
        "solver_config": asdict(context.solver_config),
        "objective_config": asdict(context.objective_config),
    }
    results: dict = {
        "config": saved_config,
        "execution": {
            "requested_device": getattr(args, "device", "auto"),
            "resolved_device": str(context.device),
            "dtype": str(context.dtype).replace("torch.", ""),
            "environment": context.device_environment,
        },
        "pipeline_timing": pipeline_timing,
        "stage1": _stage_json(stage1_outcome),
    }
    output_directory = build_output_directory(Path(args.output_dir), context.frame_indices)

    if args.stop_after_stage == "stage1":
        pipeline_timing["fitting_total_ms"] = (time.perf_counter() - fitting_start) * 1000.0
        pipeline_timing["fitting_with_oracle_ms"] = pipeline_timing["fitting_total_ms"]
        pipeline_timing["lbfgs_total_ms"] = 0.0
        start = time.perf_counter()
        arrays = collect_stage1_arrays(
            stage1_definition.initial_lm_parameters, stage1_outcome,
            context.beta_anchor, context.beta_latent0, stage1_definition.raw_groups_fn,
        )
        pipeline_timing["output_array_build_ms"] = (time.perf_counter() - start) * 1000.0
        if save_outputs:
            save_experiment_outputs(
                output_directory, results, arrays,
                pipeline_timing=pipeline_timing, command_start=command_start,
            )
        return ProductionRunResult(
            context, stage1_outcome, None, None, results, arrays,
            output_directory, pipeline_timing,
        )

    stage2_outcome = run_stage2(context, stage1_outcome, pipeline_timing)
    pose_outcome = stage2_outcome.pose_warmup
    results["handoffs"] = {key: value for key, value in stage2_outcome.handoffs.items() if value is not None}
    results["stage2"] = {"pose_warmup": _stage_json(pose_outcome)}

    if args.stop_after_stage == "stage2_pose":
        pipeline_timing["fitting_total_ms"] = (time.perf_counter() - fitting_start) * 1000.0
        pipeline_timing["fitting_with_oracle_ms"] = pipeline_timing["fitting_total_ms"]
        pipeline_timing["lbfgs_total_ms"] = 0.0
        start = time.perf_counter()
        pose_definition = pose_outcome.definition
        arrays = collect_stage2_pose_arrays(
            stage1_definition.initial_lm_parameters,
            pose_definition.initial_lm_parameters,
            None,
            stage1_outcome, pose_outcome, context.beta_anchor, context.beta_latent0,
            stage1_definition.raw_groups_fn, pose_definition.raw_groups_fn,
        )
        pipeline_timing["output_array_build_ms"] = (time.perf_counter() - start) * 1000.0
        if save_outputs:
            save_experiment_outputs(
                output_directory, results, arrays,
                pipeline_timing=pipeline_timing, command_start=command_start,
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
    beta_metrics = {"lm": {"max": float(beta_lm.abs().max()), "sat": float(beta_lm.abs().max()) / beta_abs_limit,
                           "delta": float(torch.norm(beta_lm - context.beta_anchor))}}
    beta_lbfgs = None

    pipeline_timing["fitting_total_ms"] = (
        time.perf_counter() - fitting_start
    ) * 1000.0
    if joint_definition.objective_terms_fn is None:
        raise RuntimeError("MISSING_JOINT_OBJECTIVE_TERMS_FN")
    start = time.perf_counter()
    beta_metrics["lm"]["breakdown"] = compute_objective_breakdown(
        joint_definition.objective_terms_fn,
        joint_outcome.lm_result.parameters,
        expected_true_objective=joint_outcome.lm_result.objective,
    )
    if context.device.type == "cuda":
        torch.cuda.synchronize(context.device)
    pipeline_timing["objective_breakdown_ms"] = (time.perf_counter() - start) * 1000.0
    results["stage2"]["joint_refinement"] = _stage_json(joint_outcome)
    results["stage2"]["beta"] = beta_metrics

    results["gate_c_verdict"] = {
        "verdict": "GATE_C_NOT_EVALUATED",
        "oracle_enabled": False,
        "checks": {},
    }
    pipeline_timing["gate_c_ms"] = 0.0
    pipeline_timing["fitting_with_oracle_ms"] = pipeline_timing["fitting_total_ms"]
    pipeline_timing["lbfgs_total_ms"] = 0.0
    start = time.perf_counter()
    pose_definition = pose_outcome.definition
    arrays = collect_stage2_arrays(
        stage1_definition.initial_lm_parameters, pose_definition.initial_lm_parameters,
        joint_definition.initial_lm_parameters, stage1_outcome, pose_outcome, joint_outcome,
        context.beta_anchor, context.beta_latent0, beta_lm, beta_lbfgs,
        stage1_definition.raw_groups_fn, pose_definition.raw_groups_fn, joint_definition.raw_groups_fn,
        None, None,
    )
    if context.device.type == "cuda":
        torch.cuda.synchronize(context.device)
    pipeline_timing["output_array_build_ms"] = (time.perf_counter() - start) * 1000.0
    if save_outputs:
        save_experiment_outputs(
            output_directory, results, arrays,
            pipeline_timing=pipeline_timing, command_start=command_start,
        )
    return ProductionRunResult(
        context, stage1_outcome, pose_outcome, joint_outcome, results, arrays,
        output_directory, pipeline_timing,
    )
