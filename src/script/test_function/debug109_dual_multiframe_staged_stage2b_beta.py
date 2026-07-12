"""Multi-frame staged LM fitting with IRLS Charbonnier support.

This script validates multi-frame, multi-stage LM fitting for SMPL-X
parameters.  It is the entry point for the ``debug109_staged_lm`` package.

Stages
------
Stage 1:
    Optimise per-frame dual-camera orient/transl (12 dims/frame).
    Body pose and beta are frozen to the HMR initialisation.

Stage 2:
    Perform a pose warmup with beta fixed, then pose/shape joint refinement
    with shared bounded beta active.

The primary solver is Levenberg-Marquardt (LM).  An optional L-BFGS oracle can
be enabled with ``--run_lbfgs_oracle`` to cross-check the LM result.  Gate C
checks LM/LBFGS consistency, handoff integrity and beta saturation.

Usage example
-------------
Run both stages on frames 100 through 102 with the Charbonnier IRLS objective
and the optional L-BFGS oracle::

python src/script/test_function/debug109_dual_multiframe_staged_stage2b_beta.py \
    --frames 513 \
    --hmr_dir data_process/Output/20260311_riding/hmr_cache \
    --sam3d_dense_dir data_process/Output/20260311_riding/sam3dbody/smplx_dense \
    --bbox_dir data_process/Output/20260311_riding/image/equirect_crop_rotate \
    --output_dir data_process/Vis/debug109_output \
    --reproj_objective charbonnier_irls \
    --run_lbfgs_oracle \
    --stop_after_stage stage2


Use ``--stop_after_stage stage1`` to save after rigid alignment, or
``--stop_after_stage stage2_pose`` to save after the pose warmup.
"""

from __future__ import annotations

import math
from dataclasses import asdict
from pathlib import Path

import torch

import config_script
from lib_pano.human3d.crop_upright_coord import resolve_dataset_rotate_rules
from lib_pano.human_fitting import DualCamERPFitter

from debug109_staged_lm.config import (
    BETA_ATANH_EPS,
    R_CAM4,
    STAGE1_PER_FRAME_DIM,
    ScalingConstants,
    SolverConfig,
    parse_args,
    resolve_objective_config,
)
from debug109_staged_lm.data_loading import (
    build_frozen_frame_parameters,
    build_segment_initializations,
    load_frame_observations,
    load_hmr_initializations,
    precompute_valid_points,
    to_tensor_vertex_ids_and_targets,
)
from debug109_staged_lm.output import (
    build_output_directory,
    collect_stage1_arrays,
    collect_stage2_pose_arrays,
    collect_stage2_arrays,
    compute_objective_breakdown,
    evaluate_gate_c,
    lbfgs_result_to_json,
    lm_result_to_json,
    save_experiment_outputs,
)
from debug109_staged_lm.stages import (
    build_stage1_definition,
    run_stage,
    run_stage2,
)
from debug109_staged_lm.types import StageContext


def build_context(args) -> StageContext:
    """Load all data and build the shared :class:`StageContext`."""
    frame_indices = [int(s.strip()) for s in args.frames.split(",")]
    frame_count = len(frame_indices)
    image_width, image_height = args.image_width, args.image_height
    is_irls = args.reproj_objective == "charbonnier_irls"
    mode = "charbonnier_irls" if is_irls else "raw_l2"
    print(f"[MODE] {mode}")
    objective_config = resolve_objective_config(args)

    rotate_rules_config = Path(config_script.PROJECT_ROOT) / "configs" / "rotate_upright_rules.json"
    rotate_rules = resolve_dataset_rotate_rules(rotate_rules_config, Path(args.bbox_dir))

    # Load per-frame observations and HMR initialisations for both cameras.
    frame_observations = []
    for frame_index in frame_indices:
        frame_observations.append(load_frame_observations(
            frame_index, Path(args.sam3d_dense_dir), Path(args.bbox_dir),
            image_width, image_height, rotate_rules,
        ))

    _, cam2_hmr_inits, cam4_hmr_inits = load_hmr_initializations(
        frame_indices, Path(args.hmr_dir),
    )
    cam2_segment_init, cam4_segment_init = build_segment_initializations(
        frame_indices, cam2_hmr_inits, cam4_hmr_inits,
    )

    # Fit SMPL-X initial parameters and extract shared beta / body-pose.
    fitter = DualCamERPFitter(primary_observation_source="sam3d_dense", gender="neutral", R_cam4=R_CAM4)
    cam2_smplx, cam4_smplx, shared_beta, shared_body_pose = fitter._init_params(
        cam2_segment_init, cam4_segment_init,
        [obs[0] for obs in frame_observations],
        [obs[1] for obs in frame_observations],
        (image_width, image_height),
    )

    body_pose_dim = int(shared_body_pose.shape[-1])
    beta_dim = int(shared_beta.numel())
    body_pose_anchors = [shared_body_pose.detach().clone()[t].reshape(-1) for t in range(frame_count)]
    beta_anchor = shared_beta.detach().clone().reshape(-1)

    # Frozen per-frame parameters (hand poses, scale).
    model = fitter._get_model()
    frozen_params_by_frame = [build_frozen_frame_parameters(cam2_smplx, cam4_smplx, t) for t in range(frame_count)]

    # Pre-compute valid SAM3D dense correspondences.
    sample_ids = fitter._sam3d_dense_sample_ids
    valid_points_by_frame = [
        precompute_valid_points(obs[0], obs[1], sample_ids) for obs in frame_observations
    ]
    cam2_vid, cam2_tgt, cam4_vid, cam4_tgt = to_tensor_vertex_ids_and_targets(valid_points_by_frame)

    # Beta feasibility check: the anchor must stay inside the tanh domain.
    safe_bound = objective_config.beta_abs_limit * (1.0 - BETA_ATANH_EPS)
    if float(beta_anchor.abs().max()) >= safe_bound:
        raise RuntimeError("INFEASIBLE_BETA_ANCHOR")
    beta_latent0 = torch.atanh(beta_anchor / objective_config.beta_abs_limit)

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

    return StageContext(
        args=args,
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


def _stage_json(outcome, frame_count: int) -> dict:
    definition = outcome.definition
    return {
        "lm": lm_result_to_json(outcome.lm_result),
        "lbfgs": lbfgs_result_to_json(outcome.lbfgs_result),
        "input_summary": {
            "T": frame_count,
            "param_dim": definition.parameter_dimension,
            "n_groups": 2 * frame_count,
            "total_valid_points": sum(
                group.valid_point_count
                for group in definition.raw_groups_fn(definition.initial_lm_parameters)
            ),
        },
        "verdict": outcome.verdict,
    }


def main() -> None:
    """Run Stage 1, unified Stage 2, Gate C, and save outputs."""
    args = parse_args()
    context = build_context(args)
    stage1_definition = build_stage1_definition(context)
    stage1_outcome = run_stage(stage1_definition, context)
    saved_config = {
        **vars(args),
        "solver_config": asdict(context.solver_config),
        "objective_config": asdict(context.objective_config),
    }
    results: dict = {"config": saved_config, "stage1": _stage_json(stage1_outcome, context.frame_count)}
    output_directory = build_output_directory(Path(args.output_dir), context.frame_indices)

    if args.stop_after_stage == "stage1":
        arrays = collect_stage1_arrays(
            stage1_definition.initial_lm_parameters, stage1_outcome,
            context.beta_anchor, context.beta_latent0, stage1_definition.raw_groups_fn,
        )
        save_experiment_outputs(output_directory, results, arrays)
        raise SystemExit(0)

    stage2_outcome = run_stage2(context, stage1_outcome)
    pose_outcome = stage2_outcome.pose_warmup
    results["handoffs"] = {key: value for key, value in stage2_outcome.handoffs.items() if value is not None}
    results["stage2"] = {"pose_warmup": _stage_json(pose_outcome, context.frame_count)}

    if args.stop_after_stage == "stage2_pose":
        pose_definition = pose_outcome.definition
        arrays = collect_stage2_pose_arrays(
            stage1_definition.initial_lm_parameters,
            pose_definition.initial_lm_parameters,
            pose_definition.initial_lbfgs_parameters,
            stage1_outcome, pose_outcome, context.beta_anchor, context.beta_latent0,
            stage1_definition.raw_groups_fn, pose_definition.raw_groups_fn,
        )
        save_experiment_outputs(output_directory, results, arrays)
        raise SystemExit(0)

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
    if joint_outcome.lbfgs_result is not None:
        beta_lbfgs = beta_abs_limit * torch.tanh(joint_outcome.lbfgs_result.parameters[-context.beta_dim:])
        beta_metrics["lbfgs"] = {"max": float(beta_lbfgs.abs().max()), "sat": float(beta_lbfgs.abs().max()) / beta_abs_limit,
                                 "delta": float(torch.norm(beta_lbfgs - context.beta_anchor))}

    per_frame_dim = STAGE1_PER_FRAME_DIM + context.body_pose_dim
    def pose_anchor_residual(x):
        return torch.cat([(x[t * per_frame_dim + STAGE1_PER_FRAME_DIM:(t + 1) * per_frame_dim]
                           - context.body_pose_anchors[t]) * context.scaling.pose_anchor_scale
                          for t in range(context.frame_count)])
    def shape_anchor_residual(x):
        return (beta_abs_limit * torch.tanh(x[-context.beta_dim:]) - context.beta_anchor) * context.scaling.shape_anchor_scale
    for label, result in (("lm", joint_outcome.lm_result), ("lbfgs", joint_outcome.lbfgs_result)):
        if result is not None:
            beta_metrics[label]["breakdown"] = compute_objective_breakdown(
                joint_definition.raw_groups_fn, pose_anchor_residual, shape_anchor_residual,
                joint_outcome.true_objective_fn, result.parameters,
                is_irls=context.is_irls,
                charbonnier_eps=context.charbonnier_eps,
                rep_scale=context.scaling.rep_scale,
            )
    results["stage2"]["joint_refinement"] = _stage_json(joint_outcome, context.frame_count)
    results["stage2"]["beta"] = beta_metrics

    handoffs = stage2_outcome.handoffs
    stage_matches = {
        "s1_match": stage1_outcome.verdict == "LM_MATCHES_LBFGS" if args.run_lbfgs_oracle else None,
        "stage2_pose_match": pose_outcome.verdict == "LM_MATCHES_LBFGS" if args.run_lbfgs_oracle else None,
        "stage2_joint_match": joint_outcome.verdict == "LM_MATCHES_LBFGS" if args.run_lbfgs_oracle else None,
    }
    non_oracle_checks = {
        "handoff_stage1_to_stage2_pose_lm": handoffs["stage1_to_stage2_pose_lm"]["ok"],
        "handoff_stage2_pose_to_joint_lm": handoffs["stage2_pose_to_joint_lm"]["ok"],
        "beta_sat_lt_0.9": beta_metrics["lm"]["sat"] < 0.90,
        "s1_lm_status_ok": stage1_outcome.lm_result.status not in ("LM_NO_ACCEPTED_TRIAL", "NUMERICAL_FAILURE"),
        "stage2_pose_lm_status_ok": pose_outcome.lm_result.status not in ("LM_NO_ACCEPTED_TRIAL", "NUMERICAL_FAILURE"),
        "stage2_joint_lm_status_ok": joint_outcome.lm_result.status not in ("LM_NO_ACCEPTED_TRIAL", "NUMERICAL_FAILURE"),
    }
    oracle_checks = {}
    if args.run_lbfgs_oracle:
        oracle_checks = {
            "handoff_stage1_to_stage2_pose_bf": handoffs["stage1_to_stage2_pose_lbfgs"]["ok"],
            "handoff_stage2_pose_to_joint_bf": handoffs["stage2_pose_to_joint_lbfgs"]["ok"],
        }
    results["gate_c_verdict"] = evaluate_gate_c(
        oracle_enabled=args.run_lbfgs_oracle,
        stage_matches=stage_matches,
        non_oracle_checks=non_oracle_checks,
        oracle_checks=oracle_checks,
    )
    gate_c = results["gate_c_verdict"]["verdict"]
    print(f"[GATE_C] {gate_c}")
    pose_definition = pose_outcome.definition
    arrays = collect_stage2_arrays(
        stage1_definition.initial_lm_parameters, pose_definition.initial_lm_parameters,
        joint_definition.initial_lm_parameters, stage1_outcome, pose_outcome, joint_outcome,
        context.beta_anchor, context.beta_latent0, beta_lm, beta_lbfgs,
        stage1_definition.raw_groups_fn, pose_definition.raw_groups_fn, joint_definition.raw_groups_fn,
        pose_definition.initial_lbfgs_parameters, joint_definition.initial_lbfgs_parameters,
    )
    save_experiment_outputs(output_directory, results, arrays)
    raise SystemExit(0)


if __name__ == "__main__":
    main()
