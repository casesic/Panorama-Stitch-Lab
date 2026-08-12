"""Dump stage-boundary state from the canonical frame-513 production pipeline."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from lib_pano.human_fitting.staged_lm.parameter_layout import (
    build_stage1_initial_parameters,
)
from lib_pano.human_fitting.staged_lm.pipeline import run_stage2
from lib_pano.human_fitting.staged_lm.pose_shape.shape import (
    physical_beta_from_latent,
)
from lib_pano.human_fitting.staged_lm.stage_definitions.rigid import run_stage1
from script.human_fitting.staged_lm.app import build_context


def _np(value: torch.Tensor) -> np.ndarray:
    return value.detach().cpu().contiguous().numpy()


def _summary(result) -> dict[str, object]:
    lm = result.lm_result
    return {
        "objective": lm.objective,
        "status": lm.status,
        "final_lambda": lm.final_lambda,
        "trials": len(lm.trial_history),
        "accepted_steps": lm.counters.accepted_steps,
        "rejected_trials": lm.counters.rejected_trials,
        "objective_calls": 1 + sum(
            int(record.get("solve_success", False)) for record in lm.trial_history
        ),
        "linearization_calls": lm.counters.linearization_count,
        "solve_methods": [
            record.get("solve_method", record.get("linear_solver_backend", "unknown"))
            for record in lm.trial_history
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frame", type=int, default=513)
    parser.add_argument("--hmr_dir", required=True)
    parser.add_argument("--sam3d_dense_dir", required=True)
    parser.add_argument("--bbox_dir", required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--image_width", type=int, default=5760)
    parser.add_argument("--image_height", type=int, default=2880)
    parser.add_argument("--reproj_objective", default="charbonnier_irls")
    args = parser.parse_args()
    args.frames = str(args.frame)
    args.profile_timing = True
    args.validate_device_contract = False
    args.stop_after_stage = "stage2"

    context = build_context(args)
    initial_stage1 = build_stage1_initial_parameters(
        context.cam2_smplx_parameters,
        context.cam4_smplx_parameters,
        context.frame_count,
    )
    started = time.perf_counter()
    stage1 = run_stage1(context)
    stage2 = run_stage2(context, stage1)
    wall_ms = (time.perf_counter() - started) * 1000.0
    pose = stage2.pose_warmup
    pose_shape = stage2.pose_shape_refinement
    assert pose_shape is not None

    pose_initial = torch.cat((
        stage1.lm_result.parameters,
        context.body_pose_anchors[0],
    ))
    beta_latent_initial = torch.atanh(
        context.beta_anchor / context.objective_config.beta_abs_limit
    )
    pose_shape_initial = torch.cat((
        pose.lm_result.parameters,
        beta_latent_initial,
    ))
    physical_beta_final, _ = physical_beta_from_latent(
        pose_shape.lm_result.parameters[-context.beta_dim:],
        context.objective_config.beta_abs_limit,
    )
    arrays = {
        "stage1_initial_parameters": _np(initial_stage1),
        "stage1_final_parameters": _np(stage1.lm_result.parameters),
        "pose_initial_parameters": _np(pose_initial),
        "pose_final_parameters": _np(pose.lm_result.parameters),
        "beta_anchor": _np(context.beta_anchor),
        "pose_shape_initial_parameters": _np(pose_shape_initial),
        "pose_shape_initial_beta_latent": _np(beta_latent_initial),
        "pose_shape_final_parameters": _np(pose_shape.lm_result.parameters),
        "final_frame_parameters": _np(pose_shape.lm_result.parameters[:75]),
        "final_beta_latent": _np(pose_shape.lm_result.parameters[-10:]),
        "final_physical_beta": _np(physical_beta_final),
    }
    summary = {
        "schema_version": "staged_lm_cpp_pipeline_reference_v1",
        "actual_initial_lm_order": [
            "initial_linearization", "lambda_initialization", "initial_objective"
        ],
        "stage_order": ["stage1", "pose", "pose_shape"],
        "stage1": _summary(stage1),
        "pose": _summary(pose),
        "pose_shape": _summary(pose_shape),
        "wall_ms": wall_ms,
    }
    manifest = {
        "schema_version": summary["schema_version"],
        "frame": args.frame,
        "cam2_count": int(context.cam2_vertex_ids_by_frame[0].numel()),
        "cam4_count": int(context.cam4_vertex_ids_by_frame[0].numel()),
        "stage_dimensions": [12, 75, 85],
        "fixture_policy": "raw_initialization_static_model_observations_and_config_only",
        "reference_leakage": "NONE",
        "solver_config": vars(context.solver_config),
        "objective_config": vars(context.objective_config),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.savez(args.output_dir / "pipeline_reference.npz", **arrays)
    (args.output_dir / "stage_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    (args.output_dir / "pipeline_reference_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    (args.output_dir / "handoff_summary.json").write_text(json.dumps({
        "stage1_to_pose": "rigid12 copied; body63 initialized from anchor",
        "pose_to_pose_shape": "frame75 copied; z=atanh(beta_anchor/2.5)",
    }, indent=2) + "\n")
    (args.output_dir / "pipeline_reference_report.md").write_text(
        "# Staged LM pipeline reference\n\n"
        "The canonical production pipeline was run as Stage1 → Pose → Pose/Shape.\n"
    )
    print("STAGED_LM_PIPELINE_REFERENCE_WRITTEN")


if __name__ == "__main__":
    main()
