"""Validate the final Joint objective term decomposition."""

from __future__ import annotations

import argparse
import json
from typing import Any, Callable

import torch

from lib_pano.human_fitting.staged_lm.pipeline import run_stage2
from lib_pano.human_fitting.staged_lm.stage_definitions.rigid import (
    build_stage1_definition,
)
from lib_pano.human_fitting.staged_lm.stage_runtime import run_stage
from script.human_fitting.staged_lm.app import build_context


def compute_objective_breakdown(
    objective_terms_fn: Callable[[torch.Tensor], dict[str, torch.Tensor]],
    parameters: torch.Tensor,
    *,
    expected_true_objective: float,
) -> dict[str, float]:
    """Decompose and verify the final Joint objective with one evaluation."""
    terms = objective_terms_fn(parameters)
    primary = float(terms["primary"].detach().item())
    pose_anchor = float(terms["pose_anchor"].detach().item())
    shape_anchor = float(terms["shape_anchor"].detach().item())
    total = float(terms["total"].detach().item())
    true_total = float(expected_true_objective)
    relative_error = abs(total - true_total) / max(abs(true_total), 1e-16)
    if relative_error > 1e-6:
        raise RuntimeError("OBJECTIVE_BREAKDOWN_MISMATCH")
    return {
        "primary": primary,
        "pose_anchor": pose_anchor,
        "shape_anchor": shape_anchor,
        "total": total,
        "true_objective": true_total,
        "relative_error": relative_error,
    }


def _production_args(args: Any) -> argparse.Namespace:
    return argparse.Namespace(
        frames=args.frames,
        hmr_dir=args.hmr_dir,
        sam3d_dense_dir=args.sam3d_dense_dir,
        bbox_dir=args.bbox_dir,
        image_width=args.image_width,
        image_height=args.image_height,
        output_dir="/tmp/staged_lm_objective_breakdown",
        device=args.device,
        validate_device_contract=False,
        reproj_objective="charbonnier_irls",
        stop_after_stage="stage2",
        profile_timing=False,
        export_debug029_format=False,
    )


def run_objective_breakdown_validation(args: Any) -> int:
    """Run production stages and verify the final Joint objective terms."""
    context = build_context(_production_args(args))
    stage1_outcome = run_stage(build_stage1_definition(context), context)
    stage2_outcome = run_stage2(context, stage1_outcome)
    pose_shape_outcome = stage2_outcome.pose_shape_refinement
    if pose_shape_outcome is None or pose_shape_outcome.definition is None:
        raise RuntimeError("MISSING_JOINT_OUTCOME_FOR_OBJECTIVE_BREAKDOWN")
    objective_terms_fn = pose_shape_outcome.definition.objective_terms_fn
    if objective_terms_fn is None:
        raise RuntimeError("MISSING_JOINT_OBJECTIVE_TERMS_FN")
    breakdown = compute_objective_breakdown(
        objective_terms_fn,
        pose_shape_outcome.lm_result.parameters,
        expected_true_objective=pose_shape_outcome.lm_result.objective,
    )
    print("[OBJECTIVE_BREAKDOWN] " + json.dumps(breakdown, sort_keys=True))
    print("OBJECTIVE_BREAKDOWN_VALIDATION_PASS")
    return 0
