"""Output serialisation for rigid alignment and pose/shape fitting.

Stage 1 performs dual-camera orient/transl rigid alignment. Stage 2 performs
pose warmup followed by pose/shape joint refinement.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .residuals import flatten_reprojection_groups
from .types import LBFGSResult, LMResult, StageOutcome


def build_output_directory(output_root: Path, frame_indices: list[int]) -> Path:
    """Return the per-segment output directory ``<output_root>/<first>_<last>``."""
    segment = f"{frame_indices[0]}_{frame_indices[-1]}"
    output_directory = output_root / segment
    output_directory.mkdir(parents=True, exist_ok=True)
    return output_directory


def save_experiment_outputs(
    output_directory: Path,
    results: dict[str, Any],
    arrays: dict[str, np.ndarray],
    *,
    pipeline_timing: dict[str, float] | None = None,
    command_start: float | None = None,
) -> None:
    """Write ``staged_stage2b_beta.json`` and ``staged_stage2b_beta.npz``."""
    start = time.perf_counter()
    json_path = output_directory / "staged_stage2b_beta.json"
    with json_path.open("w") as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)
    np.savez_compressed(str(output_directory / "staged_stage2b_beta.npz"), **arrays)
    if pipeline_timing is not None:
        pipeline_timing["output_save_ms"] = (time.perf_counter() - start) * 1000.0
        if command_start is not None:
            pipeline_timing["command_total_ms"] = (time.perf_counter() - command_start) * 1000.0
        results["pipeline_timing"] = pipeline_timing
        # Rewrite only the small JSON metadata so it contains the measured
        # first complete save. This rewrite is intentionally not folded back
        # into output_save_ms to avoid a self-referential timing value.
        with json_path.open("w") as f:
            json.dump(results, f, ensure_ascii=False, indent=2, default=str)
    print(f"[SAVE] {output_directory}")


def lm_result_to_json(lm_result: LMResult) -> dict[str, Any]:
    """Convert an LM result to its JSON view (excluding the parameter tensor)."""
    return lm_result.to_json_dict()


def lbfgs_result_to_json(lbfgs_result: LBFGSResult | None) -> dict[str, Any] | None:
    """Convert an L-BFGS result to its JSON view, or ``None`` if not run."""
    if lbfgs_result is None:
        return None
    return lbfgs_result.to_json_dict()


def collect_stage1_arrays(
    stage1_initial: torch.Tensor,
    stage1_outcome: StageOutcome,
    beta_anchor: torch.Tensor,
    beta_latent0: torch.Tensor,
    stage1_raw_groups_fn: Any,
) -> dict[str, np.ndarray]:
    """Collect NPZ arrays for the Stage 1 early-exit path."""
    arrays: dict[str, np.ndarray] = {
        "x0_s1": stage1_initial.cpu().numpy(),
        "x_s1_lm": stage1_outcome.lm_result.parameters.cpu().numpy(),
        "beta_anchor": beta_anchor.cpu().numpy(),
        "beta_latent0": beta_latent0.cpu().numpy(),
        "raw_reproj_s1_lm": flatten_reprojection_groups(
            stage1_raw_groups_fn(stage1_outcome.lm_result.parameters)
        ).detach().cpu().numpy(),
    }
    if stage1_outcome.lbfgs_result is not None:
        arrays["x_s1_lbfgs"] = stage1_outcome.lbfgs_result.parameters.cpu().numpy()
    return arrays


def collect_stage2_pose_arrays(
    stage1_initial: torch.Tensor,
    stage2_pose_initial_lm: torch.Tensor,
    stage2_pose_initial_lbfgs: torch.Tensor | None,
    stage1_outcome: StageOutcome,
    stage2_pose_outcome: StageOutcome,
    beta_anchor: torch.Tensor,
    beta_latent0: torch.Tensor,
    stage1_raw_groups_fn: Any,
    stage2_pose_raw_groups_fn: Any,
) -> dict[str, np.ndarray]:
    """Collect NPZ arrays for the Stage 2 pose warmup early-exit path."""
    raw_s1 = flatten_reprojection_groups(
        stage1_raw_groups_fn(stage1_outcome.lm_result.parameters)
    ).detach().cpu().numpy()
    raw_pose = flatten_reprojection_groups(
        stage2_pose_raw_groups_fn(stage2_pose_outcome.lm_result.parameters)
    ).detach().cpu().numpy()
    arrays: dict[str, np.ndarray] = {
        "x0_s1": stage1_initial.cpu().numpy(),
        "x0_stage2_pose_lm": stage2_pose_initial_lm.cpu().numpy(),
        "x_s1_lm": stage1_outcome.lm_result.parameters.cpu().numpy(),
        "x_stage2_pose_lm": stage2_pose_outcome.lm_result.parameters.cpu().numpy(),
        "beta_anchor": beta_anchor.cpu().numpy(),
        "beta_latent0": beta_latent0.cpu().numpy(),
        "raw_reproj_s1_lm": raw_s1,
        "raw_reproj_stage2_pose_lm": raw_pose,
    }
    if stage1_outcome.lbfgs_result is not None:
        arrays["x_s1_lbfgs"] = stage1_outcome.lbfgs_result.parameters.cpu().numpy()
    if stage2_pose_outcome.lbfgs_result is not None and stage2_pose_initial_lbfgs is not None:
        arrays["x_stage2_pose_lbfgs"] = stage2_pose_outcome.lbfgs_result.parameters.cpu().numpy()
        arrays["x0_stage2_pose_lbfgs"] = stage2_pose_initial_lbfgs.cpu().numpy()
    return arrays


def collect_stage2_arrays(
    stage1_initial: torch.Tensor,
    stage2_pose_initial_lm: torch.Tensor,
    stage2_joint_initial_lm: torch.Tensor,
    stage1_outcome: StageOutcome,
    stage2_pose_outcome: StageOutcome,
    stage2_joint_outcome: StageOutcome,
    beta_anchor: torch.Tensor,
    beta_latent0: torch.Tensor,
    beta_lm: torch.Tensor,
    beta_lbfgs: torch.Tensor | None,
    stage1_raw_groups_fn: Any,
    stage2_pose_raw_groups_fn: Any,
    stage2_joint_raw_groups_fn: Any,
    stage2_pose_initial_lbfgs: torch.Tensor | None,
    stage2_joint_initial_lbfgs: torch.Tensor | None,
) -> dict[str, np.ndarray]:
    """Collect NPZ arrays for the full Stage 2 joint refinement path."""
    raw_s1_lm = flatten_reprojection_groups(
        stage1_raw_groups_fn(stage1_outcome.lm_result.parameters)
    ).detach().cpu().numpy()
    raw_pose_lm = flatten_reprojection_groups(
        stage2_pose_raw_groups_fn(stage2_pose_outcome.lm_result.parameters)
    ).detach().cpu().numpy()
    raw_joint_lm = flatten_reprojection_groups(
        stage2_joint_raw_groups_fn(stage2_joint_outcome.lm_result.parameters)
    ).detach().cpu().numpy()

    arrays: dict[str, np.ndarray] = {
        "x0_s1": stage1_initial.cpu().numpy(),
        "x0_stage2_pose_lm": stage2_pose_initial_lm.cpu().numpy(),
        "x0_stage2_joint_lm": stage2_joint_initial_lm.cpu().numpy(),
        "x_s1_lm": stage1_outcome.lm_result.parameters.cpu().numpy(),
        "x_stage2_pose_lm": stage2_pose_outcome.lm_result.parameters.cpu().numpy(),
        "x_stage2_joint_lm": stage2_joint_outcome.lm_result.parameters.cpu().numpy(),
        "beta_anchor": beta_anchor.cpu().numpy(),
        "beta_latent0": beta_latent0.cpu().numpy(),
        "beta_lm": beta_lm.cpu().numpy(),
        "raw_reproj_s1_lm": raw_s1_lm,
        "raw_reproj_stage2_pose_lm": raw_pose_lm,
        "raw_reproj_stage2_joint_lm": raw_joint_lm,
    }

    if stage1_outcome.lbfgs_result is not None:
        arrays["x_s1_lbfgs"] = stage1_outcome.lbfgs_result.parameters.cpu().numpy()
        arrays["raw_reproj_s1_lbfgs"] = flatten_reprojection_groups(
            stage1_raw_groups_fn(stage1_outcome.lbfgs_result.parameters)
        ).detach().cpu().numpy()
    if stage2_pose_outcome.lbfgs_result is not None:
        arrays["x0_stage2_pose_lbfgs"] = stage2_pose_initial_lbfgs.cpu().numpy()
        arrays["x_stage2_pose_lbfgs"] = stage2_pose_outcome.lbfgs_result.parameters.cpu().numpy()
        arrays["raw_reproj_stage2_pose_lbfgs"] = flatten_reprojection_groups(
            stage2_pose_raw_groups_fn(stage2_pose_outcome.lbfgs_result.parameters)
        ).detach().cpu().numpy()
    if stage2_joint_outcome.lbfgs_result is not None:
        arrays["x0_stage2_joint_lbfgs"] = stage2_joint_initial_lbfgs.cpu().numpy()
        arrays["x_stage2_joint_lbfgs"] = stage2_joint_outcome.lbfgs_result.parameters.cpu().numpy()
        arrays["beta_lbfgs"] = beta_lbfgs.cpu().numpy()
        arrays["raw_reproj_stage2_joint_lbfgs"] = flatten_reprojection_groups(
            stage2_joint_raw_groups_fn(stage2_joint_outcome.lbfgs_result.parameters)
        ).detach().cpu().numpy()

    return arrays


def compute_objective_breakdown(
    objective_terms_fn: Any,
    parameters: torch.Tensor,
    *,
    expected_true_objective: float,
) -> dict[str, float]:
    """Decompose the final joint objective with one fast objective evaluation."""
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
