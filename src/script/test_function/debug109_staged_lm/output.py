"""Output serialisation for rigid alignment and pose/shape fitting.

Stage 1 performs dual-camera orient/transl rigid alignment. Stage 2 performs
pose warmup followed by pose/shape joint refinement.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .irls import build_primary_aggregation_spec, compute_primary_charbonnier_tensor
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
) -> None:
    """Write ``staged_stage2b_beta.json`` and ``staged_stage2b_beta.npz``."""
    with (output_directory / "staged_stage2b_beta.json").open("w") as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)
    np.savez_compressed(str(output_directory / "staged_stage2b_beta.npz"), **arrays)
    print(f"[SAVE] {output_directory}")


def lm_result_to_json(lm_result: LMResult) -> dict[str, Any]:
    """Convert an LM result to its JSON view (excluding the parameter tensor)."""
    return lm_result.to_json_dict()


def lbfgs_result_to_json(lbfgs_result: LBFGSResult | None) -> dict[str, Any] | None:
    """Convert an L-BFGS result to its JSON view, or ``None`` if not run."""
    if lbfgs_result is None:
        return None
    return lbfgs_result.to_json_dict()


def evaluate_gate_c(
    *,
    oracle_enabled: bool,
    stage_matches: dict[str, bool | None],
    non_oracle_checks: dict[str, bool],
    oracle_checks: dict[str, bool],
) -> dict[str, Any]:
    """Build Gate C's three-state verdict without treating a skipped oracle as failure."""
    checks: dict[str, bool | None] = {**stage_matches, **non_oracle_checks, **oracle_checks}
    if not oracle_enabled:
        verdict = "GATE_C_NOT_EVALUATED"
    else:
        verdict = (
            "GATE_C_PASS_READY_FOR_MULTIFRAME_GATE_D"
            if all(value is True for value in checks.values())
            else "GATE_C_BLOCKED"
        )
    return {"verdict": verdict, "oracle_enabled": oracle_enabled, "checks": checks}


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
    raw_groups_fn: Any,
    pose_anchor_fn: Any,
    shape_anchor_fn: Any,
    true_objective_fn: Any,
    parameters: torch.Tensor,
    *,
    is_irls: bool,
    charbonnier_eps: float,
    rep_scale: float,
) -> dict[str, float]:
    """Decompose and validate the active Stage 2 joint objective."""
    groups = raw_groups_fn(parameters)
    if is_irls:
        primary_tensor = compute_primary_charbonnier_tensor(
            groups, charbonnier_eps, build_primary_aggregation_spec(groups)
        )
    else:
        scaled_residual = flatten_reprojection_groups(groups) * rep_scale
        primary_tensor = 0.5 * torch.sum(scaled_residual ** 2)
    primary = float(primary_tensor.detach().item())
    pose_anchor = float(0.5 * torch.sum(pose_anchor_fn(parameters) ** 2).detach().item()) if pose_anchor_fn is not None else 0.0
    shape_anchor = float(0.5 * torch.sum(shape_anchor_fn(parameters) ** 2).detach().item()) if shape_anchor_fn is not None else 0.0
    total = primary + pose_anchor + shape_anchor
    true_total = float(true_objective_fn(parameters).detach().item())
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
