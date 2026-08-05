"""Output serialisation for rigid alignment and pose/shape fitting.

Stage 1 performs dual-camera orient/transl rigid alignment. Stage 2 performs
pose warmup followed by pose/shape joint refinement.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from lib_pano.fitting_io.fitting_result_io import save_segment_result
from lib_pano.human_fitting.smplx_sequence import SMPLX_Seq_Forward_Module

from .config import (
    PRODUCTION_JOINT_EVALUATION,
    PRODUCTION_JOINT_FK,
    PRODUCTION_JOINT_LINEARIZATION,
    PRODUCTION_POSE_EVALUATION,
    PRODUCTION_POSE_LINEARIZATION,
    PRODUCTION_STAGE1_EVALUATION,
    PRODUCTION_STAGE1_LINEARIZATION,
)
from .parameter_layout import unpack_frame_parameters
from .residuals import flatten_reprojection_groups
from .types import LMResult, StageContext, StageOutcome


_COMPAT_RESULT_KEYS = {
    "stage1": ("x_s1_lm", "beta_anchor", False),
    "pose_warmup": ("x_stage2_pose_lm", "beta_anchor", True),
    "joint_refinement": ("x_stage2_joint_lm", "beta_lm", True),
}


def stage_outcome_to_json(outcome: StageOutcome) -> dict[str, Any]:
    """Serialize one Stage outcome without evaluating its residual function."""
    definition = outcome.definition
    summary = definition.input_summary if definition is not None else None
    if summary is None:
        raise RuntimeError("MISSING_STAGE_INPUT_SUMMARY_FOR_OUTPUT")
    return {
        "lm": lm_result_to_json(outcome.lm_result),
        "input_summary": {
            "T": summary.frame_count,
            "param_dim": summary.parameter_dimension,
            "n_groups": summary.group_count,
            "total_valid_points": summary.total_valid_points,
        },
    }


def build_saved_config(args: Any, context: StageContext) -> dict[str, Any]:
    """Build the stable production configuration stored in result JSON."""
    return {
        **vars(args),
        "jacobian_backend": PRODUCTION_STAGE1_LINEARIZATION,
        "stage1_eval_backend": PRODUCTION_STAGE1_EVALUATION,
        "stage2_pose_linearization_backend": PRODUCTION_POSE_LINEARIZATION,
        "stage2_pose_eval_backend": PRODUCTION_POSE_EVALUATION,
        "stage2_joint_linearization_backend": PRODUCTION_JOINT_LINEARIZATION,
        "stage2_joint_eval_backend": PRODUCTION_JOINT_EVALUATION,
        "stage2_joint_linear_solver": "schur",
        "stage2_joint_fk_jacobian_backend": PRODUCTION_JOINT_FK,
        "validate_stage1_fast_eval": False,
        "validate_stage2_pose_fast_eval": False,
        "validate_stage2_joint_fast_eval": False,
        "validate_irls_first_order": False,
        "solver_config": asdict(context.solver_config),
        "objective_config": asdict(context.objective_config),
    }


def build_initial_results(
    args: Any,
    context: StageContext,
    pipeline_timing: dict[str, Any],
    stage1_outcome: StageOutcome,
) -> dict[str, Any]:
    """Build the result JSON root after Stage 1 has completed."""
    return {
        "config": build_saved_config(args, context),
        "execution": {
            "requested_device": getattr(args, "device", "cpu"),
            "resolved_device": str(context.device),
            "dtype": str(context.dtype).replace("torch.", ""),
            "environment": context.device_environment,
        },
        "pipeline_timing": pipeline_timing,
        "stage1": stage_outcome_to_json(stage1_outcome),
    }


def build_beta_metrics(
    beta: torch.Tensor,
    beta_anchor: torch.Tensor,
    beta_abs_limit: float,
) -> dict[str, Any]:
    """Build the compact Joint-stage beta summary."""
    beta_max = float(beta.abs().max())
    return {
        "lm": {
            "max": beta_max,
            "sat": beta_max / beta_abs_limit,
            "delta": float(torch.norm(beta - beta_anchor)),
        },
    }


def _flat_cpu_tensor(value: Any, name: str, expected_size: int) -> torch.Tensor:
    tensor = torch.as_tensor(value).detach().reshape(-1).to(device="cpu", dtype=torch.float32).clone()
    if tensor.numel() != expected_size:
        raise ValueError(
            f"{name} shape mismatch: expected {expected_size} values, got {tensor.numel()}"
        )
    return tensor


def build_compatible_smplx_sequences(
    context: Any,
    arrays: dict[str, np.ndarray],
    stage: str,
) -> tuple[Any, Any, torch.Tensor]:
    """Rebuild debug029-compatible CPU SMPL-X sequences from staged-LM arrays."""
    if stage not in _COMPAT_RESULT_KEYS:
        raise ValueError(f"Unsupported compatibility export stage: {stage}")
    parameter_key, beta_key, include_body_pose = _COMPAT_RESULT_KEYS[stage]
    if parameter_key not in arrays or beta_key not in arrays:
        raise KeyError(f"Compatibility export requires {parameter_key!r} and {beta_key!r}")

    per_frame_dim = 12 + (context.body_pose_dim if include_body_pose else 0)
    expected_parameter_count = context.frame_count * per_frame_dim
    if stage == "joint_refinement":
        expected_parameter_count += context.beta_dim
    parameters = _flat_cpu_tensor(arrays[parameter_key], parameter_key, expected_parameter_count)
    shared_beta = _flat_cpu_tensor(arrays[beta_key], beta_key, context.beta_dim)

    sequences = [
        SMPLX_Seq_Forward_Module.SMPLX_Param_Seq(
            num_batch=context.frame_count, gender="neutral",
        )
        for _ in range(2)
    ]
    scales: list[list[torch.Tensor]] = [[], []]
    for position in range(context.frame_count):
        body_pose_override = None
        if not include_body_pose:
            body_pose_override = context.body_pose_anchors[position].reshape(1, -1)
        unpacked = unpack_frame_parameters(
            parameters, position, per_frame_dim, include_body_pose,
            body_pose_override=body_pose_override,
        )
        frozen = context.frozen_params_by_frame[position]
        for camera_index, camera in enumerate(("cam2", "cam4")):
            sequence = sequences[camera_index]
            sequence.orient[position] = _flat_cpu_tensor(
                getattr(unpacked, f"{camera}_orient"), f"{camera} orient", 3,
            )
            sequence.transl[position] = _flat_cpu_tensor(
                getattr(unpacked, f"{camera}_transl"), f"{camera} transl", 3,
            )
            sequence.body_pose[position] = _flat_cpu_tensor(
                unpacked.body_pose, "body pose", context.body_pose_dim,
            )
            sequence.left_hand_pose[position] = _flat_cpu_tensor(
                getattr(frozen, f"{camera}_left_hand_pose"), f"{camera} left hand pose", 45,
            )
            sequence.right_hand_pose[position] = _flat_cpu_tensor(
                getattr(frozen, f"{camera}_right_hand_pose"), f"{camera} right hand pose", 45,
            )
            scales[camera_index].append(_flat_cpu_tensor(
                getattr(frozen, f"{camera}_scale"), f"{camera} scale", 1,
            ))

    for camera_index, sequence in enumerate(sequences):
        reference_scale = scales[camera_index][0]
        if any(not torch.equal(scale, reference_scale) for scale in scales[camera_index][1:]):
            raise ValueError(
                "Cannot export debug029 format: SMPLX_Param_Seq supports one shared scale "
                f"but cam{2 if camera_index == 0 else 4} scales differ between frames"
            )
        sequence.beta = shared_beta.clone()
        sequence.scale = reference_scale.clone()
    return sequences[0], sequences[1], shared_beta


def export_debug029_compatible_result(
    output_root: Path,
    context: Any,
    arrays: dict[str, np.ndarray],
    stage: str,
) -> None:
    """Write the cam2/cam4 pickle layout consumed by debug077."""
    frame_indices = list(context.frame_indices)
    expected = list(range(frame_indices[0], frame_indices[-1] + 1))
    if frame_indices != expected:
        raise ValueError(
            "--export_debug029_format requires contiguous ascending --frames; "
            f"got {frame_indices}"
        )
    result_c2, result_c4, shared_beta = build_compatible_smplx_sequences(
        context, arrays, stage,
    )
    valid_frame_count_c2 = sum(int(ids.numel() > 0) for ids in context.cam2_vertex_ids_by_frame)
    valid_frame_count_c4 = sum(int(ids.numel() > 0) for ids in context.cam4_vertex_ids_by_frame)
    save_segment_result(
        result_c2=result_c2,
        result_c4=result_c4,
        shared_beta=shared_beta,
        output_dir=output_root,
        frame_idxs=frame_indices,
        valid_frame_count_c2=valid_frame_count_c2,
        valid_frame_count_c4=valid_frame_count_c4,
    )
    segment_key = f"{frame_indices[0]}_{frame_indices[-1]}"
    for camera in ("cam2", "cam4"):
        meta_path = output_root / camera / f"seg_{segment_key}_meta.json"
        with meta_path.open("r", encoding="utf-8") as file:
            metadata = json.load(file)
        metadata["export_source"] = "staged_lm"
        metadata["export_stage"] = stage
        with meta_path.open("w", encoding="utf-8") as file:
            json.dump(metadata, file, ensure_ascii=False, indent=2)
    print(f"[SAVE debug029-compatible] {output_root} segment={frame_indices[0]}_{frame_indices[-1]}")


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
    pipeline_timing: dict[str, Any] | None = None,
    command_start: float | None = None,
) -> None:
    """Write ``staged_stage2b_beta.json`` and ``staged_stage2b_beta.npz``."""
    start = time.perf_counter()
    json_path = output_directory / "staged_stage2b_beta.json"
    with json_path.open("w") as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)
    np.savez_compressed(str(output_directory / "staged_stage2b_beta.npz"), **arrays)
    if pipeline_timing is not None:
        output_save_ms = (time.perf_counter() - start) * 1000.0
        profile = pipeline_timing.get("profile")
        if isinstance(profile, dict):
            profile.setdefault("after_fitting", {})["output_save_ms"] = output_save_ms
            pipeline_timing.pop("profile")
        if command_start is not None:
            pipeline_timing["command_total_ms"] = (time.perf_counter() - command_start) * 1000.0
        if isinstance(profile, dict):
            pipeline_timing["profile"] = profile
        results["pipeline_timing"] = pipeline_timing
        # Rewrite only the small JSON metadata so it contains the measured
        # first complete save. The JSON rewrite itself is intentionally excluded.
        with json_path.open("w") as f:
            json.dump(results, f, ensure_ascii=False, indent=2, default=str)
    print(f"[SAVE] {output_directory}")


def lm_result_to_json(lm_result: LMResult) -> dict[str, Any]:
    """Convert an LM result to its JSON view (excluding the parameter tensor)."""
    return lm_result.to_json_dict()


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
    return arrays


def collect_stage2_pose_arrays(
    stage1_initial: torch.Tensor,
    stage2_pose_initial_lm: torch.Tensor,
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
    stage1_raw_groups_fn: Any,
    stage2_pose_raw_groups_fn: Any,
    stage2_joint_raw_groups_fn: Any,
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

    return arrays
