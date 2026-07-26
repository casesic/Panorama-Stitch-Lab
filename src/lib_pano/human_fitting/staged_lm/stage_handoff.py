"""Parameter-state handoffs and checks between production fitting stages."""

from __future__ import annotations

from typing import Any

import torch

from .config import STAGE1_PER_FRAME_DIM


def print_handoff(
    handoff: dict[str, Any] | None,
    tag: str,
    kind: str,
) -> None:
    if handoff is None:
        return
    if kind == "pose":
        print(
            f"[HANDOFF_{tag}_S1_STAGE2_POSE] ot={handoff['ot_max']:.2e} "
            f"bp={handoff['bp_max']:.2e} ok={handoff['ok']}"
        )
    else:
        raw_value = handoff["raw_reprojection_max_diff"]
        relative_value = handoff["relative_difference_joint_init_to_pose"]
        raw_text = "NOT_RUN" if raw_value is None else f"{raw_value:.2e}"
        relative_text = (
            "NOT_RUN" if relative_value is None else f"{relative_value:.2e}"
        )
        print(
            f"[HANDOFF_{tag}_STAGE2_POSE_JOINT] "
            f"state={handoff['state_max_diff']:.2e} "
            f"bp={handoff['physical_beta_init_max_diff_to_anchor']:.2e} "
            f"raw={raw_text} "
            f"sa={handoff['shape_anchor_objective_at_init']:.2e} "
            f"rel={relative_text} "
            f"scope={handoff.get('validation_scope', 'full')} "
            f"ok={handoff['ok']}"
        )


def check_s1_to_pose(
    stage1_parameters: torch.Tensor,
    stage2_pose_parameters: torch.Tensor,
    body_pose_anchors: list[torch.Tensor],
    frame_count: int,
    body_pose_dim: int,
) -> dict[str, Any]:
    """Verify the Stage 1 to pose handoff."""
    per_frame_dim = STAGE1_PER_FRAME_DIM + body_pose_dim
    orient_transl_max_diff = 0.0
    body_pose_max_diff = 0.0
    for frame in range(frame_count):
        pose_start = frame * per_frame_dim
        stage1_start = frame * STAGE1_PER_FRAME_DIM
        orient_transl_max_diff = max(
            orient_transl_max_diff,
            float((
                stage2_pose_parameters[
                    pose_start:pose_start + STAGE1_PER_FRAME_DIM
                ] - stage1_parameters[
                    stage1_start:stage1_start + STAGE1_PER_FRAME_DIM
                ]
            ).abs().max()),
        )
        body_pose_max_diff = max(
            body_pose_max_diff,
            float((
                stage2_pose_parameters[
                    pose_start + STAGE1_PER_FRAME_DIM:(frame + 1) * per_frame_dim
                ] - body_pose_anchors[frame]
            ).abs().max()),
        )
    return {
        "ot_max": orient_transl_max_diff,
        "bp_max": body_pose_max_diff,
        "ok": orient_transl_max_diff < 1e-6 and body_pose_max_diff < 1e-6,
    }


def handoff_s1_to_pose(
    stage1_parameters: torch.Tensor,
    body_pose_anchors: list[torch.Tensor],
    frame_count: int,
    body_pose_dim: int,
) -> torch.Tensor:
    """Construct the pose-stage initial vector from Stage 1 and anchors."""
    parts: list[torch.Tensor] = []
    for frame in range(frame_count):
        if body_pose_anchors[frame].numel() != body_pose_dim:
            raise ValueError("body pose anchor dimension mismatch")
        parts.append(stage1_parameters[
            frame * STAGE1_PER_FRAME_DIM:(frame + 1) * STAGE1_PER_FRAME_DIM
        ])
        parts.append(body_pose_anchors[frame].clone())
    return torch.cat(parts).clone()


def check_pose_to_joint_state(
    stage2_pose_parameters: torch.Tensor,
    stage2_joint_parameters: torch.Tensor,
    beta_anchor: torch.Tensor,
    beta_abs_limit: float,
    shape_anchor_scale: float,
) -> dict[str, Any]:
    """Check pose state and bounded-beta initialization without model forward."""
    pose_dimension = stage2_pose_parameters.shape[0]
    beta_dimension = stage2_joint_parameters.shape[0] - pose_dimension
    state_max_diff = float((
        stage2_joint_parameters[:pose_dimension] - stage2_pose_parameters
    ).abs().max())
    beta_initial = beta_abs_limit * torch.tanh(
        stage2_joint_parameters[-beta_dimension:]
    )
    beta_max_diff = float((beta_initial - beta_anchor).abs().max())
    shape_anchor_objective = float(
        0.5 * torch.sum(
            (beta_initial - beta_anchor) ** 2 * shape_anchor_scale ** 2
        )
    )
    ok = (
        state_max_diff <= 1e-7
        and beta_max_diff <= 1e-6
        and shape_anchor_objective <= 1e-10
    )
    return {
        "state_max_diff": state_max_diff,
        "physical_beta_init_max_diff_to_anchor": beta_max_diff,
        "raw_reprojection_max_diff": None,
        "shape_anchor_objective_at_init": shape_anchor_objective,
        "relative_difference_joint_init_to_pose": None,
        "expensive_validation_performed": False,
        "validation_scope": "state_only",
        "ok": ok,
    }
