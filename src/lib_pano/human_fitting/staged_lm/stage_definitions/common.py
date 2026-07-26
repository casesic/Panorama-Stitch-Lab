"""Small construction helpers shared by production stage definitions."""

from __future__ import annotations

from typing import Any

import torch

from ..irls import build_primary_aggregation_spec_from_observations
from ..types import StageContext, StageInputSummary


def build_input_summary(
    context: StageContext,
    parameter_dimension: int,
    anchor_residual_dimension: int,
) -> StageInputSummary:
    total_valid_points = sum(
        int(ids.numel())
        for ids in (
            *context.cam2_vertex_ids_by_frame,
            *context.cam4_vertex_ids_by_frame,
        )
    )
    reprojection_dimension = 2 * total_valid_points
    return StageInputSummary(
        frame_count=context.frame_count,
        parameter_dimension=parameter_dimension,
        group_count=2 * context.frame_count,
        total_valid_points=total_valid_points,
        reprojection_residual_dimension=reprojection_dimension,
        anchor_residual_dimension=anchor_residual_dimension,
        surrogate_residual_dimension=(
            reprojection_dimension + anchor_residual_dimension
        ),
    )


def build_static_aggregation_spec(
    context: StageContext,
) -> dict[str, Any] | None:
    if not context.is_irls:
        return None
    return build_primary_aggregation_spec_from_observations(
        context.frame_indices,
        context.cam2_vertex_ids_by_frame,
        context.cam4_vertex_ids_by_frame,
    )


def synchronize_context_device(context: StageContext) -> None:
    if context.device.type == "cuda":
        torch.cuda.synchronize(context.device)
