"""Production orchestration of Stage 2 pose and pose/shape refinement."""

from __future__ import annotations

import contextlib
import time
from typing import Any

import torch

from .stage2_shared_cache import build_stage2_shared_geometry_cache
from .stage_definitions.common import synchronize_context_device
from .stage_definitions.pose import build_pose_stage_definition
from .stage_definitions.pose_shape import build_pose_shape_stage_definition
from .stage_handoff import print_handoff
from .stage_runtime import run_stage
from .stage_timing import (
    record_preparation_timing,
    record_stage2_cache_totals,
)
from .types import Stage2Outcome, StageContext, StageOutcome


def run_stage2(
    context: StageContext,
    stage1_outcome: StageOutcome,
    pipeline_timing: dict[str, Any] | None = None,
    profiler_ranges: bool = False,
) -> Stage2Outcome:
    """Run pose warmup followed by pose-and-shape refinement."""
    range_fn = (
        torch.profiler.record_function
        if profiler_ranges else lambda _name: contextlib.nullcontext()
    )
    with range_fn("stage2_shared_cache"):
        shared_cache = build_stage2_shared_geometry_cache(context)
    with range_fn("stage2_pose_definition"):
        pose_definition, pose_handoff = build_pose_stage_definition(
            context, stage1_outcome, shared_cache,
        )
    pose_lm_handoff = pose_handoff["lm"]
    if pose_lm_handoff is not None:
        print_handoff(pose_lm_handoff, "LM", "pose")
        if not pose_lm_handoff["ok"]:
            raise RuntimeError("LM handoff Stage 1->Stage 2 pose failed")
    record_preparation_timing(
        pipeline_timing, "stage2_pose", pose_definition,
    )

    synchronize_context_device(context)
    start = time.perf_counter()
    with range_fn("stage2_pose_solver"):
        pose_outcome = run_stage(pose_definition, context)
    synchronize_context_device(context)
    if pipeline_timing is not None:
        pipeline_timing["stage2_pose_run_ms"] = (
            time.perf_counter() - start
        ) * 1000.0
    handoffs = {}
    if pose_lm_handoff is not None:
        handoffs["stage1_to_stage2_pose_lm"] = pose_lm_handoff
    if context.args.stop_after_stage == "stage2_pose":
        record_stage2_cache_totals(
            pipeline_timing, shared_cache, pose_definition,
        )
        return Stage2Outcome(
            pose_warmup=pose_outcome, pose_shape_refinement=None,
            handoffs=handoffs,
        )

    with range_fn("stage2_pose_shape_definition"):
        pose_shape_definition, _pose_shape_handoff = (
            build_pose_shape_stage_definition(
            context, pose_outcome, shared_cache,
            )
        )
    record_preparation_timing(
        pipeline_timing, "stage2_pose_shape", pose_shape_definition,
    )
    record_stage2_cache_totals(
        pipeline_timing, shared_cache, pose_definition, pose_shape_definition,
    )
    synchronize_context_device(context)
    start = time.perf_counter()
    with range_fn("stage2_pose_shape_solver"):
        pose_shape_outcome = run_stage(pose_shape_definition, context)
    synchronize_context_device(context)
    if pipeline_timing is not None:
        pipeline_timing["stage2_pose_shape_run_ms"] = (
            time.perf_counter() - start
        ) * 1000.0
    return Stage2Outcome(
        pose_warmup=pose_outcome,
        pose_shape_refinement=pose_shape_outcome,
        handoffs=handoffs,
    )
