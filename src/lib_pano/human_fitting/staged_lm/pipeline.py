"""Production orchestration of Stage 2 pose warmup and joint refinement."""

from __future__ import annotations

import contextlib
import time

import torch

from .stage2_shared_cache import build_stage2_shared_geometry_cache
from .stage_definitions.common import synchronize_context_device
from .stage_definitions.joint import build_joint_stage_definition
from .stage_definitions.pose import build_pose_stage_definition
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
    pipeline_timing: dict[str, float] | None = None,
    profiler_ranges: bool = False,
) -> Stage2Outcome:
    """Run fixed production pose warmup followed by joint refinement."""
    range_fn = (
        torch.profiler.record_function
        if profiler_ranges else lambda _name: contextlib.nullcontext()
    )
    with range_fn("stage2_shared_cache"):
        shared_cache = build_stage2_shared_geometry_cache(context)
    if pipeline_timing is not None:
        pipeline_timing.update(shared_cache.timing_dict())
    with range_fn("stage2_pose_definition"):
        pose_definition, pose_handoff = build_pose_stage_definition(
            context, stage1_outcome, shared_cache,
        )
    print_handoff(pose_handoff["lm"], "LM", "pose")
    if not pose_handoff["lm"]["ok"]:
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
        run_ms = (time.perf_counter() - start) * 1000.0
        pipeline_timing["stage2_pose_run_ms"] = run_ms
        pipeline_timing["stage2_pose_run_overhead_ms"] = max(
            0.0, run_ms - pose_outcome.lm_result.wall_time_ms,
        )
    handoffs = {
        "stage1_to_stage2_pose_lm": pose_handoff["lm"],
        "stage1_to_stage2_pose_lbfgs": None,
    }
    if context.args.stop_after_stage == "stage2_pose":
        record_stage2_cache_totals(
            pipeline_timing, shared_cache, pose_definition,
        )
        return Stage2Outcome(
            pose_warmup=pose_outcome, joint_refinement=None, handoffs=handoffs,
        )

    with range_fn("stage2_joint_definition"):
        joint_definition, joint_handoff = build_joint_stage_definition(
            context, pose_outcome, shared_cache,
        )
    print_handoff(joint_handoff["lm"], "LM", "joint")
    if not joint_handoff["lm"]["ok"]:
        raise RuntimeError("LM handoff Stage 2 pose->joint failed")
    record_preparation_timing(
        pipeline_timing, "stage2_joint", joint_definition,
    )
    record_stage2_cache_totals(
        pipeline_timing, shared_cache, pose_definition, joint_definition,
    )
    handoffs.update({
        "stage2_pose_to_joint_lm": joint_handoff["lm"],
        "stage2_pose_to_joint_lbfgs": None,
    })

    synchronize_context_device(context)
    start = time.perf_counter()
    with range_fn("stage2_joint_solver"):
        joint_outcome = run_stage(joint_definition, context)
    synchronize_context_device(context)
    if pipeline_timing is not None:
        run_ms = (time.perf_counter() - start) * 1000.0
        pipeline_timing["stage2_joint_run_ms"] = run_ms
        pipeline_timing["stage2_joint_run_overhead_ms"] = max(
            0.0, run_ms - joint_outcome.lm_result.wall_time_ms,
        )
    return Stage2Outcome(
        pose_warmup=pose_outcome,
        joint_refinement=joint_outcome,
        handoffs=handoffs,
    )
