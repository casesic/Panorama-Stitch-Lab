"""Compact serialization of stage preparation timing."""

from __future__ import annotations

from typing import Any

from .stage2_shared_cache import Stage2SharedGeometryCache
from .types import StageContext, StageDefinition, StagePreparationTiming


def build_pipeline_timing(context: StageContext) -> dict[str, Any]:
    """Build the compact timing result and optional profiling structure."""
    pipeline_timing: dict[str, Any] = {}
    if not context.args.profile_timing:
        return pipeline_timing

    pipeline_timing["profile"] = {
        "units": "ms",
        "containment": {
            "fitting_total_ms": [
                "stage1_prepare", "stage1_run_ms", "stage2_prepare",
                "stage2_pose_run_ms", "stage2_joints_run_ms",
                "orchestration_overhead",
            ],
            "command_total_ms": [
                "context.total_ms", "fitting_total_ms", "after_fitting",
            ],
            "stage2_prepare": [
                "shared_cache", "stage2_pose", "stage2_joint",
            ],
        },
        "context": {
            "total_ms": context.context_timing["context_total_ms"],
            "resolve_device_ms": context.context_timing["resolve_device_ms"],
            "load_observations_ms": context.context_timing["load_observations_ms"],
            "load_hmr_ms": context.context_timing["load_hmr_ms"],
            "model_load_ms": context.context_timing["smplx_model_load_ms"],
            "model_to_device_ms": context.context_timing[
                "smplx_model_to_device_ms"
            ],
        },
    }
    return pipeline_timing


def _profile(pipeline_timing: dict[str, Any]) -> dict[str, Any] | None:
    value = pipeline_timing.get("profile")
    return value if isinstance(value, dict) else None


def record_preparation_timing(
    pipeline_timing: dict[str, Any] | None,
    prefix: str,
    stage: StageDefinition,
) -> None:
    if pipeline_timing is None:
        return
    profile = _profile(pipeline_timing)
    if profile is None:
        return
    preparation = stage.preparation_timing
    profile.setdefault("stage2_prepare_breakdown", {})[prefix] = {
        "total_ms": preparation.total_ms,
        "handoff_ms": preparation.handoff_ms,
        "cache_ms": preparation.cache_setup_ms,
        "bind_ms": preparation.definition_bind_ms,
    }


def record_stage2_cache_totals(
    pipeline_timing: dict[str, Any] | None,
    shared_cache: Stage2SharedGeometryCache,
    pose_definition: StageDefinition,
    joint_definition: StageDefinition | None = None,
) -> None:
    if pipeline_timing is None:
        return
    pose_preparation = pose_definition.preparation_timing
    joint_preparation = (
        joint_definition.preparation_timing
        if joint_definition is not None else StagePreparationTiming()
    )
    stage2_prepare = (
        shared_cache.setup_ms
        + pose_preparation.total_ms
        + joint_preparation.total_ms
    )
    # Preparation finishes after the Pose run has already been measured. Reinsert
    # that value so the serialized summary follows the pipeline reading order.
    pose_run_ms = pipeline_timing.pop("stage2_pose_run_ms", None)
    pipeline_timing["stage2_prepare"] = stage2_prepare
    if pose_run_ms is not None:
        pipeline_timing["stage2_pose_run_ms"] = pose_run_ms
    profile = _profile(pipeline_timing)
    if profile is not None:
        profile.setdefault("stage2_prepare_breakdown", {})["shared_cache"] = {
            "total_ms": shared_cache.setup_ms,
        }
