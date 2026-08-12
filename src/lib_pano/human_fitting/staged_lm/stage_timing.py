"""Build and serialize pipeline and per-stage timing summaries."""

from __future__ import annotations

from typing import Any

from .stage2_shared_cache import Stage2SharedGeometryCache
from .types import StageContext, StageDefinition, StagePreparationTiming


def build_stage_run_profile(
    raw: dict[str, Any],
    wall_time_ms: float,
    stage: StageDefinition,
) -> dict[str, Any]:
    """Convert raw LM-runner timings into the public per-stage profile."""
    accounted_ms = (
        raw["normal_equations_ms"]
        + raw["objective_eval_ms"]
        + raw["linear_solve_ms"]
        + raw["final_metrics_ms"]
    )
    return {
        "units": "ms",
        "normal_equations_ms": raw["normal_equations_ms"],
        "true_objective_ms": raw["objective_eval_ms"],
        "linear_solve_ms": raw["linear_solve_ms"],
        "final_metrics_ms": raw["final_metrics_ms"],
        "other_runner_ms": max(0.0, wall_time_ms - accounted_ms),
        "accounting": {
            "approximately_sums_to": "wall_time_ms",
            "exclusive_buckets": [
                "normal_equations_ms", "true_objective_ms",
                "linear_solve_ms", "final_metrics_ms", "other_runner_ms",
            ],
            "other_runner_includes": [
                "linearization_construction", "python_overhead",
            ],
        },
        "eval_backend": stage.eval_backend,
        "linearization_backend": stage.linearization_backend,
        "linear_solver_backend": stage.linear_solver_backend,
    }


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
                "stage2_pose_run_ms", "stage2_pose_shape_run_ms",
                "orchestration_overhead",
            ],
            "command_total_ms": [
                "context.total_ms", "fitting_total_ms", "after_fitting",
            ],
            "stage2_prepare": [
                "shared_cache", "stage2_pose", "stage2_pose_shape",
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
    pose_shape_definition: StageDefinition | None = None,
) -> None:
    if pipeline_timing is None:
        return
    pose_preparation = pose_definition.preparation_timing
    pose_shape_preparation = (
        pose_shape_definition.preparation_timing
        if pose_shape_definition is not None else StagePreparationTiming()
    )
    stage2_prepare = (
        shared_cache.setup_ms
        + pose_preparation.total_ms
        + pose_shape_preparation.total_ms
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
