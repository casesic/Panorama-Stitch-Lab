"""Serialization of stage preparation and Stage 2 cache timing metadata."""

from __future__ import annotations

from .stage2_shared_cache import Stage2SharedGeometryCache
from .types import StageDefinition, StagePreparationTiming


def record_preparation_timing(
    pipeline_timing: dict[str, float] | None,
    prefix: str,
    stage: StageDefinition,
) -> None:
    if pipeline_timing is None:
        return
    preparation = stage.preparation_timing
    pipeline_timing[f"{prefix}_prepare_total_ms"] = preparation.total_ms
    pipeline_timing[f"{prefix}_handoff_ms"] = preparation.handoff_ms
    pipeline_timing[f"{prefix}_cache_setup_ms"] = preparation.cache_setup_ms
    pipeline_timing[f"{prefix}_definition_bind_ms"] = preparation.definition_bind_ms
    if prefix in ("stage2_pose", "stage2_joint"):
        pipeline_timing[f"{prefix}_derived_cache_setup_ms"] = stage.analytic_cache.setup_ms
    if prefix == "stage2_pose":
        cache = stage.analytic_cache
        pipeline_timing["stage2_pose_fast_cache_setup_ms"] = cache.setup_ms
        pipeline_timing["stage2_pose_batch_cache_setup_ms"] = cache.batch_cache_setup_ms
        pipeline_timing["stage2_pose_batch_padding_ratio"] = cache.batch_padding_ratio
        pipeline_timing["stage2_pose_batch_persistent_bytes"] = cache.batch_persistent_bytes
        pipeline_timing["stage2_pose_batch_group_count"] = cache.batch_group_count
        pipeline_timing["stage2_pose_batch_nmax"] = cache.batch_nmax
    if prefix == "stage2_joint":
        cache = stage.analytic_cache
        pipeline_timing["stage2_joint_fast_cache_setup_ms"] = cache.setup_ms
        pipeline_timing["stage2_joint_batch_cache_setup_ms"] = cache.batch_cache_setup_ms
        pipeline_timing["stage2_joint_batch_padding_ratio"] = cache.batch_padding_ratio
        pipeline_timing["stage2_joint_batch_persistent_bytes"] = cache.batch_persistent_bytes
        pipeline_timing["stage2_joint_batch_group_count"] = cache.batch_group_count
        pipeline_timing["stage2_joint_batch_nmax"] = cache.batch_nmax


def record_stage2_cache_totals(
    pipeline_timing: dict[str, float] | None,
    shared_cache: Stage2SharedGeometryCache,
    pose_definition: StageDefinition,
    joint_definition: StageDefinition | None = None,
) -> None:
    if pipeline_timing is None:
        return
    pipeline_timing.update(shared_cache.timing_dict())
    pose_preparation = pose_definition.preparation_timing
    joint_preparation = (
        joint_definition.preparation_timing
        if joint_definition is not None else StagePreparationTiming()
    )
    pipeline_timing["stage2_total_cache_setup_ms"] = (
        shared_cache.setup_ms
        + pose_preparation.cache_setup_ms
        + joint_preparation.cache_setup_ms
    )
    pipeline_timing["stage2_prepare_total_ms"] = (
        shared_cache.setup_ms
        + pose_preparation.total_ms
        + joint_preparation.total_ms
    )
