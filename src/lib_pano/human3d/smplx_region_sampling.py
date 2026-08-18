"""Deterministic, anatomy-stratified sampling of SMPL-X surface vertices."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np


REGION_JOINTS: dict[str, tuple[int, ...]] = {
    "head_neck": (12, 15, 22, 23, 24),
    "torso": (0, 3, 6, 9, 13, 14),
    "left_upper_arm": (16,),
    "right_upper_arm": (17,),
    "left_lower_arm": (18,),
    "right_lower_arm": (19,),
    "left_hand": (20, *range(25, 40)),
    "right_hand": (21, *range(40, 55)),
    "left_upper_leg": (1,),
    "right_upper_leg": (2,),
    "left_lower_leg": (4,),
    "right_lower_leg": (5,),
    "left_foot": (7, 10),
    "right_foot": (8, 11),
}

# N=100 target allocation. Other requested sizes scale these weights while
# preserving one vertex per anatomical region and exact total cardinality.
REGION_WEIGHTS: dict[str, int] = {
    "head_neck": 10,
    "torso": 24,
    "left_upper_arm": 6,
    "right_upper_arm": 6,
    "left_lower_arm": 6,
    "right_lower_arm": 6,
    "left_hand": 5,
    "right_hand": 5,
    "left_upper_leg": 6,
    "right_upper_leg": 6,
    "left_lower_leg": 6,
    "right_lower_leg": 6,
    "left_foot": 4,
    "right_foot": 4,
}


def assign_smplx_vertex_regions(lbs_weights: np.ndarray) -> dict[str, np.ndarray]:
    """Assign each vertex using its maximum-weight SMPL-X joint."""
    weights = np.asarray(lbs_weights)
    if weights.ndim != 2 or weights.shape[1] < 55:
        raise ValueError(f"expected LBS weights [V,>=55], got {weights.shape}")
    if not np.isfinite(weights).all():
        raise ValueError("LBS weights contain non-finite values")
    dominant = np.argmax(weights[:, :55], axis=1)
    assigned: dict[str, np.ndarray] = {}
    covered = np.zeros(weights.shape[0], dtype=bool)
    for region, joints in REGION_JOINTS.items():
        ids = np.flatnonzero(np.isin(dominant, joints)).astype(np.int32)
        if ids.size == 0:
            raise ValueError(f"region {region} has no dominant-joint vertices")
        assigned[region] = ids
        covered[ids] = True
    if not covered.all():
        missing = np.flatnonzero(~covered)
        raise ValueError(f"region map leaves {missing.size} vertices unassigned")
    return assigned


def allocate_region_quotas(
    num_samples: int,
    region_sizes: Mapping[str, int],
) -> dict[str, int]:
    """Scale the N=100 allocation to an exact total with non-empty regions."""
    names = list(REGION_WEIGHTS)
    if num_samples < len(names):
        raise ValueError(f"num_samples must be at least {len(names)}")
    raw = np.asarray([REGION_WEIGHTS[name] for name in names], np.float64)
    raw *= num_samples / raw.sum()
    quota = np.maximum(1, np.floor(raw).astype(int))
    remainder = num_samples - int(quota.sum())
    fractions = raw - np.floor(raw)
    order = np.argsort(-fractions, kind="stable")
    for index in order[:remainder]:
        quota[index] += 1
    result = {name: int(value) for name, value in zip(names, quota)}
    for name, value in result.items():
        if value > int(region_sizes[name]):
            raise ValueError(f"region {name} quota {value} exceeds size")
    if sum(result.values()) != num_samples:
        raise RuntimeError("quota allocation did not preserve exact total")
    return result


def _extremal_local_ids(points: np.ndarray, limit: int) -> list[int]:
    selected: list[int] = []
    # x covers left/right in the neutral template; z covers front/back.
    for axis, reducer in ((0, np.argmin), (0, np.argmax),
                          (2, np.argmin), (2, np.argmax)):
        index = int(reducer(points[:, axis]))
        if index not in selected:
            selected.append(index)
        if len(selected) == limit:
            break
    return selected


def _fps_with_initial(
    points: np.ndarray,
    count: int,
    initial: list[int],
    seed: int,
) -> np.ndarray:
    chosen = list(initial[:count])
    rng = np.random.default_rng(seed)
    if not chosen:
        chosen.append(int(rng.integers(0, points.shape[0])))
    min_distance = np.full(points.shape[0], np.inf, dtype=np.float64)
    for index in chosen:
        diff = points - points[index]
        np.minimum(min_distance, np.einsum("ij,ij->i", diff, diff), out=min_distance)
    min_distance[chosen] = -1.0
    while len(chosen) < count:
        nxt = int(np.argmax(min_distance))
        chosen.append(nxt)
        diff = points - points[nxt]
        np.minimum(min_distance, np.einsum("ij,ij->i", diff, diff), out=min_distance)
        min_distance[chosen] = -1.0
    return np.asarray(chosen, dtype=np.int64)


def sample_smplx_vertices_by_region(
    vertices: np.ndarray,
    lbs_weights: np.ndarray,
    num_samples: int,
    seed: int = 0,
) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, int]]:
    """Select fixed extrema then region-local FPS, returning exact unique IDs."""
    points = np.asarray(vertices, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"expected vertices [V,3], got {points.shape}")
    regions = assign_smplx_vertex_regions(lbs_weights)
    quotas = allocate_region_quotas(
        num_samples, {name: ids.size for name, ids in regions.items()})
    samples: dict[str, np.ndarray] = {}
    for region_index, (name, ids) in enumerate(regions.items()):
        quota = quotas[name]
        local_points = points[ids]
        extrema_count = min(4, quota)
        initial = _extremal_local_ids(local_points, extrema_count)
        local = _fps_with_initial(
            local_points, quota, initial, seed + region_index)
        samples[name] = ids[local].astype(np.int32)
    combined = np.concatenate([samples[name] for name in REGION_WEIGHTS])
    if combined.size != num_samples or np.unique(combined).size != num_samples:
        raise RuntimeError("sample IDs are not exact and unique")
    return combined.astype(np.int32), samples, quotas
