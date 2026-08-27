"""Geometry-only identity competition for track-guided DenseKP candidates.

Cross-camera matching establishes a current physical-person candidate.  This
module deliberately performs the separate temporal identity decision across
*all* active tracks and candidates.  Detector/DenseKP confidence is absent by
design.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum
import time
from typing import Any, Sequence

import numpy as np

from .matched_person_tracks import (
    MatchedPerson, TrackingConfig, _erp_rays, optimal_cardinality_assignment,
)


class GuidedIdentityStatus(str, Enum):
    GUIDED_ACCEPTED = "GUIDED_ACCEPTED"
    GUIDED_AMBIGUOUS = "GUIDED_AMBIGUOUS"
    GUIDED_UNASSIGNED = "GUIDED_UNASSIGNED"
    FULL_RECOVERED = "FULL_RECOVERED"
    FULL_NOT_FOUND = "FULL_NOT_FOUND"


class GuidedIdentityReason(str, Enum):
    DUPLICATE_CURRENT_CANDIDATE = "DUPLICATE_CURRENT_CANDIDATE"
    TRACK_ASSIGNMENT_AMBIGUOUS = "TRACK_ASSIGNMENT_AMBIGUOUS"
    CANDIDATE_ASSIGNMENT_AMBIGUOUS = "CANDIDATE_ASSIGNMENT_AMBIGUOUS"
    NO_UNIQUE_ASSIGNMENT = "NO_UNIQUE_ASSIGNMENT"


@dataclass(frozen=True)
class GuidedPersonCandidate:
    candidate_id: int
    source_track_ids: tuple[int, ...]
    person: MatchedPerson
    cam2_person_index: int
    cam4_person_index: int
    cross_camera_valid_count: int
    cross_camera_median_angle_deg: float
    source_guided_regions: tuple[Any, ...] = ()
    raw_candidate_ids: tuple[int, ...] = ()


@dataclass(frozen=True)
class CandidateDedupConfig:
    image_width: int = 5760
    image_height: int = 2880
    semantic_median_angle_deg: float = 1.0
    minimum_shared_points: int = 4


@dataclass(frozen=True)
class GuidedIdentityConfig:
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    identity_margin_deg: float = 0.25
    identity_margin_ratio: float = 1.25


@dataclass(frozen=True)
class IdentityAssignment:
    track_id: int
    candidate_id: int | None
    status: GuidedIdentityStatus
    reason: GuidedIdentityReason | None
    temporal_cost_deg: float | None
    track_margin_deg: float | None
    track_margin_ratio: float | None
    candidate_margin_deg: float | None
    candidate_margin_ratio: float | None


@dataclass(frozen=True)
class GuidedIdentityResult:
    candidates: tuple[GuidedPersonCandidate, ...]
    assignments: tuple[IdentityAssignment, ...]
    cost_matrix: np.ndarray
    temporal_diagnostics: tuple[tuple[dict[str, float | int], ...], ...]
    dedup_groups: tuple[tuple[int, ...], ...]
    elapsed_ms: float

    @property
    def accepted(self) -> dict[int, GuidedPersonCandidate]:
        by_id = {candidate.candidate_id: candidate for candidate in self.candidates}
        return {row.track_id: by_id[row.candidate_id] for row in self.assignments
                if row.status == GuidedIdentityStatus.GUIDED_ACCEPTED
                and row.candidate_id is not None}


def _shared_geometry(a: MatchedPerson, b: MatchedPerson,
                     config: TrackingConfig) -> dict[str, float | int]:
    lookup = {int(value): index for index, value in enumerate(a.point_ids)}
    rows = [(row, lookup[int(point_id)]) for row, point_id in enumerate(b.point_ids)
            if int(point_id) in lookup]
    if not rows:
        return {"shared_count": 0, "median_deg": float("inf"),
                "p75_deg": float("inf"), "p90_deg": float("inf"),
                "max_deg": float("inf")}
    br = np.asarray([x for x, _ in rows]); ar = np.asarray([x for _, x in rows])
    valid = b.valid[br] & a.valid[ar]
    bp, ap = b.pairs[br][valid], a.pairs[ar][valid]
    finite = np.isfinite(bp).all(axis=1) & np.isfinite(ap).all(axis=1)
    bp, ap = bp[finite], ap[finite]
    errors = []
    for offset in (0, 2):
        rb = _erp_rays(bp[:, offset:offset + 2], config.image_width,
                       config.image_height)
        ra = _erp_rays(ap[:, offset:offset + 2], config.image_width,
                       config.image_height)
        errors.extend(np.degrees(np.arccos(
            np.clip(np.sum(rb * ra, axis=1), -1.0, 1.0))).tolist())
    values = np.asarray(errors, np.float64)
    if not len(values):
        values = np.asarray([np.inf])
    return {"shared_count": int(len(bp)), "median_deg": float(np.median(values)),
            "p75_deg": float(np.percentile(values, 75)),
            "p90_deg": float(np.percentile(values, 90)),
            "max_deg": float(np.max(values))}


def _candidate_duplicate(a: GuidedPersonCandidate, b: GuidedPersonCandidate,
                         config: CandidateDedupConfig) -> bool:
    geometry = _shared_geometry(a.person, b.person, TrackingConfig(
        image_width=config.image_width, image_height=config.image_height))
    return (geometry["shared_count"] >= config.minimum_shared_points
            and geometry["median_deg"] <= config.semantic_median_angle_deg)


def deduplicate_guided_candidates(
    candidates: Sequence[GuidedPersonCandidate],
    config: CandidateDedupConfig = CandidateDedupConfig(),
) -> tuple[list[GuidedPersonCandidate], list[tuple[int, ...]]]:
    """Merge duplicate observations using seam-safe spherical point geometry."""
    ordered = sorted(candidates, key=lambda c: (
        c.source_track_ids, c.cam2_person_index, c.cam4_person_index,
        c.raw_candidate_ids, c.candidate_id))
    groups: list[list[GuidedPersonCandidate]] = []
    for candidate in ordered:
        compatible = [index for index, group in enumerate(groups)
                      if any(_candidate_duplicate(candidate, item, config)
                             for item in group)]
        if not compatible:
            groups.append([candidate])
            continue
        target = compatible[0]
        groups[target].append(candidate)
        for index in reversed(compatible[1:]):
            groups[target].extend(groups.pop(index))
    unique = []
    dedup_groups = []
    for candidate_id, group in enumerate(groups):
        # Preserve the strongest cross-camera observation; merge provenance only.
        representative = min(group, key=lambda c: (
            c.cross_camera_median_angle_deg, -c.cross_camera_valid_count,
            c.source_track_ids, c.raw_candidate_ids))
        raw_ids = tuple(sorted({value for item in group
                                for value in (item.raw_candidate_ids or
                                              (item.candidate_id,))}))
        source_ids = tuple(sorted({value for item in group
                                   for value in item.source_track_ids}))
        regions = tuple(region for item in group for region in item.source_guided_regions)
        unique.append(replace(representative, candidate_id=candidate_id,
                              source_track_ids=source_ids,
                              source_guided_regions=regions,
                              raw_candidate_ids=raw_ids))
        dedup_groups.append(raw_ids)
    return unique, dedup_groups


def _margin(best: float, alternatives: np.ndarray) -> tuple[float, float]:
    finite = alternatives[np.isfinite(alternatives)]
    if not len(finite):
        return float("inf"), float("inf")
    second = float(np.min(finite))
    return second - best, second / max(best, 1e-12)


def resolve_guided_identities(
    tracks: Sequence[Any], raw_candidates: Sequence[GuidedPersonCandidate],
    config: GuidedIdentityConfig = GuidedIdentityConfig(),
    dedup_config: CandidateDedupConfig | None = None,
) -> GuidedIdentityResult:
    """Resolve current candidates globally; uncertain identities stay unaccepted."""
    started = time.perf_counter()
    tracks = sorted(tracks, key=lambda track: int(track.track_id))
    dedup_config = dedup_config or CandidateDedupConfig(
        config.tracking.image_width, config.tracking.image_height)
    candidates, groups = deduplicate_guided_candidates(raw_candidates, dedup_config)
    costs = np.full((len(tracks), len(candidates)), np.inf, np.float64)
    diagnostics: list[list[dict[str, float | int]]] = []
    for row, track in enumerate(tracks):
        diag_row = []
        for column, candidate in enumerate(candidates):
            previous = replace(track, optimized=None)
            diag = _shared_geometry(previous.last_person, candidate.person,
                                    config.tracking)
            # The production temporal cost is this same shared-semantic median.
            # Reuse the diagnostic computation instead of traversing 138 points twice.
            cost = float(diag["median_deg"])
            diag = {**diag, "cost_deg": float(cost)}
            diag_row.append(diag)
            if np.isfinite(cost) and cost <= config.tracking.max_temporal_angle_deg:
                costs[row, column] = cost
        diagnostics.append(diag_row)
    selected = {row: column for row, column in optimal_cardinality_assignment(costs)}
    assignments = []
    for row, track in enumerate(tracks):
        column = selected.get(row)
        if column is None:
            assignments.append(IdentityAssignment(
                track.track_id, None, GuidedIdentityStatus.GUIDED_UNASSIGNED,
                GuidedIdentityReason.NO_UNIQUE_ASSIGNMENT, None,
                None, None, None, None))
            continue
        best = float(costs[row, column])
        track_margin, track_ratio = _margin(best, np.delete(costs[row], column))
        candidate_margin, candidate_ratio = _margin(best, np.delete(costs[:, column], row))
        reason = None
        if (track_margin < config.identity_margin_deg
                or track_ratio < config.identity_margin_ratio):
            reason = GuidedIdentityReason.TRACK_ASSIGNMENT_AMBIGUOUS
        if (candidate_margin < config.identity_margin_deg
                or candidate_ratio < config.identity_margin_ratio):
            reason = GuidedIdentityReason.CANDIDATE_ASSIGNMENT_AMBIGUOUS
        status = (GuidedIdentityStatus.GUIDED_ACCEPTED if reason is None
                  else GuidedIdentityStatus.GUIDED_AMBIGUOUS)
        assignments.append(IdentityAssignment(
            track.track_id, candidates[column].candidate_id, status, reason, best,
            track_margin, track_ratio, candidate_margin, candidate_ratio))
    return GuidedIdentityResult(
        tuple(candidates), tuple(assignments), costs,
        tuple(tuple(row) for row in diagnostics), tuple(groups),
        (time.perf_counter() - started) * 1000.0)
