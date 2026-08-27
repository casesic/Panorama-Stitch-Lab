from __future__ import annotations

from dataclasses import replace

import numpy as np

from lib_pano.human_fitting.guided_identity import (
    CandidateDedupConfig, GuidedIdentityConfig, GuidedIdentityReason,
    GuidedIdentityStatus, GuidedPersonCandidate,
    deduplicate_guided_candidates, resolve_guided_identities,
)
from lib_pano.human_fitting.matched_person_tracks import (
    MatchedPerson, TrackState, TrackingConfig,
)


def _person(person_id: int, x: float, *, seam: bool = False) -> MatchedPerson:
    ids = np.arange(12, dtype=np.int32)
    xs = (x + ids * 2.0) % 5760 if seam else x + ids * 2.0
    p2 = np.column_stack((xs, 1100.0 + ids)).astype(np.float32)
    p4 = np.column_stack(((xs + 20.0) % 5760, 1105.0 + ids)).astype(np.float32)
    return MatchedPerson(
        person_id, np.array([x, 1000, x + 100, 1500], np.float32),
        np.array([x + 20, 1000, x + 120, 1500], np.float32),
        np.column_stack((p2, p4)), ids, np.ones(12, bool), seam, seam)


def _candidate(cid: int, source: int, x: float, *, seam=False):
    return GuidedPersonCandidate(
        cid, (source,), _person(cid, x, seam=seam), 0, 0, 12, 0.1,
        raw_candidate_ids=(cid,))


def _track(tid: int, x: float):
    return TrackState(tid, 1, None, _person(tid, x))


def _config(margin_deg=0.25, margin_ratio=1.25):
    return GuidedIdentityConfig(
        TrackingConfig(max_temporal_angle_deg=20.0), margin_deg, margin_ratio)


def test_candidate_dedup_same_person_merges_provenance():
    unique, groups = deduplicate_guided_candidates([
        _candidate(5, 0, 1000), _candidate(9, 1, 1000.2)])
    assert len(unique) == 1
    assert unique[0].source_track_ids == (0, 1)
    assert groups == [(5, 9)]


def test_candidate_non_dedup_different_people():
    unique, _ = deduplicate_guided_candidates([
        _candidate(0, 0, 1000), _candidate(1, 1, 3000)])
    assert len(unique) == 2


def test_erp_seam_candidate_dedup_uses_spherical_geometry():
    first = _candidate(0, 0, 5755, seam=True)
    second = _candidate(1, 1, -5, seam=True)
    unique, _ = deduplicate_guided_candidates(
        [first, second], CandidateDedupConfig(semantic_median_angle_deg=0.2))
    assert len(unique) == 1


def test_two_tracks_one_candidate_is_one_to_one_and_ambiguous():
    result = resolve_guided_identities(
        [_track(0, 1000), _track(1, 1002)], [_candidate(0, 0, 1001)],
        _config())
    assert sum(row.candidate_id is not None for row in result.assignments) == 1
    assert {row.status for row in result.assignments} == {
        GuidedIdentityStatus.GUIDED_AMBIGUOUS,
        GuidedIdentityStatus.GUIDED_UNASSIGNED,
    }
    assigned = next(row for row in result.assignments if row.candidate_id is not None)
    assert assigned.reason == GuidedIdentityReason.CANDIDATE_ASSIGNMENT_AMBIGUOUS


def test_two_tracks_two_people_global_assignment():
    result = resolve_guided_identities(
        [_track(0, 1000), _track(1, 3000)],
        [_candidate(10, 1, 3001), _candidate(11, 0, 1001)], _config())
    assert all(row.status == GuidedIdentityStatus.GUIDED_ACCEPTED
               for row in result.assignments)
    assigned_x = {row.track_id: result.candidates[row.candidate_id].person.pairs[0, 0]
                  for row in result.assignments}
    assert assigned_x[0] < assigned_x[1]


def test_track_side_ambiguity_triggers():
    result = resolve_guided_identities(
        [_track(0, 1000)],
        [_candidate(0, 0, 1000.0), _candidate(1, 0, 1000.2)], _config())
    # Near-identical candidates are physically deduplicated first.
    assert len(result.candidates) == 1
    # Force two distinct candidates but a deliberately strict track-side margin.
    result = resolve_guided_identities(
        [_track(0, 1000)],
        [_candidate(0, 0, 1005), _candidate(1, 0, 1030)],
        _config(margin_deg=10.0, margin_ratio=10.0),
        CandidateDedupConfig(semantic_median_angle_deg=0.01))
    assert result.assignments[0].reason == GuidedIdentityReason.TRACK_ASSIGNMENT_AMBIGUOUS


def test_candidate_and_track_order_do_not_change_physical_assignment():
    tracks = [_track(1, 3000), _track(0, 1000)]
    candidates = [_candidate(8, 1, 3001), _candidate(3, 0, 1001)]
    first = resolve_guided_identities(tracks, candidates, _config())
    second = resolve_guided_identities(list(reversed(tracks)),
                                       list(reversed(candidates)), _config())
    def physical(result):
        return {row.track_id: round(float(result.candidates[row.candidate_id]
                                          .person.pairs[0, 0]))
                for row in result.assignments}
    assert physical(first) == physical(second)


def test_source_track_is_only_provenance_and_scores_cannot_affect_identity():
    candidate = _candidate(4, 99, 1001)
    result = resolve_guided_identities([_track(0, 1000)], [candidate], _config())
    assert result.assignments[0].track_id == 0
    assert result.candidates[0].source_track_ids == (99,)
    assert "score" not in GuidedPersonCandidate.__dataclass_fields__
