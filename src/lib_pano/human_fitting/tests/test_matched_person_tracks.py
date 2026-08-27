from __future__ import annotations

import numpy as np

from lib_pano.human_fitting.matched_person_tracks import (
    MatchedPerson,
    MatchedPersonFittingOrchestrator,
    MatchedPersonTrackManager,
    OptimizedPersonState,
    TrackingConfig,
    choose_best_view,
)


def _person(person_id: int, x: float = 1000.0, *, cam2_edge=False) -> MatchedPerson:
    ids = np.arange(12, dtype=np.int32)
    p2 = np.column_stack((x + ids, 1200.0 + ids)).astype(np.float32)
    p4 = np.column_stack((x + 20.0 + ids, 1205.0 + ids)).astype(np.float32)
    return MatchedPerson(
        person_id=person_id,
        bbox_cam2=np.asarray([0.0 if cam2_edge else x, 1000, x + 200, 1600], np.float32),
        bbox_cam4=np.asarray([x, 1000, x + 180, 1550], np.float32),
        pairs=np.column_stack((p2, p4)), point_ids=ids,
        valid=np.ones(12, dtype=bool),
    )


def _fit(person, initialization, track_id):
    if isinstance(initialization, OptimizedPersonState):
        beta = initialization.beta.copy()
        parameters = initialization.frame_parameters.copy()
    else:
        beta = np.full(10, track_id + 0.25, np.float32)
        parameters = np.full(75, track_id, np.float32)
    return OptimizedPersonState(
        parameters, beta, person.pairs.copy(), person.point_ids.copy(),
        person.valid.copy(), True, "converged", 1.0,
    )


def test_unmatched_detections_never_enter_hmr_or_lm():
    calls = []

    def hmr(camera, persons):
        calls.append((camera, len(persons)))
        return [object() for _ in persons]

    orchestrator = MatchedPersonFittingOrchestrator(
        MatchedPersonTrackManager(), hmr, _fit,
    )
    # The detector may have 2+4 boxes, but only matched persons[] is passed.
    result = orchestrator.process_frame(1, [_person(0)])
    assert calls == [("cam2", 1)]
    assert result.profiling["hmr2_person_count"] == 1
    assert len(result.payload["persons"]) == 1


def test_best_view_is_geometric_and_deterministic():
    assert choose_best_view(_person(0, cam2_edge=True)).camera == "cam4"
    person = _person(0)
    person.bbox_cam4[:] = person.bbox_cam2
    assert choose_best_view(person).camera == "cam2"


def test_track_identity_warm_start_lost_reconnect_and_reinit():
    hmr_person_counts = []

    def hmr(camera, persons):
        hmr_person_counts.append(len(persons))
        return [object() for _ in persons]

    manager = MatchedPersonTrackManager(TrackingConfig(max_reconnect_gap=1))
    orchestrator = MatchedPersonFittingOrchestrator(manager, hmr, _fit)
    first = orchestrator.process_frame(10, [_person(0)])
    second = orchestrator.process_frame(11, [_person(2, 1001)])
    orchestrator.process_frame(12, [])
    reconnect = orchestrator.process_frame(13, [_person(1, 1002)])
    reinit = orchestrator.process_frame(16, [_person(7, 1003)])
    assert [first.payload["persons"][0]["track_id"],
            second.payload["persons"][0]["track_id"],
            reconnect.payload["persons"][0]["track_id"]] == [0, 0, 0]
    assert second.payload["persons"][0]["initialization"]["type"] == "previous_frame"
    assert reconnect.payload["persons"][0]["initialization"]["type"] == "previous_frame"
    assert reinit.payload["persons"][0]["track_id"] == 1
    assert reinit.payload["persons"][0]["initialization"]["type"] == "hmr2"
    assert hmr_person_counts == [1, 1]


def test_people_never_share_beta_and_output_preserves_boundaries():
    orchestrator = MatchedPersonFittingOrchestrator(
        MatchedPersonTrackManager(),
        lambda camera, persons: [object() for _ in persons], _fit,
    )
    result = orchestrator.process_frame(1, [_person(4, 1000), _person(9, 3000)])
    first, second = result.fitted_states
    assert not np.shares_memory(first.beta, second.beta)
    assert not np.array_equal(first.beta, second.beta)
    assert result.payload["schema_version"] == "densekp_multi_person_fitting_v1"
    assert [row["person_id"] for row in result.payload["persons"]] == [4, 9]
    assert all("optimized_pairs" in row and "track_id" in row
               for row in result.payload["persons"])


def test_failed_lm_state_reinitializes_same_track():
    calls = []

    def hmr(camera, persons):
        calls.append(len(persons))
        return [object() for _ in persons]

    attempts = 0

    def fit(person, initialization, track_id):
        nonlocal attempts
        attempts += 1
        result = _fit(person, initialization, track_id)
        if attempts == 1:
            return OptimizedPersonState(
                result.frame_parameters, result.beta, result.optimized_pairs,
                result.point_ids, result.valid, False, "linear_solve_failed", None,
            )
        return result

    orchestrator = MatchedPersonFittingOrchestrator(
        MatchedPersonTrackManager(), hmr, fit,
    )
    first = orchestrator.process_frame(1, [_person(0)])
    second = orchestrator.process_frame(2, [_person(8, 1001)])
    assert first.payload["persons"][0]["track_id"] == 0
    assert second.payload["persons"][0]["track_id"] == 0
    assert second.payload["persons"][0]["initialization"]["type"] == "hmr2_reinit"
    assert calls == [1, 1]
    assert orchestrator.counters.reinit_count == 1


def test_warm_failure_retries_hmr2_in_same_frame():
    hmr_calls = []
    fit_calls = 0

    def hmr(camera, persons):
        hmr_calls.append((camera, len(persons)))
        return [object() for _ in persons]

    def fit(person, initialization, track_id):
        nonlocal fit_calls
        fit_calls += 1
        result = _fit(person, initialization, track_id)
        if fit_calls == 2:  # first attempt on frame 2 is the warm start
            return OptimizedPersonState(
                result.frame_parameters, result.beta, result.optimized_pairs,
                result.point_ids, result.valid, False, "forced_failure")
        return result

    orchestrator = MatchedPersonFittingOrchestrator(
        MatchedPersonTrackManager(), hmr, fit)
    orchestrator.process_frame(1, [_person(0)])
    second = orchestrator.process_frame(2, [_person(1, 1001)])
    row = second.payload["persons"][0]
    assert row["track_id"] == 0
    assert row["initialization"]["type"] == "hmr2_reinit"
    assert row["fitting"]["attempts"] == 2
    assert row["fitting"]["success"] is True
    assert hmr_calls == [("cam2", 1), ("cam2", 1)]
