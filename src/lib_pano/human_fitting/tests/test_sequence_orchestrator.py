"""Production sequence orchestrator timeline checks (no models required)."""

from __future__ import annotations

import numpy as np

from lib_pano.human_fitting.sequence_orchestrator import (
    ErpHmr2ColdStartBatch, TrackGuidedSequenceConfig,
    TrackGuidedSequenceOrchestrator,
)
from lib_pano.fitting_io.densekp_runtime_export import PersonFrameInitialState
from lib_pano.fitting_io.track_guided_delivery import GuidedTrackObservation
from lib_pano.human_fitting.matched_person_tracks import (
    MatchedPerson, OptimizedPersonState,
)


def _valid_masks(width: int = 64, height: int = 32):
    # cam2 and cam4 valid regions overlap on a middle band, as real ERPs do.
    mask = np.zeros((height, width), bool)
    mask[:, width // 4:3 * width // 4] = True
    return mask, mask.copy()


def _person(person_id: int = 0, value: float = 100.0) -> MatchedPerson:
    pairs = np.full((8, 4), value, np.float32)
    pairs += np.arange(len(pairs), dtype=np.float32).reshape(-1, 1)
    return MatchedPerson(
        person_id=person_id,
        bbox_cam2=np.array([value - 10, value - 20, value + 10, value + 20],
                           np.float32),
        bbox_cam4=np.array([value - 10, value - 20, value + 10, value + 20],
                           np.float32),
        pairs=pairs,
        point_ids=np.arange(len(pairs), dtype=np.int32),
        valid=np.ones(len(pairs), bool),
    )


def _frozen_state() -> PersonFrameInitialState:
    state = PersonFrameInitialState(
        frame_parameters=np.zeros(75, np.float32), beta=np.zeros(10, np.float32),
        cam2_left_hand_pose=np.zeros(45, np.float32),
        cam2_right_hand_pose=np.zeros(45, np.float32),
        cam4_left_hand_pose=np.zeros(45, np.float32),
        cam4_right_hand_pose=np.zeros(45, np.float32),
        cam2_scale=1.0, cam4_scale=1.0)
    state.validate()
    return state


class _StubHmr(ErpHmr2ColdStartBatch):
    """Bypasses HMR2Initializer entirely; records per-camera batch calls."""

    def __init__(self):
        self.calls = []

    def __call__(self, camera, persons):
        self.calls.append((camera, [p.person_id for p in persons]))
        return [_frozen_state() for _ in persons]


class _StubRuntime:
    profile_timing = {}


def _orchestrator(config: TrackGuidedSequenceConfig):
    builder = TrackGuidedSequenceOrchestrator(
        _StubRuntime(), _StubHmr(), _no_fitter, _valid_masks(), config,
        sequence_start_frame=10)
    return builder


def _no_fitter(person, initialization, track_id):
    raise AssertionError("stub must be replaced by the test")


def test_full_frames_keep_a_continuous_in_memory_warm_start_timeline():
    calls = []
    fits = []

    def fit(person: MatchedPerson, initialization, track_id: int):
        fits.append((track_id, type(initialization).__name__))
        offset = float(len(fits))
        return OptimizedPersonState(
            frame_parameters=np.full(75, offset, np.float32),
            beta=np.full(10, offset, np.float32),
            optimized_pairs=person.pairs.copy(),
            point_ids=person.point_ids.copy(),
            valid=person.valid.copy(),
            success=True, status="ok",
            warm_state_exact=isinstance(initialization,
                                        OptimizedPersonState))

    builder = _orchestrator(TrackGuidedSequenceConfig(detection_interval=1))
    person_counts = {frame: (2 if frame == 13 else 1) for frame in range(10, 16)}

    def full(cam2, cam4):
        calls.append("full")
        frame = 10 + len(calls) - 1
        count = person_counts[frame]
        if count == 2:
            return [_person(0), _person(1, value=50.0)], {
                "detector_calls": 4, "densekp_calls": 2}
        return [_person(0)], {"detector_calls": 4, "densekp_calls": 2}

    builder._full_delivery = full
    builder.fitting.fit_person = fit
    types_per_frame = []
    for frame in range(10, 16):
        record = builder.process_frame(frame, object(), object())
        assert record.actual_mode.value == "FULL_DETECTION"
        types_per_frame.append([row["initialization"]["type"]
                                for row in record.payload["persons"]])
    # Cold start only on the two genuinely new identities.
    assert types_per_frame[0] == ["hmr2"]
    assert types_per_frame[3] == ["previous_frame", "hmr2"]
    assert all(types == ["previous_frame"]
               for index, types in enumerate(types_per_frame)
               if index not in (0, 3))
    summary = builder.sequence_result({})
    assert summary["unique_tracks"] == 2
    assert summary["new_track_count"] == 2
    assert summary["warm_state_exact"] is True
    assert summary["lm_successes"] == sum(len(t) for t in types_per_frame)


def test_guided_identity_gate_binding_is_used_verbatim():
    builder = _orchestrator(TrackGuidedSequenceConfig(detection_interval=3))

    def full(cam2, cam4):
        return [_person(0)], {"detector_calls": 4, "densekp_calls": 2}

    builder._full_delivery = full

    def fit(person, initialization, track_id):
        return OptimizedPersonState(
            frame_parameters=np.zeros(75, np.float32),
            beta=np.zeros(10, np.float32),
            optimized_pairs=person.pairs.copy(),
            point_ids=person.point_ids.copy(),
            valid=person.valid.copy(),
            success=True, status="ok",
            warm_state_exact=isinstance(initialization, OptimizedPersonState))

    builder.fitting.fit_person = fit
    builder.process_frame(10, object(), object())

    def detector_must_not_run(*_args, **_kwargs):
        raise AssertionError("detector ran on an accepted guided frame")

    def guided(cam2, cam4, tracks):
        candidate = _person(int(tracks[0].track_id))
        row = GuidedTrackObservation(tracks[0].track_id, candidate, None, None,
                                     identity={"status": "GUIDED_ACCEPTED"})
        return [row], [], {"detector_calls": 0, "densekp_calls": 2}

    builder._full_delivery = detector_must_not_run
    builder._guided_delivery = guided
    record = builder.process_frame(11, object(), object())
    assert record.actual_mode.value == "TRACK_GUIDED"
    assert record.detector_calls == 0
    assert record.payload["persons"][0]["initialization"]["type"] == \
           "previous_frame"
    assert record.payload["persons"][0]["initialization"]["source_frame"] == 10


def test_guided_failure_recovers_with_full_detection_same_frame():
    builder = _orchestrator(TrackGuidedSequenceConfig(detection_interval=3))

    def fit(person, initialization, track_id):
        return OptimizedPersonState(
            frame_parameters=np.zeros(75, np.float32),
            beta=np.zeros(10, np.float32),
            optimized_pairs=person.pairs.copy(),
            point_ids=person.point_ids.copy(),
            valid=person.valid.copy(),
            success=True, status="ok",
            warm_state_exact=isinstance(initialization, OptimizedPersonState))

    builder.fitting.fit_person = fit

    def full(_cam2, _cam4):
        return [_person(0)], {"detector_calls": 4, "densekp_calls": 2}

    builder._full_delivery = full
    builder.process_frame(10, object(), object())

    def failing_guided(cam2, cam4, tracks):
        row = GuidedTrackObservation(tracks[0].track_id, None, None, None,
                                     failure_reason="LM_FAIL")
        return [row], [row], {"detector_calls": 0, "densekp_calls": 2}

    builder._guided_delivery = failing_guided
    record = builder.process_frame(11, object(), object())
    assert record.actual_mode.value == "RECOVERY"
    assert record.requested_mode.value == "TRACK_GUIDED"
    assert record.detector_calls == 4
    assert record.payload["persons"][0]["fitting"]["success"] is True

    def accepted_guided(cam2, cam4, tracks):
        candidate = _person(int(tracks[0].track_id))
        row = GuidedTrackObservation(tracks[0].track_id, candidate, None, None,
                                     identity={"status": "GUIDED_ACCEPTED"})
        return [row], [], {"detector_calls": 0, "densekp_calls": 2}

    builder._guided_delivery = accepted_guided
    next_record = builder.process_frame(12, object(), object())
    assert next_record.actual_mode.value == "TRACK_GUIDED"
    assert next_record.detector_calls == 0
