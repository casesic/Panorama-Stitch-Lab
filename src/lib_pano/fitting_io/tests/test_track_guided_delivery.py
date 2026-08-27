import numpy as np

from lib_pano.fitting_io.track_guided_delivery import build_guided_crop_spec
from lib_pano.human_fitting.track_guided_policy import (
    DeliveryFrameMode, TrackGuidedScheduler,
    run_delivery_with_same_frame_recovery,
)


def test_seam_aware_bbox_does_not_expand_to_full_erp():
    points = np.array([[5750, 1000], [5, 1010], [20, 1200], [5735, 1190]], np.float32)
    spec = build_guided_crop_spec(
        0, "cam2", points, np.ones(4, bool), 5760, 2880, 0.2, 10)
    assert spec.roi.wraps_seam
    assert spec.roi.width < 200


def test_margin_expansion_and_minimum_margin():
    points = np.array([[100, 100], [200, 100], [200, 300], [100, 300]], np.float32)
    spec = build_guided_crop_spec(
        0, "cam2", points, np.ones(4, bool), 5760, 2880, 0.1, 30)
    assert spec.roi.width == 160
    assert spec.roi.height == 260


def test_scheduler_skips_detector_and_keeps_periodic_schedule():
    scheduler = TrackGuidedScheduler(3, 100)
    assert scheduler.mode(100, 1) == DeliveryFrameMode.FULL_DETECTION
    assert scheduler.mode(101, 1) == DeliveryFrameMode.TRACK_GUIDED
    assert scheduler.mode(102, 1) == DeliveryFrameMode.TRACK_GUIDED
    assert scheduler.mode(103, 1) == DeliveryFrameMode.FULL_DETECTION
    assert scheduler.mode(102, 0) == DeliveryFrameMode.FULL_DETECTION


def test_guided_success_has_zero_detector_and_no_recovery():
    decision = run_delivery_with_same_frame_recovery(
        DeliveryFrameMode.TRACK_GUIDED, ["track"],
        lambda: (_ for _ in ()).throw(AssertionError("unexpected full path")),
        lambda tracks: (["guided"], [], {"densekp_calls": 2}),
    )
    assert decision.actual_mode == DeliveryFrameMode.TRACK_GUIDED
    assert decision.detector_calls == 0
    assert decision.densekp_calls == 2


def test_guided_failure_recovers_same_frame_without_score_semantics():
    # The policy accepts only explicit geometry failures; there is no score or
    # confidence argument in either the guided acceptance or recovery API.
    failure = {"reason": "CROSS_CAMERA_GEOMETRY_FAIL", "raw_score": 1.0}
    decision = run_delivery_with_same_frame_recovery(
        DeliveryFrameMode.TRACK_GUIDED, ["track"],
        lambda: (["recovered"], {"detector_calls": 4, "densekp_calls": 6}),
        lambda tracks: ([], [failure], {"densekp_calls": 2}),
    )
    assert decision.actual_mode == DeliveryFrameMode.RECOVERY
    assert decision.observations == ["recovered"]
    assert decision.detector_calls == 4
    assert decision.densekp_calls == 8


def test_periodic_full_can_discover_new_person():
    decision = run_delivery_with_same_frame_recovery(
        DeliveryFrameMode.FULL_DETECTION, ["old"],
        lambda: (["old", "new"], {"detector_calls": 4, "densekp_calls": 4}),
        lambda tracks: ([], [], {}),
    )
    assert decision.observations == ["old", "new"]


def test_multiple_identity_failures_share_one_same_frame_full_recovery():
    calls = 0
    def full():
        nonlocal calls
        calls += 1
        return ["track0", "track1"], {"detector_calls": 4, "densekp_calls": 4}
    decision = run_delivery_with_same_frame_recovery(
        DeliveryFrameMode.TRACK_GUIDED, ["track0", "track1"], full,
        lambda tracks: ([], [
            {"track": 0, "reason": "CANDIDATE_ASSIGNMENT_AMBIGUOUS"},
            {"track": 1, "reason": "NO_UNIQUE_ASSIGNMENT"},
        ], {"densekp_calls": 4}),
    )
    assert calls == 1
    assert decision.actual_mode == DeliveryFrameMode.RECOVERY
    assert decision.densekp_calls == 8
