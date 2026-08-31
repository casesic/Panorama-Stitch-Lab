"""Regression tests for the HMR2 pinhole -> spherical translation fix."""

import numpy as np
import pytest

from lib_pano.fitting_io.densekp_runtime_export import PersonFrameInitialState
from lib_pano.human_fitting.hmr2_staged_init import (
    SPHERICAL_TRANSLATION_RADIUS, erp_pixel_direction,
    project_model_points, spherical_translation_initialization)
from lib_pano.human_fitting.matched_person_tracks import OptimizedPersonState
from lib_pano.human_fitting.per_person_cpp import CppPerPersonFitter


def _observations(center_uv, spread_px, count=64, seed=7):
    rng = np.random.default_rng(seed)
    return np.stack([center_uv[0] + rng.uniform(-spread_px, spread_px, count),
                     center_uv[1] + rng.uniform(-spread_px, spread_px, count)],
                    axis=1)


def _frozen_state():
    state = PersonFrameInitialState(
        frame_parameters=np.arange(75, dtype=np.float32) * 0.01,
        beta=np.linspace(-0.5, 0.5, 10, dtype=np.float32),
        cam2_left_hand_pose=np.full(45, 0.1, np.float32),
        cam2_right_hand_pose=np.full(45, 0.2, np.float32),
        cam4_left_hand_pose=np.full(45, 0.3, np.float32),
        cam4_right_hand_pose=np.full(45, 0.4, np.float32),
        cam2_scale=1.25, cam4_scale=1.25)
    state.validate()
    return state


def test_pinhole_depth_not_directly_used():
    """A pinhole-style depth (~hundreds) must never reach the translation."""
    points2 = _observations((2000.0, 1400.0), 250.0)
    points4 = _observations((1900.0, 1350.0), 250.0)
    valid = np.ones(len(points2), bool)
    t2, t4 = spherical_translation_initialization(
        points2, points4, valid,
        bbox_cam2=np.array([1900.0, 1200.0, 2100.0, 1600.0]),
        bbox_cam4=np.array([1800.0, 1200.0, 2000.0, 1600.0]))
    for translation in (t2, t4):
        assert np.isfinite(translation).all()
        # HMR2 pinhole translations carry depth ~O(100); the spherical
        # initialization must produce the fixed basin radius instead.
        assert np.linalg.norm(translation) == pytest.approx(
            SPHERICAL_TRANSLATION_RADIUS, rel=1e-9)


def test_spherical_initial_projection_extent_does_not_collapse():
    """Synthetic body: spherical init spreads, pinhole-depth init collapses."""
    rng = np.random.default_rng(3)
    body = rng.normal(size=(300, 3))          # synthetic body at the origin,
    body *= 0.6 / np.linalg.norm(body, axis=1, keepdims=True)  # ~0.6 units
    # real degenerate values from the phase22-era fixture
    pinhole_translation = np.array([-291.5, -194.8, 10.0])  # real degenerate final state
    t2, _ = spherical_translation_initialization(
        _observations((2000.0, 1400.0), 250.0),
        _observations((1900.0, 1350.0), 250.0), np.ones(64, bool),
        bbox_cam2=np.array([1900.0, 1200.0, 2100.0, 1600.0]),
        bbox_cam4=np.array([1800.0, 1200.0, 2000.0, 1600.0]))
    spherical_uv = project_model_points(body + t2, 5760, 2880)
    pinhole_uv = project_model_points(body + pinhole_translation, 5760, 2880)

    def extent(uv):
        return (uv[:, 0].max() - uv[:, 0].min()) * (uv[:, 1].max() - uv[:, 1].min())

    assert extent(spherical_uv) > 2000.0         # body-sized, not collapsed
    assert extent(pinhole_uv) < 200.0            # reproduces the original bug


def test_translation_direction_matches_observation_centroid_ray():
    points2 = _observations((2000.0, 1400.0), 100.0)
    points4 = _observations((1900.0, 1350.0), 100.0)
    valid = np.ones(len(points2), bool)
    t2, t4 = spherical_translation_initialization(
        points2, points4, valid,
        bbox_cam2=np.array([1900.0, 1200.0, 2100.0, 1600.0]),
        bbox_cam4=np.array([1800.0, 1200.0, 2000.0, 1600.0]))
    expected2 = erp_pixel_direction(
        np.asarray([[2000.0, 1400.0]]), 5760, 2880)[0]
    expected4 = erp_pixel_direction(
        np.asarray([[1900.0, 1350.0]]), 5760, 2880)[0]
    # mean-of-rays over a finite patch deviates from the center-pixel ray by a
    # second-order amount (~0.1 deg here); the bug this guards against is the
    # ~90 deg pinhole-direction error, so the tolerance stays far below that.
    assert np.degrees(np.arccos(np.clip(
        t2 @ expected2 / np.linalg.norm(t2), -1, 1))) < 0.5
    assert np.degrees(np.arccos(np.clip(
        t4 @ expected4 / np.linalg.norm(t4), -1, 1))) < 0.5


def test_warm_state_path_unchanged_by_spherical_initialization():
    """Warm start must keep merging frozen hand/scale state exactly as before."""
    fitter = CppPerPersonFitter(type("C", (), {
        "base_fixture": None, "executable": None, "output_root": None,
        "evaluator_backend": "cpu"})())
    frozen = _frozen_state()
    merged = fitter._initial_state(frozen, 3)
    assert merged is frozen                      # cold path archives frozen state
    optimized = OptimizedPersonState(
        frame_parameters=np.full(75, 2.0, np.float32),
        beta=np.full(10, -1.0, np.float32),
        optimized_pairs=np.zeros((4, 4), np.float32),
        point_ids=np.arange(4, dtype=np.int32),
        valid=np.ones(4, bool), success=True, status="ok")
    warm = fitter._initial_state(optimized, 3)
    assert np.array_equal(warm.frame_parameters, optimized.frame_parameters)
    assert np.array_equal(warm.beta, optimized.beta)
    assert np.array_equal(warm.cam2_left_hand_pose, frozen.cam2_left_hand_pose)
    assert np.array_equal(warm.cam4_right_hand_pose, frozen.cam4_right_hand_pose)
    assert warm.cam2_scale == frozen.cam2_scale
    with pytest.raises(RuntimeError):
        fitter._initial_state(optimized, 99)     # unknown track stays fail-fast


_BBOX2 = np.array([1900.0, 1200.0, 2100.0, 1600.0])
_BBOX4 = np.array([1800.0, 1200.0, 2000.0, 1600.0])


@pytest.mark.parametrize("valid_count", [0, 1, 2, 3])
def test_valid_below_minimum_uses_bbox_center_not_pinhole(valid_count):
    """valid<4 must fall to the bbox-center spherical ray, never to pinhole."""
    rng = np.random.default_rng(valid_count)
    points2 = _observations((2000.0, 1400.0), 300.0)
    points4 = _observations((1900.0, 1350.0), 300.0)
    valid = np.zeros(len(points2), bool)
    if valid_count:
        valid[:valid_count] = True       # a few stray/invalid points only
        points2 = points2.copy(); points4 = points4.copy()
        points2[:valid_count] += rng.normal(0, 2000, (valid_count, 2))
        points4[:valid_count] += rng.normal(0, 2000, (valid_count, 2))
    t2, t4 = spherical_translation_initialization(
        points2, points4, valid,
        bbox_cam2=_BBOX2, bbox_cam4=_BBOX4)
    for translation in (t2, t4):
        assert np.isfinite(translation).all()
        assert np.linalg.norm(translation) == pytest.approx(
            SPHERICAL_TRANSLATION_RADIUS, rel=1e-9)
    # direction follows the bbox-center ray, not the stray points
    from densekp_delivery.camera.erp_rotate import rotate_erp_points
    center2 = rotate_erp_points(np.asarray([[2000.0, 1400.0]]), 5760, 2880)[0]
    expected2 = erp_pixel_direction(center2[None, :], 5760, 2880)[0]
    assert np.degrees(np.arccos(np.clip(
        t2 @ expected2 / np.linalg.norm(t2), -1, 1))) < 2.0


def test_valid_below_minimum_without_bbox_fails_fast():
    """No points AND no bbox: refuse instead of silently keeping pinhole."""
    import pytest as _pytest
    with _pytest.raises(ValueError, match="refusing to fall back"):
        spherical_translation_initialization(
            _observations((2000.0, 1400.0), 100.0, count=8),
            _observations((1900.0, 1350.0), 100.0, count=8),
            np.zeros(8, bool))


def test_pinhole_depth_never_reaches_spherical_state():
    """PINHOLE_DEPTH_NEVER_REACHES_SPHERICAL_STATE invariant.

    A huge pinhole translation may only enter as metadata (it is passed in
    nowhere here); whatever the inputs, the constructed cold-start
    translation norm must be the fixed basin radius.
    """
    rng = np.random.default_rng(11)
    for seed_valid in (0, 1, 4, 30):
        n = 64
        points2 = _observations((1500.0, 2000.0), 250.0, count=n, seed=seed_valid)
        points4 = _observations((1400.0, 1900.0), 250.0, count=n, seed=seed_valid)
        valid = np.zeros(n, bool) if seed_valid == 0 else (
            (rng.random(n) < (seed_valid / 30)))
        valid[:1] = seed_valid > 0            # guarantee >=1 when requested
        if seed_valid == 0:
            valid[:] = False
        kwargs = {}
        if valid.sum() < 4:
            kwargs = {"bbox_cam2": _BBOX2, "bbox_cam4": _BBOX4}
        t2, t4 = spherical_translation_initialization(points2, points4, valid,
                                                      **kwargs)
        for translation in (t2, t4):
            assert np.linalg.norm(translation) < 10.0   # never O(100)


def test_seam_wrapped_bbox_center_uses_minimal_arc():
    """A seam-wrapping bbox center must stay on the wrapped side of the ERP."""
    points2 = np.full((6, 2), np.nan)        # ignored: valid mask all False
    points4 = np.full((6, 2), np.nan)
    t2, _ = spherical_translation_initialization(
        points2, points4, np.zeros(6, bool),
        bbox_cam2=np.array([5700.0, 1400.0, 60.0, 1600.0]),   # wraps 5760
        bbox_cam4=_BBOX4, bbox_cam2_wraps_seam=True)
    # bbox spans [5700, 5760+60]: arc center ≈ 30 (+ seam) -> near lon ±pi
    # the center pixel must NOT be (5700+60)/2=2880 (image middle)
    from densekp_delivery.camera.erp_rotate import rotate_erp_points
    arc_center_erp = (2880.0, 1500.0)         # x=5760+... wraps to ~30: min arc
    # verify via direction: expected ray from ERP x≈5730 (center of [5700,60])
    expected_erp = np.asarray([[5730.0, 1500.0]])
    expected_rot = rotate_erp_points(expected_erp, 5760, 2880)
    expected = erp_pixel_direction(expected_rot, 5760, 2880)[0]
    angle = np.degrees(np.arccos(np.clip(
        t2 @ expected / np.linalg.norm(t2), -1, 1)))
    assert angle < 2.0


def test_corrected_fitting_extent_ratio_not_tiny():
    """Diagnostic guard: extent ratio on the corrected path is not near zero."""
    from lib_pano.human_fitting.per_person_cpp import projection_extent_ratio
    rng = np.random.default_rng(5)
    raw = np.zeros((100, 4))
    raw[:, 0] = 1700 + rng.uniform(0, 500, 100)
    raw[:, 1] = 1300 + rng.uniform(0, 220, 100)
    raw[:, 2] = 1600 + rng.uniform(0, 480, 100)
    raw[:, 3] = 1250 + rng.uniform(0, 260, 100)
    valid = np.ones(100, bool)
    collapsed = raw.mean(axis=0, keepdims=True) + np.zeros((100, 4))
    ratio_collapsed = projection_extent_ratio(collapsed, raw, valid)
    healthy = raw + rng.normal(0, 3, raw.shape)   # few-pixel fitting residual
    ratio_healthy = projection_extent_ratio(healthy, raw, valid)
    assert ratio_collapsed < 0.02
    assert ratio_healthy > 0.9
