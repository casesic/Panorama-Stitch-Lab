from __future__ import annotations

from pathlib import Path

import numpy as np

from lib_pano.fitting_io.densekp_runtime_export import PersonFrameInitialState
from lib_pano.human_fitting.matched_person_tracks import OptimizedPersonState
from lib_pano.human_fitting.per_person_cpp import (
    CppPerPersonFitter,
    CppPersonFitterConfig,
)


def _initial(offset: float = 0) -> PersonFrameInitialState:
    return PersonFrameInitialState(
        frame_parameters=np.full(75, offset, np.float32),
        beta=np.full(10, offset, np.float32),
        cam2_left_hand_pose=np.full(45, 2, np.float32),
        cam2_right_hand_pose=np.full(45, 3, np.float32),
        cam4_left_hand_pose=np.full(45, 4, np.float32),
        cam4_right_hand_pose=np.full(45, 5, np.float32),
        cam2_scale=1.1, cam4_scale=1.2,
    )


def test_warm_start_uses_optimized_state_and_track_frozen_values(tmp_path: Path) -> None:
    fitter = CppPerPersonFitter(CppPersonFitterConfig(
        base_fixture=tmp_path, executable=tmp_path / "unused",
        output_root=tmp_path / "out", evaluator_backend="cpu"))
    cold = fitter._initial_state(_initial(), 7)
    optimized = OptimizedPersonState(
        frame_parameters=np.arange(75, dtype=np.float32),
        beta=np.arange(10, dtype=np.float32),
        optimized_pairs=np.zeros((1, 4), np.float32),
        point_ids=np.zeros(1, np.int32), valid=np.ones(1, bool), success=True)
    warm = fitter._initial_state(optimized, 7)
    np.testing.assert_array_equal(warm.frame_parameters, optimized.frame_parameters)
    np.testing.assert_array_equal(warm.beta, optimized.beta)
    np.testing.assert_array_equal(warm.cam2_left_hand_pose, cold.cam2_left_hand_pose)
    assert warm.cam4_scale == cold.cam4_scale


def test_hmr_reinit_replaces_track_frozen_values(tmp_path: Path) -> None:
    fitter = CppPerPersonFitter(CppPersonFitterConfig(
        base_fixture=tmp_path, executable=tmp_path / "unused",
        output_root=tmp_path / "out", evaluator_backend="cpu"))
    fitter._initial_state(_initial(), 3)
    refreshed = fitter._initial_state(_initial(9), 3)
    np.testing.assert_array_equal(
        fitter._frozen_by_track[3].frame_parameters, refreshed.frame_parameters)
