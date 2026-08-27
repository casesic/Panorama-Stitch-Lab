from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from lib_pano.fitting_io.densekp_runtime_export import (
    person_initial_state_from_smplx,
)


def _camera(*, translation: float = 0.0, body_offset: float = 0.0):
    return SimpleNamespace(
        orient=np.arange(3, dtype=np.float32).reshape(1, 3),
        transl=np.full((1, 3), translation, np.float32),
        body_pose=(np.arange(63, dtype=np.float32) + body_offset).reshape(1, 63),
        beta=np.arange(10, dtype=np.float32),
        left_hand_pose=np.zeros((1, 45), np.float32),
        right_hand_pose=np.ones((1, 45), np.float32),
        scale=np.asarray([1.25], np.float32),
    )


def test_person_state_uses_established_75_parameter_layout() -> None:
    cam2, cam4 = _camera(translation=2), _camera(translation=4)
    state = person_initial_state_from_smplx(cam2, cam4)
    assert state.frame_parameters.shape == (75,)
    np.testing.assert_array_equal(state.frame_parameters[:3], cam2.orient[0])
    np.testing.assert_array_equal(state.frame_parameters[3:6], cam2.transl[0])
    np.testing.assert_array_equal(state.frame_parameters[6:9], cam4.orient[0])
    np.testing.assert_array_equal(state.frame_parameters[9:12], cam4.transl[0])
    np.testing.assert_array_equal(state.frame_parameters[12:], cam2.body_pose[0])


def test_person_state_rejects_nonshared_body_state() -> None:
    with pytest.raises(ValueError, match="share body pose"):
        person_initial_state_from_smplx(_camera(), _camera(body_offset=1))
