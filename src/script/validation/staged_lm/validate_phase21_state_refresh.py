"""Verify previous-frame warm start and HMR2 reinitialization refresh geometry."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess

import numpy as np

from lib_pano.fitting_io.densekp_delivery_io import load_densekp_delivery_observation
from lib_pano.fitting_io.densekp_runtime_export import (
    PersonFrameInitialState,
    materialize_person_runtime_fixture,
)


def _read(path: Path, count: int) -> np.ndarray:
    value = np.fromfile(path, dtype="<f4")
    if value.size != count:
        raise RuntimeError(f"{path}: expected {count}, got {value.size}")
    return value


def _state(frame: np.ndarray, beta: np.ndarray, frozen: Path) -> PersonFrameInitialState:
    return PersonFrameInitialState(
        frame_parameters=frame, beta=beta,
        cam2_left_hand_pose=_read(frozen / "model_cam2_left_hand_pose.f32", 45),
        cam2_right_hand_pose=_read(frozen / "model_cam2_right_hand_pose.f32", 45),
        cam4_left_hand_pose=_read(frozen / "model_cam4_left_hand_pose.f32", 45),
        cam4_right_hand_pose=_read(frozen / "model_cam4_right_hand_pose.f32", 45),
        cam2_scale=float(_read(frozen / "model_cam2_scale.f32", 1)[0]),
        cam4_scale=float(_read(frozen / "model_cam4_scale.f32", 1)[0]),
    )


def _run(executable: Path, fixture: Path, output: Path, backend: str) -> None:
    subprocess.run([
        str(executable), "--fixtures", str(fixture), "--output", str(output),
        "--runtime-prepare-stage1", "--evaluator-backend", backend,
    ], check=True, capture_output=True, text=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--densekp", type=Path, required=True)
    parser.add_argument("--base-fixture", type=Path, required=True)
    parser.add_argument("--hmr-fixture", type=Path, required=True)
    parser.add_argument("--hmr-output", type=Path, required=True)
    parser.add_argument("--cpp-executable", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--evaluator-backend", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    observation = load_densekp_delivery_observation(args.densekp)

    previous_frame = _read(
        args.hmr_output / "frame_0/pose_shape_frame_parameters.f32", 75)
    previous_beta = _read(args.hmr_output / "shared_physical_beta.f32", 10)
    warm_state = _state(previous_frame, previous_beta, args.hmr_fixture)
    warm_fixture, warm_output = args.output / "warm_fixture", args.output / "warm_output"
    materialize_person_runtime_fixture(
        args.base_fixture, warm_fixture, observation, warm_state)
    _run(args.cpp_executable, warm_fixture, warm_output, args.evaluator_backend)
    serialized_frame = np.concatenate((
        _read(warm_fixture / "stage1_initial_parameters.f32", 12),
        _read(warm_fixture / "model_body_pose_anchor.f32", 63),
    ))
    serialized_beta = _read(warm_fixture / "model_beta_anchor.f32", 10)

    # Reinitialize from the original HMR state and demand a fresh preparation.
    hmr_frame = np.concatenate((
        _read(args.hmr_fixture / "stage1_initial_parameters.f32", 12),
        _read(args.hmr_fixture / "model_body_pose_anchor.f32", 63),
    ))
    hmr_beta = _read(args.hmr_fixture / "model_beta_anchor.f32", 10)
    reinit_fixture, reinit_output = (
        args.output / "reinit_fixture", args.output / "reinit_output")
    materialize_person_runtime_fixture(
        args.base_fixture, reinit_fixture, observation,
        _state(hmr_frame, hmr_beta, args.hmr_fixture))
    _run(args.cpp_executable, reinit_fixture, reinit_output, args.evaluator_backend)

    def geometry(output: Path) -> np.ndarray:
        return _read(
            output / "runtime_prepared_geometry/frame_0/cam2_support_local_vertices.f32",
            204 * 3,
        )
    first_geometry = geometry(args.hmr_output)
    warm_geometry, reinit_geometry = geometry(warm_output), geometry(reinit_output)
    report = {
        "warm_start_frame_parameters_exact": bool(np.array_equal(previous_frame, serialized_frame)),
        "warm_start_beta_exact": bool(np.array_equal(previous_beta, serialized_beta)),
        "warm_geometry_changed_from_hmr": bool(not np.array_equal(first_geometry, warm_geometry)),
        "reinit_geometry_exact_to_hmr": bool(np.array_equal(first_geometry, reinit_geometry)),
        "warm_output_finite": bool(np.isfinite(warm_geometry).all()),
        "reinit_output_finite": bool(np.isfinite(reinit_geometry).all()),
    }
    report["pass"] = all(report.values())
    (args.output / "state_refresh_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"PHASE21_STATE_REFRESH={'PASS' if report['pass'] else 'FAIL'}")
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
