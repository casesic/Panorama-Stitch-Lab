"""Concrete per-person adapter from tracking state to the existing C++ LM."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import subprocess
import time

import numpy as np

from lib_pano.fitting_io.densekp_delivery_io import DenseKPDeliveryObservation
from lib_pano.fitting_io.densekp_runtime_export import (
    PersonFrameInitialState,
    materialize_person_runtime_fixture,
)
from .matched_person_tracks import MatchedPerson, OptimizedPersonState


def _metadata(path: Path) -> dict[str, str]:
    return dict(line.split("=", 1) for line in path.read_text().splitlines() if line)


@dataclass(frozen=True)
class CppPersonFitterConfig:
    base_fixture: Path
    executable: Path
    output_root: Path
    # Build-compatible default for current callers; not a performance claim.
    evaluator_backend: str = "cpu"


class CppPerPersonFitter:
    """Stateful callable used directly as ``MatchedPersonFittingOrchestrator.fit_person``.

    Static model constants come from one shared base fixture. Frozen hand/scale
    state is retained per track, while every call serializes the current
    pose/beta and forces C++ Stage1 geometry regeneration.
    """

    def __init__(self, config: CppPersonFitterConfig):
        if config.evaluator_backend not in {"cpu", "cuda"}:
            raise ValueError("evaluator_backend must be cpu or cuda")
        self.config = config
        self._frozen_by_track: dict[int, PersonFrameInitialState] = {}
        self.call_count = 0

    @staticmethod
    def _observation(person: MatchedPerson) -> DenseKPDeliveryObservation:
        return DenseKPDeliveryObservation(
            points_cam2=np.ascontiguousarray(person.pairs[:, :2], np.float32),
            points_cam4=np.ascontiguousarray(person.pairs[:, 2:], np.float32),
            point_ids=np.ascontiguousarray(person.point_ids, np.int32),
            valid=np.ascontiguousarray(person.valid, bool),
        )

    def _initial_state(self, initialization: object,
                       track_id: int) -> PersonFrameInitialState:
        if isinstance(initialization, PersonFrameInitialState):
            self._frozen_by_track[track_id] = initialization
            return initialization
        if not isinstance(initialization, OptimizedPersonState):
            raise TypeError(
                "initialization must be PersonFrameInitialState or OptimizedPersonState")
        frozen = self._frozen_by_track.get(track_id)
        if frozen is None:
            raise RuntimeError("warm start has no frozen hand/scale state for track")
        state = PersonFrameInitialState(
            frame_parameters=np.asarray(initialization.frame_parameters, np.float32),
            beta=np.asarray(initialization.beta, np.float32),
            cam2_left_hand_pose=frozen.cam2_left_hand_pose,
            cam2_right_hand_pose=frozen.cam2_right_hand_pose,
            cam4_left_hand_pose=frozen.cam4_left_hand_pose,
            cam4_right_hand_pose=frozen.cam4_right_hand_pose,
            cam2_scale=frozen.cam2_scale, cam4_scale=frozen.cam4_scale,
        )
        state.validate()
        return state

    def __call__(self, person: MatchedPerson, initialization: object,
                 track_id: int) -> OptimizedPersonState:
        initial = self._initial_state(initialization, track_id)
        call = self.call_count
        self.call_count += 1
        directory = self.config.output_root / f"track_{track_id:04d}" / f"call_{call:06d}"
        fixture, output = directory / "runtime_input", directory / "cpp_output"
        directory.mkdir(parents=True, exist_ok=False)
        materialize_person_runtime_fixture(
            self.config.base_fixture, fixture, self._observation(person), initial)
        warm_state_exact = None
        if isinstance(initialization, OptimizedPersonState):
            serialized_frame = np.concatenate((
                np.fromfile(fixture / "stage1_initial_parameters.f32", dtype="<f4"),
                np.fromfile(fixture / "model_body_pose_anchor.f32", dtype="<f4"),
            ))
            serialized_beta = np.fromfile(
                fixture / "model_beta_anchor.f32", dtype="<f4")
            warm_state_exact = bool(
                np.array_equal(serialized_frame, initialization.frame_parameters)
                and np.array_equal(serialized_beta, initialization.beta))
            if not warm_state_exact:
                raise RuntimeError("WARM_STATE_SERIALIZATION_MISMATCH")
        started = time.perf_counter()
        completed = subprocess.run([
            str(self.config.executable), "--fixtures", str(fixture),
            "--output", str(output), "--runtime-prepare-stage1",
            "--evaluator-backend", self.config.evaluator_backend,
        ], text=True, capture_output=True)
        (directory / "cpp_process.log").write_text(
            completed.stdout + completed.stderr, encoding="utf-8")
        if completed.returncode != 0:
            return OptimizedPersonState(
                frame_parameters=initial.frame_parameters.copy(), beta=initial.beta.copy(),
                optimized_pairs=person.pairs.copy(), point_ids=person.point_ids.copy(),
                valid=person.valid.copy(), success=False,
                status=f"CPP_EXIT_{completed.returncode}",
                cpp_preparation_success=False,
            )
        summary = _metadata(output / "multiframe_summary.txt")
        frame_summary = _metadata(output / "frame_0/fitting_summary.txt")
        optimized = json.loads(
            (output / "frame_0/optimized_densekp.json").read_text())
        all_pairs = np.asarray(optimized["pairs"], np.float32)
        if np.any(person.point_ids < 0) or np.any(person.point_ids >= len(all_pairs)):
            raise RuntimeError("optimized DenseKP IDs outside output range")
        frame = np.fromfile(
            output / "frame_0/pose_shape_frame_parameters.f32", dtype="<f4")
        beta = np.fromfile(output / "shared_physical_beta.f32", dtype="<f4")
        finite = bool(
            frame.shape == (75,) and beta.shape == (10,)
            and np.isfinite(frame).all() and np.isfinite(beta).all()
            and np.isfinite(all_pairs).all())
        (directory / "adapter_timing.json").write_text(json.dumps({
            "cpp_process_wall_ms": (time.perf_counter() - started) * 1000.0,
            "cpp_prepare_ms": float(summary["runtime_preparation_ms"]),
            "cpp_lm_ms": float(summary["total_ms"]),
        }, indent=2) + "\n")
        return OptimizedPersonState(
            frame_parameters=frame, beta=beta,
            optimized_pairs=np.ascontiguousarray(all_pairs[person.point_ids]),
            point_ids=person.point_ids.copy(), valid=person.valid.copy(),
            success=finite, status=summary["pose_shape_status"],
            final_objective=float(summary["pose_shape_final_objective"]),
            cpp_preparation_success=True,
            cpp_prepare_ms=float(summary["runtime_preparation_ms"]),
            initial_objective=float(frame_summary["stage1_initial_objective"]),
            lm_iterations=(int(frame_summary["stage1_trials"])
                           + int(frame_summary["pose_trials"])
                           + int(summary["pose_shape_trials"])),
            cpp_lm_ms=float(summary["total_ms"]),
            warm_state_exact=warm_state_exact,
            projection_extent_ratio=projection_extent_ratio(
                np.ascontiguousarray(all_pairs[person.point_ids]),
                person.pairs, person.valid),
        )


def projection_extent_ratio(optimized_pairs: np.ndarray, raw_pairs: np.ndarray,
                             valid: np.ndarray) -> float | None:
    """Dimensionless collapse diagnostic: optimized/raw spatial extent.

    Values well below 1 mean the fitted projections collapsed toward a point
    even though the LM terminated successfully (see the Phase-A translation
    semantic audit).  Diagnostic only; it never changes fitting behaviour.
    """
    valid = np.asarray(valid, bool)
    if not valid.any():
        return None
    optimized = np.asarray(optimized_pairs, np.float64)[valid]
    raw = np.asarray(raw_pairs, np.float64)[valid]
    ratios = []
    for axis in (0, 2):
        raw_span = float(np.hypot(
            np.ptp(raw[:, axis]), np.ptp(raw[:, axis + 1])))
        if raw_span <= 0.0:
            return None
        optimized_span = float(np.hypot(
            np.ptp(optimized[:, axis]), np.ptp(optimized[:, axis + 1])))
        ratios.append(optimized_span / raw_span)
    return float(sum(ratios) / len(ratios))
