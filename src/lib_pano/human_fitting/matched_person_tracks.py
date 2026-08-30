"""Matched-person tracking and cold-start orchestration for staged LM.

This module deliberately knows nothing about detector confidences and does not
change the staged-LM solver.  Its input is the matched ``persons[]`` portion of
``densekp_pairs_v2``; unmatched detections therefore cannot enter HMR2 or LM.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Mapping, Sequence
import time

import numpy as np


class TrackStatus(str, Enum):
    ACTIVE = "active"
    LOST = "lost"


@dataclass(frozen=True)
class MatchedPerson:
    person_id: int
    bbox_cam2: np.ndarray
    bbox_cam4: np.ndarray
    pairs: np.ndarray
    point_ids: np.ndarray
    valid: np.ndarray
    bbox_cam2_wraps_seam: bool = False
    bbox_cam4_wraps_seam: bool = False

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "MatchedPerson":
        pairs = np.asarray(payload["pairs"], dtype=np.float32)
        point_ids = np.asarray(
            payload.get("point_ids", np.arange(len(pairs))), dtype=np.int32,
        )
        valid = np.asarray(payload.get("valid", np.ones(len(pairs))), dtype=bool)
        if pairs.ndim != 2 or pairs.shape[1] != 4:
            raise ValueError(f"person pairs must be [N,4], got {pairs.shape}")
        if point_ids.shape != (len(pairs),) or valid.shape != (len(pairs),):
            raise ValueError("person point_ids/valid length mismatch")
        return cls(
            person_id=int(payload["person_id"]),
            bbox_cam2=np.asarray(payload["bbox_cam2"], dtype=np.float32),
            bbox_cam4=np.asarray(payload["bbox_cam4"], dtype=np.float32),
            pairs=pairs,
            point_ids=point_ids,
            valid=valid,
            bbox_cam2_wraps_seam=bool(payload.get("bbox_cam2_wraps_seam", False)),
            bbox_cam4_wraps_seam=bool(payload.get("bbox_cam4_wraps_seam", False)),
        )


@dataclass(frozen=True)
class OptimizedPersonState:
    """The actual previous C++ result used for warm start, not raw HMR2."""

    frame_parameters: np.ndarray  # [75]
    beta: np.ndarray  # [10], physical beta
    optimized_pairs: np.ndarray  # [N,4], ERP_ROTATE
    point_ids: np.ndarray
    valid: np.ndarray
    success: bool
    status: str = "unknown"
    final_objective: float | None = None
    cpp_preparation_success: bool = True
    cpp_prepare_ms: float | None = None
    initial_objective: float | None = None
    lm_iterations: int | None = None
    cpp_lm_ms: float | None = None
    warm_state_exact: bool | None = None

    def __post_init__(self) -> None:
        if np.asarray(self.frame_parameters).shape != (75,):
            raise ValueError("frame_parameters must have shape [75]")
        if np.asarray(self.beta).shape != (10,):
            raise ValueError("beta must have shape [10]")


@dataclass
class TrackState:
    track_id: int
    last_frame: int
    optimized: OptimizedPersonState | None
    last_person: MatchedPerson
    status: TrackStatus = TrackStatus.ACTIVE
    consecutive_lost: int = 0
    initialization_source: str = "hmr2"
    created_frame: int = 0
    last_guided_bbox_cam2: tuple[float, float, float, float] | None = None
    last_guided_bbox_cam4: tuple[float, float, float, float] | None = None
    last_full_detection_frame: int | None = None
    last_recovery_frame: int | None = None
    guided_success_count: int = 0
    guided_failure_count: int = 0


@dataclass(frozen=True)
class TrackAssignment:
    person: MatchedPerson
    track_id: int
    initialization_source: str
    previous: OptimizedPersonState | None
    previous_frame: int | None = None
    is_reinit: bool = False


@dataclass(frozen=True)
class BestView:
    camera: str
    bbox_xyxy: np.ndarray
    quality: tuple[int, int, float, int]


@dataclass(frozen=True)
class TrackingConfig:
    image_width: int = 5760
    image_height: int = 2880
    max_reconnect_gap: int = 2
    max_temporal_angle_deg: float = 20.0
    boundary_margin_px: float = 2.0
    bootstrap_hmr_frames: int = 1


def _view_quality(person: MatchedPerson, camera: str,
                  config: TrackingConfig) -> BestView:
    if camera == "cam2":
        bbox, points, wraps = (
            person.bbox_cam2, person.pairs[:, :2], person.bbox_cam2_wraps_seam,
        )
        deterministic_tie_break = 1
    elif camera == "cam4":
        bbox, points, wraps = (
            person.bbox_cam4, person.pairs[:, 2:], person.bbox_cam4_wraps_seam,
        )
        deterministic_tie_break = 0
    else:
        raise ValueError(f"unsupported camera: {camera}")
    margin = config.boundary_margin_px
    truncated = bool(
        wraps or bbox[0] <= margin or bbox[1] <= margin
        or bbox[2] >= config.image_width - margin
        or bbox[3] >= config.image_height - margin
    )
    finite = np.isfinite(points).all(axis=1)
    in_bounds = (
        (points[:, 0] >= 0) & (points[:, 0] < config.image_width)
        & (points[:, 1] >= 0) & (points[:, 1] < config.image_height)
    )
    strict_valid = int(np.count_nonzero(person.valid & finite & in_bounds))
    area = float(max(0.0, bbox[2] - bbox[0]) * max(0.0, bbox[3] - bbox[1]))
    return BestView(
        camera, bbox.copy(),
        (int(not truncated), strict_valid, area, deterministic_tie_break),
    )


def choose_best_view(person: MatchedPerson,
                     config: TrackingConfig = TrackingConfig()) -> BestView:
    """Choose one view using geometry only; cam2 is the deterministic tie-break."""
    return max(
        (_view_quality(person, "cam2", config),
         _view_quality(person, "cam4", config)),
        key=lambda item: item.quality,
    )


def _erp_rays(points: np.ndarray, width: int, height: int) -> np.ndarray:
    longitude = points[:, 0] / width * (2.0 * np.pi) - np.pi
    latitude = points[:, 1] / height * np.pi - np.pi / 2.0
    cos_latitude = np.cos(latitude)
    return np.stack((
        cos_latitude * np.sin(longitude), np.sin(latitude),
        cos_latitude * np.cos(longitude),
    ), axis=1)


def temporal_person_cost_deg(previous: TrackState, current: MatchedPerson,
                             config: TrackingConfig) -> float:
    reference = (
        previous.optimized.optimized_pairs
        if previous.optimized is not None else previous.last_person.pairs
    )
    reference_ids = (
        previous.optimized.point_ids
        if previous.optimized is not None else previous.last_person.point_ids
    )
    reference_valid = (
        previous.optimized.valid
        if previous.optimized is not None else previous.last_person.valid
    )
    lookup = {int(value): index for index, value in enumerate(reference_ids)}
    rows = [(row, lookup[int(point_id)]) for row, point_id in enumerate(current.point_ids)
            if int(point_id) in lookup]
    if not rows:
        return float("inf")
    current_rows = np.asarray([row[0] for row in rows], dtype=np.int64)
    reference_rows = np.asarray([row[1] for row in rows], dtype=np.int64)
    valid = current.valid[current_rows] & reference_valid[reference_rows]
    current_points = current.pairs[current_rows][valid]
    reference_points = reference[reference_rows][valid]
    finite = np.isfinite(current_points).all(axis=1) & np.isfinite(reference_points).all(axis=1)
    current_points, reference_points = current_points[finite], reference_points[finite]
    if len(current_points) < 4:
        return float("inf")
    errors = []
    for offset in (0, 2):
        rays_a = _erp_rays(current_points[:, offset:offset + 2], config.image_width,
                           config.image_height)
        rays_b = _erp_rays(reference_points[:, offset:offset + 2], config.image_width,
                           config.image_height)
        errors.append(np.degrees(np.arccos(np.clip(np.sum(rays_a * rays_b, axis=1), -1, 1))))
    return float(np.median(np.concatenate(errors)))


def optimal_cardinality_assignment(costs: np.ndarray) -> list[tuple[int, int]]:
    """Maximise finite-pair cardinality, then minimise total assignment cost."""
    best_key: tuple[Any, ...] | None = None
    best: list[tuple[int, int]] = []

    def visit(row: int, used: set[int], pairs: list[tuple[int, int]], total: float) -> None:
        nonlocal best_key, best
        if row == costs.shape[0]:
            key = (-len(pairs), total, tuple(pairs))
            if best_key is None or key < best_key:
                best_key, best = key, list(pairs)
            return
        visit(row + 1, used, pairs, total)
        for column in range(costs.shape[1]):
            value = float(costs[row, column])
            if column in used or not np.isfinite(value):
                continue
            used.add(column); pairs.append((row, column))
            visit(row + 1, used, pairs, total + value)
            pairs.pop(); used.remove(column)

    visit(0, set(), [], 0.0)
    return best


# Backwards-compatible private alias for older callers/tests.
_optimal_assignment = optimal_cardinality_assignment


class MatchedPersonTrackManager:
    def __init__(self, config: TrackingConfig = TrackingConfig()):
        self.config = config
        self.tracks: dict[int, TrackState] = {}
        self._next_track_id = 0

    def associate(self, frame: int,
                  persons: Sequence[MatchedPerson], *,
                  bound: Sequence[tuple[int, int]] | None = None
                  ) -> list[TrackAssignment]:
        """Associate persons with tracks.

        With ``bound=None`` the association is computed from temporal angular
        costs.  Alternatively ``bound`` supplies identity-resolved
        ``(track_id, person_index)`` pairs from a caller that already ran the
        Phase 4.1 guided identity gate; the gate's global one-to-one decision
        is then trusted verbatim instead of being recomputed.  Unbound book-
        keeping (lost marking, initialization source, reinit flags) is shared.
        """
        if bound is None:
            candidates = [track for track in self.tracks.values()
                          if frame - track.last_frame <= self.config.max_reconnect_gap + 1]
            costs = np.full((len(candidates), len(persons)), np.inf, dtype=np.float64)
            for row, track in enumerate(candidates):
                for column, person in enumerate(persons):
                    value = temporal_person_cost_deg(track, person, self.config)
                    if value <= self.config.max_temporal_angle_deg:
                        costs[row, column] = value
            matched = optimal_cardinality_assignment(costs)
        else:
            unknown = sorted({track_id for track_id, _ in bound
                              if track_id not in self.tracks})
            if unknown:
                raise ValueError(f"bound references unknown tracks: {unknown}")
            indexes = sorted(person_index for _, person_index in bound)
            if indexes != list(range(len(persons))):
                raise ValueError(
                    "bound must resolve every current person exactly once")
            candidates = [self.tracks[track_id] for track_id, _ in bound]
            matched = [(row, person_index)
                       for row, (_, person_index) in enumerate(bound)]
        by_person = {column: candidates[row] for row, column in matched}
        assignments = []
        for index, person in enumerate(persons):
            track = by_person.get(index)
            if track is None:
                track = TrackState(
                    track_id=self._next_track_id, last_frame=frame,
                    optimized=None, last_person=person, created_frame=frame,
                )
                self.tracks[track.track_id] = track
                self._next_track_id += 1
                source = "hmr2"
            else:
                source = "previous_frame" if track.optimized and track.optimized.success else "hmr2"
                track.status = TrackStatus.ACTIVE
                track.consecutive_lost = 0
            is_reinit = bool(track.optimized is not None and source == "hmr2")
            assignments.append(TrackAssignment(
                person, track.track_id, source,
                track.optimized if source == "previous_frame" else None,
                track.last_frame if source == "previous_frame" else None,
                is_reinit,
            ))
        matched_ids = {assignment.track_id for assignment in assignments}
        for track in self.tracks.values():
            if track.track_id not in matched_ids and track.last_frame < frame:
                track.status = TrackStatus.LOST
                track.consecutive_lost = frame - track.last_frame
        return assignments

    def update(self, frame: int, assignment: TrackAssignment,
               result: OptimizedPersonState) -> None:
        track = self.tracks[assignment.track_id]
        track.last_frame = frame
        track.last_person = assignment.person
        track.optimized = result
        track.status = TrackStatus.ACTIVE if result.success else TrackStatus.LOST
        track.consecutive_lost = 0 if result.success else 1
        track.initialization_source = assignment.initialization_source


@dataclass
class SequenceCounters:
    frames: int = 0
    matched_person_observations: int = 0
    hmr2_calls: int = 0
    hmr2_person_count: int = 0
    warm_start_count: int = 0
    reinit_count: int = 0
    hmr2_ms: float = 0.0
    lost_event_count: int = 0
    reconnect_count: int = 0
    lm_attempt_count: int = 0
    lm_success_count: int = 0
    lm_failure_count: int = 0


@dataclass(frozen=True)
class FrameOrchestrationResult:
    payload: dict[str, Any]
    fitted_states: tuple[OptimizedPersonState, ...]
    profiling: dict[str, float | int]


class MatchedPersonFittingOrchestrator:
    """Policy layer with injectable HMR2 conversion and existing C++ runner.

    ``hmr2_batch`` is called once per selected source camera and must preserve
    request order. ``fit_person`` is deliberately per-person; it may launch the
    existing T=1 C++ executable without changing any C++ mathematics.
    """

    def __init__(
        self,
        track_manager: MatchedPersonTrackManager,
        hmr2_batch: Callable[[str, Sequence[MatchedPerson]], Sequence[Any]],
        fit_person: Callable[[MatchedPerson, Any, int], OptimizedPersonState],
    ):
        self.track_manager = track_manager
        self.hmr2_batch = hmr2_batch
        self.fit_person = fit_person
        self.counters = SequenceCounters()

    def process_frame(self, frame: int,
                      persons: Sequence[MatchedPerson], *,
                      bound: Sequence[tuple[int, int]] | None = None
                      ) -> FrameOrchestrationResult:
        active_before = {
            track_id for track_id, track in self.track_manager.tracks.items()
            if track.status == TrackStatus.ACTIVE
        }
        association_started = time.perf_counter()
        assignments = self.track_manager.associate(frame, persons, bound=bound)
        association_ms = (time.perf_counter() - association_started) * 1000.0
        cold_by_camera: dict[str, list[TrackAssignment]] = {"cam2": [], "cam4": []}
        views: dict[int, BestView] = {}
        for assignment in assignments:
            if assignment.initialization_source == "hmr2":
                view = choose_best_view(assignment.person, self.track_manager.config)
                views[assignment.track_id] = view
                cold_by_camera[view.camera].append(assignment)

        initializers: dict[int, Any] = {
            assignment.track_id: assignment.previous
            for assignment in assignments if assignment.previous is not None
        }
        hmr_started = time.perf_counter()
        hmr_call_count = 0
        hmr_person_count = 0
        for camera, cold in cold_by_camera.items():
            if not cold:
                continue
            hmr_call_count += 1
            hmr_person_count += len(cold)
            converted = list(self.hmr2_batch(camera, [item.person for item in cold]))
            if len(converted) != len(cold):
                raise RuntimeError(
                    f"HMR2 batch output mismatch camera={camera}: "
                    f"expected={len(cold)} actual={len(converted)}"
                )
            for assignment, initialization in zip(cold, converted):
                initializers[assignment.track_id] = initialization
        hmr_ms = (time.perf_counter() - hmr_started) * 1000.0

        rows: list[dict[str, Any]] = []
        results: list[OptimizedPersonState] = []
        fitting_started = time.perf_counter()
        for assignment in assignments:
            initialization = initializers[assignment.track_id]
            fitted = self.fit_person(
                assignment.person, initialization, assignment.track_id,
            )
            lm_attempts = 1
            reinitialized = False
            if (not fitted.success
                    and assignment.initialization_source == "previous_frame"):
                # Retry in the same frame with a fresh HMR2 state.  The C++
                # adapter replaces the track's frozen state and forces Stage1
                # runtime preparation again.
                view = choose_best_view(
                    assignment.person, self.track_manager.config)
                retry_started = time.perf_counter()
                retry_values = list(self.hmr2_batch(
                    view.camera, [assignment.person]))
                retry_ms = (time.perf_counter() - retry_started) * 1000.0
                if len(retry_values) != 1:
                    raise RuntimeError("HMR2 reinit must return exactly one state")
                fitted = self.fit_person(
                    assignment.person, retry_values[0], assignment.track_id)
                lm_attempts += 1
                reinitialized = True
                hmr_call_count += 1
                hmr_person_count += 1
                hmr_ms += retry_ms
                self.counters.reinit_count += 1
            self.track_manager.update(frame, assignment, fitted)
            results.append(fitted)
            init_type = (
                "hmr2_reinit"
                if reinitialized or assignment.is_reinit
                else assignment.initialization_source
            )
            init = {"type": init_type}
            if init_type in {"hmr2", "hmr2_reinit"}:
                if reinitialized:
                    views[assignment.track_id] = view
                init["source_camera"] = views[assignment.track_id].camera
            else:
                init.update({
                    "source_frame": assignment.previous_frame,
                    "track_id": assignment.track_id,
                })
            rows.append({
                "person_id": assignment.person.person_id,
                "track_id": assignment.track_id,
                "initialization": init,
                "point_ids": assignment.person.point_ids.astype(int).tolist(),
                "raw_pairs": assignment.person.pairs.astype(float).tolist(),
                "optimized_pairs": fitted.optimized_pairs.astype(float).tolist(),
                "valid": fitted.valid.astype(np.uint8).tolist(),
                "fitting": {
                    "success": bool(fitted.success), "status": fitted.status,
                    "initial_objective": fitted.initial_objective,
                    "final_objective": fitted.final_objective,
                    "iterations": fitted.lm_iterations,
                    "lm_ms": fitted.cpp_lm_ms,
                    "attempts": lm_attempts,
                },
                "cpp_preparation": {
                    "success": bool(fitted.cpp_preparation_success),
                    "prepare_ms": fitted.cpp_prepare_ms,
                    "warm_state_exact": fitted.warm_state_exact,
                },
            })
        fitting_ms = (time.perf_counter() - fitting_started) * 1000.0
        self.counters.frames += 1
        self.counters.matched_person_observations += len(persons)
        self.counters.hmr2_calls += hmr_call_count
        self.counters.hmr2_person_count += hmr_person_count
        self.counters.warm_start_count += sum(
            item.initialization_source == "previous_frame"
            for item in assignments)
        self.counters.reinit_count += sum(
            item.is_reinit for item in assignments)
        self.counters.hmr2_ms += hmr_ms
        self.counters.lm_attempt_count += sum(
            int(row["fitting"]["attempts"]) for row in rows)
        self.counters.lm_success_count += sum(
            int(row["fitting"]["success"]) for row in rows)
        self.counters.lm_failure_count += sum(
            int(not row["fitting"]["success"]) for row in rows)
        active_after_association = {item.track_id for item in assignments}
        self.counters.lost_event_count += len(
            active_before - active_after_association)
        reconnects = sum(
            int(item.previous_frame is not None and item.previous_frame < frame - 1)
            for item in assignments)
        self.counters.reconnect_count += reconnects
        return FrameOrchestrationResult(
            payload=build_fitting_output(frame, rows),
            fitted_states=tuple(results),
            profiling={
                "temporal_association_ms": association_ms,
                "hmr2_calls": hmr_call_count,
                "hmr2_person_count": hmr_person_count,
                "hmr2_ms": hmr_ms,
                "per_person_fitting_ms": fitting_ms,
                "matched_person_count": len(persons),
                "reconnect_count": reconnects,
                "reinit_count": int(sum(
                    row["initialization"]["type"] == "hmr2_reinit"
                    for row in rows)),
            },
        )


def build_fitting_output(frame: int, rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "schema_version": "densekp_multi_person_fitting_v1",
        "frame": int(frame),
        "persons": [dict(row) for row in rows],
    }
