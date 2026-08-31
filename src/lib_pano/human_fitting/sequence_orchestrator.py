"""Formal Phase 4.1 production sequence timeline.

Composes verified pieces without reimplementing any math:

- ``track_guided_policy``       : FULL/TRACK_GUIDED scheduling + same-frame recovery
- ``track_guided_delivery``     : guided crops, all-candidate retention, identity gate
- ``matched_person_tracks``     : temporal tracks, cold/warm decision, fitting outputs
- ``per_person_cpp``            : Python -> C++ staged-LM boundary
- ``densekp_delivery_boundary`` : detector/DenseKP seams at the ERP input boundary

The orchestrator owns one continuous timeline: frame t's fitted state stays in
memory (TrackState) and is consumed by frame t+1.  No frame state is rebuilt
from disk results.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import time
from typing import Any, Callable, Sequence

import numpy as np

from lib_pano.fitting_io.densekp_delivery_boundary import (
    build_component_detection_rois, densekp_calls_since, detector_call_count,
    matched_persons_from_full_erp, match_full_erp, run_full_detection,
    runtime_profile_delta,
)
from lib_pano.fitting_io.track_guided_delivery import (
    run_track_guided_densekp,
)
from lib_pano.fitting_io.densekp_runtime_export import (
    PersonFrameInitialState, person_initial_state_from_smplx,
)
from lib_pano.human3d.hmr2_initializer import HMR2Initializer
from lib_pano.human_fitting.hmr2_staged_init import (
    convert_hmr2_to_staged_initialization, spherical_translation_initialization,
    to_smplx_camera_initializations,
)
from lib_pano.human_fitting.matched_person_tracks import (
    MatchedPersonFittingOrchestrator, MatchedPersonTrackManager, TrackingConfig,
    TrackStatus,
)
from lib_pano.human_fitting.track_guided_policy import (
    DeliveryFrameMode, TrackGuidedScheduler, run_delivery_with_same_frame_recovery,
)


@dataclass(frozen=True)
class TrackGuidedSequenceConfig:
    """Validated Phase 4.1 production-candidate policy in one place."""

    detection_interval: int = 3            # K=3 verified sparse-delivery keyframes
    margin_ratio: float = 0.30             # guided-bbox expansion ratio (Phase 4 bake-off)
    minimum_pixel_margin: int = 32
    temporal_max_angle_deg: float = 20.0   # temporal association gating angle
    identity_margin_deg: float = 0.25      # absolute ambiguity margin
    identity_margin_ratio: float = 1.25    # relative ambiguity margin
    candidate_dedup_angle_deg: float = 1.0


class ErpHmr2ColdStartBatch:
    """Best-view HMR2 cold start over the current frame's normal-ERP images.

    ``MatchedPersonFittingOrchestrator`` invokes this once per source camera for
    every new/reinit track.  The conversion path is the established
    HMR2 -> SMPL69 axis-angle -> SMPL-X two-camera normalization; nothing here
    changes model math.
    """

    def __init__(self, initializer: HMR2Initializer,
                 image_width: int, image_height: int):
        self.hmr = initializer
        config = initializer.model_cfg
        self.focal_length = (
            float(config.EXTRA.FOCAL_LENGTH)
            / float(config.MODEL.IMAGE_SIZE) * image_width
        )
        self.image_width = image_width
        self.image_height = image_height
        self._images: tuple[np.ndarray, np.ndarray] | None = None

    def update_frame_images(self, erp_cam2: np.ndarray,
                            erp_cam4: np.ndarray) -> None:
        self._images = (erp_cam2, erp_cam4)

    def __call__(self, camera: str,
                 persons: Sequence[Any]) -> list[PersonFrameInitialState]:
        if self._images is None:
            raise RuntimeError("current frame ERP images were never provided")
        image = self._images[0 if camera == "cam2" else 1]
        boxes = np.stack([
            person.bbox_cam2 if camera == "cam2" else person.bbox_cam4
            for person in persons])
        outputs = self.hmr.infer(image, boxes)
        normalized = []
        for person, value in zip(persons, outputs):
            staged = convert_hmr2_to_staged_initialization(
                value, source_camera=camera,
                bboxes={"cam2": person.bbox_cam2, "cam4": person.bbox_cam4},
                focal_length=self.focal_length,
            )
            state = person_initial_state_from_smplx(
                *to_smplx_camera_initializations(staged))
            # Replace the pinhole-semantics HMR2 translation (u, v, depth) with
            # an observation-derived spherical translation: DenseKP centroid
            # when enough valid points exist, else the matched-person ERP bbox
            # center direction (seam-safe).  The pinhole depth component
            # collapses all vertex directions under the directional ERP
            # projection (see hmr2_staged_init docs).  Raises when neither
            # source is usable -- a pinhole translation never falls back in.
            replacement = spherical_translation_initialization(
                person.pairs[:, 0:2], person.pairs[:, 2:], person.valid,
                self.image_width, self.image_height,
                bbox_cam2=person.bbox_cam2, bbox_cam4=person.bbox_cam4,
                bbox_cam2_wraps_seam=person.bbox_cam2_wraps_seam,
                bbox_cam4_wraps_seam=person.bbox_cam4_wraps_seam)
            frame = np.array(state.frame_parameters, dtype=np.float32,
                             copy=True)
            frame[3:6], frame[9:12] = replacement
            state = replace(state, frame_parameters=frame)
            normalized.append(state)
        return normalized


@dataclass(frozen=True)
class FrameSequenceRecord:
    """One timeline entry: delivery decision + existing v1 fitting payload."""

    frame: int
    requested_mode: DeliveryFrameMode
    actual_mode: DeliveryFrameMode
    delivery_ms: float
    detector_calls: int
    densekp_calls: int
    delivery_info: dict[str, Any]
    payload: dict[str, Any]
    profiling: dict[str, float | int]


class TrackGuidedSequenceOrchestrator:
    """Owns the scheduler, tracker, HMR2 cold start and C++ fitter for a run."""

    def __init__(self, runtime: Any, hmr2_batch: ErpHmr2ColdStartBatch,
                 fit_person: Callable[..., Any],
                 valid_masks: tuple[np.ndarray, np.ndarray],
                 config: TrackGuidedSequenceConfig = TrackGuidedSequenceConfig(),
                 sequence_start_frame: int = 0):
        height, width = np.asarray(valid_masks[0]).shape[:2]
        self.runtime = runtime
        self.config = config
        self.valid_masks = valid_masks
        self.image_width = width
        self.image_height = height
        self.scheduler = TrackGuidedScheduler(
            config.detection_interval, int(sequence_start_frame))
        self.track_manager = MatchedPersonTrackManager(TrackingConfig(
            image_width=width, image_height=height,
            max_temporal_angle_deg=config.temporal_max_angle_deg))
        self.fitting = MatchedPersonFittingOrchestrator(
            self.track_manager, hmr2_batch, fit_person)
        self.hmr2_batch = hmr2_batch
        self.detection_rois = build_component_detection_rois(*valid_masks)
        self.frame_records: list[FrameSequenceRecord] = []
        self._last_full_counts: dict[str, Any] | None = None
        self._last_guided_timing: dict[str, float | int] | None = None

    def active_tracks(self) -> list[Any]:
        return [track for track in self.track_manager.tracks.values()
                if track.status == TrackStatus.ACTIVE]

    def _full_delivery(self, erp_cam2: np.ndarray, erp_cam4: np.ndarray
                       ) -> tuple[list[MatchedPerson], dict[str, Any]]:
        started = time.perf_counter()
        before = dict(self.runtime.profile_timing)
        pair = run_full_detection(
            self.runtime, erp_cam2, erp_cam4, self.detection_rois)
        inference_wall_ms = (time.perf_counter() - started) * 1000.0
        matching_started = time.perf_counter()
        match_data = match_full_erp(pair[0], pair[1], *self.valid_masks)
        matching_ms = (time.perf_counter() - matching_started) * 1000.0
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        runtime_delta = runtime_profile_delta(
            before, dict(self.runtime.profile_timing))
        detector_ms = runtime_delta.get("detector_inference_ms")
        densekp_parts = (
            "bbox_postprocess_ms", "densekp_preprocess_ms",
            "densekp_inference_ms", "densekp_postprocess_ms",
        )
        densekp_ms = (
            sum(runtime_delta[name] for name in densekp_parts)
            if all(name in runtime_delta for name in densekp_parts) else None
        )
        counts = {
            "detector_calls": detector_call_count(self.detection_rois),
            "densekp_calls": densekp_calls_since(self.runtime, before),
            "inference_ms": elapsed_ms,
            "full_inference_wall_ms": inference_wall_ms,
            "full_matching_ms": matching_ms,
            "full_detector_ms": detector_ms,
            "full_densekp_ms": densekp_ms,
            "full_runtime_profile_delta": runtime_delta,
            "cam2_person_count": len(pair[0].persons),
            "cam4_person_count": len(pair[1].persons),
            "matched_person_count": len(match_data.get("matches", [])),
            "matched_bboxes": [
                {"bbox_cam2": [float(v) for v in match["bbox_cam2"]],
                 "bbox_cam4": [float(v) for v in match["bbox_cam4"]]}
                for match in match_data.get("matches", [])
            ],
        }
        self._last_full_counts = counts
        return matched_persons_from_full_erp(match_data, self.image_width,
                                             self.image_height), counts

    def _guided_delivery(self, erp_cam2: np.ndarray, erp_cam4: np.ndarray,
                         tracks: Sequence[Any]):
        started = time.perf_counter()
        observations, timing = run_track_guided_densekp(
            self.runtime, erp_cam2, erp_cam4, tracks, self.valid_masks,
            margin_ratio=self.config.margin_ratio,
            minimum_pixel_margin=self.config.minimum_pixel_margin,
            temporal_max_angle_deg=self.config.temporal_max_angle_deg,
            identity_margin_deg=self.config.identity_margin_deg,
            identity_margin_ratio=self.config.identity_margin_ratio,
            candidate_dedup_angle_deg=self.config.candidate_dedup_angle_deg,
        )
        timing["wall_ms"] = (time.perf_counter() - started) * 1000.0
        self._last_guided_timing = timing
        return observations, [
            observation for observation in observations if not observation.success
        ], {key: timing[key] for key in ("detector_calls", "densekp_calls")}

    def process_frame(self, frame: int, erp_cam2: np.ndarray,
                      erp_cam4: np.ndarray) -> FrameSequenceRecord:
        """Advance one true timeline step for one frame."""
        frame_started = time.perf_counter()
        active = self.active_tracks()
        requested = self.scheduler.mode(frame, len(active))
        self.hmr2_batch.update_frame_images(erp_cam2, erp_cam4)
        self._last_guided_timing = None

        started = time.perf_counter()
        decision = run_delivery_with_same_frame_recovery(
            requested, active,
            full_delivery=lambda: self._full_delivery(erp_cam2, erp_cam4),
            guided_delivery=lambda tracks: self._guided_delivery(
                erp_cam2, erp_cam4, tracks),
        )
        delivery_ms = (time.perf_counter() - started) * 1000.0

        delivery_info: dict[str, Any] = {}
        if decision.actual_mode == DeliveryFrameMode.TRACK_GUIDED:
            successes = [row for row in decision.observations if row.success]
            persons = [row.person for row in successes]
            bound = [(row.track_id, index) for index, row in enumerate(successes)]
            delivery_info["identity"] = [
                {"track_id": row.track_id, **(row.identity or {})}
                for row in decision.observations
            ]
            delivery_info["guided_crops"] = [
                {
                    "track_id": row.track_id,
                    "camera": spec.camera,
                    "predicted_bbox_erp": spec.predicted_bbox_erp,
                    "status": ("accepted" if row.success
                               else (row.failure_reason.value
                                     if row.failure_reason else "unknown")),
                }
                for row in decision.observations
                for spec in (row.crop_cam2, row.crop_cam4) if spec is not None
            ]
            guided = self._last_guided_timing or {}
            delivery_info["guided_timing"] = {
                "guided_bbox_projection_ms": guided.get("bbox_projection_ms"),
                "guided_crop_ms": guided.get("guided_crop_ms"),
                "guided_densekp_ms": guided.get("guided_densekp_ms"),
                "guided_validation_ms": guided.get("guided_validation_ms"),
                "guided_identity_ms": guided.get("identity_gate_ms"),
                "guided_wall_ms": guided.get("wall_ms"),
            }
        else:
            persons = list(decision.observations)
            bound = None
            delivery_info.update(self._last_full_counts or {})
            if decision.actual_mode == DeliveryFrameMode.RECOVERY:
                guided = self._last_guided_timing or {}
                delivery_info["guided_timing_before_recovery"] = {
                    "guided_bbox_projection_ms": guided.get("bbox_projection_ms"),
                    "guided_crop_ms": guided.get("guided_crop_ms"),
                    "guided_densekp_ms": guided.get("guided_densekp_ms"),
                    "guided_validation_ms": guided.get("guided_validation_ms"),
                    "guided_identity_ms": guided.get("identity_gate_ms"),
                    "guided_wall_ms": guided.get("wall_ms"),
                }
            self._last_full_counts = None

        measured_components = [
            delivery_info.get("full_detector_ms"),
            delivery_info.get("full_densekp_ms"),
            delivery_info.get("full_matching_ms"),
        ]
        if (decision.actual_mode == DeliveryFrameMode.FULL_DETECTION
                and all(value is not None for value in measured_components)):
            delivery_info["full_other_ms"] = (
                delivery_ms - sum(float(value)
                                  for value in measured_components))
        guided_timing = delivery_info.get("guided_timing")
        if guided_timing is not None:
            disjoint = [
                guided_timing.get("guided_bbox_projection_ms"),
                guided_timing.get("guided_crop_ms"),
                guided_timing.get("guided_densekp_ms"),
                guided_timing.get("guided_validation_ms"),
            ]
            if all(value is not None for value in disjoint):
                guided_timing["guided_residual_ms"] = (
                    delivery_ms - sum(float(value) for value in disjoint))

        result = self.fitting.process_frame(frame, persons, bound=bound)
        record = FrameSequenceRecord(
            frame=int(frame),
            requested_mode=requested,
            actual_mode=decision.actual_mode,
            delivery_ms=delivery_ms,
            detector_calls=decision.detector_calls,
            densekp_calls=decision.densekp_calls,
            delivery_info=delivery_info,
            payload=result.payload,
            profiling=result.profiling,
        )
        result.profiling["frame_wall_ms"] = (
            time.perf_counter() - frame_started) * 1000.0
        self.frame_records.append(record)
        return record

    def sequence_result(self, metadata: dict[str, Any]) -> dict[str, Any]:
        """Summarize the run, preserving Phase 2.2 invariants and v1 rows."""
        counters = self.fitting.counters
        rows = [{**{"frame": record.frame}, **person}
                for record in self.frame_records
                for person in record.payload["persons"]]
        new_track_count = sum(row["initialization"]["type"] == "hmr2"
                              for row in rows)
        lm_successes = sum(bool(row["fitting"]["success"]) for row in rows)
        warm_rows = [row for row in rows
                     if row["initialization"]["type"] == "previous_frame"]
        warm_state_exact = all(
            row["cpp_preparation"]["warm_state_exact"] is True for row in warm_rows)
        if counters.hmr2_calls > new_track_count + counters.reinit_count:
            raise RuntimeError("MATCHED_ONLY_HMR2_CALL_INVARIANT_VIOLATION")
        if warm_rows and not warm_state_exact:
            raise RuntimeError("WARM_STATE_EXACT_PROPAGATION_FAILED")

        mode_counts = {mode.name.lower(): sum(
            record.actual_mode == mode for record in self.frame_records)
            for mode in DeliveryFrameMode}
        frames = []
        for record in self.frame_records:
            frames.append({
                "frame": record.frame,
                "requested_mode": record.requested_mode.value,
                "mode": record.actual_mode.value,
                "delivery_ms": record.delivery_ms,
                "frame_wall_ms": record.profiling.get("frame_wall_ms"),
                "detector_calls": record.detector_calls,
                "densekp_calls": record.densekp_calls,
                "delivery": record.delivery_info,
                "profiling": record.profiling,
                "persons": record.payload["persons"],
            })
        return {
            "schema_version": "densekp_track_guided_sequence_v1",
            "num_frames": len(self.frame_records),
            "frame_modes": mode_counts,
            "matched_person_observations": counters.matched_person_observations,
            "unique_tracks": len(self.track_manager.tracks),
            "new_track_count": new_track_count,
            "lost_event_count": counters.lost_event_count,
            "reconnect_count": counters.reconnect_count,
            "reinit_count": counters.reinit_count,
            "warm_start_count": counters.warm_start_count,
            "hmr2_call_count": counters.hmr2_calls,
            "hmr2_person_count": counters.hmr2_person_count,
            "lm_attempts": counters.lm_attempt_count,
            "lm_successes": lm_successes,
            "lm_failures": len(rows) - lm_successes,
            "warm_state_exact": warm_state_exact,
            "frames": frames,
            **metadata,
        }
