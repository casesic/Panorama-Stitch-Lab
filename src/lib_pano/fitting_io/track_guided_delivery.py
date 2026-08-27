"""Detector-free DenseKP delivery driven by the previous fitted track state."""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import replace
from enum import Enum
import time
from typing import Any, Sequence

import numpy as np


class GuidedFailureReason(str, Enum):
    INVALID_PROJECTION = "INVALID_PROJECTION"
    EMPTY_GUIDED_CROP = "EMPTY_GUIDED_CROP"
    TOO_FEW_DENSEKP = "TOO_FEW_DENSEKP"
    CROSS_CAMERA_GEOMETRY_FAIL = "CROSS_CAMERA_GEOMETRY_FAIL"
    TEMPORAL_GEOMETRY_FAIL = "TEMPORAL_GEOMETRY_FAIL"
    LM_FAIL = "LM_FAIL"
    TRACK_NOT_FOUND_ON_RECOVERY = "TRACK_NOT_FOUND_ON_RECOVERY"
    DUPLICATE_CURRENT_CANDIDATE = "DUPLICATE_CURRENT_CANDIDATE"
    TRACK_ASSIGNMENT_AMBIGUOUS = "TRACK_ASSIGNMENT_AMBIGUOUS"
    CANDIDATE_ASSIGNMENT_AMBIGUOUS = "CANDIDATE_ASSIGNMENT_AMBIGUOUS"
    NO_UNIQUE_ASSIGNMENT = "NO_UNIQUE_ASSIGNMENT"


@dataclass(frozen=True)
class GuidedCropSpec:
    track_id: int
    camera: str
    roi: Any
    predicted_bbox_erp: tuple[float, float, float, float]
    densekp_bbox_local: tuple[float, float, float, float]


@dataclass
class GuidedTrackObservation:
    track_id: int
    person: Any | None
    crop_cam2: GuidedCropSpec | None
    crop_cam4: GuidedCropSpec | None
    valid_count: int = 0
    median_angular_error_deg: float | None = None
    temporal_error_deg: float | None = None
    failure_reason: GuidedFailureReason | None = None
    identity: dict[str, Any] | None = None

    @property
    def success(self) -> bool:
        return self.person is not None and self.failure_reason is None


def _minimum_circular_interval(values: np.ndarray, width: int) -> tuple[float, float]:
    values = np.sort(np.mod(np.asarray(values, np.float64), float(width)))
    if not len(values):
        raise ValueError("empty circular coordinate set")
    if len(values) == 1:
        return float(values[0]), 1.0
    wrapped = np.r_[values, values[0] + width]
    gap_index = int(np.argmax(np.diff(wrapped)))
    start = float(values[(gap_index + 1) % len(values)])
    end = float(values[gap_index] + (width if gap_index < len(values) - 1 else 0))
    if end < start:
        end += width
    return start % width, max(1.0, end - start)


def build_guided_crop_spec(
    track_id: int,
    camera: str,
    points_erp: np.ndarray,
    valid: np.ndarray,
    width: int,
    height: int,
    margin_ratio: float,
    minimum_pixel_margin: int,
) -> GuidedCropSpec:
    """Create one seam-aware expanded bbox from projected body keypoints."""
    from densekp_delivery.overlap_roi import ERPDetectionROI

    points = np.asarray(points_erp, np.float32)
    keep = np.asarray(valid, bool) & np.isfinite(points).all(axis=1)
    points = points[keep]
    if len(points) < 4:
        raise ValueError(GuidedFailureReason.INVALID_PROJECTION.value)
    start, circular_width = _minimum_circular_interval(points[:, 0], width)
    y0, y1 = float(points[:, 1].min()), float(points[:, 1].max())
    body_height = max(1.0, y1 - y0)
    margin_x = max(float(minimum_pixel_margin), circular_width * margin_ratio)
    margin_y = max(float(minimum_pixel_margin), body_height * margin_ratio)
    expanded_width = min(width, int(np.ceil(circular_width + 2.0 * margin_x)))
    x_start = int(np.floor(start - margin_x)) % width
    expanded_y0 = max(0, int(np.floor(y0 - margin_y)))
    expanded_y1 = min(height, int(np.ceil(y1 + margin_y)))
    if expanded_width <= 1 or expanded_y1 - expanded_y0 <= 1:
        raise ValueError(GuidedFailureReason.EMPTY_GUIDED_CROP.value)
    roi = ERPDetectionROI(
        x_start=x_start, width=expanded_width,
        y_start=expanded_y0, height=expanded_y1 - expanded_y0,
        erp_width=width, erp_height=height,
    )
    return GuidedCropSpec(
        track_id=track_id, camera=camera, roi=roi,
        predicted_bbox_erp=(float(x_start), float(expanded_y0),
                            float((x_start + expanded_width) % width),
                            float(expanded_y1)),
        densekp_bbox_local=(0.0, 0.0, float(expanded_width),
                            float(expanded_y1 - expanded_y0)),
    )


def run_track_guided_densekp(
    runtime: Any,
    erp_cam2: np.ndarray,
    erp_cam4: np.ndarray,
    tracks: Sequence[Any],
    masks: tuple[np.ndarray, np.ndarray],
    *,
    margin_ratio: float = 0.30,
    minimum_pixel_margin: int = 32,
    temporal_max_angle_deg: float = 20.0,
    identity_margin_deg: float = 0.25,
    identity_margin_ratio: float = 1.25,
    candidate_dedup_angle_deg: float = 1.0,
    return_identity_result: bool = False,
) -> tuple[list[GuidedTrackObservation], dict[str, float | int]]:
    """Run B1 DenseKP only for established active matched tracks."""
    from densekp_delivery.camera.erp_rotate import (
        DEFAULT_ERP_TO_ERP_ROTATE_R, rotate_erp_points,
    )
    from densekp_delivery.overlap_roi import crop_erp_roi, restore_result_to_full_erp
    from densekp_delivery.person_match import match_all_persons
    from lib_pano.human_fitting.guided_identity import (
        CandidateDedupConfig, GuidedIdentityConfig, GuidedIdentityReason,
        GuidedPersonCandidate,
        resolve_guided_identities,
    )
    from lib_pano.human_fitting.matched_person_tracks import (
        MatchedPerson, TrackingConfig,
    )

    height, width = erp_cam2.shape[:2]
    timing: dict[str, float | int] = {
        "bbox_projection_ms": 0.0, "guided_crop_ms": 0.0,
        "guided_densekp_ms": 0.0, "guided_validation_ms": 0.0,
        "identity_gate_ms": 0.0,
        "detector_calls": 0, "densekp_calls": 0,
    }
    observations = {track.track_id: GuidedTrackObservation(
        track.track_id, None, None, None) for track in tracks}
    items = []
    item_specs = []
    started = time.perf_counter()
    raw_candidates = []
    raw_candidate_id = 0
    for track in tracks:
        optimized = track.optimized
        if optimized is None or not optimized.success:
            observations[track.track_id].failure_reason = (
                GuidedFailureReason.INVALID_PROJECTION)
            continue
        # Current C++ output exposes fitted DenseKP projections, but those can
        # collapse to a tiny extent and are not a safe body-envelope proxy.
        # Use the previous *matched observation* geometry retained by this
        # established track until final support/body extent is exported.
        pairs = np.asarray(track.last_person.pairs, np.float32)
        valid = np.asarray(track.last_person.valid, bool)
        for camera, offset, image in (
            ("cam2", 0, erp_cam2), ("cam4", 2, erp_cam4),
        ):
            # C++ output is ERP_ROTATE; detector input remains normal ERP.
            points_erp = rotate_erp_points(
                pairs[:, offset:offset + 2], width, height,
                R=DEFAULT_ERP_TO_ERP_ROTATE_R.T,
            )
            try:
                spec = build_guided_crop_spec(
                    track.track_id, camera, points_erp, valid, width, height,
                    margin_ratio, minimum_pixel_margin)
            except ValueError:
                observations[track.track_id].failure_reason = (
                    GuidedFailureReason.INVALID_PROJECTION)
                continue
            if camera == "cam2":
                observations[track.track_id].crop_cam2 = spec
            else:
                observations[track.track_id].crop_cam4 = spec
            item_specs.append(spec)
            items.append((image, spec))
    timing["bbox_projection_ms"] = (time.perf_counter() - started) * 1000.0

    started = time.perf_counter()
    runner_items = []
    for image, spec in items:
        crop = crop_erp_roi(image, spec.roi)
        runner_items.append((
            crop, f"track_{spec.track_id}_{spec.camera}", spec.camera,
            [list(spec.densekp_bbox_local)],
        ))
    timing["guided_crop_ms"] = (time.perf_counter() - started) * 1000.0
    started = time.perf_counter()
    raw_results = runtime.run_bboxes(runner_items, densekp_batch_size=1)
    timing["guided_densekp_ms"] = (time.perf_counter() - started) * 1000.0
    timing["densekp_calls"] = len(runner_items)

    restored: dict[tuple[int, str], Any] = {}
    for result, spec in zip(raw_results, item_specs):
        restored[(spec.track_id, spec.camera)] = restore_result_to_full_erp(
            result, spec.roi)
    started = time.perf_counter()
    for track in tracks:
        row = observations[track.track_id]
        if row.failure_reason is not None:
            continue
        cam2 = restored.get((track.track_id, "cam2"))
        cam4 = restored.get((track.track_id, "cam4"))
        if cam2 is None or cam4 is None or not cam2.persons or not cam4.persons:
            row.failure_reason = GuidedFailureReason.TOO_FEW_DENSEKP
            continue
        matched = match_all_persons(
            cam2, cam4, erp_mask_cam2=masks[0], erp_mask_cam4=masks[1])
        if not matched["matches"]:
            row.failure_reason = GuidedFailureReason.CROSS_CAMERA_GEOMETRY_FAIL
            continue
        # Retain every cross-camera-valid person from this guided crop.  The
        # source track is provenance only; global identity is resolved below.
        for match in matched["matches"]:
            pairs_erp = np.concatenate(
                (match["kps_cam2"], match["kps_cam4"]), axis=1)
            pairs_rotate = np.concatenate((
                rotate_erp_points(pairs_erp[:, :2], width, height),
                rotate_erp_points(pairs_erp[:, 2:], width, height),
            ), axis=1)
            person = MatchedPerson(
                person_id=raw_candidate_id,
                bbox_cam2=np.asarray(match["bbox_cam2"], np.float32),
                bbox_cam4=np.asarray(match["bbox_cam4"], np.float32),
                pairs=pairs_rotate.astype(np.float32),
                point_ids=np.arange(len(pairs_rotate), dtype=np.int32),
                valid=np.asarray(match["keep"], bool),
                bbox_cam2_wraps_seam=bool(match["bbox_cam2_wraps_seam"]),
                bbox_cam4_wraps_seam=bool(match["bbox_cam4_wraps_seam"]),
            )
            raw_candidates.append(GuidedPersonCandidate(
                candidate_id=raw_candidate_id,
                source_track_ids=(int(track.track_id),), person=person,
                cam2_person_index=int(match["cam2_person_index"]),
                cam4_person_index=int(match["cam4_person_index"]),
                cross_camera_valid_count=int(match["valid_count"]),
                cross_camera_median_angle_deg=float(
                    match["median_angular_error_deg"]),
                source_guided_regions=(row.crop_cam2, row.crop_cam4),
                raw_candidate_ids=(raw_candidate_id,),
            ))
            raw_candidate_id += 1
    identity = resolve_guided_identities(
        tracks, raw_candidates,
        GuidedIdentityConfig(tracking=TrackingConfig(
            image_width=width, image_height=height,
            max_temporal_angle_deg=temporal_max_angle_deg),
            identity_margin_deg=identity_margin_deg,
            identity_margin_ratio=identity_margin_ratio),
        CandidateDedupConfig(
            image_width=width, image_height=height,
            semantic_median_angle_deg=candidate_dedup_angle_deg))
    timing["identity_gate_ms"] = identity.elapsed_ms
    accepted = identity.accepted
    assignment_by_track = {item.track_id: item for item in identity.assignments}
    reason_map = {
        GuidedIdentityReason.TRACK_ASSIGNMENT_AMBIGUOUS:
            GuidedFailureReason.TRACK_ASSIGNMENT_AMBIGUOUS,
        GuidedIdentityReason.CANDIDATE_ASSIGNMENT_AMBIGUOUS:
            GuidedFailureReason.CANDIDATE_ASSIGNMENT_AMBIGUOUS,
        GuidedIdentityReason.NO_UNIQUE_ASSIGNMENT:
            GuidedFailureReason.NO_UNIQUE_ASSIGNMENT,
    }
    for track in tracks:
        row = observations[track.track_id]
        assignment = assignment_by_track[track.track_id]
        row.identity = {
            "status": assignment.status.value,
            "candidate_id": assignment.candidate_id,
            "reason": None if assignment.reason is None else assignment.reason.value,
            "temporal_cost_deg": assignment.temporal_cost_deg,
            "track_margin_deg": assignment.track_margin_deg,
            "track_margin_ratio": assignment.track_margin_ratio,
            "candidate_margin_deg": assignment.candidate_margin_deg,
            "candidate_margin_ratio": assignment.candidate_margin_ratio,
        }
        candidate = accepted.get(track.track_id)
        if candidate is not None:
            row.person = replace(candidate.person, person_id=track.track_id)
            row.valid_count = candidate.cross_camera_valid_count
            row.median_angular_error_deg = candidate.cross_camera_median_angle_deg
            row.temporal_error_deg = assignment.temporal_cost_deg
        elif row.failure_reason is None:
            row.failure_reason = reason_map.get(
                assignment.reason, GuidedFailureReason.NO_UNIQUE_ASSIGNMENT)
    timing["guided_validation_ms"] = (time.perf_counter() - started) * 1000.0
    timing["raw_candidate_count"] = len(raw_candidates)
    timing["unique_candidate_count"] = len(identity.candidates)
    if return_identity_result:
        timing["identity_result"] = identity
    return list(observations.values()), timing


def track_manager_config(track: Any):
    """Build the default geometry config without coupling delivery to a manager."""
    from lib_pano.human_fitting.matched_person_tracks import TrackingConfig
    return TrackingConfig()
