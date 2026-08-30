"""ERP-boundary seam between production orchestration and the delivered runtime.

The formal sequence entry consumes ready ERP images; fisheye/LUT/remap stays in
the delivered package (upstream concern).  This module is the one place where
production code touches delivery symbols, so it can collapse into the delivery
package's own public API once that package is version controlled.  Until then
no other production module may import ``densekp_delivery`` directly.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from lib_pano.delivery_bootstrap import ensure_importable
from lib_pano.human_fitting.matched_person_tracks import MatchedPerson


def load_erp_frame_pair(
    erp_cam2_path: str | Path, erp_cam4_path: str | Path,
) -> tuple[np.ndarray, np.ndarray]:
    """Read one prepared normal-ERP pair (BGR) without any remap."""
    ensure_importable()
    from densekp_delivery.image_io import read_image_pair_profiled

    cam2, cam4, _ = read_image_pair_profiled(erp_cam2_path, erp_cam4_path)
    return cam2, cam4


def build_component_detection_rois(
    mask_cam2: np.ndarray, mask_cam4: np.ndarray,
):
    """Production ROI policy: component bboxes over strict overlap."""
    ensure_importable()
    from densekp_delivery.overlap_roi import build_detection_rois

    _, _, component_rois = build_detection_rois(mask_cam2, mask_cam4)
    return component_rois


def run_full_detection(runtime, erp_cam2: np.ndarray, erp_cam4: np.ndarray,
                       detection_rois):
    """Detector + DenseKP over the detection ROIs, restored to full ERP.

    The delivered pipeline function is still private; this is the single
    documented seam until the package exposes a public equivalent.
    """
    ensure_importable()
    from densekp_delivery.pipeline import _run_densekp

    return _run_densekp(runtime, erp_cam2, erp_cam4, detection_rois)


def match_full_erp(result_cam2, result_cam4, mask_cam2, mask_cam4) -> dict:
    """Cross-camera one-to-one matching with spherical-geometry costs."""
    ensure_importable()
    from densekp_delivery.person_match import match_all_persons

    return match_all_persons(
        result_cam2, result_cam4,
        erp_mask_cam2=mask_cam2, erp_mask_cam4=mask_cam4,
    )


def matched_persons_from_full_erp(match_data: dict, image_width: int,
                                  image_height: int) -> list[MatchedPerson]:
    """Convert full-ERP matches to the shared v2 person convention.

    Pairs are rotated into ERP_ROTATE exactly as
    ``fitting_io.track_guided_delivery`` and the delivered JSON schema do;
    bboxes stay in normal ERP space where HMR2 consumes them.
    """
    ensure_importable()
    from densekp_delivery.camera.erp_rotate import rotate_erp_points

    persons: list[MatchedPerson] = []
    for person_id, match in enumerate(match_data["matches"]):
        points2 = np.asarray(match["kps_cam2"], np.float32)
        points4 = np.asarray(match["kps_cam4"], np.float32)
        pairs_rotate = np.concatenate((
            rotate_erp_points(points2, image_width, image_height),
            rotate_erp_points(points4, image_width, image_height),
        ), axis=1).astype(np.float32)
        persons.append(MatchedPerson(
            person_id=int(person_id),
            bbox_cam2=np.asarray(match["bbox_cam2"], np.float32),
            bbox_cam4=np.asarray(match["bbox_cam4"], np.float32),
            pairs=pairs_rotate,
            point_ids=np.arange(len(pairs_rotate), dtype=np.int32),
            valid=np.asarray(match["keep"], bool),
            bbox_cam2_wraps_seam=bool(match["bbox_cam2_wraps_seam"]),
            bbox_cam4_wraps_seam=bool(match["bbox_cam4_wraps_seam"]),
        ))
    return persons


def detector_call_count(detection_rois) -> int:
    """YOLOX runs B=1 once per camera per ROI; the policy is fixed."""
    return 2 * len(detection_rois)


def densekp_calls_since(runtime, before: dict) -> int:
    after = runtime.profile_timing
    return int(round(float(after["densekp_ort_call_count"])
                     - float(before.get("densekp_ort_call_count", 0.0))))
