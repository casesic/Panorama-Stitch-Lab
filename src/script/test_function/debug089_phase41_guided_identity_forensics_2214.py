"""Read-only Phase 4.1 identity audit for the frame-2214 guided ambiguity."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DELIVERY_ROOT = PROJECT_ROOT / "Circular_Video/01_recv/20260708"
for root in (PROJECT_ROOT / "src", DELIVERY_ROOT):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from densekp_delivery.config import DenseKPConfig
from densekp_delivery.person_match import match_all_persons
from densekp_delivery.pipeline import _run_densekp, load_erp_pair, run_pipeline
from densekp_delivery.runtime import DenseKPRuntime
from lib_pano.fitting_io.track_guided_delivery import run_track_guided_densekp
from lib_pano.human_fitting.guided_identity import resolve_guided_identities
from lib_pano.human_fitting.matched_person_tracks import (
    MatchedPerson, OptimizedPersonState, TrackState,
)


def _bbox(points, valid):
    points = np.asarray(points, np.float32)[np.asarray(valid, bool)]
    points = points[np.isfinite(points).all(axis=1)]
    return np.asarray([points[:, 0].min(), points[:, 1].min(),
                       points[:, 0].max(), points[:, 1].max()], np.float32)


def _track(frame, row):
    pairs = np.asarray(row["raw_pairs"], np.float32)
    valid = np.asarray(row["valid"], bool)
    person = MatchedPerson(
        int(row["person_id"]), _bbox(pairs[:, :2], valid),
        _bbox(pairs[:, 2:], valid), pairs,
        np.asarray(row["point_ids"], np.int32), valid)
    optimized_pairs = np.asarray(row["optimized_pairs"], np.float32)
    optimized = OptimizedPersonState(
        np.zeros(75, np.float32), np.zeros(10, np.float32), optimized_pairs,
        np.asarray(row["point_ids"], np.int32), valid, True)
    # Delivery requires a successful fitted state for guided projection, while
    # its current body-envelope proxy and temporal gate use last_person.
    return TrackState(int(row["track_id"]), frame, optimized, person,
                      created_frame=frame)


def _finite(value):
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _identity_payload(result):
    return {
        "raw_candidate_count": sum(len(group) for group in result.dedup_groups),
        "unique_candidate_count": len(result.candidates),
        "dedup_groups": [list(group) for group in result.dedup_groups],
        "candidates": [{
            "candidate_id": row.candidate_id,
            "source_track_ids": list(row.source_track_ids),
            "raw_candidate_ids": list(row.raw_candidate_ids),
            "cam2_person_index": row.cam2_person_index,
            "cam4_person_index": row.cam4_person_index,
            "valid_count": row.cross_camera_valid_count,
            "cross_camera_median_angle_deg": row.cross_camera_median_angle_deg,
            "bbox_cam2": row.person.bbox_cam2.tolist(),
            "bbox_cam4": row.person.bbox_cam4.tolist(),
        } for row in result.candidates],
        "cost_matrix_deg": [[_finite(float(x)) for x in row]
                            for row in result.cost_matrix],
        "temporal_diagnostics": [[{key: _finite(value) for key, value in cell.items()}
                                  for cell in row]
                                 for row in result.temporal_diagnostics],
        "assignments": [{key: (_finite(value.value) if hasattr(value, "value")
                                else _finite(value))
                         for key, value in asdict(row).items()}
                        for row in result.assignments],
        "identity_gate_ms": result.elapsed_ms,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--phase22-result", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--onnx-path", type=Path, required=True)
    parser.add_argument("--erp-remap-cuda-library", type=Path, required=True)
    parser.add_argument("--calibration-cache-dir", type=Path, required=True)
    parser.add_argument("--camera", default="new")
    parser.add_argument("--previous-frame", type=int, default=2213)
    parser.add_argument("--frame", type=int, default=2214)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    sequence = json.loads(args.phase22_result.read_text())
    previous = next(row for row in sequence["frames"]
                    if int(row["frame"]) == args.previous_frame)
    tracks = [_track(args.previous_frame, row) for row in previous["persons"]]
    template = DenseKPConfig(
        backend="ort", fisheye_cam2="", fisheye_cam4="", out_dir="",
        onnx_path=str(args.onnx_path), camera=args.camera,
        erp_remap_backend="cuda",
        erp_remap_cuda_library=str(args.erp_remap_cuda_library),
        calibration_cache_dir=str(args.calibration_cache_dir),
        output_coord_space="erp_rotate", vis_option="none", save_option="none",
        device="cuda", batch_size=1, profile_runtime=True)
    config = template.for_frame(
        args.dataset_root / f"image/fisheye/cam2/{args.frame:05d}.jpg",
        args.dataset_root / f"image/fisheye/cam4/{args.frame:05d}.jpg",
        args.output_dir / "runtime")
    runtime = DenseKPRuntime.from_config(template).load()
    run_pipeline(config, runtime=runtime)  # warm/static setup
    images = load_erp_pair(config, runtime)
    entry = list(runtime._static_geometry.values())[0]
    masks = entry["valid_masks"]
    rois = entry["overlap_roi_component_bbox"][-1]
    observations, timing = run_track_guided_densekp(
        runtime, *images, tracks, masks, margin_ratio=0.10,
        return_identity_result=True)
    identity = timing.pop("identity_result")
    full_pair = _run_densekp(runtime, images[0], images[1], rois)
    full = match_all_persons(
        full_pair[0], full_pair[1], erp_mask_cam2=masks[0],
        erp_mask_cam4=masks[1])
    payload = {
        "schema_version": "phase41_guided_identity_forensics_v1",
        "previous_frame": args.previous_frame, "frame": args.frame,
        "previous_track_ids": [track.track_id for track in tracks],
        "identity": _identity_payload(identity),
        "guided_observations": [{
            "track_id": row.track_id, "success": row.success,
            "identity": row.identity,
            "failure_reason": (None if row.failure_reason is None
                               else row.failure_reason.value),
        } for row in observations],
        "full_reference": {
            "cam2_person_count": full["cam2_person_count"],
            "cam4_person_count": full["cam4_person_count"],
            "matched_person_count": len(full["matches"]),
            "matches": [{
                "person_id": row["person_id"],
                "cam2_person_index": row["cam2_person_index"],
                "cam4_person_index": row["cam4_person_index"],
                "valid_count": row["valid_count"],
                "median_angular_error_deg": row["median_angular_error_deg"],
            } for row in full["matches"]],
        },
        "timing": {key: value for key, value in timing.items()
                   if isinstance(value, (int, float))},
    }
    path = args.output_dir / "phase41_2214_identity_forensics.json"
    path.write_text(json.dumps(payload, indent=2) + "\n")
    runtime.close()
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
