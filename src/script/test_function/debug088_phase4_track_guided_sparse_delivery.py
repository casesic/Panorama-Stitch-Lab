"""Phase 4 track-guided sparse DenseKP delivery bake-off on real frames."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import sys
import time

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DELIVERY_ROOT = PROJECT_ROOT / "Circular_Video/01_recv/20260708"
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))
if str(DELIVERY_ROOT) not in sys.path:
    sys.path.insert(0, str(DELIVERY_ROOT))

from densekp_delivery.config import DenseKPConfig
from densekp_delivery.person_match import match_all_persons
from densekp_delivery.pipeline import _run_densekp, load_erp_pair, run_pipeline
from densekp_delivery.runtime import DenseKPRuntime
from lib_pano.fitting_io.track_guided_delivery import run_track_guided_densekp
from lib_pano.fitting_io.densekp_runtime_export import PersonFrameInitialState
from lib_pano.human_fitting.matched_person_tracks import (
    MatchedPerson, OptimizedPersonState, TrackState,
)
from lib_pano.human_fitting.per_person_cpp import (
    CppPerPersonFitter, CppPersonFitterConfig,
)


def _stats(values):
    values = np.asarray(values, np.float64)
    return {"mean": float(values.mean()), "p50": float(np.percentile(values, 50)),
            "p90": float(np.percentile(values, 90)), "count": int(len(values))}


def _bbox(points, valid):
    points = np.asarray(points, np.float32)[np.asarray(valid, bool)]
    finite = points[np.isfinite(points).all(axis=1)]
    return np.array([finite[:, 0].min(), finite[:, 1].min(),
                     finite[:, 0].max(), finite[:, 1].max()], np.float32)


def _track_from_row(frame, row):
    raw = np.asarray(row["raw_pairs"], np.float32)
    optimized = np.asarray(row["optimized_pairs"], np.float32)
    valid = np.asarray(row["valid"], bool)
    ids = np.asarray(row["point_ids"], np.int32)
    person = MatchedPerson(
        int(row["person_id"]), _bbox(raw[:, :2], valid), _bbox(raw[:, 2:], valid),
        raw, ids, valid)
    state = OptimizedPersonState(
        np.zeros(75, np.float32), np.zeros(10, np.float32), optimized, ids,
        valid, True, cpp_lm_ms=float(row["fitting"]["lm_ms"]))
    return TrackState(int(row["track_id"]), int(frame), state, person,
                      created_frame=int(frame))


def _read_f32(path, count):
    value = np.fromfile(path, dtype="<f4")
    if value.shape != (count,):
        raise RuntimeError(f"unexpected {path} shape {value.shape}")
    return value


def _load_phase22_track0_state(phase22_root, first_row):
    call = phase22_root / "per_person_cpp/track_0000/call_000000"
    fixture, output = call / "runtime_input", call / "cpp_output"
    frozen = PersonFrameInitialState(
        frame_parameters=_read_f32(output / "frame_0/pose_shape_frame_parameters.f32", 75),
        beta=_read_f32(output / "shared_physical_beta.f32", 10),
        cam2_left_hand_pose=_read_f32(fixture / "model_cam2_left_hand_pose.f32", 45),
        cam2_right_hand_pose=_read_f32(fixture / "model_cam2_right_hand_pose.f32", 45),
        cam4_left_hand_pose=_read_f32(fixture / "model_cam4_left_hand_pose.f32", 45),
        cam4_right_hand_pose=_read_f32(fixture / "model_cam4_right_hand_pose.f32", 45),
        cam2_scale=float(_read_f32(fixture / "model_cam2_scale.f32", 1)[0]),
        cam4_scale=float(_read_f32(fixture / "model_cam4_scale.f32", 1)[0]),
    )
    track = _track_from_row(int(first_row["frame"]), first_row["persons"][0])
    track.optimized = OptimizedPersonState(
        frozen.frame_parameters, frozen.beta,
        np.asarray(first_row["persons"][0]["optimized_pairs"], np.float32),
        np.asarray(first_row["persons"][0]["point_ids"], np.int32),
        np.asarray(first_row["persons"][0]["valid"], bool), True)
    return frozen, track


def _full_delivery(runtime, images, rois, masks):
    before = dict(runtime.profile_timing); started = time.perf_counter()
    pair = _run_densekp(runtime, images[0], images[1], rois)
    matched = match_all_persons(
        pair[0], pair[1], erp_mask_cam2=masks[0], erp_mask_cam4=masks[1])
    elapsed = (time.perf_counter() - started) * 1000.0
    after = runtime.profile_timing
    return matched, {
        "delivery_ms": elapsed, "detector_calls": 4,
        "densekp_calls": int(round(float(after["densekp_ort_call_count"])
                                    - float(before["densekp_ort_call_count"]))),
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
    parser.add_argument("--warmup-cycles", type=int, default=1)
    parser.add_argument("--repeat-cycles", type=int, default=7)
    parser.add_argument("--base-fixture", type=Path)
    parser.add_argument("--cpp-executable", type=Path)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    phase22 = json.loads(args.phase22_result.read_text())
    frame_rows = {int(row["frame"]): row for row in phase22["frames"]}
    frames = sorted(frame_rows)
    template = DenseKPConfig(
        backend="ort", fisheye_cam2="", fisheye_cam4="", out_dir="",
        onnx_path=str(args.onnx_path), camera=args.camera,
        erp_remap_backend="cuda",
        erp_remap_cuda_library=str(args.erp_remap_cuda_library),
        calibration_cache_dir=str(args.calibration_cache_dir),
        output_coord_space="erp_rotate", vis_option="none", save_option="none",
        device="cuda", batch_size=1, profile_runtime=True)
    runtime = DenseKPRuntime.from_config(template).load()
    loaded = {}
    for frame in frames:
        config = template.for_frame(
            args.dataset_root / f"image/fisheye/cam2/{frame:05d}.jpg",
            args.dataset_root / f"image/fisheye/cam4/{frame:05d}.jpg",
            args.output_dir / "setup" / f"frame_{frame:06d}")
        if frame == frames[0]:
            run_pipeline(config, runtime=runtime)
        loaded[frame] = load_erp_pair(config, runtime)
    entry = list(runtime._static_geometry.values())[0]
    masks = entry["valid_masks"]
    rois = entry["overlap_roi_component_bbox"][-1]

    margin_results = {}
    for margin in (0.10, 0.20, 0.30, 0.40):
        rows = []
        for previous, frame in zip(frames[:-1], frames[1:]):
            tracks = [_track_from_row(previous, row)
                      for row in frame_rows[previous]["persons"]]
            observations, timing = run_track_guided_densekp(
                runtime, *loaded[frame], tracks, masks, margin_ratio=margin)
            expected = {int(row["track_id"]) for row in frame_rows[frame]["persons"]}
            for observation in observations:
                rows.append({
                    "frame": frame, "track_id": observation.track_id,
                    "expected_present": observation.track_id in expected,
                    "success": observation.success,
                    "valid_count": observation.valid_count,
                    "median_angle_deg": observation.median_angular_error_deg,
                    "temporal_error_deg": observation.temporal_error_deg,
                    "failure_reason": (None if observation.failure_reason is None
                                       else observation.failure_reason.value),
                    "identity": observation.identity,
                    "bbox_cam2": (None if observation.crop_cam2 is None else
                                  observation.crop_cam2.predicted_bbox_erp),
                    "bbox_cam4": (None if observation.crop_cam4 is None else
                                  observation.crop_cam4.predicted_bbox_erp),
                    "crop_area_ratio": (
                        0.0 if observation.crop_cam2 is None or observation.crop_cam4 is None
                        else (observation.crop_cam2.roi.pixel_count
                              + observation.crop_cam4.roi.pixel_count)
                        / (2.0 * loaded[frame][0].shape[0] * loaded[frame][0].shape[1])),
                    "timing": timing,
                })
        present = [row for row in rows if row["expected_present"]]
        margin_results[str(margin)] = {
            "rows": rows, "guided_attempts": len(rows),
            "expected_present_attempts": len(present),
            "guided_successes": sum(row["success"] for row in rows),
            "present_successes": sum(row["success"] for row in present),
            "present_success_rate": (sum(row["success"] for row in present)
                                     / max(1, len(present))),
            "fallback_count": sum(not row["success"] for row in rows),
            "mean_valid_points": statistics.mean(
                [row["valid_count"] for row in rows]) if rows else 0.0,
            "mean_crop_area_ratio": statistics.mean(
                [row["crop_area_ratio"] for row in rows]) if rows else 0.0,
        }
    selected_margin = max(
        margin_results, key=lambda key: (
            margin_results[key]["present_success_rate"], -float(key)))

    # One unique correctness pass; performance below executes the model again
    # for every repeated real-equivalent frame.
    guided_cost = {}; full_cost = {}
    for previous, frame in zip(frames[:-1], frames[1:]):
        tracks = [_track_from_row(previous, row)
                  for row in frame_rows[previous]["persons"]]
        observations, timing = run_track_guided_densekp(
            runtime, *loaded[frame], tracks, masks,
            margin_ratio=float(selected_margin))
        guided_cost[frame] = {
            "delivery_ms": sum(float(timing[key]) for key in (
                "bbox_projection_ms", "guided_crop_ms", "guided_densekp_ms",
                "guided_validation_ms")),
            "detector_calls": 0, "densekp_calls": int(timing["densekp_calls"]),
            "success": all(row.success for row in observations)
                       and len(observations) == len(tracks),
            "identity_gate_ms": float(timing["identity_gate_ms"]),
            "failures": [{"track_id": row.track_id,
                          "reason": row.failure_reason.value}
                         for row in observations if not row.success],
        }
    for frame in frames:
        _, full_cost[frame] = _full_delivery(runtime, loaded[frame], rois, masks)

    interval_results = {}
    for interval in (1, 3, 5, 10):
        timeline = []
        for _ in range(args.warmup_cycles + args.repeat_cycles):
            for index, frame in enumerate(frames):
                periodic = index % interval == 0
                if periodic or frame == frames[0]:
                    _, cost = _full_delivery(runtime, loaded[frame], rois, masks)
                    mode = "FULL_DETECTION"
                else:
                    previous = frames[index - 1]
                    tracks = [_track_from_row(previous, row)
                              for row in frame_rows[previous]["persons"]]
                    observations, timing = run_track_guided_densekp(
                        runtime, *loaded[frame], tracks, masks,
                        margin_ratio=float(selected_margin))
                    guided = {
                        "delivery_ms": sum(float(timing[key]) for key in (
                            "bbox_projection_ms", "guided_crop_ms",
                            "guided_densekp_ms", "guided_validation_ms")),
                        "detector_calls": 0,
                        "densekp_calls": int(timing["densekp_calls"]),
                        "identity_gate_ms": float(timing["identity_gate_ms"]),
                        "success": all(row.success for row in observations)
                                   and len(observations) == len(tracks),
                    }
                    if guided["success"]:
                        cost = guided; mode = "TRACK_GUIDED"
                    else:
                        _, recovery = _full_delivery(
                            runtime, loaded[frame], rois, masks)
                        cost = {
                            "delivery_ms": guided["delivery_ms"] + recovery["delivery_ms"],
                            "detector_calls": recovery["detector_calls"],
                            "densekp_calls": guided["densekp_calls"] + recovery["densekp_calls"],
                            "identity_gate_ms": guided["identity_gate_ms"],
                        }; mode = "RECOVERY"
                timeline.append({"frame": frame, "mode": mode, **cost})
        discard = args.warmup_cycles * len(frames)
        steady = timeline[discard:]
        delivery = [row["delivery_ms"] for row in steady]
        lm = [sum(float(person["fitting"]["lm_ms"])
                  for person in frame_rows[row["frame"]]["persons"])
              for row in steady]
        serial = [a + b for a, b in zip(delivery, lm)]
        interval_results[str(interval)] = {
            "timeline_unique": timeline[:len(frames)],
            "full_frames": sum(row["mode"] == "FULL_DETECTION" for row in steady),
            "guided_frames": sum(row["mode"] == "TRACK_GUIDED" for row in steady),
            "recovery_frames": sum(row["mode"] == "RECOVERY" for row in steady),
            "detector_calls_per_frame": statistics.mean(row["detector_calls"] for row in steady),
            "densekp_calls_per_frame": statistics.mean(row["densekp_calls"] for row in steady),
            "identity_gate_ms_per_frame": statistics.mean(
                float(row.get("identity_gate_ms", 0.0)) for row in steady),
            "full_recovery_rate": sum(row["mode"] == "RECOVERY" for row in steady)
                                  / len(steady),
            "delivery": _stats(delivery), "serial_e2e": _stats(serial),
            "delivery_fps": 1000.0 / statistics.mean(delivery),
            "serial_fps": 1000.0 / statistics.mean(serial),
            "new_person_detection_delay_upper_bound_frames": interval - 1,
            "first_10_mean_ms": statistics.mean(delivery[:10]),
            "middle_10_mean_ms": statistics.mean(
                delivery[max(0, len(delivery)//2-5):len(delivery)//2+5]),
            "last_10_mean_ms": statistics.mean(delivery[-10:]),
        }
    main_track_integration = None
    if args.base_fixture is not None and args.cpp_executable is not None:
        frozen, track = _load_phase22_track0_state(
            args.phase22_result.parent, frame_rows[frames[0]])
        fitter = CppPerPersonFitter(CppPersonFitterConfig(
            args.base_fixture, args.cpp_executable,
            args.output_dir / "main_track_cpp", "cpu"))
        fitter._frozen_by_track[0] = frozen
        integration_rows = []
        for index, frame in enumerate(frames[1:], start=1):
            started = time.perf_counter()
            if index % 3 == 0:
                _full_delivery(runtime, loaded[frame], rois, masks)
                source = next(row for row in frame_rows[frame]["persons"]
                              if int(row["track_id"]) == 0)
                person = _track_from_row(frame, source).last_person
                mode = "FULL_DETECTION"
                guided_valid = None
            else:
                guided, _ = run_track_guided_densekp(
                    runtime, *loaded[frame], [track], masks,
                    margin_ratio=float(selected_margin))
                if not guided[0].success:
                    raise RuntimeError(
                        f"main Track 0 guided failure frame={frame}: "
                        f"{guided[0].failure_reason}")
                person = guided[0].person
                guided_valid = guided[0].valid_count
                mode = "TRACK_GUIDED"
            delivery_ms = (time.perf_counter() - started) * 1000.0
            fit_started = time.perf_counter()
            fitted = fitter(person, track.optimized, 0)
            process_wall_ms = (time.perf_counter() - fit_started) * 1000.0
            integration_rows.append({
                "frame": frame, "track_id": 0, "mode": mode,
                "guided_valid_count": guided_valid,
                "delivery_ms": delivery_ms, "cpp_process_wall_ms": process_wall_ms,
                "cpp_prepare_ms": fitted.cpp_prepare_ms,
                "cpp_lm_ms": fitted.cpp_lm_ms, "lm_success": fitted.success,
                "warm_state_exact": fitted.warm_state_exact,
            })
            track.last_frame = frame; track.last_person = person; track.optimized = fitted
        main_track_integration = {
            "detect_interval": 3, "rows": integration_rows,
            "all_lm_success": all(row["lm_success"] for row in integration_rows),
            "all_warm_state_exact": all(row["warm_state_exact"] for row in integration_rows),
            "serial_ms": _stats([row["delivery_ms"] + row["cpp_process_wall_ms"]
                                 for row in integration_rows]),
        }
    result = {
        "schema_version": "track_guided_sparse_delivery_bakeoff_v2_identity",
        "correctness_frames": frames,
        "performance_workload": "precomputed real-frame costs repeated",
        "erp_remap_included": False, "densekp_batch_enabled": False,
        "detector_batch_enabled": False,
        "margin_bakeoff": margin_results, "selected_margin": float(selected_margin),
        "guided_frame_costs": guided_cost, "full_frame_costs": full_cost,
        "interval_bakeoff": interval_results,
        "main_track_cpp_integration": main_track_integration,
    }
    (args.output_dir / "phase4_bakeoff.json").write_text(json.dumps(result, indent=2) + "\n")
    identity_reasons = {
        "TRACK_ASSIGNMENT_AMBIGUOUS", "CANDIDATE_ASSIGNMENT_AMBIGUOUS",
        "NO_UNIQUE_ASSIGNMENT",
    }
    ambiguities = [
        {"margin": margin, **row}
        for margin, summary in margin_results.items() for row in summary["rows"]
        if row["failure_reason"] in identity_reasons
    ]
    (args.output_dir / "guided_failure_cases.json").write_text(
        json.dumps({"identity_ambiguities": ambiguities}, indent=2) + "\n")
    runtime.close()
    print(json.dumps({"selected_margin": selected_margin,
                      "interval_bakeoff": interval_results}, indent=2))


if __name__ == "__main__":
    main()
