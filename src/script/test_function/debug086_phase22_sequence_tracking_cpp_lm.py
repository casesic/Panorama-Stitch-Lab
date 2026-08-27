"""Real DenseKP-v2 sequence -> temporal tracks -> HMR2/previous -> C++ LM."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import sys
import time

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DELIVERY_ROOT = PROJECT_ROOT / "Circular_Video/01_recv/20260708"
if str(DELIVERY_ROOT) not in sys.path:
    sys.path.insert(0, str(DELIVERY_ROOT))

from densekp_delivery.config import DenseKPConfig
from densekp_delivery.pipeline import load_erp_pair
from densekp_delivery.runtime import DenseKPRuntime

from lib_pano.human3d.hmr2_initializer import HMR2Initializer
from lib_pano.human_fitting.hmr2_staged_init import (
    convert_hmr2_to_staged_initialization,
    to_smplx_camera_initializations,
)
from lib_pano.fitting_io.densekp_runtime_export import (
    person_initial_state_from_smplx,
)
from lib_pano.human_fitting.matched_person_tracks import (
    MatchedPerson,
    MatchedPersonFittingOrchestrator,
    MatchedPersonTrackManager,
    TrackingConfig,
)
from lib_pano.human_fitting.per_person_cpp import (
    CppPerPersonFitter,
    CppPersonFitterConfig,
)


def _frames(text: str) -> list[int]:
    values: list[int] = []
    for part in text.split(","):
        if "-" in part:
            start, end = map(int, part.split("-", 1))
            values.extend(range(start, end + 1))
        else:
            values.append(int(part))
    return list(dict.fromkeys(values))


def _percentile(values: list[float], percentile: float) -> float:
    return float(np.percentile(np.asarray(values, np.float64), percentile)) if values else 0.0


def _visualize(frame: int, dataset_root: Path, frame_payload: dict,
               output: Path) -> None:
    images = [cv2.imread(str(dataset_root / f"image/equirect_rotate/{camera}/{frame:05d}.jpg"))
              for camera in ("cam2", "cam4")]
    if any(image is None for image in images):
        return
    rows = []
    for mode in ("raw_pairs", "optimized_pairs"):
        canvas = np.concatenate(tuple(image.copy() for image in images), axis=1)
        width = images[0].shape[1]
        for person in frame_payload["persons"]:
            pairs = np.asarray(person[mode], np.float32)
            valid = np.asarray(person["valid"], bool)
            color = ((37 * (person["track_id"] + 3)) % 255, 255,
                     (91 * (person["track_id"] + 1)) % 255)
            for x2, y2, x4, y4 in pairs[valid]:
                cv2.circle(canvas, (round(x2), round(y2)), 3, color, -1)
                cv2.circle(canvas, (round(x4) + width, round(y4)), 3, color, -1)
            label = (f"T{person['track_id']} P{person['person_id']} "
                     f"{person['initialization']['type']} "
                     f"LM={'OK' if person['fitting']['success'] else 'FAIL'}")
            cv2.putText(canvas, label, (30, 55 + 36 * person["track_id"]),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
        cv2.putText(canvas, mode, (30, canvas.shape[0] - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        rows.append(canvas)
    output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output), np.concatenate(rows, axis=0))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--frames", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--onnx-path", type=Path, required=True)
    parser.add_argument("--hmr2-checkpoint", type=Path, required=True)
    parser.add_argument("--base-fixture", type=Path, required=True)
    parser.add_argument("--cpp-executable", type=Path, required=True)
    parser.add_argument("--erp-remap-cuda-library", type=Path, required=True)
    parser.add_argument("--calibration-cache-dir", type=Path, required=True)
    parser.add_argument("--camera", choices=("old", "new"), default="new")
    parser.add_argument("--cpp-backend", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    requested = _frames(args.frames)
    frames = [frame for frame in requested if all(
        (args.dataset_root / f"image/fisheye/{camera}/{frame:05d}.jpg").is_file()
        for camera in ("cam2", "cam4"))]
    missing = sorted(set(requested) - set(frames))
    if not frames:
        raise RuntimeError("no requested frame has both fisheye images")
    args.output_dir.mkdir(parents=True, exist_ok=False)

    template = DenseKPConfig(
        backend="ort", fisheye_cam2="", fisheye_cam4="", out_dir="",
        onnx_path=str(args.onnx_path), camera=args.camera,
        erp_remap_backend="cuda",
        erp_remap_cuda_library=str(args.erp_remap_cuda_library),
        calibration_cache_dir=str(args.calibration_cache_dir),
        output_coord_space="erp_rotate", vis_option="none", save_option="none",
        device=args.device, profile_runtime=True,
    )
    dense_runtime = DenseKPRuntime.from_config(template)
    dense_started = time.perf_counter()
    dense_runtime.load()
    dense_startup_ms = (time.perf_counter() - dense_started) * 1000.0
    hmr_started = time.perf_counter()
    hmr = HMR2Initializer(args.hmr2_checkpoint, device=args.device)
    hmr_startup_ms = (time.perf_counter() - hmr_started) * 1000.0
    focal = (float(hmr.model_cfg.EXTRA.FOCAL_LENGTH)
             / float(hmr.model_cfg.MODEL.IMAGE_SIZE) * 5760.0)

    cpp = CppPerPersonFitter(CppPersonFitterConfig(
        args.base_fixture, args.cpp_executable,
        args.output_dir / "per_person_cpp", args.cpp_backend))
    manager = MatchedPersonTrackManager(TrackingConfig())
    current: dict[str, object] = {}
    hmr_forward_times: list[float] = []
    hmr_erp_prepare_times: list[float] = []

    def hmr2_batch(camera: str, persons: list[MatchedPerson]):
        config = current["config"]
        erp_pair = current.get("erp_pair")
        if erp_pair is None:
            erp_started = time.perf_counter()
            erp_pair = load_erp_pair(config, dense_runtime)
            hmr_erp_prepare_times.append(
                (time.perf_counter() - erp_started) * 1000.0)
            current["erp_pair"] = erp_pair
        image = erp_pair[0 if camera == "cam2" else 1]
        boxes = np.stack([
            person.bbox_cam2 if camera == "cam2" else person.bbox_cam4
            for person in persons])
        forward_started = time.perf_counter()
        outputs = hmr.infer(image, boxes)
        hmr_forward_times.append((time.perf_counter() - forward_started) * 1000.0)
        normalized = []
        for person, value in zip(persons, outputs):
            staged = convert_hmr2_to_staged_initialization(
                value, source_camera=camera,
                bboxes={"cam2": person.bbox_cam2, "cam4": person.bbox_cam4},
                focal_length=focal,
            )
            normalized.append(person_initial_state_from_smplx(
                *to_smplx_camera_initializations(staged)))
        return normalized

    orchestrator = MatchedPersonFittingOrchestrator(manager, hmr2_batch, cpp)
    sequence_frames = []
    delivery_times: list[float] = []
    delivery_without_erp_times: list[float] = []
    frame_fitting_times: list[float] = []
    all_rows: list[dict] = []
    total_cam2 = total_cam4 = 0
    for frame in frames:
        frame_output = args.output_dir / "observations" / f"frame_{frame:06d}"
        config = template.for_frame(
            args.dataset_root / f"image/fisheye/cam2/{frame:05d}.jpg",
            args.dataset_root / f"image/fisheye/cam4/{frame:05d}.jpg",
            frame_output,
        )
        current.clear(); current["config"] = config
        delivery_started = time.perf_counter()
        dense_runtime.process(config)
        delivery_ms = (time.perf_counter() - delivery_started) * 1000.0
        delivery_times.append(delivery_ms)
        runtime_profile = json.loads((
            frame_output / "result/densekp_runtime_profile.json").read_text())
        delivery_without_erp_times.append(
            float(runtime_profile["runtime_total_ms"])
            - float(runtime_profile.get("erp_remap_ms", 0.0)))
        path = frame_output / "result/densekp_pairs_erp_rotate.json"
        delivery = json.loads(path.read_text())
        payloads = delivery.get("persons", [])
        persons = [MatchedPerson.from_payload(item) for item in payloads]
        cam2_indices = {int(p["cam2_person_index"]) for p in payloads}
        cam4_indices = {int(p["cam4_person_index"]) for p in payloads}
        total_cam2 += len(cam2_indices) + len(delivery.get("unmatched_cam2_person_indices", []))
        total_cam4 += len(cam4_indices) + len(delivery.get("unmatched_cam4_person_indices", []))
        fitting_started = time.perf_counter()
        result = orchestrator.process_frame(frame, persons)
        fitting_ms = (time.perf_counter() - fitting_started) * 1000.0
        frame_fitting_times.append(fitting_ms)
        frame_row = {
            "frame": frame, "observation_path": str(path),
            "densekp_delivery_ms": delivery_ms,
            "fitting_ms": fitting_ms,
            "unmatched_cam2": delivery.get("unmatched_cam2_person_indices", []),
            "unmatched_cam4": delivery.get("unmatched_cam4_person_indices", []),
            "persons": result.payload["persons"],
            "profiling": result.profiling,
        }
        sequence_frames.append(frame_row)
        for person in result.payload["persons"]:
            all_rows.append({"frame": frame, **person})
        frame_dir = args.output_dir / "fitting" / f"frame_{frame:06d}"
        frame_dir.mkdir(parents=True)
        (frame_dir / "result.json").write_text(json.dumps(frame_row, indent=2) + "\n")
        _visualize(frame, args.dataset_root, result.payload,
                   args.output_dir / "visualization" / f"frame_{frame:06d}.jpg")
    dense_runtime.close()

    prepares = [float(row["cpp_preparation"]["prepare_ms"])
                for row in all_rows if row["cpp_preparation"]["prepare_ms"] is not None]
    lm_times = [float(row["fitting"]["lm_ms"])
                for row in all_rows if row["fitting"]["lm_ms"] is not None]
    warm_rows = [row for row in all_rows
                 if row["initialization"]["type"] == "previous_frame"]
    cold_rows = [row for row in all_rows
                 if row["initialization"]["type"] == "hmr2"]
    successful = sum(bool(row["fitting"]["success"]) for row in all_rows)
    counters = orchestrator.counters
    summary = {
        "schema_version": "densekp_multi_person_sequence_fitting_v1",
        "dataset_root": str(args.dataset_root), "requested_frames": requested,
        "actual_frames": frames, "missing_frames": missing,
        "num_frames": len(frames), "total_cam2_detections": total_cam2,
        "total_cam4_detections": total_cam4,
        "matched_person_observations": len(all_rows),
        "unique_tracks": len(manager.tracks),
        "new_track_count": sum(row["initialization"]["type"] == "hmr2" for row in all_rows),
        "lost_event_count": counters.lost_event_count,
        "reconnect_count": counters.reconnect_count,
        "reinit_count": counters.reinit_count,
        "hmr2_call_count": counters.hmr2_calls,
        "hmr2_person_count": counters.hmr2_person_count,
        "previous_frame_warm_start_count": counters.warm_start_count,
        "lm_attempts": counters.lm_attempt_count,
        "lm_successes": successful,
        "lm_failures": len(all_rows) - successful,
        "lm_success_rate": successful / max(1, len(all_rows)),
        "hmr2_startup_ms": hmr_startup_ms,
        "hmr2_callback_total_ms": counters.hmr2_ms,
        "hmr2_erp_prepare_total_ms": sum(hmr_erp_prepare_times),
        "hmr2_true_runtime_ms": sum(hmr_forward_times),
        "hmr2_forward_p50_ms": statistics.median(hmr_forward_times),
        "hmr2_amortized_ms_per_frame": sum(hmr_forward_times) / len(frames),
        "densekp_startup_ms": dense_startup_ms,
        "densekp_delivery_p50_ms": statistics.median(delivery_times),
        "densekp_delivery_p90_ms": _percentile(delivery_times, 90),
        "densekp_without_erp_p50_ms": statistics.median(delivery_without_erp_times),
        "densekp_without_erp_p90_ms": _percentile(delivery_without_erp_times, 90),
        "prepare_p50_ms": statistics.median(prepares) if prepares else 0.0,
        "prepare_p90_ms": _percentile(prepares, 90),
        "lm_p50_ms": statistics.median(lm_times) if lm_times else 0.0,
        "lm_p90_ms": _percentile(lm_times, 90),
        "total_fitting_p50_ms_per_frame": statistics.median(frame_fitting_times),
        "total_fitting_p90_ms_per_frame": _percentile(frame_fitting_times, 90),
        "warm_state_exact": all(
            row["cpp_preparation"]["warm_state_exact"] is True for row in warm_rows),
        "cold_initial_objectives": [row["fitting"]["initial_objective"] for row in cold_rows],
        "warm_initial_objectives": [row["fitting"]["initial_objective"] for row in warm_rows],
        "final_objectives": [row["fitting"]["final_objective"] for row in all_rows],
        "frames": sequence_frames,
    }
    if counters.hmr2_calls > summary["new_track_count"] + counters.reinit_count:
        raise RuntimeError("MATCHED_ONLY_HMR2_CALL_INVARIANT_VIOLATION")
    if warm_rows and not summary["warm_state_exact"]:
        raise RuntimeError("WARM_STATE_EXACT_PROPAGATION_FAILED")
    (args.output_dir / "sequence_result.json").write_text(
        json.dumps(summary, indent=2) + "\n")
    print("PHASE22_SEQUENCE_CPP_LM=PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
