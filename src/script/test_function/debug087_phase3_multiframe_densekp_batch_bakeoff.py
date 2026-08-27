"""Phase 3 in-memory ERP detector/DenseKP cross-frame batch bake-off."""

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

from densekp_delivery.camera.erp_rotate import rotate_erp_points
from densekp_delivery.config import DenseKPConfig
from densekp_delivery.person_match import match_all_persons
from densekp_delivery.pipeline import (
    _run_densekp, load_erp_pair, run_densekp_erp_window, run_pipeline,
)
from densekp_delivery.runtime import DenseKPRuntime


def _frames(text: str) -> list[int]:
    result = []
    for part in text.split(","):
        if "-" in part:
            lo, hi = map(int, part.split("-", 1)); result.extend(range(lo, hi + 1))
        else:
            result.append(int(part))
    return list(dict.fromkeys(result))


def _delta(after: dict, before: dict, name: str) -> float:
    return float(after.get(name, 0.0)) - float(before.get(name, 0.0))


def _stats(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, np.float64)
    return {
        "mean": float(array.mean()), "p50": float(np.percentile(array, 50)),
        "p90": float(np.percentile(array, 90)), "min": float(array.min()),
        "max": float(array.max()),
    }


def _match(pair, masks):
    return match_all_persons(
        pair[0], pair[1], erp_mask_cam2=masks[0], erp_mask_cam4=masks[1])


def _compare_pairs(reference, candidate) -> dict:
    result = {"camera": {}, "max_keypoint_abs": 0.0, "bbox_max_abs": 0.0}
    for camera_index, camera in enumerate(("cam2", "cam4")):
        left, right = reference[camera_index], candidate[camera_index]
        row = {
            "reference_count": len(left.persons),
            "candidate_count": len(right.persons), "persons": [],
        }
        for p, q in zip(left.persons, right.persons):
            kp = float(np.max(np.abs(p.keypoints - q.keypoints)))
            box = float(np.max(np.abs(np.asarray(p.bbox) - np.asarray(q.bbox))))
            row["persons"].append({"keypoint_max_abs": kp, "bbox_max_abs": box})
            result["max_keypoint_abs"] = max(result["max_keypoint_abs"], kp)
            result["bbox_max_abs"] = max(result["bbox_max_abs"], box)
        result["camera"][camera] = row
    return result


def _run_window(runtime, window, rois, masks, mode, batch_size):
    profile_before = dict(runtime.profile_timing)
    started = time.perf_counter()
    auxiliary = {"roi_extraction_ms": 0.0, "metadata_restoration_ms": 0.0}
    if mode == "baseline":
        pairs = {frame: _run_densekp(runtime, cam2, cam4, rois)
                 for frame, cam2, cam4 in window}
    elif mode == "densekp_batch":
        pairs, auxiliary = run_densekp_erp_window(
            runtime, window, rois, batch_size)
    else:
        raise ValueError(mode)
    matches = {frame: _match(pairs[frame], masks) for frame, _, _ in window}
    # Exercise the same coordinate transform as production, without disk JSON.
    for frame, cam2, _ in window:
        width, height = cam2.shape[1], cam2.shape[0]
        for match in matches[frame]["matches"]:
            rotate_erp_points(match["kps_cam2"], width, height)
            rotate_erp_points(match["kps_cam4"], width, height)
    wall_ms = (time.perf_counter() - started) * 1000.0
    after = dict(runtime.profile_timing)
    metrics = {
        "window_ms": wall_ms,
        "detector_total_ms": _delta(after, profile_before, "detector_inference_ms"),
        "detector_preprocess_ms": _delta(after, profile_before, "detector_preprocess_ms"),
        "detector_ort_ms": _delta(after, profile_before, "detector_ort_ms"),
        "detector_postprocess_ms": _delta(after, profile_before, "detector_postprocess_ms"),
        "densekp_crop_affine_ms": _delta(after, profile_before, "densekp_crop_resize_ms"),
        "densekp_normalize_ms": _delta(after, profile_before, "densekp_normalize_ms"),
        "densekp_pack_ms": _delta(after, profile_before, "densekp_tensor_prepare_ms"),
        "densekp_ort_ms": _delta(after, profile_before, "densekp_inference_ms"),
        "densekp_postprocess_ms": _delta(after, profile_before, "densekp_postprocess_ms"),
        "densekp_calls": int(round(_delta(after, profile_before, "densekp_ort_call_count"))),
        "densekp_items": int(round(_delta(after, profile_before, "densekp_ort_batch_item_count"))),
        **auxiliary,
    }
    metrics["persons"] = sum(len(pair[0].persons) + len(pair[1].persons)
                             for pair in pairs.values())
    return pairs, matches, metrics


def _benchmark(runtime, loaded, rois, masks, mode, window_size, batch_size,
               warmup, repeat):
    windows = [[loaded[(offset + i) % len(loaded)] for i in range(window_size)]
               for offset in range(len(loaded))]
    for index in range(warmup):
        _run_window(runtime, windows[index % len(windows)], rois, masks, mode, batch_size)
    rows = []
    for index in range(repeat):
        _, _, row = _run_window(
            runtime, windows[index % len(windows)], rois, masks, mode, batch_size)
        rows.append(row)
    fields = [key for key, value in rows[0].items() if isinstance(value, (int, float))]
    aggregate = {key: _stats([float(row[key]) / window_size for row in rows])
                 for key in fields if key not in ("densekp_calls", "densekp_items", "persons")}
    mean_ms = aggregate["window_ms"]["mean"]
    return {
        "mode": mode, "window_size": window_size,
        "densekp_batch_size": batch_size,
        "detector_batch_size": 1,
        "detector_calls_per_window": 4 * window_size,
        "detector_calls_per_frame": 4.0,
        "densekp_calls_per_window_mean": statistics.mean(r["densekp_calls"] for r in rows),
        "densekp_calls_per_frame_mean": statistics.mean(r["densekp_calls"] for r in rows) / window_size,
        "persons_per_window_mean": statistics.mean(r["persons"] for r in rows),
        "amortized": aggregate,
        "window_latency_ms": _stats([r["window_ms"] for r in rows]),
        "delivery_ms_per_frame": mean_ms,
        "fps": 1000.0 / mean_ms,
        "peak_gpu_memory": "UNAVAILABLE_ORTRUNTIME_NO_PEAK_ALLOCATOR_COUNTER",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--frames", default="2213-2221")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--onnx-path", type=Path, required=True)
    parser.add_argument("--erp-remap-cuda-library", type=Path, required=True)
    parser.add_argument("--calibration-cache-dir", type=Path, required=True)
    parser.add_argument("--camera", choices=("old", "new"), default="new")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeat", type=int, default=20)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    frame_ids = _frames(args.frames)
    template = DenseKPConfig(
        backend="ort", fisheye_cam2="", fisheye_cam4="", out_dir="",
        onnx_path=str(args.onnx_path), camera=args.camera,
        erp_remap_backend="cuda",
        erp_remap_cuda_library=str(args.erp_remap_cuda_library),
        calibration_cache_dir=str(args.calibration_cache_dir),
        output_coord_space="erp_rotate", vis_option="none", save_option="none",
        device="cuda", batch_size=1, profile_runtime=True,
    )
    runtime = DenseKPRuntime.from_config(template).load()
    loaded = []
    # Setup is intentionally outside benchmark: it creates static geometry and
    # preloads normal-ERP arrays from the real fisheye sequence.
    for frame in frame_ids:
        config = template.for_frame(
            args.dataset_root / f"image/fisheye/cam2/{frame:05d}.jpg",
            args.dataset_root / f"image/fisheye/cam4/{frame:05d}.jpg",
            args.output_dir / "setup" / f"frame_{frame:06d}")
        run_pipeline(config, runtime=runtime)
        cam2, cam4 = load_erp_pair(config, runtime)
        loaded.append((frame, cam2, cam4))
    entries = list(runtime._static_geometry.values())
    if len(entries) != 1:
        raise RuntimeError(f"expected one calibration entry, got {len(entries)}")
    entry = entries[0]
    masks = entry["valid_masks"]
    rois = entry["overlap_roi_component_bbox"][-1]

    baseline_pairs, baseline_matches, _ = _run_window(
        runtime, loaded, rois, masks, "baseline", 1)
    batch_pairs, batch_matches, _ = _run_window(
        runtime, loaded, rois, masks, "densekp_batch", 16)
    correctness = {}
    for frame in frame_ids:
        compare = _compare_pairs(baseline_pairs[frame], batch_pairs[frame])
        left, right = baseline_matches[frame], batch_matches[frame]
        compare.update({
            "baseline_matches": len(left["matches"]),
            "batch_matches": len(right["matches"]),
            "baseline_counts": [left["cam2_person_count"], left["cam4_person_count"]],
            "batch_counts": [right["cam2_person_count"], right["cam4_person_count"]],
            "baseline_valid": [m["valid_count"] for m in left["matches"]],
            "batch_valid": [m["valid_count"] for m in right["matches"]],
            "baseline_median_angle": [m["median_angular_error_deg"] for m in left["matches"]],
            "batch_median_angle": [m["median_angular_error_deg"] for m in right["matches"]],
        })
        correctness[str(frame)] = compare
    (args.output_dir / "correctness_t4.json").write_text(
        json.dumps({key: correctness[key] for key in map(str, frame_ids[:4])}, indent=2) + "\n")

    benchmark = {
        "schema_version": "densekp_phase3_batch_bakeoff_v1",
        "correctness_sequence": frame_ids,
        "performance_workload": "preloaded real frames repeated cyclically",
        "startup_excluded": True, "image_load_excluded": True,
        "erp_remap_excluded": True,
        "onnx_batch_capability": {
            "detector": {"input_shape": [1, 3, 640, 640], "dynamic": False,
                         "available_batch_sizes": [1],
                         "batch_status": "UNAVAILABLE_REEXPORT_REQUIRED"},
            "densekp": {"input_shape": ["batch", 3, 256, 256], "dynamic": True,
                        "tested_batch_sizes": [1, 2, 4, 8, 16, 32]},
        },
        "correctness": correctness,
        "modes": {}, "densekp_batch_size_bakeoff": {},
    }
    for size in (1, 2, 4, 8, 16, 32):
        benchmark["densekp_batch_size_bakeoff"][str(size)] = _benchmark(
            runtime, loaded, rois, masks, "densekp_batch", 8, size,
            args.warmup, args.repeat)
    recommended_batch_size = int(min(
        benchmark["densekp_batch_size_bakeoff"].items(),
        key=lambda item: item[1]["delivery_ms_per_frame"],
    )[0])
    for window_size in (1, 2, 4, 8):
        benchmark["modes"][f"T{window_size}"] = {
            "baseline": _benchmark(runtime, loaded, rois, masks, "baseline",
                                   window_size, 1, args.warmup, args.repeat),
            "detector_batch": {"status": "UNAVAILABLE_REEXPORT_REQUIRED"},
            "densekp_batch": _benchmark(
                runtime, loaded, rois, masks, "densekp_batch",
                window_size, recommended_batch_size, args.warmup, args.repeat),
            "both_batch": {"status": "UNAVAILABLE_REEXPORT_REQUIRED"},
        }
    candidates = [row["densekp_batch"] for row in benchmark["modes"].values()]
    best = min(candidates, key=lambda row: row["delivery_ms_per_frame"])
    benchmark["best"] = best
    benchmark["recommended_densekp_batch_size"] = recommended_batch_size
    benchmark["decision"] = {
        "DELIVERY_BATCH_GATE1": "PASS" if best["delivery_ms_per_frame"] < 250 else "FAIL",
        "DELIVERY_BATCH_GATE2": "PASS" if best["delivery_ms_per_frame"] < 150 else "FAIL",
        "DELIVERY_10FPS_GATE": "PASS" if best["delivery_ms_per_frame"] <= 100 else "FAIL",
    }
    (args.output_dir / "benchmark.json").write_text(json.dumps(benchmark, indent=2) + "\n")
    runtime.close()
    print(json.dumps({"best": best, "decision": benchmark["decision"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
