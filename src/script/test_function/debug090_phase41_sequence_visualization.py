"""Render Phase 4.1 sequence timeline: guided crops, identity gate, keypoints.

Re-runs the production orchestrator on an ERP sequence and saves one
side-by-side (cam2 | cam4) image per frame showing:

- mode label (FULL_DETECTION / TRACK_GUIDED / RECOVERY)
- yellow detection ROIs on FULL frames
- orange/cyan guided crop boxes per track with identity status
- green fitted keypoints (ERP_ROTATE pairs drawn back in normal ERP)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", required=True)
    parser.add_argument("--erp-root", type=Path, required=True)
    parser.add_argument("--valid-masks", type=Path, required=True)
    parser.add_argument("--onnx-path", type=Path, required=True)
    parser.add_argument("--hmr2-checkpoint", type=Path, required=True)
    parser.add_argument("--base-fixture", type=Path, required=True)
    parser.add_argument("--cpp-executable", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--detect-interval", type=int, default=3)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--cpp-backend", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--scale", type=float, default=0.25)
    return parser.parse_args()


def _frame_list(value: str) -> list[int]:
    frames: list[int] = []
    for part in value.split(","):
        if "-" in part:
            first, last = (int(item) for item in part.split("-", 1))
            frames.extend(range(first, last + 1))
        else:
            frames.append(int(part))
    return sorted(dict.fromkeys(frames))


def _draw_box(image, bbox, color, label: str, scale: float, erp_width: int):
    x0, y0, x1, y1 = bbox
    full = (x1 % erp_width) < (x0 % erp_width) or x1 > erp_width
    for start in ((x0, x1) if full else (x0,)):
        p0 = (int(start * scale), int(y0 * scale))
        p1 = (min(erp_width, int(x1 * scale)) if start == x0
              else int((x1 - erp_width) * scale), int(y1 * scale))
        cv2.rectangle(image, p0, p1, color, 2)
        cv2.putText(image, label, (p0[0], max(12, p0[1] - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)


def _keypoints_erp(pairs, valid, width: int, height: int):
    from densekp_delivery.camera.erp_rotate import (
        DEFAULT_ERP_TO_ERP_ROTATE_R, rotate_erp_points)

    pairs = np.asarray(pairs, np.float32)
    valid = np.asarray(valid, bool)
    points = np.concatenate((pairs[valid][:, :2], pairs[valid][:, 2:]), axis=0)
    return rotate_erp_points(points, width, height,
                             R=DEFAULT_ERP_TO_ERP_ROTATE_R.T)


def main() -> int:
    from lib_pano.delivery_bootstrap import ensure_importable
    from lib_pano.fitting_io.densekp_delivery_boundary import load_erp_frame_pair
    from lib_pano.human3d.hmr2_initializer import HMR2Initializer
    from lib_pano.human_fitting.per_person_cpp import (
        CppPerPersonFitter, CppPersonFitterConfig)
    from lib_pano.human_fitting.sequence_orchestrator import (
        ErpHmr2ColdStartBatch, TrackGuidedSequenceConfig,
        TrackGuidedSequenceOrchestrator)

    args = parse_args()
    ensure_importable()
    from densekp_delivery.config import DenseKPConfig
    from densekp_delivery.runtime import DenseKPRuntime

    frames = _frame_list(args.frames)
    masks = dict(np.load(args.valid_masks).items())
    valid_masks = (masks["valid_masks_cam2"], masks["valid_masks_cam4"])
    height, width = np.asarray(valid_masks[0]).shape[:2]
    template = DenseKPConfig(
        backend="ort", fisheye_cam2="", fisheye_cam4="", out_dir="",
        onnx_path=str(args.onnx_path), device=args.device,
        batch_size=1, profile_runtime=True)
    runtime = DenseKPRuntime.from_config(template).load()
    hmr = HMR2Initializer(args.hmr2_checkpoint, device=args.device)
    cold_start = ErpHmr2ColdStartBatch(hmr, width, height)
    fitter = CppPerPersonFitter(CppPersonFitterConfig(
        args.base_fixture, args.cpp_executable,
        args.output_dir / "per_person_cpp", args.cpp_backend))
    orchestrator = TrackGuidedSequenceOrchestrator(
        runtime, cold_start, fitter, valid_masks,
        TrackGuidedSequenceConfig(detection_interval=args.detect_interval),
        sequence_start_frame=frames[0])

    vis_dir = args.output_dir / "visualization"
    vis_dir.mkdir(parents=True, exist_ok=True)
    mode_colors = {"FULL_DETECTION": (80, 200, 80), "TRACK_GUIDED": (0, 140, 255),
                   "RECOVERY": (0, 0, 255)}
    colors = {0: (0, 170, 255), 1: (255, 170, 0), 2: (200, 0, 200)}

    for frame in frames:
        erp2 = cv2.imread(str(args.erp_root / "cam2" / f"{frame:05d}.png"))
        erp4 = cv2.imread(str(args.erp_root / "cam4" / f"{frame:05d}.png"))
        record = orchestrator.process_frame(frame, erp2, erp4)
        info = record.delivery_info
        panels = []
        for camera, image in (("cam2", erp2), ("cam4", erp4)):
            canvas = cv2.resize(image, None, fx=args.scale, fy=args.scale,
                                interpolation=cv2.INTER_AREA)
            for index, roi in enumerate(orchestrator.detection_rois):
                _draw_box(canvas, (roi.x_start, roi.y_start,
                                   roi.x_start + roi.width,
                                   roi.y_start + roi.height),
                          (180, 180, 180), f"roi{index}", args.scale, width)
            for crop in info.get("guided_crops", []):
                if crop["camera"] != camera:
                    continue
                color = colors.get(crop["track_id"], (255, 255, 255))
                _draw_box(canvas, crop["predicted_bbox_erp"], color,
                          f"t{crop['track_id']}:{crop['status'][:12]}",
                          args.scale, width)
            for person in record.payload["persons"]:
                points = _keypoints_erp(person["optimized_pairs"],
                                        person["valid"], width, height)
                for x, y in points:
                    cv2.circle(canvas, (int(x * args.scale), int(y * args.scale)),
                               2, (0, 255, 0), -1, cv2.LINE_AA)
            cv2.putText(canvas, f"cam{camera[-1]}", (8, 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2,
                        cv2.LINE_AA)
            panels.append(canvas)
        banner = np.zeros((34, panels[0].shape[1] * 2, 3), np.uint8)
        text = (f"frame {frame}  {record.actual_mode.value}  "
                f"detector={record.detector_calls} densekp={record.densekp_calls} "
                f"persons={len(record.payload['persons'])}")
        cv2.putText(banner, text, (8, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    mode_colors.get(record.actual_mode.value, (255, 255, 255)),
                    2, cv2.LINE_AA)
        side = np.concatenate((banner, np.concatenate((panels[0], panels[1]),
                                                      axis=1)), axis=0)
        cv2.imwrite(str(vis_dir / f"frame_{frame:05d}.jpg"), side,
                    [cv2.IMWRITE_JPEG_QUALITY, 92])
        print(f"rendered {frame} {record.actual_mode.value}", flush=True)

    (args.output_dir / "sequence_result.json").write_text(
        json.dumps(orchestrator.sequence_result({}), indent=2) + "\n")
    runtime.close()
    print("PHASE41_SEQUENCE_VISUALIZATION=PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
