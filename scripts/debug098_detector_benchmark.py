#!/usr/bin/env python3
"""debug098 detector benchmark worker (ONE backend per process).

Protocol (per spec):
    warmup = 20, runs = 100, isolated process, model load + first ORT session
    build NOT counted, no image/json/per-run logging inside the loop, each
    timed iteration = detector forward + postprocess only.

Writes: <out_root>/benchmark_<backend>/result.json

This measures DETECTOR-ONLY latency. It is NOT the end-to-end DenseKP latency.

Run (separately for each backend, in its own process):
    python scripts/debug098_detector_benchmark.py --backend rtmlib_yolox
    python scripts/debug098_detector_benchmark.py --backend detectron2_vitdet
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
TOOL_ROOT = REPO_ROOT / "tools" / "CameraHMR"
DEFAULT_IMAGE = (REPO_ROOT / "data_process/Output/20260311_riding/"
                 "image/equirect_crop_rotate_upright/cam2/00513.jpg")
DEFAULT_OUT = REPO_ROOT / "artifacts" / "densekp_detector_bakeoff"


def _stats(ms):
    a = np.asarray(ms, dtype=np.float64)
    return {
        "mean": float(a.mean()), "p50": float(np.percentile(a, 50)),
        "p95": float(np.percentile(a, 95)), "min": float(a.min()),
        "max": float(a.max()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", required=True,
                    choices=["detectron2_vitdet", "rtmlib_yolox"])
    ap.add_argument("--image", type=str, default=str(DEFAULT_IMAGE))
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--runs", type=int, default=100)
    ap.add_argument("--threshold", type=float, default=0.25)
    ap.add_argument("--out_root", type=str, default=str(DEFAULT_OUT))
    args = ap.parse_args()

    sys.path.insert(0, str(TOOL_ROOT))
    os.chdir(str(TOOL_ROOT))

    import cv2
    import torch
    from detectors import build_person_detector

    img = cv2.imread(args.image)
    if img is None:
        raise FileNotFoundError(args.image)
    h, w = img.shape[:2]

    # --- build detector (NOT timed) ---
    detector = build_person_detector(backend=args.backend, threshold=args.threshold)
    md = detector.metadata

    # --- raw vs valid detection counts (probed once, outside timing) ---
    raw_rows = None
    if args.backend == "rtmlib_yolox":
        padded, ratio = detector._yolox.preprocess(img)
        outs = detector._yolox.inference(padded)
        raw_rows = int(np.asarray(outs[0]).shape[1])  # model top-k rows (incl. padding)
    else:
        det_out = detector._predictor(img)
        raw_rows = int(len(det_out["instances"]))      # all classes, pre person-filter

    valid_person = len(detector.detect(img))
    sel_bbox = None
    dets0 = detector.detect(img)
    if dets0:
        sel_bbox = [float(x) for x in dets0[0]["bbox_xyxy"]]  # top-score target

    use_cuda_event = (args.backend == "detectron2_vitdet") and torch.cuda.is_available()

    # --- warmup (NOT timed) ---
    for _ in range(args.warmup):
        detector.detect(img)
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    # --- timed loop: detector forward + postprocess only ---
    wall_ms = []
    cuda_ms = []
    for _ in range(args.runs):
        if use_cuda_event:
            ev0 = torch.cuda.Event(enable_timing=True)
            ev1 = torch.cuda.Event(enable_timing=True)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        import time
        if use_cuda_event:
            ev0.record()
        t0 = time.perf_counter()
        detector.detect(img)
        t1 = time.perf_counter()
        if use_cuda_event:
            ev1.record()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        wall_ms.append((t1 - t0) * 1000.0)
        if use_cuda_event:
            cuda_ms.append(ev0.elapsed_time(ev1))

    wall = _stats(wall_ms)
    fps = 1000.0 / wall["mean"] if wall["mean"] > 0 else 0.0

    result = {
        "backend": args.backend,
        "warmup": args.warmup,
        "runs": args.runs,
        "measurement": "detector-only (forward + postprocess); device-synced wall",
        "wall_ms": wall,
        "cuda_ms": _stats(cuda_ms) if cuda_ms else None,
        "cuda_ms_note": (
            "torch CUDA events (detectron2/torch backend)"
            if cuda_ms else
            "n/a for ONNXRuntime session; wall is device-synced and includes GPU"
        ),
        "fps": fps,
        "gpu_provider": md.get("selected_provider", "torch.cuda"),
        "image": os.path.relpath(args.image, REPO_ROOT),
        "image_size_hw": [h, w],
        "detector_input_size": md.get("input_size", "native(ResizeShortestEdge)"),
        "num_raw_detections": raw_rows,
        "num_valid_person_detections": valid_person,
        "selected_target_bbox_xyxy": sel_bbox,
        "metadata": md,
    }

    out_dir = Path(args.out_root) / f"benchmark_{args.backend}"
    out_dir.mkdir(parents=True, exist_ok=True)
    json.dump(result, open(out_dir / "result.json", "w"), indent=2)
    print(f"[bench:{args.backend}] wall mean={wall['mean']:.3f}ms p50={wall['p50']:.3f} "
          f"p95={wall['p95']:.3f} fps={fps:.1f} provider={result['gpu_provider']} "
          f"raw={raw_rows} valid={valid_person}")
    print(f"  -> {out_dir/'result.json'}")


if __name__ == "__main__":
    main()
