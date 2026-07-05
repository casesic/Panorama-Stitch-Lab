#!/usr/bin/env python3
"""End-to-end DenseKP per-frame benchmark (detector backend selectable).

Measures the FULL production per-image path (as in demo_keypoints.process_image):
    detector.detect -> bbox->Dataset crop/preprocess -> DenseKP forward -> npz write

Reports per-stage ms + total ms + end-to-end FPS. Excludes model load and the
first ORT session build. Stages other than the detector are detector-independent,
so an e2e estimate for the other backend (using its measured detector time) is
also printed.

Run:
    python scripts/debug098_densekp_e2e_benchmark.py --detector_backend rtmlib_yolox
"""

import argparse, json, os, sys, time
from argparse import Namespace
from pathlib import Path
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
TOOL_ROOT = REPO_ROOT / "tools" / "CameraHMR"
DEFAULT_IMAGE = (REPO_ROOT / "data_process/Output/20260311_riding/"
                 "image/equirect_crop_rotate_upright/cam2/00513.jpg")
OUT = REPO_ROOT / "artifacts" / "densekp_detector_bakeoff"


def _stats(ms):
    a = np.asarray(ms, np.float64)
    return {"mean": float(a.mean()), "p50": float(np.percentile(a, 50)),
            "p95": float(np.percentile(a, 95))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--detector_backend", default="rtmlib_yolox",
                    choices=["detectron2_vitdet", "rtmlib_yolox"])
    ap.add_argument("--image", default=str(DEFAULT_IMAGE))
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--runs", type=int, default=100)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--threshold", type=float, default=0.25)
    args = ap.parse_args()

    sys.path.insert(0, str(TOOL_ROOT)); os.chdir(str(TOOL_ROOT))
    import cv2, torch
    from core.densekp_trainer import DenseKP
    from core.constants import DENSEKP_CKPT, IMAGE_SIZE
    from core.datasets.dataset import Dataset
    from core.utils import recursive_to
    from demo_keypoints import save_densekp_npz
    from detectors import build_person_detector

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    img = cv2.imread(args.image); h, w = img.shape[:2]
    model = DenseKP.load_from_checkpoint(DENSEKP_CKPT, strict=False).to(dev).eval()
    detector = build_person_detector(backend=args.detector_backend, threshold=args.threshold)

    tmp_npz = OUT / "_e2e_tmp"; tmp_npz.mkdir(parents=True, exist_ok=True)

    def sync():
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    def one_iter(record):
        sync(); t0 = time.perf_counter()
        dets = detector.detect(img)
        sync(); t1 = time.perf_counter()
        boxes = np.stack([d["bbox_xyxy"] for d in dets]).astype(np.float32)
        bbox_scale = (boxes[:, 2:4] - boxes[:, 0:2]) / 200.0
        bbox_center = (boxes[:, 2:4] + boxes[:, 0:2]) / 2.0
        ds = Dataset(img, bbox_center, bbox_scale, None, False, args.image)
        dl = torch.utils.data.DataLoader(ds, batch_size=args.batch_size,
                                         shuffle=False, num_workers=0)
        batch = next(iter(dl))
        batch = recursive_to(batch, dev)
        sync(); t2 = time.perf_counter()
        with torch.no_grad():
            out = model(batch)
        pred = out["pred_keypoints"]
        sync(); t3 = time.perf_counter()
        pk = pred[0].detach().cpu().numpy()
        save_densekp_npz(tmp_npz / "bench.npz", pk, boxes[0],
                         batch["box_center"][0].detach().cpu().numpy(),
                         float(batch["box_size"][0].detach().cpu().numpy()), (h, w))
        sync(); t4 = time.perf_counter()
        if record is not None:
            record["detector"].append((t1 - t0) * 1000)
            record["crop_preprocess"].append((t2 - t1) * 1000)
            record["forward"].append((t3 - t2) * 1000)
            record["restore_npz"].append((t4 - t3) * 1000)
            record["total"].append((t4 - t0) * 1000)

    for _ in range(args.warmup):
        one_iter(None)
    rec = {k: [] for k in ["detector", "crop_preprocess", "forward", "restore_npz", "total"]}
    for _ in range(args.runs):
        one_iter(rec)

    stages = {k: _stats(v) for k, v in rec.items()}
    total_mean = stages["total"]["mean"]
    fps = 1000.0 / total_mean

    # detector-independent (shared) stage cost
    shared_mean = total_mean - stages["detector"]["mean"]
    # cross-backend e2e estimate using measured detector means from benchmark files
    est = {}
    for bk in ["detectron2_vitdet", "rtmlib_yolox"]:
        f = OUT / f"benchmark_{bk}" / "result.json"
        if f.exists():
            dmean = json.load(open(f))["wall_ms"]["mean"]
            est[bk] = {"detector_ms": round(dmean, 2),
                       "e2e_total_ms_est": round(shared_mean + dmean, 2),
                       "e2e_fps_est": round(1000.0 / (shared_mean + dmean), 2)}

    result = {
        "detector_backend": args.detector_backend,
        "image": os.path.relpath(args.image, REPO_ROOT), "image_size_hw": [h, w],
        "warmup": args.warmup, "runs": args.runs, "batch_size": args.batch_size,
        "densekp_network": "PyTorch CUDA (demo_keypoints default)",
        "per_stage_ms": stages,
        "end_to_end_total_ms": round(total_mean, 2),
        "end_to_end_fps": round(fps, 2),
        "shared_non_detector_ms": round(shared_mean, 2),
        "cross_backend_e2e_estimate": est,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(result, open(OUT / f"densekp_e2e_{args.detector_backend}.json", "w"), indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
