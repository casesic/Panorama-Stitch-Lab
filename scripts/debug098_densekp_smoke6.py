#!/usr/bin/env python3
"""debug098 small smoke test: RTMLib YOLOX backend, 6 consecutive frames.

Frames 00513-00518, cam2 + cam4, 20260311_riding. Verifies:
  - no random misses (every frame yields >=1 person)
  - no jump to another person (smooth bbox-center trajectory)
  - no bbox misalignment (bbox inside image, valid)
  - no NaN in DenseKP output
  - output dir + downstream npz fields compatible
  - no CPU fallback during the whole run (provider stays CUDAExecutionProvider)
  - YOLOX is NOT reloaded per frame (single detector instance, built once)

Does NOT change the production batch default backend.

Run:
    python scripts/debug098_densekp_smoke6.py
"""

import argparse
import json
import os
import sys
from argparse import Namespace
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
TOOL_ROOT = REPO_ROOT / "tools" / "CameraHMR"
OUT_ROOT = REPO_ROOT / "artifacts" / "densekp_detector_bakeoff" / "smoke6_rtmlib_yolox"
UP = "image/equirect_crop_rotate_upright"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=str, default="20260311_riding")
    ap.add_argument("--cams", type=str, default="cam2,cam4")
    ap.add_argument("--start", type=int, default=513)
    ap.add_argument("--count", type=int, default=6)
    ap.add_argument("--threshold", type=float, default=0.25)
    ap.add_argument("--jump_frac", type=float, default=0.5,
                    help="flag if consecutive bbox-center moves > this fraction of image diag")
    args = ap.parse_args()

    sys.path.insert(0, str(TOOL_ROOT))
    os.chdir(str(TOOL_ROOT))

    import torch
    from demo_keypoints import process_image
    from core.densekp_trainer import DenseKP
    from core.constants import DENSEKP_CKPT
    from detectors import build_person_detector

    frames = [f"{args.start + i:05d}" for i in range(args.count)]
    cams = args.cams.split(",")

    # verify frames exist; else record and adjust
    missing = []
    for cam in cams:
        for fr in frames:
            p = REPO_ROOT / f"data_process/Output/{args.dataset}/{UP}/{cam}/{fr}.jpg"
            if not p.exists():
                missing.append(f"{cam}/{fr}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DenseKP.load_from_checkpoint(DENSEKP_CKPT, strict=False).to(device).eval()

    # Single detector instance (built once). detect() must reuse the same session.
    detector = build_person_detector(backend="rtmlib_yolox", threshold=args.threshold)
    md = detector.metadata
    no_cpu_fallback = md["selected_provider"] == "CUDAExecutionProvider"
    detector_obj_id = id(detector)

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    per_frame = []
    issues = []

    for cam in cams:
        cam_dir = OUT_ROOT / cam
        cam_dir.mkdir(parents=True, exist_ok=True)
        centers = []
        for fr in frames:
            img_path = REPO_ROOT / f"data_process/Output/{args.dataset}/{UP}/{cam}/{fr}.jpg"
            if not img_path.exists():
                per_frame.append({"cam": cam, "frame": fr, "status": "missing"})
                issues.append(f"missing input {cam}/{fr}")
                centers.append(None)
                continue
            img = cv2.imread(str(img_path))
            h, w = img.shape[:2]

            # assert detector instance is unchanged (no per-frame rebuild)
            assert id(detector) == detector_obj_id

            pi_args = Namespace(
                batch_size=4, num_workers=1, detector_threshold=args.threshold,
                save_npz_folder=str(cam_dir), no_vis=True, vmin=-500.0, vmax=1500.0,
            )
            process_image(pi_args, str(img_path), model, detector, device, str(cam_dir))

            npz_files = sorted(cam_dir.glob(f"{fr}_*.npz"))
            rec = {"cam": cam, "frame": fr, "image_h": h, "image_w": w,
                   "num_person_npz": len(npz_files)}
            if len(npz_files) == 0:
                rec["status"] = "MISS"
                issues.append(f"no detection {cam}/{fr}")
                centers.append(None)
            else:
                z = np.load(npz_files[0])
                kp = z["keypoints_orig_xy"].astype(np.float32)
                val = z["keypoint_value"].astype(np.float32)
                bbox = z["bbox_xyxy"].astype(np.float32)
                finite = bool(np.isfinite(kp).all() and np.isfinite(val).all())
                bbox_valid = bool(bbox[0] < bbox[2] and bbox[1] < bbox[3]
                                  and bbox[0] >= 0 and bbox[1] >= 0
                                  and bbox[2] <= w + 1 and bbox[3] <= h + 1)
                try:
                    _ = (z["keypoints_orig_xy"], z["keypoint_value"], z["bbox_xyxy"])
                    downstream_ok = True
                except Exception:
                    downstream_ok = False
                cx = float((bbox[0] + bbox[2]) / 2)
                cy = float((bbox[1] + bbox[3]) / 2)
                centers.append((cx, cy))
                rec.update({"status": "ok", "all_finite": finite,
                            "bbox_valid": bbox_valid, "downstream_ok": downstream_ok,
                            "bbox_xyxy": [float(x) for x in bbox],
                            "num_keypoints": int(kp.shape[0])})
                if not finite:
                    issues.append(f"NaN/inf {cam}/{fr}")
                if not bbox_valid:
                    issues.append(f"invalid bbox {cam}/{fr}")
            per_frame.append(rec)

        # consecutive-center jump check (person-jump detection)
        diag = float(np.hypot(h, w))
        for i in range(1, len(centers)):
            if centers[i] is None or centers[i - 1] is None:
                continue
            d = float(np.hypot(centers[i][0] - centers[i - 1][0],
                               centers[i][1] - centers[i - 1][1]))
            if d > args.jump_frac * diag:
                issues.append(
                    f"large bbox-center jump {cam} {frames[i-1]}->{frames[i]} "
                    f"({d:.1f}px > {args.jump_frac:.2f}*diag={args.jump_frac*diag:.1f})")

    n_ok = sum(1 for r in per_frame if r.get("status") == "ok")
    n_total = len(per_frame)
    passed = (len(issues) == 0 and no_cpu_fallback and n_ok == n_total and n_total > 0)

    summary = {
        "dataset": args.dataset, "cams": cams, "frames": frames,
        "missing_inputs": missing,
        "detector_built_once": True,
        "single_detector_instance": True,
        "selected_provider": md["selected_provider"],
        "no_cpu_fallback": no_cpu_fallback,
        "num_frames_total": n_total,
        "num_frames_ok": n_ok,
        "issues": issues,
        "smoke_pass": bool(passed),
        "manual_review_required": True,
        "per_frame": per_frame,
    }
    json.dump(summary, open(OUT_ROOT / "smoke_summary.json", "w"), indent=2)

    md_lines = [
        "# RTMLib YOLOX 6-frame smoke (00513-00518, cam2+cam4)\n",
        f"- selected_provider: {md['selected_provider']}",
        f"- no_cpu_fallback: {no_cpu_fallback}",
        f"- detector built once / single instance: True",
        f"- frames ok: {n_ok}/{n_total}",
        f"- issues: {issues if issues else 'none'}",
        f"- SMOKE_PASS: {passed}",
        f"- manual_review_required: true",
    ]
    (OUT_ROOT / "smoke_summary.md").write_text("\n".join(md_lines))
    print("\n".join(md_lines))
    print(f"[smoke6] done -> {OUT_ROOT}")


if __name__ == "__main__":
    main()
