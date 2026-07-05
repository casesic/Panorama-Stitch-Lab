#!/usr/bin/env python3
"""debug098 DenseKP single-frame end-to-end compatibility (00513 cam2/cam4).

Runs the REAL production `demo_keypoints.process_image` so the DenseKP network,
weights, crop/resize/normalize, batch size and NPZ save format are identical;
the detector backend is the ONLY variable.

For each (cam, backend) it writes:
    artifacts/densekp_detector_bakeoff/densekp_single_frame/00513_<cam>_<backend>/
        <stem>_0.npz            (densekp_original_output, untouched save format)
        detector_metadata.json
        bbox_overlay.jpg
        densekp_overlay.jpg
Plus per-cam compare_00513_<cam>.jpg and compatibility_summary.md.

Quality is NOT judged here: only `manual_review_required = true`.

Run:
    python scripts/debug098_densekp_single_frame.py
"""

import argparse
import json
import os
import shutil
import sys
from argparse import Namespace
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
TOOL_ROOT = REPO_ROOT / "tools" / "CameraHMR"
OUT_ROOT = REPO_ROOT / "artifacts" / "densekp_detector_bakeoff" / "densekp_single_frame"
UP = "image/equirect_crop_rotate_upright"

EXPECTED_NPZ = {
    "keypoints_crop_xy": ((138, 2), np.float32),
    "keypoints_orig_xy": ((138, 2), np.float32),
    "keypoint_value": ((138,), np.float32),
    "bbox_xyxy": ((4,), np.float32),
    "bbox_center": ((2,), np.float32),
    "box_size": ((), np.float32),
    "image_size": ((2,), np.int32),
    "crop_trans": ((2, 3), np.float32),
    "crop_inv_trans": ((2, 3), np.float32),
}


def overlay_keypoints(img, kpts_xy, values):
    vis = img.copy()
    v = np.asarray(values, np.float32)
    lo, hi = float(np.nanmin(v)), float(np.nanmax(v))
    rng = (hi - lo) if hi > lo else 1.0
    for (x, y), val in zip(kpts_xy, v):
        if not (np.isfinite(x) and np.isfinite(y)):
            continue
        t = (float(val) - lo) / rng
        color = (int(255 * (1 - t)), int(120), int(255 * t))  # blue->red by value
        cv2.circle(vis, (int(round(x)), int(round(y))), 2, color, -1, cv2.LINE_AA)
    return vis


def overlay_bbox(img, bbox_xyxy):
    vis = img.copy()
    x1, y1, x2, y2 = [int(round(float(v))) for v in bbox_xyxy]
    cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 200, 0), 2)
    return vis


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frame", type=str, default="00513")
    ap.add_argument("--cams", type=str, default="cam2,cam4")
    ap.add_argument("--dataset", type=str, default="20260311_riding")
    ap.add_argument("--threshold", type=float, default=0.25)
    ap.add_argument("--batch_size", type=int, default=4)
    args_cli = ap.parse_args()

    sys.path.insert(0, str(TOOL_ROOT))
    os.chdir(str(TOOL_ROOT))

    import torch
    from demo_keypoints import process_image
    from core.densekp_trainer import DenseKP
    from core.constants import DENSEKP_CKPT
    from detectors import build_person_detector

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[single_frame] loading DenseKP network (shared, identical weights)...")
    model = DenseKP.load_from_checkpoint(DENSEKP_CKPT, strict=False).to(device).eval()

    print("[single_frame] building detectors...")
    detectors = {
        "vitdet": build_person_detector(backend="detectron2_vitdet", threshold=args_cli.threshold),
        "rtmlib_yolox": build_person_detector(backend="rtmlib_yolox", threshold=args_cli.threshold),
    }

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    cams = args_cli.cams.split(",")
    frame = args_cli.frame
    results = {}

    for cam in cams:
        img_path = REPO_ROOT / f"data_process/Output/{args_cli.dataset}/{UP}/{cam}/{frame}.jpg"
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"  WARN: missing {img_path}")
            continue
        h, w = img.shape[:2]
        legacy_npz = (REPO_ROOT / f"data_process/Output/{args_cli.dataset}/"
                      f"camerahmr/densekp_npz/{cam}/{frame}_0.npz")

        for backend, detector in detectors.items():
            tag = f"{frame}_{cam}_{backend}"
            res_dir = OUT_ROOT / tag
            res_dir.mkdir(parents=True, exist_ok=True)

            # Build a production-equivalent args object; detector is the only variable.
            pi_args = Namespace(
                batch_size=args_cli.batch_size, num_workers=1,
                detector_threshold=args_cli.threshold,
                save_npz_folder=str(res_dir), no_vis=True,
                vmin=-500.0, vmax=1500.0,
            )
            process_image(pi_args, str(img_path), model, detector, device, str(res_dir))

            npz_files = sorted(res_dir.glob(f"{frame}_*.npz"))
            check = {
                "backend": backend, "cam": cam, "image_path": str(img_path.relative_to(REPO_ROOT)),
                "image_h": h, "image_w": w,
                "person_detected": len(npz_files) > 0,
                "num_person_npz": len(npz_files),
                "only_target_person": len(npz_files) == 1,
            }

            json.dump(detector.metadata, open(res_dir / "detector_metadata.json", "w"), indent=2)

            if npz_files:
                z = np.load(npz_files[0])
                keys = set(z.files)
                fields_ok = True
                field_report = {}
                for k, (shp, dt) in EXPECTED_NPZ.items():
                    present = k in keys
                    arr = z[k] if present else None
                    shp_ok = present and tuple(arr.shape) == shp
                    dt_ok = present and arr.dtype == dt
                    field_report[k] = {"present": present,
                                       "shape": (list(arr.shape) if present else None),
                                       "shape_ok": bool(shp_ok),
                                       "dtype": (str(arr.dtype) if present else None),
                                       "dtype_ok": bool(dt_ok)}
                    fields_ok = fields_ok and present and shp_ok and dt_ok

                kp = z["keypoints_orig_xy"].astype(np.float32)
                kpc = z["keypoints_crop_xy"].astype(np.float32)
                val = z["keypoint_value"].astype(np.float32)
                finite = bool(np.isfinite(kp).all() and np.isfinite(kpc).all()
                              and np.isfinite(val).all())
                in_x = (kp[:, 0] >= 0) & (kp[:, 0] <= w)
                in_y = (kp[:, 1] >= 0) & (kp[:, 1] <= h)
                in_range_frac = float((in_x & in_y).mean())

                # downstream load test (mirror lib_pano densekp_io field access)
                try:
                    _ = z["keypoints_orig_xy"].astype(np.float32)
                    _ = z["keypoint_value"].astype(np.float32)
                    _ = z["bbox_xyxy"].astype(np.float32)
                    downstream_loadable = True
                except Exception as e:
                    downstream_loadable = False
                    check["downstream_error"] = str(e)

                check.update({
                    "num_keypoints": int(kp.shape[0]),
                    "num_keypoints_is_138": int(kp.shape[0]) == 138,
                    "all_finite": finite,
                    "orig_xy_in_image_fraction": round(in_range_frac, 4),
                    "npz_fields_shape_dtype_ok": bool(fields_ok),
                    "npz_field_report": field_report,
                    "downstream_loadable": downstream_loadable,
                    "bbox_xyxy": [float(x) for x in z["bbox_xyxy"]],
                })

                # overlays
                cv2.imwrite(str(res_dir / "bbox_overlay.jpg"),
                            overlay_bbox(img, z["bbox_xyxy"]),
                            [int(cv2.IMWRITE_JPEG_QUALITY), 95])
                cv2.imwrite(str(res_dir / "densekp_overlay.jpg"),
                            overlay_keypoints(img, kp, val),
                            [int(cv2.IMWRITE_JPEG_QUALITY), 95])

                # vitdet parity vs legacy stored npz (non-detector path unchanged)
                if backend == "vitdet" and legacy_npz.exists():
                    lz = np.load(legacy_npz)
                    parity = {}
                    identical_all = True
                    for k in EXPECTED_NPZ:
                        if k in lz.files and k in z.files:
                            same = bool(np.array_equal(z[k], lz[k]))
                            parity[k] = same
                            identical_all = identical_all and same
                    check["vitdet_vs_legacy_npz_identical"] = identical_all
                    check["vitdet_vs_legacy_field_parity"] = parity

            results[tag] = check
            print(f"  {tag}: persons={check['num_person_npz']} "
                  f"finite={check.get('all_finite')} "
                  f"fields_ok={check.get('npz_fields_shape_dtype_ok')} "
                  f"kp138={check.get('num_keypoints_is_138')}")

        # per-cam compare image (densekp overlays side by side)
        a = OUT_ROOT / f"{frame}_{cam}_vitdet" / "densekp_overlay.jpg"
        b = OUT_ROOT / f"{frame}_{cam}_rtmlib_yolox" / "densekp_overlay.jpg"
        if a.exists() and b.exists():
            ia, ib = cv2.imread(str(a)), cv2.imread(str(b))
            hh = max(ia.shape[0], ib.shape[0])
            def pad(im):
                if im.shape[0] < hh:
                    im = cv2.copyMakeBorder(im, 0, hh - im.shape[0], 0, 0,
                                            cv2.BORDER_CONSTANT, value=(0, 0, 0))
                cv2.putText(im, "", (0, 0), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 1)
                return im
            ia, ib = pad(ia), pad(ib)
            cv2.putText(ia, "vitdet", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(ib, "rtmlib_yolox", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 128, 255), 2)
            cv2.imwrite(str(OUT_ROOT / f"compare_{frame}_{cam}.jpg"),
                        cv2.hconcat([ia, ib]), [int(cv2.IMWRITE_JPEG_QUALITY), 95])

    # summary
    lines = ["# DenseKP single-frame compatibility (detector backend = only variable)\n"]
    lines.append("DenseKP network/weights/crop/normalize/batch/save-format identical; "
                 "only the detector backend differs.\n")
    lines.append("`manual_review_required = true` (quality verdict deferred to human review).\n")
    for tag, c in results.items():
        lines.append(f"\n## {tag}")
        for k in ["person_detected", "num_person_npz", "only_target_person",
                  "num_keypoints_is_138", "all_finite", "orig_xy_in_image_fraction",
                  "npz_fields_shape_dtype_ok", "downstream_loadable", "bbox_xyxy",
                  "vitdet_vs_legacy_npz_identical"]:
            if k in c:
                lines.append(f"- {k}: {c[k]}")
    lines.append("\n## Verdict")
    lines.append("- manual_review_required: true")
    (OUT_ROOT / "compatibility_summary.md").write_text("\n".join(lines))
    json.dump(results, open(OUT_ROOT / "compatibility_results.json", "w"), indent=2)
    print(f"[single_frame] done -> {OUT_ROOT}")


if __name__ == "__main__":
    main()
