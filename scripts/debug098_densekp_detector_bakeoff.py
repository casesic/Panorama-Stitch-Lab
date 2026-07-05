#!/usr/bin/env python3
"""debug098: DenseKP detector-only bake-off (ViTDet-H vs RTMLib YOLOX).

Detector-only comparison. Does NOT run the DenseKP network, crop, fitting, CPW,
or any reprojection. Does NOT judge quality from bbox differences.

For each selected image it writes:
    artifacts/densekp_detector_bakeoff/detector_only/<image_id>/
        vitdet.json          (all detectron2_vitdet person detections + metadata)
        rtmlib_yolox.json     (all rtmlib_yolox person detections + metadata)
        compare_boxes.jpg     (original + both backends' bboxes + scores + legend)
        metadata.json         (image info + both backend metadata + selection rationale)

Selection of representative frames is data-driven (recorded in
artifacts/densekp_detector_bakeoff/detector_only/selection.json), not guessed
from filenames. See source_audit.md.

Run:
    python scripts/debug098_densekp_detector_bakeoff.py
"""

import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
TOOL_ROOT = REPO_ROOT / "tools" / "CameraHMR"
ART_DIR = REPO_ROOT / "artifacts" / "densekp_detector_bakeoff" / "detector_only"

# Data-driven selection (see selection rationale fields). Paths are repo-relative.
UP = "image/equirect_crop_rotate_upright"
SELECTION = [
    {
        "image_id": "00513_cam2",
        "category": "mandatory",
        "path": f"data_process/Output/20260311_riding/{UP}/cam2/00513.jpg",
        "rationale": "Required frame: 20260311_riding cam2 00513.",
    },
    {
        "image_id": "00513_cam4",
        "category": "mandatory",
        "path": f"data_process/Output/20260311_riding/{UP}/cam4/00513.jpg",
        "rationale": "Required frame: 20260311_riding cam4 00513.",
    },
    {
        "image_id": "normal_frontal__20260311_cam4_01905",
        "category": "normal_frontal",
        "path": f"data_process/Output/20260311_riding/{UP}/cam4/01905.jpg",
        "rationale": "Fully-inside (0 borders touched), high score 0.9465, "
                     "full-body area_frac 0.875 -> clean frontal/normal person.",
    },
    {
        "image_id": "riding_leg_occlusion__20260311_cam2_01917",
        "category": "riding_leg_occlusion",
        "path": f"data_process/Output/20260311_riding/{UP}/cam2/01917.jpg",
        "rationale": "Max full-body area_frac 0.908 riding frame (legs within "
                     "bbox). Leg occlusion is a property of the riding context "
                     "(bike occludes legs); proxy, not auto-verified.",
    },
    {
        "image_id": "truncated_boundary__20260320_ds1_cam4_05118",
        "category": "truncated_boundary",
        "path": f"data_process/Output/20260320_riding_dataset1/{UP}/cam4/05118.jpg",
        "rationale": "2 borders touched (left+right), area_frac 0.98 -> person "
                     "wider than crop, sides truncated at image boundary.",
    },
    {
        "image_id": "complex_multi__20260320_ds3_cam2_07905",
        "category": "complex_multi_background",
        "path": f"data_process/Output/20260320_riding_dataset3/{UP}/cam2/07905.jpg",
        "rationale": "9 person detections (most in dataset), small top area_frac "
                     "0.246 -> complex background with multiple candidate people.",
    },
]

# Drawing colors (BGR)
COLOR_VITDET = (0, 200, 0)       # green
COLOR_RTMLIB = (0, 128, 255)     # orange


def _jsonable_dets(dets):
    out = []
    for d in dets:
        out.append({
            "bbox_xyxy": [float(x) for x in d["bbox_xyxy"]],
            "score": float(d["score"]),
            "class_id": int(d["class_id"]),
            "class_name": d["class_name"],
            "source": d["source"],
        })
    return out


def _draw_boxes(img, dets, color, label_prefix):
    for i, d in enumerate(dets):
        b = d["bbox_xyxy"]
        x1, y1, x2, y2 = int(round(b[0])), int(round(b[1])), int(round(b[2])), int(round(b[3]))
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        txt = f"{label_prefix} {d['score']:.2f}"
        ty = max(0, y1 - 4)
        cv2.putText(img, txt, (x1, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)


def _draw_legend(img):
    h, w = img.shape[:2]
    pad = 6
    cv2.rectangle(img, (0, 0), (260, 52), (0, 0, 0), -1)
    cv2.putText(img, "ViTDet-H (detectron2)", (pad, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_VITDET, 2, cv2.LINE_AA)
    cv2.putText(img, "RTMLib YOLOX-s", (pad, 44),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_RTMLIB, 2, cv2.LINE_AA)


def main():
    ap = argparse.ArgumentParser(description="DenseKP detector-only bake-off")
    ap.add_argument("--out_dir", type=str, default=str(ART_DIR))
    ap.add_argument("--threshold", type=float, default=0.25)
    args = ap.parse_args()

    # bootstrap imports exactly like the production entry (cwd=tool root)
    sys.path.insert(0, str(TOOL_ROOT))
    os.chdir(str(TOOL_ROOT))

    from detectors import build_person_detector

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[debug098] building detectors (detectron2_vitdet may be heavy)...")
    det_vitdet = build_person_detector(backend="detectron2_vitdet", threshold=args.threshold)
    det_rtmlib = build_person_detector(backend="rtmlib_yolox", threshold=args.threshold)
    md_vitdet = det_vitdet.metadata
    md_rtmlib = det_rtmlib.metadata

    summary = []
    for sel in SELECTION:
        img_path = REPO_ROOT / sel["path"]
        image_id = sel["image_id"]
        d_out = out_dir / image_id
        d_out.mkdir(parents=True, exist_ok=True)

        img = cv2.imread(str(img_path))
        if img is None:
            print(f"  WARN: cannot read {img_path}; skipping")
            summary.append({"image_id": image_id, "status": "missing", "path": sel["path"]})
            continue
        h, w = img.shape[:2]

        dets_v = det_vitdet.detect(img)
        dets_r = det_rtmlib.detect(img)

        json.dump({
            "backend": "detectron2_vitdet", "image_id": image_id,
            "image_path": sel["path"], "image_h": h, "image_w": w,
            "num_detections": len(dets_v), "detections": _jsonable_dets(dets_v),
            "metadata": md_vitdet,
        }, open(d_out / "vitdet.json", "w"), indent=2)

        json.dump({
            "backend": "rtmlib_yolox", "image_id": image_id,
            "image_path": sel["path"], "image_h": h, "image_w": w,
            "num_detections": len(dets_r), "detections": _jsonable_dets(dets_r),
            "metadata": md_rtmlib,
        }, open(d_out / "rtmlib_yolox.json", "w"), indent=2)

        vis = img.copy()
        _draw_boxes(vis, dets_v, COLOR_VITDET, "ViT")
        _draw_boxes(vis, dets_r, COLOR_RTMLIB, "YOLOX")
        _draw_legend(vis)
        cv2.imwrite(str(d_out / "compare_boxes.jpg"), vis,
                    [int(cv2.IMWRITE_JPEG_QUALITY), 95])

        json.dump({
            "image_id": image_id, "category": sel["category"],
            "selection_rationale": sel["rationale"],
            "image_path": sel["path"], "image_h": h, "image_w": w,
            "num_detections_vitdet": len(dets_v),
            "num_detections_rtmlib_yolox": len(dets_r),
            "vitdet_metadata": md_vitdet,
            "rtmlib_yolox_metadata": md_rtmlib,
            "note": "Detector-only comparison. No DenseKP/crop/fitting/CPW. "
                    "bbox differences are NOT a quality verdict.",
        }, open(d_out / "metadata.json", "w"), indent=2)

        summary.append({
            "image_id": image_id, "category": sel["category"], "status": "ok",
            "path": sel["path"], "h": h, "w": w,
            "n_vitdet": len(dets_v), "n_rtmlib_yolox": len(dets_r),
        })
        print(f"  {image_id}: vitdet={len(dets_v)} rtmlib_yolox={len(dets_r)} -> {d_out}")

    json.dump({
        "selection": SELECTION,
        "results": summary,
        "scan_procedure": (
            "RTMLib YOLOX (thr 0.25) was run over all 2006 upright frames across "
            "4 datasets x {cam2,cam4}; per-frame features (n_dets, top score, "
            "area_frac, border-touch flags) drove the category picks. See "
            "source_audit.md and the rationale per selection entry."
        ),
        "vitdet_metadata": md_vitdet,
        "rtmlib_yolox_metadata": md_rtmlib,
    }, open(out_dir / "selection.json", "w"), indent=2)

    print(f"[debug098] done -> {out_dir}")


if __name__ == "__main__":
    main()
