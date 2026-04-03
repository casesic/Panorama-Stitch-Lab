#!/usr/bin/env python3
"""
Generate person segmentation masks from input images with SAM.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

MASK_ALPHA = 0.5
SEG_COLOR = (0, 120, 255)
PERSON_CLASS_ID = 0
PERSON_BBOX_MARGIN = 0.15

'''
python tools/SingleFitting/scripts/debug010_segmentation.py `
    --image data_process/Input/20260311_riding/camera4/frame_00515.png `
    --output-dir tools/SingleFitting/outputs/segmentation `
    --camera cam4 `
'''


def parse_args():
    p = argparse.ArgumentParser(description="Segment people from images and save mask/seg files.")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--image", type=Path, default=None)
    src.add_argument("--image-dir", type=Path, default=None)
    p.add_argument("--camera", type=str, default="cam2")
    p.add_argument("--output-dir", type=Path, default=Path("tools\\SingleFitting\\outputs\\segmentation"))
    p.add_argument("--det-model", type=str, default="yolo11n.pt")
    p.add_argument("--sam-model", type=str, default="sam2_b.pt")
    return p.parse_args()


def list_image_paths(args) -> list[Path]:
    if args.image is not None:
        return [args.image]

    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    return sorted(p for p in args.image_dir.iterdir() if p.suffix.lower() in exts)


def load_models(det_model_name: str, sam_model_name: str):
    from ultralytics import SAM, YOLO

    return YOLO(det_model_name), SAM(sam_model_name)


def choose_primary_mask(result, image_shape) -> np.ndarray | None:
    if result.masks is None or result.masks.data is None or len(result.masks.data) == 0:
        return None

    masks = result.masks.data.detach().cpu().numpy().astype(bool)
    h, w = image_shape[:2]
    best_mask = None
    best_area = -1

    for mask in masks:
        if mask.shape != (h, w):
            mask = cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
        area = int(mask.sum())
        if area > best_area:
            best_area = area
            best_mask = mask
    return best_mask


def expand_bbox(bbox: list[float], image_shape, margin: float = PERSON_BBOX_MARGIN) -> list[float]:
    h, w = image_shape[:2]
    x1, y1, x2, y2 = bbox
    bw = max(1.0, x2 - x1)
    bh = max(1.0, y2 - y1)
    mx = bw * margin
    my = bh * margin
    return [
        max(0.0, x1 - mx),
        max(0.0, y1 - my),
        min(float(w - 1), x2 + mx),
        min(float(h - 1), y2 + my),
    ]


def detect_person_bbox(det_model, image_bgr: np.ndarray) -> list[float] | None:
    results = det_model(image_bgr, verbose=False)
    if not results:
        return None

    boxes = results[0].boxes
    if boxes is None or boxes.xyxy is None or len(boxes) == 0:
        return None

    xyxy = boxes.xyxy.detach().cpu().numpy()
    cls = boxes.cls.detach().cpu().numpy().astype(int) if boxes.cls is not None else None
    conf = boxes.conf.detach().cpu().numpy() if boxes.conf is not None else None

    best_bbox = None
    best_score = -1.0
    for idx, bbox in enumerate(xyxy):
        if cls is not None and cls[idx] != PERSON_CLASS_ID:
            continue
        x1, y1, x2, y2 = bbox.tolist()
        area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        score = area if conf is None else float(conf[idx]) * area
        if score > best_score:
            best_score = score
            best_bbox = [x1, y1, x2, y2]
    return best_bbox


def segment_person_mask(det_model, sam_model, image_bgr: np.ndarray) -> tuple[np.ndarray | None, list[float] | None]:
    person_bbox = detect_person_bbox(det_model, image_bgr)
    if person_bbox is None:
        return None, None
    person_bbox = expand_bbox(person_bbox, image_bgr.shape)

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    results = sam_model(image_rgb, bboxes=[person_bbox], verbose=False)
    if not results:
        return None, person_bbox
    return choose_primary_mask(results[0], image_bgr.shape), person_bbox


def render_segmentation(
    image_bgr: np.ndarray,
    mask: np.ndarray,
    prompt_bbox: list[float] | None = None,
    alpha: float = MASK_ALPHA,
    color: tuple[int, int, int] = SEG_COLOR,
) -> np.ndarray:
    h, w = image_bgr.shape[:2]
    if mask.shape != (h, w):
        mask = cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)

    seg = image_bgr.copy()
    overlay = seg.copy()
    overlay[mask] = color
    seg = cv2.addWeighted(overlay, alpha, seg, 1 - alpha, 0)

    contours, _ = cv2.findContours(mask.astype(np.uint8) * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(seg, contours, -1, (0, 255, 255), 2)
    if prompt_bbox is not None:
        x1, y1, x2, y2 = [int(v) for v in prompt_bbox]
        cv2.rectangle(seg, (x1, y1), (x2, y2), (0, 255, 0), 2)
    return seg


def main():
    args = parse_args()
    image_paths = list_image_paths(args)
    if not image_paths:
        print("[warn] No input images found.")
        return 1

    det_model, sam_model = load_models(args.det_model, args.sam_model)
    camera_dir = args.output_dir / args.camera
    mask_dir = camera_dir / "mask"
    mask_dir.mkdir(parents=True, exist_ok=True)

    saved_count = 0
    for idx, image_path in enumerate(image_paths):
        image_bgr = cv2.imread(str(image_path))
        if image_bgr is None:
            print(f"[warn] Failed to read image: {image_path}")
            continue

        mask, person_bbox = segment_person_mask(det_model, sam_model, image_bgr)
        if mask is None:
            print(f"[warn] Failed to get person mask: {image_path}")
            continue

        seg = render_segmentation(image_bgr, mask, prompt_bbox=person_bbox)

        stem = image_path.stem
        mask_path = mask_dir / f"mask_{stem}.png"
        seg_path = mask_dir / f"seg_{stem}.jpg"

        cv2.imwrite(str(mask_path), mask.astype(np.uint8) * 255)
        cv2.imwrite(str(seg_path), seg, [cv2.IMWRITE_JPEG_QUALITY, 92])

        saved_count += 1
        print(f"[{idx + 1}/{len(image_paths)}] saved {mask_path.name}, {seg_path.name}")

    return 0 if saved_count else 1


if __name__ == "__main__":
    raise SystemExit(main())
