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
pythonRTMLIB_DEVICE = 'cpu'
BBOX_YMAX_RATIO = 0.9  # bbox 底部 y 超过图像高度此比例的检测框直接排除

_wholebody_model = None

'''
python tools/SingleFitting/scripts/debug010_segmentation.py `
    --video-path Circular_Video/00_send/20260311_riding/VID_20181003_034909_00_982.0.mp4 `
    --index-start 490 `
    --num-frames 30 `
    --output-dir  data_process/Output/20260311_riding/segmentation/ `
    --camera cam2 `
'''


def parse_args():
    p = argparse.ArgumentParser(description="Segment people from images and save mask/seg files.")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--image", type=Path, default=None)
    src.add_argument("--image-dir", type=Path, default=None)
    src.add_argument("--video-path", type=Path, default=None)
    p.add_argument("--camera", type=str, default="cam2")
    p.add_argument("--output-dir", type=Path, default=Path("tools\\SingleFitting\\outputs\\segmentation"))
    p.add_argument("--index-start", type=int, default=0)
    p.add_argument("--num-frames", type=int, default=1)
    p.add_argument("--sam-model", type=str, default="sam2_b.pt")
    return p.parse_args()


def load_frames(args) -> list[tuple[str, np.ndarray]]:
    if args.video_path is not None:
        cap = cv2.VideoCapture(str(args.video_path))
        if not cap.isOpened():
            return []

        cap.set(cv2.CAP_PROP_POS_FRAMES, args.index_start)
        frames: list[tuple[str, np.ndarray]] = []
        for i in range(args.num_frames):
            ok, image_bgr = cap.read()
            if not ok or image_bgr is None:
                break
            stem = f"frame_{args.index_start + i:05d}"
            frames.append((stem, image_bgr))
        cap.release()
        return frames

    if args.image is not None:
        image_bgr = cv2.imread(str(args.image))
        if image_bgr is None:
            return []
        return [(args.image.stem, image_bgr)]

    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    frames: list[tuple[str, np.ndarray]] = []
    for image_path in sorted(p for p in args.image_dir.iterdir() if p.suffix.lower() in exts):
        image_bgr = cv2.imread(str(image_path))
        if image_bgr is not None:
            frames.append((image_path.stem, image_bgr))
    return frames


def load_models(sam_model_name: str):
    from ultralytics import SAM

    return SAM(sam_model_name)


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


def detect_person_bbox(image_bgr: np.ndarray, det_model=None) -> list[float] | None:
    del det_model

    global _wholebody_model
    if _wholebody_model is None:
        from rtmlib import Wholebody

        _wholebody_model = Wholebody(device=pythonRTMLIB_DEVICE)

    bboxes = _wholebody_model.det_model(image_bgr)
    if bboxes is None or len(bboxes) == 0:
        return None

    ymax_thresh = BBOX_YMAX_RATIO * image_bgr.shape[0]
    filtered_bboxes = bboxes[bboxes[:, 3] <= ymax_thresh]
    candidate_bboxes = filtered_bboxes if len(filtered_bboxes) > 0 else bboxes

    areas = (candidate_bboxes[:, 2] - candidate_bboxes[:, 0]) * (candidate_bboxes[:, 3] - candidate_bboxes[:, 1])
    best_bbox = candidate_bboxes[np.argmax(areas)]
    return best_bbox[:4].tolist()


def segment_person_mask(sam_model, image_bgr: np.ndarray) -> tuple[np.ndarray | None, list[float] | None]:
    person_bbox = detect_person_bbox(image_bgr)
    if person_bbox is None:
        return None, None

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
    frames = load_frames(args)
    if not frames:
        print("[warn] No input images found.")
        return 1

    sam_model = load_models(args.sam_model)
    camera_dir = args.output_dir / args.camera
    mask_dir = camera_dir / "mask"
    mask_dir.mkdir(parents=True, exist_ok=True)

    saved_count = 0
    for idx, (stem, image_bgr) in enumerate(frames):
        mask, person_bbox = segment_person_mask(sam_model, image_bgr)
        if mask is None:
            print(f"[warn] Failed to get person mask: {stem}")
            continue

        seg = render_segmentation(image_bgr, mask, prompt_bbox=person_bbox)

        mask_path = mask_dir / f"mask_{stem}.png"
        seg_path = mask_dir / f"seg_{stem}.jpg"

        cv2.imwrite(str(mask_path), mask.astype(np.uint8) * 255)
        cv2.imwrite(str(seg_path), seg, [cv2.IMWRITE_JPEG_QUALITY, 92])

        saved_count += 1
        print(f"[{idx + 1}/{len(frames)}] saved {mask_path.name}, {seg_path.name}")

    return 0 if saved_count else 1


if __name__ == "__main__":
    raise SystemExit(main())
