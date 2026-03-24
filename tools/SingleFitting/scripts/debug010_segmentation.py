#!/usr/bin/env python3
"""
Segment the person using the projected SMPL-X mesh as prompt.

Usage:
  python tools/SingleFitting/scripts/debug010_segmentation.py ^
    --result-pkl tools/SingleFitting/outputs/fisheye_smplx_cam4/fisheye_cam4_s0_i0_29_seq.pkl ^
    --camera cam4 ^
    --image-dir data/fisheye_cam4_frames ^
    --index-start 0 ^
    --output-dir tools/SingleFitting/outputs/segmentation
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

from single_fitting.debug_fit_shared import SingleCamFisheyeFitter, forward_param_to_numpy, unpack_smplx_result

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from single_fitting.debug_shared import DS_PARAMS, ds_project_np, load_pickle


def get_mesh_bbox(
    smplx_param,
    fitter,
    ds_params: dict,
    img_h: int,
    img_w: int,
    margin: float = 0.05,
) -> Optional[np.ndarray]:
    verts, _, _ = forward_param_to_numpy(fitter, smplx_param)
    v2d = ds_project_np(verts, **ds_params)

    in_img = (
        (v2d[:, 0] >= 0) & (v2d[:, 0] < img_w) &
        (v2d[:, 1] >= 0) & (v2d[:, 1] < img_h)
    )
    if in_img.sum() < 50:
        return None

    v2d_in = v2d[in_img]
    x1, y1 = float(v2d_in[:, 0].min()), float(v2d_in[:, 1].min())
    x2, y2 = float(v2d_in[:, 0].max()), float(v2d_in[:, 1].max())
    mx, my = (x2 - x1) * margin, (y2 - y1) * margin
    return np.array(
        [max(0, x1 - mx), max(0, y1 - my), min(img_w - 1, x2 + mx), min(img_h - 1, y2 + my)],
        dtype=np.float32,
    )


def get_mesh_keypoints(
    smplx_param,
    fitter,
    ds_params: dict,
    img_h: int,
    img_w: int,
) -> Optional[np.ndarray]:
    key_joints = [0, 1, 2, 12, 15, 16, 17]
    _, _, joints_body = forward_param_to_numpy(fitter, smplx_param)
    key_pts_2d = ds_project_np(joints_body[key_joints], **ds_params)
    in_img = (
        (key_pts_2d[:, 0] >= 0) & (key_pts_2d[:, 0] < img_w) &
        (key_pts_2d[:, 1] >= 0) & (key_pts_2d[:, 1] < img_h)
    )
    return key_pts_2d[in_img] if in_img.any() else None


def segment_with_ultralytics(
    image_bgr: np.ndarray,
    bbox: np.ndarray,
    model_name: str = "sam2_b.pt",
) -> Optional[np.ndarray]:
    from ultralytics import SAM

    model = SAM(model_name)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    results = model(image_rgb, bboxes=[bbox.tolist()])
    if results and results[0].masks is not None:
        return results[0].masks.data[0].cpu().numpy().astype(bool)
    return None


def segment_human(
    image_bgr: np.ndarray,
    bbox: np.ndarray,
    key_pts: Optional[np.ndarray] = None,
) -> Optional[np.ndarray]:
    return segment_with_ultralytics(image_bgr, bbox)


def visualize_segmentation(
    img: np.ndarray,
    mask: np.ndarray,
    bbox: np.ndarray,
    key_pts: Optional[np.ndarray] = None,
    alpha: float = 0.5,
    mask_color: Tuple[int, int, int] = (0, 120, 255),
) -> np.ndarray:
    h, w = img.shape[:2]
    if mask.shape != (h, w):
        mask = cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)

    vis = img.copy()
    overlay = vis.copy()
    overlay[mask] = mask_color
    vis = cv2.addWeighted(overlay, alpha, vis, 1 - alpha, 0)

    mask_u8 = mask.astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(vis, contours, -1, (0, 255, 255), 2)

    x1, y1, x2, y2 = bbox.astype(int)
    cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
    if key_pts is not None:
        for u, v in key_pts:
            cv2.circle(vis, (int(u), int(v)), 5, (255, 0, 0), -1)
    return vis


def parse_args():
    p = argparse.ArgumentParser(description="Use SMPL-X mesh bbox as SAM2 prompt.")
    p.add_argument("--result-pkl", type=Path, required=True)
    p.add_argument("--camera", type=str, default="cam4", choices=["cam2", "cam4"])
    p.add_argument("--index-start", type=int, default=0)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--image", type=Path, default=None)
    src.add_argument("--image-dir", type=Path, default=None)
    p.add_argument("--output-dir", type=Path, default=Path("outputs/segmentation"))
    p.add_argument("--bbox-margin", type=float, default=0.05)
    p.add_argument("--mask-alpha", type=float, default=0.5)
    p.add_argument("--sam-model", type=str, default="sam2_b.pt")
    return p.parse_args()


def main():
    args = parse_args()

    result = load_pickle(args.result_pkl)
    params = unpack_smplx_result(result)

    if args.image is not None:
        img_paths = [args.image] * len(params)
    else:
        exts = {".jpg", ".jpeg", ".png", ".bmp"}
        all_imgs = sorted(p for p in args.image_dir.iterdir() if p.suffix.lower() in exts)
        img_paths = all_imgs[args.index_start : args.index_start + len(params)]
        params = params[: len(img_paths)]

    fitter = SingleCamFisheyeFitter(gender="neutral", camera=args.camera)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    saved = []
    for fi, (param, img_path) in enumerate(zip(params, img_paths)):
        image_bgr = cv2.imread(str(img_path))
        if image_bgr is None:
            print(f"[warn] Failed to read image: {img_path}")
            continue

        img_h, img_w = image_bgr.shape[:2]
        bbox = get_mesh_bbox(param, fitter, DS_PARAMS, img_h, img_w, margin=args.bbox_margin)
        if bbox is None:
            print(f"[warn] Invalid projected bbox for {img_path}")
            continue

        key_pts = get_mesh_keypoints(param, fitter, DS_PARAMS, img_h, img_w)
        mask = segment_with_ultralytics(image_bgr, bbox, model_name=args.sam_model)
        if mask is None:
            print(f"[warn] SAM2 returned empty mask for {img_path}")
            continue

        vis = visualize_segmentation(image_bgr, mask, bbox, key_pts=key_pts, alpha=args.mask_alpha)
        stem = img_path.stem if args.image is None else f"frame_{fi:04d}"
        seg_path = args.output_dir / f"seg_{args.camera}_{stem}.jpg"
        mask_path = args.output_dir / f"mask_{args.camera}_{stem}.png"
        meta_path = args.output_dir / f"meta_{args.camera}_{stem}.json"

        cv2.imwrite(str(seg_path), vis, [cv2.IMWRITE_JPEG_QUALITY, 92])
        cv2.imwrite(str(mask_path), mask.astype(np.uint8) * 255)
        with meta_path.open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "camera": args.camera,
                    "image": str(img_path),
                    "bbox": bbox.tolist(),
                    "key_pts": None if key_pts is None else key_pts.tolist(),
                    "mask_path": str(mask_path),
                    "seg_path": str(seg_path),
                },
                f,
                indent=2,
                ensure_ascii=False,
            )
        saved.append((seg_path, mask_path, meta_path))
        print(f"[{fi + 1}/{len(params)}] -> {seg_path}")

    return 0 if saved else 1


if __name__ == "__main__":
    raise SystemExit(main())
