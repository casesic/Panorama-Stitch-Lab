#!/usr/bin/env python3
"""
Visualize CameraHMR DenseKP npz results.

Expected latest npz fields:
  - keypoints_crop_xy: (138, 2), float32
  - keypoints_orig_xy: (138, 2), float32
  - keypoint_value: (138,), float32
  - bbox_xyxy: (4,), float32
  - bbox_center: (2,), float32
  - box_size: scalar float32
  - image_size: (2,), int32, [H, W]
  - crop_trans: (2, 3), float32, original image -> crop image
  - crop_inv_trans: (2, 3), float32, crop image -> original image

This script does not support older npz formats.
"""

from __future__ import annotations

from pathlib import Path
import argparse

import cv2
import numpy as np


IMAGE_EXTS = [".jpg", ".jpeg", ".png", ".bmp", ".webp"]


REQUIRED_KEYS = [
    "keypoints_crop_xy",
    "keypoints_orig_xy",
    "keypoint_value",
    "bbox_xyxy",
    "bbox_center",
    "box_size",
    "image_size",
    "crop_trans",
    "crop_inv_trans",
]


def find_image(image_dir: Path, stem: str) -> Path:
    for ext in IMAGE_EXTS:
        p = image_dir / f"{stem}{ext}"
        if p.exists():
            return p
    raise FileNotFoundError(f"No image found for stem={stem} in {image_dir}")


def parse_stem_from_npz(npz_path: Path) -> str:
    """
    00513_0.npz -> 00513
    """
    name = npz_path.stem
    if "_" in name:
        return name.rsplit("_", 1)[0]
    return name


def load_densekp_npz(npz_path: Path) -> dict[str, np.ndarray]:
    data = np.load(npz_path)
    missing = [k for k in REQUIRED_KEYS if k not in data]
    if missing:
        raise KeyError(f"{npz_path} missing keys: {missing}")

    return {
        "keypoints_crop_xy": data["keypoints_crop_xy"].astype(np.float32),
        "keypoints_orig_xy": data["keypoints_orig_xy"].astype(np.float32),
        "keypoint_value": data["keypoint_value"].astype(np.float32),
        "bbox_xyxy": data["bbox_xyxy"].astype(np.float32),
        "bbox_center": data["bbox_center"].astype(np.float32),
        "box_size": np.asarray(data["box_size"]).astype(np.float32),
        "image_size": data["image_size"].astype(np.int32),
        "crop_trans": data["crop_trans"].astype(np.float32),
        "crop_inv_trans": data["crop_inv_trans"].astype(np.float32),
    }


def normalize_values(values: np.ndarray) -> np.ndarray:
    values = values.astype(np.float32)
    vmin = float(np.nanmin(values))
    vmax = float(np.nanmax(values))
    if vmax - vmin < 1e-6:
        return np.ones_like(values, dtype=np.float32) * 0.5
    return (values - vmin) / (vmax - vmin)


def draw_keypoints(
    image: np.ndarray,
    keypoints_xy: np.ndarray,
    keypoint_value: np.ndarray,
    radius: int = 3,
    draw_index: bool = False,
) -> np.ndarray:
    vis = image.copy()
    h, w = vis.shape[:2]
    norm_values = normalize_values(keypoint_value)

    for i, (xy, nv) in enumerate(zip(keypoints_xy, norm_values)):
        x, y = float(xy[0]), float(xy[1])

        if not np.isfinite(x) or not np.isfinite(y):
            continue

        xi, yi = int(round(x)), int(round(y))
        if xi < 0 or xi >= w or yi < 0 or yi >= h:
            continue

        color = cv2.applyColorMap(
            np.array([[int(np.clip(nv, 0, 1) * 255)]], dtype=np.uint8),
            cv2.COLORMAP_VIRIDIS,
        )[0, 0].tolist()

        cv2.circle(vis, (xi, yi), radius, color, -1)
        cv2.circle(vis, (xi, yi), radius + 1, (0, 0, 0), 1)

        if draw_index:
            cv2.putText(
                vis,
                str(i),
                (xi + 3, yi - 3),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

    return vis


def draw_bbox(image: np.ndarray, bbox_xyxy: np.ndarray) -> np.ndarray:
    vis = image.copy()
    if bbox_xyxy[0] >= 0:
        x1, y1, x2, y2 = [int(round(v)) for v in bbox_xyxy.tolist()]
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 255), 2)
    return vis


def make_crop_image(image: np.ndarray, crop_trans: np.ndarray, crop_size: int) -> np.ndarray:
    return cv2.warpAffine(
        image,
        crop_trans,
        (crop_size, crop_size),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )


def save_vis(
    npz_path: Path,
    image_dir: Path,
    output_dir: Path,
    coord: str,
    crop_size: int,
    radius: int,
    draw_index: bool,
    draw_bbox_on_orig: bool,
) -> None:
    item = load_densekp_npz(npz_path)

    stem = parse_stem_from_npz(npz_path)
    image_path = find_image(image_dir, stem)
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Failed to read image: {image_path}")

    expected_h, expected_w = item["image_size"].tolist()
    if image.shape[:2] != (expected_h, expected_w):
        print(
            f"[WARN] image size mismatch for {npz_path.name}: "
            f"image={image.shape[:2]}, npz_image_size={(expected_h, expected_w)}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    if coord in {"orig", "both"}:
        orig_img = draw_bbox(image, item["bbox_xyxy"]) if draw_bbox_on_orig else image
        vis = draw_keypoints(
            image=orig_img,
            keypoints_xy=item["keypoints_orig_xy"],
            keypoint_value=item["keypoint_value"],
            radius=radius,
            draw_index=draw_index,
        )
        suffix = "" if coord == "orig" else "_orig"
        out_path = output_dir / f"{npz_path.stem}{suffix}.jpg"
        cv2.imwrite(str(out_path), vis)
        print(f"[DONE] {npz_path.name} orig -> {out_path}")

    if coord in {"crop", "both"}:
        crop_img = make_crop_image(image, item["crop_trans"], crop_size)
        vis = draw_keypoints(
            image=crop_img,
            keypoints_xy=item["keypoints_crop_xy"],
            keypoint_value=item["keypoint_value"],
            radius=radius,
            draw_index=draw_index,
        )
        suffix = "" if coord == "crop" else "_crop"
        out_path = output_dir / f"{npz_path.stem}{suffix}.jpg"
        cv2.imwrite(str(out_path), vis)
        print(f"[DONE] {npz_path.name} crop -> {out_path}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz_path", default=None)
    parser.add_argument("--npz_dir", default=None)
    parser.add_argument("--image_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--coord", choices=["orig", "crop", "both"], default="orig")
    parser.add_argument("--crop_size", type=int, default=256)
    parser.add_argument("--radius", type=int, default=3)
    parser.add_argument("--draw_index", action="store_true")
    parser.add_argument("--no_bbox", action="store_true")
    parser.add_argument("--max_files", type=int, default=None)
    args = parser.parse_args()

    image_dir = Path(args.image_dir)
    output_dir = Path(args.output_dir)

    if args.npz_path is not None:
        npz_paths = [Path(args.npz_path)]
    elif args.npz_dir is not None:
        npz_paths = sorted(Path(args.npz_dir).glob("*.npz"))
        if args.max_files is not None:
            npz_paths = npz_paths[: args.max_files]
    else:
        raise ValueError("Need --npz_path or --npz_dir")

    print(f"[INFO] files: {len(npz_paths)}")

    for npz_path in npz_paths:
        save_vis(
            npz_path=npz_path,
            image_dir=image_dir,
            output_dir=output_dir,
            coord=args.coord,
            crop_size=args.crop_size,
            radius=args.radius,
            draw_index=args.draw_index,
            draw_bbox_on_orig=not args.no_bbox,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())


r'''
python src/script/test_function/debug004_vis_densekp_npz.py \
  --npz_path data_process/Output/20260311_riding/camerahmr/densekp_npz/cam2/00513_0.npz \
  --image_dir data_process/Output/20260311_riding/image/equirect_crop_rotate_upright/cam2 \
  --output_dir data_process/Output/20260311_riding/camerahmr/densekp_vis/cam2 \
  --coord orig \
  --draw_index
'''

r'''
python src/script/test_function/debug004_vis_densekp_npz.py \
  --npz_dir data_process/Output/20260311_riding/camerahmr/densekp_npz/cam4 \
  --image_dir data_process/Output/20260311_riding/image/equirect_crop_rotate_upright/cam4 \
  --output_dir data_process/Output/20260311_riding/camerahmr/densekp_vis/cam4 \
  --coord orig \
  --max_files 20
'''

