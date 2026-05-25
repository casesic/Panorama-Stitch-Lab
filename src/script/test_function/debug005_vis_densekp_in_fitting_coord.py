#!/usr/bin/env python3
"""
将 CameraHMR DenseKP 点转换到 full ERP fitting 坐标并可视化。
坐标与 debug029 传给 DualCamERPFitter.fit 的 DenseKPResult.keypoints_2d 完全一致。
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import config_script
from lib_pano.fitting.densekp_io import load_densekp_result
from lib_pano.pipeline.bbox_io import load_bbox_dict


def normalize_values(values: np.ndarray) -> np.ndarray:
    vmin = float(np.nanmin(values))
    vmax = float(np.nanmax(values))
    if vmax - vmin < 1e-6:
        return np.ones_like(values, dtype=np.float32) * 0.5
    return (values.astype(np.float32) - vmin) / (vmax - vmin)


def draw_bbox(image: np.ndarray, bbox: list[int], color: tuple[int, int, int] = (0, 255, 255)) -> np.ndarray:
    x1, y1, x2, y2 = [int(v) for v in bbox]
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
    return image


def draw_keypoints(
    image: np.ndarray,
    keypoints_xy: np.ndarray,
    keypoint_value: np.ndarray,
    valid: np.ndarray,
    radius: int = 3,
) -> np.ndarray:
    vis = image.copy()
    h, w = vis.shape[:2]
    norm_values = normalize_values(keypoint_value)

    for xy, nv, v in zip(keypoints_xy, norm_values, valid):
        if not v:
            continue

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

    return vis


def process_one(
    npz_path: Path,
    densekp_dir_parent: Path,
    cam_id: str,
    bboxes: dict[str, list[int]],
    image_width: int,
    image_height: int,
    output_dir: Path,
    full_image_dir: Path | None,
    radius: int,
) -> None:
    stem = npz_path.stem
    frame_id = int(stem.split("_")[0])
    frame_key = f"{frame_id:05d}"

    crop_bbox = bboxes.get(str(frame_id))
    if crop_bbox is None:
        print(f"[SKIP] {npz_path.name}: no bbox for frame {frame_id}")
        return

    result = load_densekp_result(
        frame_idx=frame_id,
        densekp_dir=densekp_dir_parent,
        cam_id=cam_id,
        image_width=image_width,
        image_height=image_height,
        crop_bbox=crop_bbox,
    )
    if result is None:
        print(f"[SKIP] {npz_path.name}: load_densekp_result returned None")
        return

    if full_image_dir is not None:
        img_path = full_image_dir / f"{frame_key}.jpg"
        if not img_path.exists():
            print(f"[WARN] {npz_path.name}: full image not found at {img_path}, using black canvas")
            img = np.zeros((image_height, image_width, 3), dtype=np.uint8)
        else:
            img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    else:
        img = np.zeros((image_height, image_width, 3), dtype=np.uint8)

    vis = draw_bbox(img, crop_bbox)
    vis = draw_keypoints(vis, result.keypoints_2d, result.values, result.valid, radius=radius)

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{stem}.jpg"
    cv2.imwrite(str(out_path), vis)
    print(f"[DONE] {npz_path.name} -> {out_path}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--densekp_npz_dir", required=True)
    parser.add_argument("--bbox_json", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--image_width", type=int, default=5760)
    parser.add_argument("--image_height", type=int, default=2880)
    parser.add_argument("--full_image_dir", default=None)
    parser.add_argument("--max_files", type=int, default=None)
    parser.add_argument("--radius", type=int, default=3)
    args = parser.parse_args()

    npz_dir = Path(args.densekp_npz_dir)
    bbox_json = Path(args.bbox_json)
    output_dir = Path(args.output_dir)
    full_image_dir = Path(args.full_image_dir) if args.full_image_dir is not None else None

    densekp_dir_parent = npz_dir.parent
    cam_id = npz_dir.name

    bboxes = load_bbox_dict(bbox_json)

    npz_paths = sorted(npz_dir.glob("*.npz"))
    if not npz_paths:
        print(f"[ERROR] no npz files in {npz_dir}")
        return 1

    if args.max_files is not None:
        npz_paths = npz_paths[: args.max_files]

    print(f"[INFO] processing {len(npz_paths)} files, cam={cam_id}")

    for npz_path in npz_paths:
        try:
            process_one(
                npz_path=npz_path,
                densekp_dir_parent=densekp_dir_parent,
                cam_id=cam_id,
                bboxes=bboxes,
                image_width=args.image_width,
                image_height=args.image_height,
                output_dir=output_dir,
                full_image_dir=full_image_dir,
                radius=args.radius,
            )
        except Exception as e:
            print(f"[ERROR] {npz_path.name}: {e}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())


r'''
python src/script/test_function/debug005_vis_densekp_in_fitting_coord.py \
  --densekp_npz_dir data_process/Output/20260311_riding/camerahmr/densekp_npz/cam2 \
  --bbox_json data_process/Output/20260311_riding/image/equirect_crop_rotate/cam2_bboxes.json \
  --output_dir data_process/Output/20260311_riding/camerahmr/densekp_vis_in_fitting_coord/cam2 \
  --max_files 20

python src/script/test_function/debug005_vis_densekp_in_fitting_coord.py \
  --densekp_npz_dir data_process/Output/20260311_riding/camerahmr/densekp_npz/cam2 \
  --bbox_json data_process/Output/20260311_riding/image/equirect_crop_rotate/cam2_bboxes.json \
  --output_dir data_process/Output/20260311_riding/camerahmr/densekp_vis_in_fitting_coord/cam2 \
  --full_image_dir data_process/Output/20260311_riding/image/equirect_rotate/cam2 \
  --max_files 20
'''
