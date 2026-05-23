#!/usr/bin/env python3
"""
将 CameraHMR DenseKP 点从 equirect_crop_rotate_upright 坐标系
转换回 equirect_crop_rotate 坐标系并可视化。

仅使用新版 npz 的 keypoints_orig_xy 和 keypoint_value 字段。
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from debug001_rotate_cam_by_frame_rule import get_rotate_code


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def parse_frame_id(npz_stem: str) -> int:
    """从 npz 文件名 stem 解析帧号，如 00513_0 -> 513"""
    return int(npz_stem.split("_")[0])


def find_image(image_dir: Path, stem: str) -> Path:
    for ext in IMAGE_EXTS:
        p = image_dir / f"{stem}{ext}"
        if p.exists():
            return p
    raise FileNotFoundError(f"No image found for stem={stem} in {image_dir}")


def normalize_values(values: np.ndarray) -> np.ndarray:
    vmin = float(np.nanmin(values))
    vmax = float(np.nanmax(values))
    if vmax - vmin < 1e-6:
        return np.ones_like(values, dtype=np.float32) * 0.5
    return (values.astype(np.float32) - vmin) / (vmax - vmin)


def convert_upright_to_original(
    keypoints_upright: np.ndarray,
    rotate_code: int | None,
    W_orig: int,
    H_orig: int,
) -> np.ndarray:
    """
    将 upright 图像坐标系的点转回原始 (equirect_crop_rotate) 坐标系。
    """
    out = keypoints_upright.copy()
    if rotate_code == cv2.ROTATE_90_COUNTERCLOCKWISE:
        out[:, 0] = W_orig - 1 - keypoints_upright[:, 1]  # x_orig = W - 1 - y_upright
        out[:, 1] = keypoints_upright[:, 0]                # y_orig = x_upright
    elif rotate_code == cv2.ROTATE_90_CLOCKWISE:
        out[:, 0] = keypoints_upright[:, 1]                 # x_orig = y_upright
        out[:, 1] = H_orig - 1 - keypoints_upright[:, 0]    # y_orig = H - 1 - x_upright
    return out


def draw_keypoints(
    image: np.ndarray,
    keypoints_xy: np.ndarray,
    keypoint_value: np.ndarray,
    radius: int = 3,
) -> np.ndarray:
    vis = image.copy()
    h, w = vis.shape[:2]
    norm_values = normalize_values(keypoint_value)

    for xy, nv in zip(keypoints_xy, norm_values):
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
    image_dir: Path,
    output_dir: Path,
    radius: int,
) -> None:
    # 1. 读取 npz
    data = np.load(npz_path)
    keypoints_upright = data["keypoints_orig_xy"].astype(np.float32)
    keypoint_value = data["keypoint_value"].astype(np.float32)

    # 2. 解析帧号
    frame_id = parse_frame_id(npz_path.stem)
    rotate_code = get_rotate_code(frame_id)

    # 3. 找到对应原图并读取尺寸
    frame_stem = f"{frame_id:05d}"
    image_path = find_image(image_dir, frame_stem)
    img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Failed to read: {image_path}")
    H_orig, W_orig = img.shape[:2]

    # 4. 坐标转换
    keypoints_orig = convert_upright_to_original(
        keypoints_upright, rotate_code, W_orig, H_orig
    )

    # 5. 画点
    vis = draw_keypoints(img, keypoints_orig, keypoint_value, radius=radius)

    # 6. 保存
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{npz_path.stem}.jpg"
    cv2.imwrite(str(out_path), vis)
    print(f"[DONE] {npz_path.name} -> {out_path}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--densekp_npz_dir", required=True)
    parser.add_argument("--target_image_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--cam", required=True)
    parser.add_argument("--max_files", type=int, default=None)
    parser.add_argument("--radius", type=int, default=3)
    args = parser.parse_args()

    npz_dir = Path(args.densekp_npz_dir)
    image_dir = Path(args.target_image_dir)
    output_dir = Path(args.output_dir)

    npz_paths = sorted(npz_dir.glob("*.npz"))
    if not npz_paths:
        print(f"[ERROR] no npz files in {npz_dir}")
        return 1

    if args.max_files is not None:
        npz_paths = npz_paths[: args.max_files]

    print(f"[INFO] processing {len(npz_paths)} files")

    for npz_path in npz_paths:
        try:
            process_one(npz_path, image_dir, output_dir, args.radius)
        except Exception as e:
            print(f"[ERROR] {npz_path.name}: {e}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())


r'''
python src/script/test_function/debug005_vis_densekp_in_fitting_coord.py \
  --densekp_npz_dir data_process/Output/20260311_riding/camerahmr/densekp_npz/cam2 \
  --target_image_dir data_process/Output/20260311_riding/image/equirect_crop_rotate/cam2 \
  --output_dir data_process/Output/20260311_riding/camerahmr/densekp_vis_in_fitting_coord/cam2 \
  --cam cam2 \
  --max_files 20
'''
