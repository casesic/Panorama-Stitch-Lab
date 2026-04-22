from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from tqdm import tqdm
import os
import config_script
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')

PROJECT_ROOT = Path(config_script.PROJECT_ROOT)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

SINGLE_FITTING_ROOT = PROJECT_ROOT / "tools" / "SingleFitting"
if str(SINGLE_FITTING_ROOT) not in sys.path:
    sys.path.insert(0, str(SINGLE_FITTING_ROOT))

from tools.fisheye_extractor.config import DS_PARAMS_CAM2, DS_PARAMS_CAM4
from tools.fisheye_extractor.ds_model import unproject

ROTATION_MATRIX = np.array([1, 0, 0, 0, 0, 1, 0, -1, 0], dtype=np.float64).reshape(3, 3)
R_EXTRINSIC_CAM4 = np.array([-1, 0, 0, 0, -1, 0, 0, 0, 1], dtype=np.float64).reshape(3, 3)
JPEG_QUALITY = 95

def _ray_to_erp_pixel(pts: np.ndarray, w: int, h: int) -> np.ndarray:
    x = pts[:, 0]
    y = pts[:, 1]
    z = pts[:, 2]
    lon = np.arctan2(x, z)
    lat = np.arcsin(np.clip(y, -1 + 1e-6, 1 - 1e-6))
    u = (lon / (2.0 * np.pi) + 0.5) * w
    v = (0.5 - lat / np.pi) * h
    return np.stack([u, v], axis=1)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Map fisheye RTMLib bbox to equirect crops for 4D-Humans."
    )
    parser.add_argument("--reviewed", type=str, required=True, help="Path to reviewed.json.")
    parser.add_argument(
        "--bbox_dir",
        type=str,
        required=True,
        help="Directory containing cam2_bboxes.json and cam4_bboxes.json.",
    )
    parser.add_argument(
        "--equirect_dir",
        type=str,
        required=True,
        help="Equirect root directory containing cam2/cam4 images.",
    )
    parser.add_argument("--output_dir", type=str, required=True, help="Output root directory.")
    parser.add_argument(
        "--invalid_detections",
        type=str,
        default=None,
        help="Optional path to invalid_detections.json.",
    )
    parser.add_argument(
        "--padding_x",
        type=int,
        default=0,
        help="左右 padding（像素），默认0不扩展。",
    )
    parser.add_argument(
        "--padding_y",
        type=int,
        default=130,
        help="上下 padding（像素）。",
    )
    parser.add_argument(
        "--fisheye_size",
        type=int,
        default=2880,
        help="Actual fisheye image size used with DS intrinsics scaled from 2880.",
    )
    parser.add_argument(
        "--apply_extrinsic",
        action="store_true",
        help="Apply optional extrinsic rotation when mapping fisheye bbox to equirect_rotate.",
    )
    return parser.parse_args()


def load_bbox_dict(json_path: Path) -> dict[str, list[int]]:
    with json_path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, dict):
        raise ValueError(f"bbox json must contain a dict: {json_path}")
    return data


def _normalize_main_camera_path(video_path: str) -> str | None:
    if video_path.endswith(".0.mp4"):
        return video_path
    if video_path.endswith(".1.mp4"):
        return video_path[:-6] + ".0.mp4"
    return None


def load_reviewed_frame_indices(reviewed_path: Path) -> list[int]:
    with reviewed_path.open("r", encoding="utf-8") as file:
        reviewed = json.load(file)

    if not isinstance(reviewed, list):
        raise ValueError("reviewed.json must contain a list")

    frame_indices: set[int] = set()
    for item in reviewed:
        if not isinstance(item, dict):
            raise ValueError("Each reviewed entry must be a dict")
        video_path = _normalize_main_camera_path(str(item.get("video_path", "")))
        if video_path is None:
            print(f"Warning: unsupported video path pattern, skipping: {item.get('video_path')}")
            continue
        frame_indices.add(int(item["frame_idx"]))

    return sorted(frame_indices)


def _extract_frame_indices(obj: Any) -> set[int]:
    frame_indices: set[int] = set()

    if isinstance(obj, int):
        frame_indices.add(int(obj))
        return frame_indices

    if isinstance(obj, list):
        for item in obj:
            frame_indices.update(_extract_frame_indices(item))
        return frame_indices

    if isinstance(obj, dict):
        if "frame_idx" in obj:
            frame_indices.add(int(obj["frame_idx"]))
        if "frame_idxs" in obj:
            frame_indices.update(_extract_frame_indices(obj["frame_idxs"]))
        if "frames" in obj:
            frame_indices.update(_extract_frame_indices(obj["frames"]))
        return frame_indices

    return frame_indices


def load_invalid_frame_sets(invalid_path: Path | None) -> tuple[set[int], set[int]]:
    if invalid_path is None:
        return set(), set()
    if not invalid_path.exists():
        raise FileNotFoundError(f"invalid_detections file not found: {invalid_path}")

    with invalid_path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    cam2_invalid = _extract_frame_indices(data.get("cam2", {}).get("fisheye", []))
    cam4_invalid = _extract_frame_indices(data.get("cam4", {}).get("fisheye", []))
    return cam2_invalid, cam4_invalid
def map_fisheye_bbox_to_equirect(
    bbox_xywh: np.ndarray,
    ds_params: dict,
    erp_w: int,
    erp_h: int,
    padding_x: int = 0,
    padding_y: int = 20,
    R_extrinsic: np.ndarray | None = None,
    pad_top: bool = False,
) -> tuple[int, int, int, int]:
    xmin, ymin, xmax, ymax = np.asarray(bbox_xywh, dtype=np.float64).reshape(4)

    xs = np.linspace(xmin, xmax, num=10, dtype=np.float64)
    ys = np.linspace(ymin, ymax, num=10, dtype=np.float64)
    edge_points = np.concatenate(
        [
            np.stack([xs, np.full_like(xs, ymin)], axis=1),
            np.stack([xs, np.full_like(xs, ymax)], axis=1),
            np.stack([np.full_like(ys, xmin), ys], axis=1),
            np.stack([np.full_like(ys, xmax), ys], axis=1),
        ],
        axis=0,
    )

    pts_3d, valid_mask = unproject(edge_points, ds_params)
    valid_mask = np.asarray(valid_mask, dtype=bool)
    if not np.any(valid_mask):
        raise ValueError("No valid sampled bbox edge points after fisheye unprojection")

    pts_raw = np.asarray(pts_3d[valid_mask], dtype=np.float64)
    if R_extrinsic is not None:
        pts_raw = pts_raw @ R_extrinsic.T
        pts_rotated = pts_raw @ ROTATION_MATRIX.T
    else:
        pts_rotated = pts_raw @ ROTATION_MATRIX
    pts_erp = _ray_to_erp_pixel(pts_rotated, erp_w, erp_h)
    

    x1 = max(0, int(np.floor(np.min(pts_erp[:, 0]) - padding_x)))
    x2 = min(erp_w, int(np.ceil(np.max(pts_erp[:, 0]) + padding_x)))
    if pad_top:
        y1 = max(0, int(np.floor(np.min(pts_erp[:, 1]) + padding_y)))
        y2 = min(erp_h, int(np.ceil(np.max(pts_erp[:, 1]) + padding_y)))
    else:
        y1 = max(0, int(np.floor(np.min(pts_erp[:, 1]) - padding_y)))
        y2 = min(erp_h, int(np.ceil(np.max(pts_erp[:, 1]) - padding_y)))

    x2 = max(x1 + 1, x2)
    y2 = max(y1 + 1, y2)
    x2 = min(erp_w, x2)
    y2 = min(erp_h, y2)
    return x1, y1, x2, y2


def _encode_jpeg(image: np.ndarray) -> bytes | None:
    ok, encoded = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
    if not ok:
        return None
    return encoded.tobytes()


def _process_single_camera(
    camera_name: str,
    frame_idx: int,
    equirect_dir: Path,
    bbox_xyxy: list[int] | None,
    output_dir: Path,
    invalid_frames: set[int],
    padding_x: int,
    padding_y: int,
    erp_w: int,
    erp_h: int,
    R_extrinsic: np.ndarray | None = None,
    pad_top: bool = False,
) -> tuple[int, int, int, int] | None:
    if frame_idx in invalid_frames:
        return

    if bbox_xyxy is None:
        print(f"Warning: bbox not found, skipping {camera_name} frame {frame_idx:05d}")
        return

    ds_params = DS_PARAMS_CAM2 if camera_name == "cam2" else DS_PARAMS_CAM4

    image_path = equirect_dir / camera_name / f"{frame_idx:05d}.jpg"
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"Warning: equirect image not found, skipping {camera_name} frame {frame_idx:05d}")
        return

    try:
        x1, y1, x2, y2 = map_fisheye_bbox_to_equirect(
            np.asarray(bbox_xyxy, dtype=np.float64),
            ds_params=ds_params,
            erp_w=erp_w,
            erp_h=erp_h,
            padding_x=padding_x,
            padding_y=padding_y,
            R_extrinsic=R_extrinsic,
            pad_top=pad_top,
        )
    except ValueError as exc:
        print(f"Warning: failed to map bbox for {camera_name} frame {frame_idx:05d}: {exc}")
        return

    cropped = image[y1:y2, x1:x2]
    if cropped.size == 0:
        print(f"Warning: empty crop, skipping {camera_name} frame {frame_idx:05d}")
        return

    encoded = _encode_jpeg(cropped)
    if encoded is None:
        print(f"Warning: failed to encode crop, skipping {camera_name} frame {frame_idx:05d}")
        return

    output_path = output_dir / f"{frame_idx:05d}.jpg"
    output_path.write_bytes(encoded)

    vis = image.copy()
    cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 255), 3)
    vis_dir = output_dir.parent / "vis" / camera_name
    vis_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(
        str(vis_dir / f"{frame_idx:05d}.jpg"),
        vis,
        [cv2.IMWRITE_JPEG_QUALITY, 95],
    )
    return x1, y1, x2, y2


def process_frame(
    frame_idx: int,
    equirect_dir: Path,
    cam2_bbox: list[int] | None,
    cam4_bbox_input: list[int] | None,
    output_cam2_dir: Path,
    output_cam4_dir: Path,
    invalid_cam2: set[int],
    invalid_cam4: set[int],
    padding_x: int,
    padding_y: int,
    erp_w: int,
    erp_h: int,
    cam2_bboxes: dict[str, list[int]],
    cam4_bboxes: dict[str, list[int]],
    apply_extrinsic: bool = False,
) -> None:
    cam2_bbox = _process_single_camera(
        camera_name="cam2",
        frame_idx=frame_idx,
        equirect_dir=equirect_dir,
        bbox_xyxy=cam2_bbox,
        output_dir=output_cam2_dir,
        invalid_frames=invalid_cam2,
        padding_x=padding_x,
        padding_y=padding_y,
        erp_w=erp_w,
        erp_h=erp_h,
        R_extrinsic=None,
        pad_top=False,
    )
    if cam2_bbox is not None:
        cam2_bboxes[str(frame_idx)] = list(cam2_bbox)

    cam4_bbox = _process_single_camera(
        camera_name="cam4",
        frame_idx=frame_idx,
        equirect_dir=equirect_dir,
        bbox_xyxy=cam4_bbox_input,
        output_dir=output_cam4_dir,
        invalid_frames=invalid_cam4,
        padding_x=padding_x,
        padding_y=padding_y,
        erp_w=erp_w,
        erp_h=erp_h,
        R_extrinsic=R_EXTRINSIC_CAM4 if apply_extrinsic else None,
        pad_top=apply_extrinsic,
    )
    if cam4_bbox is not None:
        cam4_bboxes[str(frame_idx)] = list(cam4_bbox)


def infer_equirect_size(frame_indices: list[int], equirect_dir: Path) -> tuple[int, int]:
    for frame_idx in frame_indices:
        for camera_name in ("cam2", "cam4"):
            image_path = equirect_dir / camera_name / f"{frame_idx:05d}.jpg"
            image = cv2.imread(str(image_path))
            if image is None:
                continue
            height, width = image.shape[:2]
            return width, height
    raise RuntimeError("Failed to read any equirect image to infer image size")


def main() -> None:
    args = parse_args()

    reviewed_path = Path(args.reviewed)
    bbox_dir = Path(args.bbox_dir)
    equirect_dir = Path(args.equirect_dir)
    output_dir = Path(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    invalid_path = Path(args.invalid_detections) if args.invalid_detections is not None else None

    frame_indices = load_reviewed_frame_indices(reviewed_path)
    cam2_bboxes_input = load_bbox_dict(bbox_dir / "cam2_bboxes.json")
    cam4_bboxes_input = load_bbox_dict(bbox_dir / "cam4_bboxes.json")
    cam2_bboxes: dict[str, list[int]] = {}
    cam4_bboxes: dict[str, list[int]] = {}
    invalid_cam2, invalid_cam4 = load_invalid_frame_sets(invalid_path)

    output_cam2_dir = output_dir / "cam2"
    output_cam4_dir = output_dir / "cam4"
    output_cam2_dir.mkdir(parents=True, exist_ok=True)
    output_cam4_dir.mkdir(parents=True, exist_ok=True)

    erp_w, erp_h = infer_equirect_size(frame_indices, equirect_dir)

    with tqdm(frame_indices, desc="Cropping equirect frames", unit="frame") as progress:
        for frame_idx in progress:
            process_frame(
                frame_idx=frame_idx,
                equirect_dir=equirect_dir,
                cam2_bbox=cam2_bboxes_input.get(str(frame_idx)),
                cam4_bbox_input=cam4_bboxes_input.get(str(frame_idx)),
                output_cam2_dir=output_cam2_dir,
                output_cam4_dir=output_cam4_dir,
                invalid_cam2=invalid_cam2,
                invalid_cam4=invalid_cam4,
                padding_x=args.padding_x,
                padding_y=args.padding_y,
                erp_w=erp_w,
                erp_h=erp_h,
                cam2_bboxes=cam2_bboxes,
                cam4_bboxes=cam4_bboxes,
                apply_extrinsic=args.apply_extrinsic,
            )

    with (output_dir / "cam2_bboxes.json").open("w", encoding="utf-8") as file:
        json.dump(cam2_bboxes, file, indent=2)
    with (output_dir / "cam4_bboxes.json").open("w", encoding="utf-8") as file:
        json.dump(cam4_bboxes, file, indent=2)


if __name__ == "__main__":
    main()
'''
python src\script\test_function\debug025_crop_equirect_for_4dhumans.py `
  --reviewed data_process\Input\20260311_riding\review.json `
  --bbox_dir data_process\Output\20260311_riding\bbox_grounding_dino `
  --equirect_dir data_process\Output\20260311_riding\image\equirect_rotate `
  --output_dir data_process\Output\20260311_riding\image\equirect_rotate_crop `
  --apply_extrinsic `
  --padding_x 0 `
  --padding_y 90
'''
