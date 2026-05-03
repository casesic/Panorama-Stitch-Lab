from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from tqdm import tqdm

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

JPEG_QUALITY = 95


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Crop equirect_rotate images from equirect-coordinate bboxes for 4D-Humans."
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
        "--invalid_json",
        type=str,
        default=None,
        help="Optional invalid.json containing invalid_frames to skip.",
    )
    parser.add_argument(
        "--padding_left",
        type=int,
        default=250,
        help="Padding to expand the left side of each equirect bbox.",
    )
    parser.add_argument(
        "--padding_right",
        type=int,
        default=250,
        help="Padding to expand the right side of each equirect bbox.",
    )
    parser.add_argument(
        "--padding_top",
        type=int,
        default=150,
        help="Padding to expand the top side of each equirect bbox.",
    )
    parser.add_argument(
        "--padding_bottom",
        type=int,
        default=150,
        help="Padding to expand the bottom side of each equirect bbox.",
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
        if "invalid_frames" in obj:
            frame_indices.update(_extract_frame_indices(obj["invalid_frames"]))
            return frame_indices
        if "frame_idx" in obj:
            frame_indices.add(int(obj["frame_idx"]))
        if "frame_idxs" in obj:
            frame_indices.update(_extract_frame_indices(obj["frame_idxs"]))
        if "frames" in obj:
            frame_indices.update(_extract_frame_indices(obj["frames"]))
        for key, value in obj.items():
            if key == "problem_frames":
                continue
            if key in ("frame_idx", "frame_idxs", "frames", "invalid_frames"):
                continue
            frame_indices.update(_extract_frame_indices(value))
        return frame_indices

    return frame_indices


def load_invalid_frames(invalid_path: Path | None) -> set[int]:
    if invalid_path is None:
        return set()
    if not invalid_path.exists():
        raise FileNotFoundError(f"invalid json file not found: {invalid_path}")

    with invalid_path.open("r", encoding="utf-8-sig") as file:
        data = json.load(file)

    if isinstance(data, dict) and "invalid_frames" in data:
        return _extract_frame_indices(data["invalid_frames"])

    return _extract_frame_indices(data)


def expand_erp_bbox(
    bbox_xyxy,
    erp_w: int,
    erp_h: int,
    padding_left: int,
    padding_right: int,
    padding_top: int,
    padding_bottom: int,
) -> tuple[int, int, int, int] | None:
    x1, y1, x2, y2 = np.asarray(bbox_xyxy, dtype=np.float64).reshape(4)
    x1 = max(0, int(np.floor(x1 - padding_left)))
    y1 = max(0, int(np.floor(y1 - padding_top)))
    x2 = min(erp_w, int(np.ceil(x2 + padding_right)))
    y2 = min(erp_h, int(np.ceil(y2 + padding_bottom)))
    if x2 <= x1 or y2 <= y1:
        return None
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
    erp_w: int,
    erp_h: int,
    padding_left: int,
    padding_right: int,
    padding_top: int,
    padding_bottom: int,
) -> tuple[int, int, int, int] | None:
    if frame_idx in invalid_frames:
        return None

    if bbox_xyxy is None:
        print(f"Warning: bbox not found, skipping {camera_name} frame {frame_idx:05d}")
        return None

    image_path = equirect_dir / camera_name / f"{frame_idx:05d}.jpg"
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"Warning: equirect image not found, skipping {camera_name} frame {frame_idx:05d}")
        return None

    expanded_bbox = expand_erp_bbox(
        bbox_xyxy,
        erp_w=erp_w,
        erp_h=erp_h,
        padding_left=padding_left,
        padding_right=padding_right,
        padding_top=padding_top,
        padding_bottom=padding_bottom,
    )
    if expanded_bbox is None:
        print(f"Warning: invalid expanded bbox, skipping {camera_name} frame {frame_idx:05d}: {bbox_xyxy}")
        return None

    x1, y1, x2, y2 = expanded_bbox
    cropped = image[y1:y2, x1:x2]
    if cropped.size == 0:
        print(f"Warning: empty crop, skipping {camera_name} frame {frame_idx:05d}")
        return None

    encoded = _encode_jpeg(cropped)
    if encoded is None:
        print(f"Warning: failed to encode crop, skipping {camera_name} frame {frame_idx:05d}")
        return None

    output_path = output_dir / f"{frame_idx:05d}.jpg"
    output_path.write_bytes(encoded)

    vis = image.copy()
    cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 255), 3)
    vis_dir = output_dir.parent / "vis" / camera_name
    vis_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(vis_dir / f"{frame_idx:05d}.jpg"), vis, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    return x1, y1, x2, y2


def process_frame(
    frame_idx: int,
    equirect_dir: Path,
    cam2_bbox: list[int] | None,
    cam4_bbox_input: list[int] | None,
    output_cam2_dir: Path,
    output_cam4_dir: Path,
    invalid_frames: set[int],
    erp_w: int,
    erp_h: int,
    padding_left: int,
    padding_right: int,
    padding_top: int,
    padding_bottom: int,
    cam2_bboxes: dict[str, list[int]],
    cam4_bboxes: dict[str, list[int]],
) -> None:
    cam2_expanded_bbox = _process_single_camera(
        camera_name="cam2",
        frame_idx=frame_idx,
        equirect_dir=equirect_dir,
        bbox_xyxy=cam2_bbox,
        output_dir=output_cam2_dir,
        invalid_frames=invalid_frames,
        erp_w=erp_w,
        erp_h=erp_h,
        padding_left=padding_left,
        padding_right=padding_right,
        padding_top=padding_top,
        padding_bottom=padding_bottom,
    )
    if cam2_expanded_bbox is not None:
        cam2_bboxes[str(frame_idx)] = list(cam2_expanded_bbox)

    cam4_expanded_bbox = _process_single_camera(
        camera_name="cam4",
        frame_idx=frame_idx,
        equirect_dir=equirect_dir,
        bbox_xyxy=cam4_bbox_input,
        output_dir=output_cam4_dir,
        invalid_frames=invalid_frames,
        erp_w=erp_w,
        erp_h=erp_h,
        padding_left=padding_left,
        padding_right=padding_right,
        padding_top=padding_top,
        padding_bottom=padding_bottom,
    )
    if cam4_expanded_bbox is not None:
        cam4_bboxes[str(frame_idx)] = list(cam4_expanded_bbox)


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
    output_dir.mkdir(parents=True, exist_ok=True)
    invalid_path = Path(args.invalid_json) if args.invalid_json is not None else None

    frame_indices = load_reviewed_frame_indices(reviewed_path)
    cam2_bboxes_input = load_bbox_dict(bbox_dir / "cam2_bboxes.json")
    cam4_bboxes_input = load_bbox_dict(bbox_dir / "cam4_bboxes.json")
    cam2_bboxes: dict[str, list[int]] = {}
    cam4_bboxes: dict[str, list[int]] = {}
    invalid_frames = load_invalid_frames(invalid_path)
    print(f"loaded invalid frames: {len(invalid_frames)}")

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
                invalid_frames=invalid_frames,
                erp_w=erp_w,
                erp_h=erp_h,
                padding_left=args.padding_left,
                padding_right=args.padding_right,
                padding_top=args.padding_top,
                padding_bottom=args.padding_bottom,
                cam2_bboxes=cam2_bboxes,
                cam4_bboxes=cam4_bboxes,
            )

    with (output_dir / "cam2_bboxes.json").open("w", encoding="utf-8") as file:
        json.dump(cam2_bboxes, file, indent=2)
    with (output_dir / "cam4_bboxes.json").open("w", encoding="utf-8") as file:
        json.dump(cam4_bboxes, file, indent=2)


if __name__ == "__main__":
    main()

r"""
python src\script\test_function\debug025_crop_equirect_rotate_from_bboxes.py `
  --reviewed data_process\Input\20260320_riding\dataset1\review.json `
  --bbox_dir data_process\Output\20260320_riding_dataset1\bbox_grounding_dino\equirect_rotate_smooth `
  --equirect_dir data_process\Output\20260320_riding_dataset1\image\equirect_rotate `
  --output_dir data_process\Output\20260320_riding_dataset1\image\equirect_crop_rotate `
  --invalid_json data_process\Output\20260320_riding_dataset1\bbox_grounding_dino\equirect_rotate_smooth\invalid.json `
  --padding_left 30 `
  --padding_right 30 `
  --padding_top 20 `
  --padding_bottom 20
  
python src\script\test_function\debug025_crop_equirect_rotate_from_bboxes.py `
  --reviewed data_process\Input\20260311_riding\review.json `
  --bbox_dir data_process\Output\20260311_riding\bbox_grounding_dino\equirect_rotate_smooth `
  --equirect_dir data_process\Output\20260311_riding\image\equirect_rotate `
  --output_dir data_process\Output\20260311_riding\image\equirect_crop_rotate `
  --invalid_json data_process\Output\20260311_riding\bbox_grounding_dino\equirect_rotate_smooth\invalid.json `
  --padding_left 30 `
  --padding_right 30 `
  --padding_top 20 `
  --padding_bottom 20
"""
