from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np


CAMERAS = ("cam2", "cam4")
IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp")


def get_bbox(bboxes: dict[str, list[int]], frame_idx: int) -> list[int] | None:
    return bboxes.get(str(frame_idx)) or bboxes.get(f"{frame_idx:05d}")


def find_image_path(image_dir: Path, frame_idx: int) -> Path | None:
    for suffix in IMAGE_SUFFIXES:
        path = image_dir / f"{frame_idx:05d}{suffix}"
        if path.exists():
            return path
    return None


def discover_fusion_frames(
    crop_image_dir: Path,
    seg_dir: Path,
    cam2_bboxes: dict[str, list[int]],
    cam4_bboxes: dict[str, list[int]],
) -> list[int]:
    frame_indices: set[int] = set()
    for camera in CAMERAS:
        for root in (crop_image_dir / camera, seg_dir / camera / "mask"):
            if not root.exists():
                continue
            for path in root.iterdir():
                if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
                    try:
                        frame_indices.add(int(path.stem))
                    except ValueError:
                        pass
    for bboxes in (cam2_bboxes, cam4_bboxes):
        for key in bboxes:
            try:
                frame_indices.add(int(key))
            except ValueError:
                pass
    return sorted(frame_indices)


def read_image_bgr(path: Path) -> np.ndarray | None:
    return cv2.imread(str(path), cv2.IMREAD_COLOR)


def read_mask(path: Path) -> np.ndarray | None:
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return None
    return ((mask > 0).astype(np.uint8) * 255)


def ensure_fusion_output_dirs(output_dir: Path) -> None:
    for camera in CAMERAS:
        for name in ("mask", "full_original_mask", "full_completed_mask", "vis"):
            (output_dir / camera / name).mkdir(parents=True, exist_ok=True)
    (output_dir / "cam2" / "full_warped_from_cam4").mkdir(parents=True, exist_ok=True)
    (output_dir / "cam4" / "full_warped_from_cam2").mkdir(parents=True, exist_ok=True)
    (output_dir / "matches").mkdir(parents=True, exist_ok=True)


def save_original_fallback(seg_dir: Path, output_dir: Path, frame_idx: int, reason: str) -> None:
    frame_name = f"{frame_idx:05d}.png"
    for camera in CAMERAS:
        source = seg_dir / camera / "mask" / frame_name
        target = output_dir / camera / "mask" / frame_name
        target.parent.mkdir(parents=True, exist_ok=True)
        mask = read_mask(source)
        if mask is not None:
            cv2.imwrite(str(target), mask)
    print(f"Warning: fallback original masks for frame {frame_idx:05d}: {reason}")


def validate_fusion_inputs(
    crop_image_dir: Path,
    seg_dir: Path,
    cam2_bboxes: dict[str, list[int]],
    cam4_bboxes: dict[str, list[int]],
    frame_idx: int,
) -> tuple[dict[str, Any] | None, list[str]]:
    missing: list[str] = []
    data: dict[str, Any] = {}
    for camera, bboxes in (("cam2", cam2_bboxes), ("cam4", cam4_bboxes)):
        image_path = find_image_path(crop_image_dir / camera, frame_idx)
        mask_path = seg_dir / camera / "mask" / f"{frame_idx:05d}.png"
        bbox = get_bbox(bboxes, frame_idx)
        if image_path is None:
            missing.append(f"{camera}_crop_image")
        if not mask_path.exists():
            missing.append(f"{camera}_mask")
        if bbox is None:
            missing.append(f"{camera}_bbox")
        data[camera] = {"image_path": image_path, "mask_path": mask_path, "bbox": bbox}

    if missing:
        return None, missing

    for camera in CAMERAS:
        image = read_image_bgr(data[camera]["image_path"])
        mask = read_mask(data[camera]["mask_path"])
        if image is None:
            missing.append(f"{camera}_crop_image_read_failed")
        if mask is None:
            missing.append(f"{camera}_mask_read_failed")
        data[camera]["image"] = image
        data[camera]["mask"] = mask

    if missing:
        return None, missing
    return data, []


def write_translation_meta(output_dir: Path, meta: dict[str, Any]) -> None:
    with (output_dir / "translation_meta.json").open("w", encoding="utf-8") as file:
        json.dump(meta, file, ensure_ascii=False, indent=2)
