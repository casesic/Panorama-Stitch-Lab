from __future__ import annotations

import argparse
import importlib
import json
import pickle
import sys
from pathlib import Path

import cv2 as cv
import numpy as np
from rtmlib import Body
from tqdm import tqdm
import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import config_script
from batch_utils import load_reviewed_entries
from lib_pano.human3d.keypoints_detector import Result_Single_Human_Single_Frame, draw_bbox


def _ensure_paths() -> Path:
    """Ensure the SingleFitting package is importable."""
    repo_root = Path(__file__).resolve().parents[3]
    sf_root = repo_root / "tools" / "SingleFitting"
    if str(sf_root) not in sys.path:
        sys.path.insert(0, str(sf_root))
    fisheye_root = repo_root / "tools" / "fisheye_extractor"
    if str(fisheye_root) not in sys.path:
        sys.path.insert(0, str(fisheye_root))
    return repo_root


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for batch RTMLib detection."""
    parser = argparse.ArgumentParser(
        description="Detect person bbox directly in fisheye images."
    )
    parser.add_argument(
        "--reviewed",
        type=str,
        required=True,
        help="Path to reviewed.json.",
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Input fisheye root directory containing cam2/ and cam4/.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output root directory, e.g. data_process/Output/20260311_riding/single_frame_cache.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device for RTMLib inference, cuda or cpu.",
    )
    parser.add_argument(
        "--click_points",
        type=str,
        default=None,
        help="Optional path to click_points.json.",
    )
    return parser.parse_args()


def _build_multi_frame_det_sequence(det_results: list[Result_Single_Human_Single_Frame]):
    """Build the detection sequence object expected by SingleFitting."""
    base_engine_mod = importlib.import_module("lib_dynamic_hvip.detection.engine.base_engine")
    Base_Detect_Engine = base_engine_mod.Base_Detect_Engine

    if not det_results:
        raise RuntimeError("det_results is empty")

    num_frames = len(det_results)
    bbox = np.zeros((num_frames, 4), dtype=np.int32)
    score = np.zeros((num_frames,), dtype=np.float32)
    kps = np.zeros((num_frames, 133, 3), dtype=np.float32)
    valid = np.zeros((num_frames,), dtype=np.bool_)

    keypoints_format = None
    for i, det_result in enumerate(det_results):
        is_valid = bool(getattr(det_result, "flag_valid", False))
        valid[i] = is_valid
        if not is_valid:
            continue

        if keypoints_format is None:
            keypoints_format = det_result.keypoints_format

        bbox_i = np.asarray(det_result.bbox, dtype=np.int32).reshape(-1)
        if bbox_i.size >= 4:
            bbox[i] = bbox_i[:4]
        score[i] = float(det_result.bbox_score)

        kps_i = getattr(det_result, "keypoints", None)
        if kps_i is not None:
            kps_i = np.asarray(kps_i, dtype=np.float32)
            if kps_i.ndim == 2:
                n = min(133, kps_i.shape[0])
                c = min(3, kps_i.shape[1])
                kps[i, :n, :c] = kps_i[:n, :c]

    if keypoints_format is None:
        raise RuntimeError("No valid detection result with keypoints_format found")

    return Base_Detect_Engine.Result_Single_Human_Sequence(
        array_valid=valid,
        array_bbox_h1h2w1w2=bbox,
        array_bbox_score=score,
        array_keypoints=kps,
        keypoints_format=keypoints_format,
        flag_with_keypoints=True,
    )


def _build_single_frame_det_seq(det_result: Result_Single_Human_Single_Frame):
    """Wrap one detection result as a single-frame sequence."""
    return _build_multi_frame_det_sequence([det_result])


def load_click_points(click_points_path: Path | None) -> dict[str, dict[str, list[float]]]:
    if click_points_path is None:
        return {}
    if not click_points_path.exists():
        raise FileNotFoundError(f"click_points.json not found: {click_points_path}")

    with click_points_path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    if not isinstance(data, dict):
        raise ValueError("click_points.json must contain a dict")

    normalized: dict[str, dict[str, list[float]]] = {}
    for camera_name, frame_map in data.items():
        if not isinstance(frame_map, dict):
            continue
        normalized[camera_name] = {}
        for frame_idx, point in frame_map.items():
            if (
                isinstance(point, list)
                and len(point) == 2
                and all(isinstance(value, (int, float)) for value in point)
            ):
                normalized[camera_name][str(frame_idx)] = [float(point[0]), float(point[1])]
    return normalized


def split_frame_segments(frame_indices: list[int], gap_threshold: int = 2) -> list[list[int]]:
    if not frame_indices:
        return []

    segments: list[list[int]] = [[frame_indices[0]]]
    for frame_idx in frame_indices[1:]:
        if frame_idx - segments[-1][-1] > gap_threshold:
            segments.append([frame_idx])
        else:
            segments[-1].append(frame_idx)
    return segments


def build_frame_to_click_point(
    frame_indices: list[int],
    click_points: dict[str, dict[str, list[float]]],
) -> dict[str, dict[str, list[float]]]:
    frame_to_click_point: dict[str, dict[str, list[float]]] = {
        camera_name: {} for camera_name in ("cam2", "cam4")
    }
    segments = split_frame_segments(sorted(set(frame_indices)))

    for segment in segments:
        if not segment:
            continue
        first_frame_idx = segment[0]
        for camera_name in ("cam2", "cam4"):
            click_point = click_points.get(camera_name, {}).get(str(first_frame_idx))
            if click_point is None:
                continue
            for frame_idx in segment:
                frame_to_click_point[camera_name][str(frame_idx)] = click_point
    return frame_to_click_point


def crop_roi(
    image: np.ndarray,
    center_xy: tuple[float, float] | list[float],
    crop_size: int,
) -> tuple[np.ndarray, int, int]:
    """Crop a square ROI centered at center_xy with clamp-to-image bounds."""
    image_h, image_w = image.shape[:2]
    center_x = int(round(float(center_xy[0])))
    center_y = int(round(float(center_xy[1])))
    half_size = crop_size // 2

    x_offset = center_x - half_size
    y_offset = center_y - half_size
    max_x_offset = max(0, image_w - crop_size)
    max_y_offset = max(0, image_h - crop_size)
    x_offset = min(max(x_offset, 0), max_x_offset)
    y_offset = min(max(y_offset, 0), max_y_offset)

    x_end = min(x_offset + crop_size, image_w)
    y_end = min(y_offset + crop_size, image_h)
    cropped_image = image[y_offset:y_end, x_offset:x_end].copy()
    return cropped_image, x_offset, y_offset


def pick_closest_bbox(
    bboxes: np.ndarray,
    click_x: float,
    click_y: float,
) -> np.ndarray | None:
    """Pick the bbox whose center is closest to the click point."""
    bbox_array = np.asarray(bboxes, dtype=np.float32)
    if bbox_array.size == 0:
        return None
    if bbox_array.ndim != 2 or bbox_array.shape[1] < 4:
        raise ValueError(f"Expected bboxes shape (N, >=4), got {bbox_array.shape}")

    centers_x = (bbox_array[:, 0] + bbox_array[:, 2]) * 0.5
    centers_y = (bbox_array[:, 1] + bbox_array[:, 3]) * 0.5
    distances = (centers_x - float(click_x)) ** 2 + (centers_y - float(click_y)) ** 2
    closest_index = int(np.argmin(distances))
    return bbox_array[closest_index]


def _extract_detector_bboxes(detector: Body, image: np.ndarray) -> np.ndarray:
    """Run RTMLib detector and normalize bbox output to shape (N, 5)."""
    bboxes = detector.det_model(image)
    bbox_array = np.asarray(bboxes, dtype=np.float32)
    if bbox_array.size == 0:
        return np.empty((0, 5), dtype=np.float32)
    if bbox_array.ndim == 1:
        bbox_array = bbox_array.reshape(1, -1)
    if bbox_array.shape[1] < 4:
        raise ValueError(f"Expected bboxes shape (N, >=4), got {bbox_array.shape}")
    return bbox_array


def detect_person_bbox_on_fisheye(
    image: np.ndarray,
    detector: Body,
    click_point: list[float] | None,
    crop_size: int = 1000,
) -> tuple[int, int, int, int] | None:
    """Detect one person bbox in fisheye image, optionally guided by click point."""
    image_h, image_w = image.shape[:2]

    if click_point is not None:
        cropped_image, x_offset, y_offset = crop_roi(image, click_point, crop_size)
        bboxes = _extract_detector_bboxes(detector, cropped_image)
        if bboxes.size == 0:
            return None

        local_click_x = float(click_point[0]) - float(x_offset)
        local_click_y = float(click_point[1]) - float(y_offset)
        selected_bbox = pick_closest_bbox(bboxes, local_click_x, local_click_y)
        if selected_bbox is None:
            return None

        x1, y1, x2, y2 = [float(value) for value in selected_bbox[:4]]
        x1 += float(x_offset)
        y1 += float(y_offset)
        x2 += float(x_offset)
        y2 += float(y_offset)
    else:
        bboxes = _extract_detector_bboxes(detector, image)
        if bboxes.size == 0:
            return None

        if bboxes.shape[1] >= 5:
            selected_bbox = bboxes[int(np.argmax(bboxes[:, 4]))]
        else:
            selected_bbox = bboxes[0]
        x1, y1, x2, y2 = [float(value) for value in selected_bbox[:4]]

    x1 = int(np.clip(np.floor(x1), 0, image_w - 1))
    y1 = int(np.clip(np.floor(y1), 0, image_h - 1))
    x2 = int(np.clip(np.ceil(x2), x1 + 1, image_w))
    y2 = int(np.clip(np.ceil(y2), y1 + 1, image_h))
    return x1, y1, x2, y2


def _process_camera_frame(
    detector: Body,
    image_path: Path,
    output_dir: Path,
    vis_dir: Path,
    frame_idx: int,
    camera_name: str,
    camera_suffix: str,
    frame_to_click_point: dict[str, dict[str, list[float]]],
) -> None:
    """Detect person bbox in one fisheye image and save RTM cache."""
    image = cv.imread(str(image_path))
    if image is None:
        print(f"Warning: image not found, skipping: {image_path}")
        return

    click_point = frame_to_click_point.get(camera_name, {}).get(str(frame_idx))
    bbox_fisheye = detect_person_bbox_on_fisheye(
        image=image,
        detector=detector,
        click_point=click_point,
    )
    if bbox_fisheye is None:
        print(f"Warning: no detection for frame {frame_idx:05d} {camera_suffix} ({image_path})")
        return

    x1, y1, x2, y2 = np.asarray(bbox_fisheye, dtype=np.float32).reshape(4)
    det_result = Result_Single_Human_Single_Frame(
        bbox=(int(y1), int(y2), int(x1), int(x2)),
        bbox_score=1.0,
        keypoints=None,
        keypoints_format={},
    )
    vis_image = draw_bbox(image, det_result)
    vis_dir.mkdir(parents=True, exist_ok=True)
    cv.imwrite(str(vis_dir / f"vis_{frame_idx:05d}_{camera_suffix}.jpg"), vis_image)

    det_seq = _build_single_frame_det_seq(det_result)

    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / f"rtm_s{frame_idx:05d}_{camera_suffix}.pkl").open("wb") as file:
        pickle.dump(det_seq, file)


def main() -> None:
    """Batch-detect person bbox directly in fisheye images."""
    _ensure_paths()
    from detector import build_detector

    args = parse_args()

    reviewed_path = Path(args.reviewed)
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    click_points_path = Path(args.click_points) if args.click_points is not None else None
    reviewed_pairs = load_reviewed_entries(reviewed_path)
    click_points = load_click_points(click_points_path)
    frame_indices = [frame_idx for _, frame_idx in reviewed_pairs]
    frame_to_click_point = build_frame_to_click_point(frame_indices, click_points)

    detector = build_detector(device=args.device)
    total_steps = len(reviewed_pairs) * 2

    with tqdm(total=total_steps, desc="Processing RTMLib", unit="frame") as progress:
        for _, frame_idx in reviewed_pairs:
            _process_camera_frame(
                detector=detector,
                image_path=input_dir / "cam2" / f"{frame_idx:05d}.jpg",
                output_dir=output_dir / "cam2",
                vis_dir=output_dir / "vis" / "cam2",
                frame_idx=frame_idx,
                camera_name="cam2",
                camera_suffix="c2",
                frame_to_click_point=frame_to_click_point,
            )
            progress.update(1)

            _process_camera_frame(
                detector=detector,
                image_path=input_dir / "cam4" / f"{frame_idx:05d}.jpg",
                output_dir=output_dir / "cam4",
                vis_dir=output_dir / "vis" / "cam4",
                frame_idx=frame_idx,
                camera_name="cam4",
                camera_suffix="c4",
                frame_to_click_point=frame_to_click_point,
            )
            progress.update(1)


if __name__ == "__main__":
    main()
