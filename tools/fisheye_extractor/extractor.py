from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from tqdm import tqdm

import config
from detector import build_detector, run_detector_on_view
from edge_mask import get_or_build_cache
from perspective_views import build_all_luts, build_rotation_matrix


def _to_serializable(obj: Any) -> Any:
    """Convert numpy objects into JSON-serializable Python values."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


def _bbox_iou(a: list[float] | np.ndarray, b: list[float] | np.ndarray) -> float:
    """Compute IoU for two axis-aligned bounding boxes [x1, y1, x2, y2]."""
    box_a = np.asarray(a, dtype=np.float32)
    box_b = np.asarray(b, dtype=np.float32)

    ix1 = max(float(box_a[0]), float(box_b[0]))
    iy1 = max(float(box_a[1]), float(box_b[1]))
    ix2 = min(float(box_a[2]), float(box_b[2]))
    iy2 = min(float(box_a[3]), float(box_b[3]))

    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, float(box_a[2] - box_a[0])) * max(0.0, float(box_a[3] - box_a[1]))
    area_b = max(0.0, float(box_b[2] - box_b[0])) * max(0.0, float(box_b[3] - box_b[1]))
    union = area_a + area_b - inter
    return inter / union if union > 0.0 else 0.0


def _sample_theta_values(
    pixels: np.ndarray,
    theta_map: np.ndarray,
    valid_mask: np.ndarray,
    image_size: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    """Sample theta values at rounded pixel locations and report validity."""
    points = np.asarray(pixels, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError(f"Expected pixels shape (N, 2), got {points.shape}")

    height, width = image_size
    x_idx = np.rint(points[:, 0]).astype(np.int32)
    y_idx = np.rint(points[:, 1]).astype(np.int32)
    in_bounds = (x_idx >= 0) & (x_idx < width) & (y_idx >= 0) & (y_idx < height)

    sampled_theta = np.full(points.shape[0], -1.0, dtype=np.float32)
    sampled_valid = np.zeros(points.shape[0], dtype=bool)
    if np.any(in_bounds):
        valid_inside = valid_mask[y_idx[in_bounds], x_idx[in_bounds]]
        sampled_valid[in_bounds] = valid_inside
        if np.any(valid_inside):
            idx = np.where(in_bounds)[0][valid_inside]
            sampled_theta[idx] = theta_map[y_idx[idx], x_idx[idx]]
    return sampled_theta, sampled_valid


def compute_detection_stats(
    detection: dict[str, Any],
    theta_map: np.ndarray,
    valid_mask: np.ndarray,
    image_size: tuple[int, int],
) -> dict[str, Any] | None:
    """Compute fisheye-space statistics for one detection and apply frame-level filtering."""
    keypoints_fisheye = np.asarray(detection["keypoints_fisheye"], dtype=np.float32)
    keypoint_scores = np.asarray(detection["keypoint_scores"], dtype=np.float32)
    bbox_fisheye = np.asarray(detection["bbox_fisheye"], dtype=np.float32)
    keypoints_valid = np.asarray(detection["keypoints_valid"], dtype=bool)

    if keypoints_fisheye.shape != (17, 2):
        raise ValueError(f"Expected keypoints_fisheye shape (17, 2), got {keypoints_fisheye.shape}")
    if keypoint_scores.shape != (17,):
        raise ValueError(f"Expected keypoint_scores shape (17,), got {keypoint_scores.shape}")
    if bbox_fisheye.shape != (4,):
        raise ValueError(f"Expected bbox_fisheye shape (4,), got {bbox_fisheye.shape}")
    if keypoints_valid.shape != (17,):
        raise ValueError(f"Expected keypoints_valid shape (17,), got {keypoints_valid.shape}")

    visible_mask = (keypoint_scores > 0.3) & keypoints_valid
    visible_keypoints_count = int(visible_mask.sum())
    if visible_keypoints_count < config.MIN_VISIBLE_KPS:
        return None

    bbox_center = np.array(
        [[(bbox_fisheye[0] + bbox_fisheye[2]) / 2.0, (bbox_fisheye[1] + bbox_fisheye[3]) / 2.0]],
        dtype=np.float32,
    )
    bbox_center_theta_arr, bbox_center_valid = _sample_theta_values(
        bbox_center,
        theta_map,
        valid_mask,
        image_size,
    )
    if not bool(bbox_center_valid[0]):
        return None
    bbox_center_theta = float(bbox_center_theta_arr[0])
    if bbox_center_theta <= config.THETA_THRESHOLD:
        return None

    visible_keypoints = keypoints_fisheye[visible_mask]
    visible_theta, visible_theta_valid = _sample_theta_values(
        visible_keypoints,
        theta_map,
        valid_mask,
        image_size,
    )
    if not np.any(visible_theta_valid):
        return None
    visible_theta = visible_theta[visible_theta_valid]
    keypoints_mean_theta = float(np.mean(visible_theta))
    keypoints_max_theta = float(np.max(visible_theta))
    if keypoints_mean_theta <= config.THETA_THRESHOLD:
        return None

    head_theta = -1.0
    if keypoint_scores[0] > 0.3 and keypoints_valid[0]:
        head_theta_arr, head_valid = _sample_theta_values(
            keypoints_fisheye[0:1],
            theta_map,
            valid_mask,
            image_size,
        )
        if bool(head_valid[0]):
            head_theta = float(head_theta_arr[0])

    foot_mask = np.array(
        [
            (keypoint_scores[15] > 0.3) and bool(keypoints_valid[15]),
            (keypoint_scores[16] > 0.3) and bool(keypoints_valid[16]),
        ],
        dtype=bool,
    )
    feet_theta = -1.0
    if np.any(foot_mask):
        foot_points = keypoints_fisheye[[15, 16]][foot_mask]
        foot_theta_arr, foot_valid = _sample_theta_values(
            foot_points,
            theta_map,
            valid_mask,
            image_size,
        )
        if np.any(foot_valid):
            feet_theta = float(np.mean(foot_theta_arr[foot_valid]))

    height, width = image_size
    x1i = int(np.clip(np.rint(bbox_fisheye[0]), 0, width - 1))
    y1i = int(np.clip(np.rint(bbox_fisheye[1]), 0, height - 1))
    x2i = int(np.clip(np.rint(bbox_fisheye[2]), 0, width))
    y2i = int(np.clip(np.rint(bbox_fisheye[3]), 0, height))
    bbox_area = max(0, x2i - x1i) * max(0, y2i - y1i)
    if bbox_area <= 0:
        bbox_completeness = 0.0
    else:
        region = valid_mask[y1i:y2i, x1i:x2i]
        bbox_completeness = float(region.sum()) / float(bbox_area)
    if bbox_completeness <= config.BBOX_COMPLETENESS_THRESHOLD:
        return None

    return {
        "source_view": detection["source_view"],
        "fisheye_bbox": bbox_fisheye.astype(np.float32),
        "keypoints_fisheye": keypoints_fisheye.astype(np.float32),
        "keypoint_scores": keypoint_scores.astype(np.float32),
        "bbox_center_theta": bbox_center_theta,
        "keypoints_mean_theta": keypoints_mean_theta,
        "keypoints_max_theta": keypoints_max_theta,
        "head_theta": float(head_theta),
        "feet_theta": float(feet_theta),
        "bbox_completeness": float(bbox_completeness),
        "visible_keypoints_count": visible_keypoints_count,
    }


def process_frame(
    fisheye_img: np.ndarray,
    frame_idx: int,
    video_path: str,
    luts: dict[str, dict[str, np.ndarray]],
    rotations: dict[str, np.ndarray],
    theta_map: np.ndarray,
    valid_mask: np.ndarray,
    body: Any,
    ds_params: dict[str, Any],
) -> list[dict[str, Any]]:
    """Process one fisheye frame across all perspective views and keep filtered candidates."""
    image = np.asarray(fisheye_img)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"Expected fisheye image shape (H, W, 3), got {image.shape}")

    candidates: list[dict[str, Any]] = []
    image_size = image.shape[:2]

    for view_name, lut in luts.items():
        detections = run_detector_on_view(
            image,
            lut,
            view_name,
            rotations[view_name],
            config.FOV_DEG,
            config.OUT_SIZE,
            ds_params,
            body,
        )
        for detection in detections:
            stats = compute_detection_stats(detection, theta_map, valid_mask, image_size)
            if stats is None:
                continue
            stats["video_path"] = video_path
            stats["frame_idx"] = frame_idx
            candidates.append(stats)

    if not candidates:
        return []

    sorted_candidates = sorted(candidates, key=lambda item: item["keypoints_mean_theta"], reverse=True)
    deduped: list[dict[str, Any]] = []
    for candidate in sorted_candidates:
        overlaps_existing = any(
            _bbox_iou(candidate["fisheye_bbox"], kept["fisheye_bbox"]) > 0.5 for kept in deduped
        )
        if not overlaps_existing:
            deduped.append(candidate)
    return deduped


def process_video(
    video_path: str,
    output_dir: str,
    body: Any,
    luts: dict[str, dict[str, np.ndarray]],
    rotations: dict[str, np.ndarray],
    theta_map: np.ndarray,
    valid_mask: np.ndarray,
    ds_params: dict[str, Any],
) -> list[dict[str, Any]]:
    """Process one video, save candidate fisheye frames, and return all accepted candidates."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Warning: failed to open video: {video_path}")
        return []

    video_fps = float(cap.get(cv2.CAP_PROP_FPS))
    if video_fps <= 0.0:
        print(f"Warning: could not read FPS for {video_path}, assuming 30fps")
        video_fps = 30.0
    frame_step = max(1, int(round(video_fps / config.SAMPLE_FPS)))

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    total_steps = max(1, int(np.ceil(total_frames / frame_step))) if total_frames > 0 else None

    all_candidates: list[dict[str, Any]] = []
    video_stem = Path(video_path).stem

    progress = tqdm(total=total_steps, desc=video_stem, unit="sampled_frame")
    try:
        sampled_idx = 0
        while True:
            frame_idx = sampled_idx * frame_step
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            success, frame = cap.read()
            if not success:
                break

            frame_candidates = process_frame(
                frame,
                frame_idx,
                video_path,
                luts,
                rotations,
                theta_map,
                valid_mask,
                body,
                ds_params,
            )
            if frame_candidates:
                image_path = output_path / f"{video_stem}_{frame_idx:06d}.jpg"
                cv2.imwrite(str(image_path), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
                all_candidates.extend(frame_candidates)

            sampled_idx += 1
            progress.update(1)
    finally:
        progress.close()
        cap.release()

    return all_candidates


def run_pipeline(
    video_paths: list[str],
    output_dir: str,
    cache_dir: str,
    ds_params: dict[str, float],
) -> None:
    """Run the end-to-end candidate-frame extraction pipeline over multiple videos."""
    output_path = Path(output_dir)
    cache_path = Path(cache_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    cache_path.mkdir(parents=True, exist_ok=True)

    theta_cache = cache_path / "theta_map_cache.npz"
    theta_map, valid_mask, _ = get_or_build_cache(
        ds_params,
        config.FISHEYE_SIZE,
        str(theta_cache),
        config.THETA_THRESHOLD,
    )

    luts = build_all_luts(ds_params, config.VIEWS, config.FOV_DEG, config.OUT_SIZE)
    rotations = {
        str(view["name"]): build_rotation_matrix(view["yaw"], view["pitch"]) for view in config.VIEWS
    }
    body = build_detector(config.DEVICE)

    all_candidates: list[dict[str, Any]] = []
    for video_path in video_paths:
        all_candidates.extend(
            process_video(
                video_path,
                str(output_path),
                body,
                luts,
                rotations,
                theta_map,
                valid_mask,
                ds_params,
            )
        )

    candidates_json = [
        {key: _to_serializable(value) for key, value in candidate.items()}
        for candidate in all_candidates
    ]
    candidates_path = output_path / "candidates.json"
    with candidates_path.open("w", encoding="utf-8") as file:
        json.dump(candidates_json, file, ensure_ascii=False, indent=2)

    print(f"Total candidate frames: {len(all_candidates)}")
    print(f"Candidates saved to {candidates_path}")
