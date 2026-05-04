from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import cv2 as cv
import numpy as np
from PIL import Image
from tqdm import tqdm

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from batch_utils import load_reviewed_entries, split_frame_segments
from lib_pano.pipeline.paths import ensure_single_fitting_path


def _ensure_paths() -> Path:
    """Ensure the SingleFitting package is importable."""
    repo_root = ensure_single_fitting_path()
    fisheye_root = repo_root / "tools" / "fisheye_extractor"
    if str(fisheye_root) not in sys.path:
        sys.path.insert(0, str(fisheye_root))
    return repo_root


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for batch GroundingDINO detection."""
    parser = argparse.ArgumentParser(
        description="Detect person bbox directly in fisheye images with GroundingDINO."
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
        help="Output root directory for bbox json and visualizations.",
    )
    parser.add_argument(
        "--click_points",
        type=str,
        default=None,
        help="Optional path to click_points.json.",
    )
    parser.add_argument(
        "--grounding_dino_config",
        type=str,
        default="tools/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py",
        help="GroundingDINO config path.",
    )
    parser.add_argument(
        "--grounding_dino_weights",
        type=str,
        default="tools/GroundingDINO/weights/groundingdino_swint_ogc.pth",
        help="GroundingDINO weights path.",
    )
    parser.add_argument("--box_threshold", type=float, default=0.35)
    parser.add_argument("--text_threshold", type=float, default=0.25)
    parser.add_argument("--text_prompt", type=str, default="person .")
    return parser.parse_args()


def load_grounding_dino(config_path: str, weights_path: str):
    from groundingdino.util.inference import load_model

    return load_model(config_path, weights_path)


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


def detect_person_bbox_grounding_dino(
    image_bgr: np.ndarray,
    model,
    box_threshold: float,
    text_threshold: float,
    text_prompt: str,
    click_point: list[float] | None,
) -> tuple[int, int, int, int] | None:
    from groundingdino.datasets import transforms as T
    from groundingdino.util.inference import predict

    image_rgb = cv.cvtColor(image_bgr, cv.COLOR_BGR2RGB)
    pil_image = Image.fromarray(image_rgb)

    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image_transformed, _ = transform(pil_image, None)

    boxes, _, phrases = predict(
        model=model,
        image=image_transformed,
        caption=text_prompt,
        box_threshold=box_threshold,
        text_threshold=text_threshold,
    )

    person_boxes: list[np.ndarray] = []
    for box, phrase in zip(boxes, phrases):
        if "person" not in str(phrase).lower():
            continue
        person_boxes.append(np.asarray(box, dtype=np.float32))

    if not person_boxes:
        return None

    h, w = image_bgr.shape[:2]
    boxes_xyxy_list: list[list[float]] = []
    for box in person_boxes:
        cx, cy, bw, bh = box
        x1 = (cx - bw / 2.0) * w
        y1 = (cy - bh / 2.0) * h
        x2 = (cx + bw / 2.0) * w
        y2 = (cy + bh / 2.0) * h
        boxes_xyxy_list.append([float(x1), float(y1), float(x2), float(y2)])

    boxes_xyxy = np.asarray(boxes_xyxy_list, dtype=np.float32)

    if click_point is not None:
        selected_bbox = pick_closest_bbox(boxes_xyxy, click_point[0], click_point[1])
        if selected_bbox is None:
            return None
    else:
        areas = (boxes_xyxy[:, 2] - boxes_xyxy[:, 0]) * (boxes_xyxy[:, 3] - boxes_xyxy[:, 1])
        selected_bbox = boxes_xyxy[int(np.argmax(areas))]

    x1, y1, x2, y2 = [float(value) for value in selected_bbox[:4]]
    x1 = int(np.clip(np.floor(x1), 0, w - 1))
    y1 = int(np.clip(np.floor(y1), 0, h - 1))
    x2 = int(np.clip(np.ceil(x2), x1 + 1, w))
    y2 = int(np.clip(np.ceil(y2), y1 + 1, h))
    return x1, y1, x2, y2


def draw_detection(
    image_bgr: np.ndarray,
    best_bbox: tuple[int, int, int, int] | None,
    click_point: list[float] | None,
) -> np.ndarray:
    vis_image = image_bgr.copy()
    if best_bbox is not None:
        x1, y1, x2, y2 = best_bbox
        cv.rectangle(vis_image, (x1, y1), (x2, y2), (0, 255, 0), 3)
    if click_point is not None:
        click_x = int(round(float(click_point[0])))
        click_y = int(round(float(click_point[1])))
        cv.circle(vis_image, (click_x, click_y), 8, (0, 0, 255), -1)
    return vis_image


def _process_camera_frame(
    model,
    image_path: Path,
    vis_dir: Path,
    frame_idx: int,
    camera_name: str,
    click_point: list[float] | None,
    box_threshold: float,
    text_threshold: float,
    text_prompt: str,
) -> list[int] | None:
    """Detect person bbox in one fisheye image and save visualization."""
    image = cv.imread(str(image_path))
    if image is None:
        print(f"Warning: image not found, skipping: {image_path}")
        return None

    best_bbox = detect_person_bbox_grounding_dino(
        image_bgr=image,
        model=model,
        box_threshold=box_threshold,
        text_threshold=text_threshold,
        text_prompt=text_prompt,
        click_point=click_point,
    )
    if best_bbox is None:
        print(f"Warning: no detection for frame {frame_idx:05d} {camera_name} ({image_path})")

    vis_image = draw_detection(
        image_bgr=image,
        best_bbox=best_bbox,
        click_point=click_point,
    )
    camera_vis_dir = vis_dir / camera_name
    camera_vis_dir.mkdir(parents=True, exist_ok=True)
    cv.imwrite(str(camera_vis_dir / f"{frame_idx:05d}.jpg"), vis_image)

    if best_bbox is not None:
        return list(best_bbox)
    return None


def main() -> None:
    _ensure_paths()
    args = parse_args()

    model = load_grounding_dino(args.grounding_dino_config, args.grounding_dino_weights)
    reviewed_pairs = load_reviewed_entries(Path(args.reviewed))
    click_points = load_click_points(
        Path(args.click_points) if args.click_points is not None else None
    )
    frame_indices = [frame_idx for _, frame_idx in reviewed_pairs]
    frame_to_click_point = build_frame_to_click_point(frame_indices, click_points)

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    vis_dir = output_dir / "vis"

    cam2_bboxes: dict[str, list[int]] = {}
    cam4_bboxes: dict[str, list[int]] = {}

    with tqdm(total=len(reviewed_pairs) * 2, desc="Detecting persons", unit="frame") as progress:
        for _, frame_idx in reviewed_pairs:
            for camera_name, bbox_dict in [("cam2", cam2_bboxes), ("cam4", cam4_bboxes)]:
                click_point = frame_to_click_point.get(camera_name, {}).get(str(frame_idx))
                bbox = _process_camera_frame(
                    model=model,
                    image_path=input_dir / camera_name / f"{frame_idx:05d}.jpg",
                    vis_dir=vis_dir,
                    frame_idx=frame_idx,
                    camera_name=camera_name,
                    click_point=click_point,
                    box_threshold=args.box_threshold,
                    text_threshold=args.text_threshold,
                    text_prompt=args.text_prompt,
                )
                if bbox is not None:
                    bbox_dict[str(frame_idx)] = bbox
                progress.update(1)

    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "cam2_bboxes.json").open("w", encoding="utf-8") as f:
        json.dump(cam2_bboxes, f, indent=2)
    with (output_dir / "cam4_bboxes.json").open("w", encoding="utf-8") as f:
        json.dump(cam4_bboxes, f, indent=2)
    print(f"✅ cam2: {len(cam2_bboxes)} 帧, cam4: {len(cam4_bboxes)} 帧")
    print(f"结果保存到 {output_dir}")


if __name__ == "__main__":
    main()

'''
python src\script\test_function\debug040_detect_grounding_dino.py `
  --reviewed data_process\Input\20260311_riding\review.json `
  --input_dir data_process\Output\20260311_riding\image\fisheye `
  --output_dir data_process\Output\20260311_riding\bbox_grounding_dino `
  --click_points data_process\Output\20260311_riding\click_points.json `
  --box_threshold 0.35 `
  --text_prompt "person ."
'''
