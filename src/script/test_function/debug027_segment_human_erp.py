from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm
import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from batch_utils import load_reviewed_entries


MASK_ALPHA = 0.5
SEG_COLOR = (0, 120, 255)




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
    parser = argparse.ArgumentParser(
        description="Segment humans from ERP crop regions and save mask, RGBA, and visualization outputs."
    )
    parser.add_argument(
        "--crop_dir",
        type=str,
        required=True,
        help="ERP crop root directory containing cam2/ and cam4/ subdirectories",
    )
    parser.add_argument("--output_dir", type=str, required=True, help="Output root directory")
    parser.add_argument(
        "--rtm_dir",
        type=str,
        required=True,
        help="RTM result root directory containing cam2/ and cam4/ subdirectories",
    )
    parser.add_argument(
        "--bbox_dir",
        type=str,
        required=True,
        help="crop bbox json 所在目录（即包含 cam2_bboxes.json 和 cam4_bboxes.json 的目录）",
    )
    parser.add_argument(
        "--sam_model",
        type=str,
        default="sam2_b.pt",
        help='SAM2 model name, default "sam2_b.pt"',
    )
    parser.add_argument("--reviewed", type=str, required=True, help="Path to reviewed.json")
    return parser.parse_args()


def choose_primary_mask(result, image_shape) -> np.ndarray | None:
    if result.masks is None or result.masks.data is None or len(result.masks.data) == 0:
        return None

    masks = result.masks.data.detach().cpu().numpy().astype(bool)
    h, w = image_shape[:2]
    best_mask = None
    best_area = -1

    for mask in masks:
        if mask.shape != (h, w):
            mask = cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
        area = int(mask.sum())
        if area > best_area:
            best_area = area
            best_mask = mask
    return best_mask


def render_segmentation(
    image_bgr: np.ndarray,
    mask: np.ndarray,
    positive_points: list[list[int]] | None = None,
    alpha: float = MASK_ALPHA,
    color: tuple[int, int, int] = SEG_COLOR,
) -> np.ndarray:
    h, w = image_bgr.shape[:2]
    if mask.shape != (h, w):
        mask = cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)

    seg = image_bgr.copy()
    overlay = seg.copy()
    overlay[mask] = color
    seg = cv2.addWeighted(overlay, alpha, seg, 1 - alpha, 0)

    contours, _ = cv2.findContours(mask.astype(np.uint8) * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(seg, contours, -1, (0, 255, 255), 2)
    if positive_points is not None:
        for x, y in positive_points:
            cv2.circle(seg, (int(x), int(y)), 5, (0, 255, 0), -1, lineType=cv2.LINE_AA)
            cv2.circle(seg, (int(x), int(y)), 8, (255, 255, 255), 1, lineType=cv2.LINE_AA)
    return seg


def load_models(sam_model_name: str):
    from ultralytics import SAM

    return SAM(sam_model_name)


def load_bbox_dict(bbox_dir: Path) -> tuple[dict[str, list[int]], dict[str, list[int]]]:
    with (bbox_dir / "cam2_bboxes.json").open("r", encoding="utf-8") as file:
        cam2_bboxes = json.load(file)
    with (bbox_dir / "cam4_bboxes.json").open("r", encoding="utf-8") as file:
        cam4_bboxes = json.load(file)

    if not isinstance(cam2_bboxes, dict):
        raise ValueError(f"bbox json must contain a dict: {bbox_dir / 'cam2_bboxes.json'}")
    if not isinstance(cam4_bboxes, dict):
        raise ValueError(f"bbox json must contain a dict: {bbox_dir / 'cam4_bboxes.json'}")
    return cam2_bboxes, cam4_bboxes


def segment_with_keypoint_prompt(
    sam_model,
    image_bgr: np.ndarray,
    positive_points: list[list[int]],
) -> np.ndarray | None:
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    points = positive_points
    labels = [1] * len(positive_points)
    results = sam_model(image_rgb, points=points, labels=labels, verbose=False)
    if not results:
        return None
    return choose_primary_mask(results[0], image_bgr.shape)


def _process_camera_frame(
    sam_model,
    crop_dir: Path,
    rtm_dir: Path,
    output_dir: Path,
    camera_name: str,
    camera_suffix: str,
    frame_idx: int,
    bbox_dict: dict[str, list[int]],
) -> None:
    image_path = crop_dir / camera_name / f"{frame_idx:05d}.jpg"
    image_bgr = cv2.imread(str(image_path))
    if image_bgr is None:
        print(f"Warning: crop image not found, skipping {camera_name} frame {frame_idx:05d}")
        return

    bbox_xyxy = bbox_dict.get(str(frame_idx))
    if bbox_xyxy is None:
        print(f"Warning: bbox not found, skipping {camera_name} frame {frame_idx:05d}")
        return

    rtm_path = rtm_dir / camera_name / f"rtm_s{frame_idx:05d}_{camera_suffix}.pkl"
    if not rtm_path.exists():
        print(f"Warning: RTM pkl not found, skipping {camera_name} frame {frame_idx:05d}")
        return

    with rtm_path.open("rb") as file:
        det_seq = pickle.load(file)

    if not bool(det_seq.array_valid[0]):
        print(f"Warning: invalid RTM detection, skipping {camera_name} frame {frame_idx:05d}")
        return

    h, w = image_bgr.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in bbox_xyxy]
    del x2, y2
    body_keypoints = np.asarray(det_seq.array_keypoints[0], dtype=np.float32)[:17]
    valid_points: list[list[int]] = []
    for x, y, conf in body_keypoints:
        if conf <= 0.6:
            continue
        x_local = x - x1
        y_local = y - y1
        if not (0 <= x_local < w and 0 <= y_local < h):
            continue
        pixel = image_bgr[int(round(y_local)), int(round(x_local))]
        if int(pixel[0]) < 10 and int(pixel[1]) < 10 and int(pixel[2]) < 10:
            continue
        valid_points.append([int(round(x_local)), int(round(y_local))])

    if not valid_points:
        print(f"Warning: no valid RTM body keypoints, skipping {camera_name} frame {frame_idx:05d}")
        return

    mask = segment_with_keypoint_prompt(sam_model, image_bgr, positive_points=valid_points)
    if mask is None:
        print(f"Warning: failed to segment human, skipping {camera_name} frame {frame_idx:05d}")
        return

    mask_dir = output_dir / camera_name / "mask"
    rgba_dir = output_dir / camera_name / "rgba"
    vis_dir = output_dir / camera_name / "vis"
    mask_dir.mkdir(parents=True, exist_ok=True)
    rgba_dir.mkdir(parents=True, exist_ok=True)
    vis_dir.mkdir(parents=True, exist_ok=True)

    mask_uint8 = mask.astype(np.uint8) * 255
    cv2.imwrite(str(mask_dir / f"{frame_idx:05d}.png"), mask_uint8)

    rgba_image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2BGRA)
    rgba_image[~mask, 3] = 0
    cv2.imwrite(str(rgba_dir / f"{frame_idx:05d}.png"), rgba_image)

    vis_image = render_segmentation(image_bgr, mask, positive_points=valid_points)
    cv2.imwrite(
        str(vis_dir / f"{frame_idx:05d}.jpg"),
        vis_image,
        [cv2.IMWRITE_JPEG_QUALITY, 92],
    )


def main() -> None:
    _ensure_paths()
    args = parse_args()

    crop_dir = Path(args.crop_dir)
    output_dir = Path(args.output_dir)
    rtm_dir = Path(args.rtm_dir)
    bbox_dir = Path(args.bbox_dir)
    reviewed_path = Path(args.reviewed)

    reviewed_pairs = load_reviewed_entries(reviewed_path)
    cam2_bboxes, cam4_bboxes = load_bbox_dict(bbox_dir)
    sam_model = load_models(args.sam_model)

    total_steps = len(reviewed_pairs) * 2
    with tqdm(total=total_steps, desc="Segmenting ERP crops", unit="frame") as progress:
        for _, frame_idx in reviewed_pairs:
            _process_camera_frame(
                sam_model=sam_model,
                crop_dir=crop_dir,
                rtm_dir=rtm_dir,
                output_dir=output_dir,
                camera_name="cam2",
                camera_suffix="c2",
                frame_idx=frame_idx,
                bbox_dict=cam2_bboxes,
            )
            progress.update(1)

            _process_camera_frame(
                sam_model=sam_model,
                crop_dir=crop_dir,
                rtm_dir=rtm_dir,
                output_dir=output_dir,
                camera_name="cam4",
                camera_suffix="c4",
                frame_idx=frame_idx,
                bbox_dict=cam4_bboxes,
            )
            progress.update(1)


if __name__ == "__main__":
    main()

'''
python src\script\test_function\debug027_segment_human_erp.py `
  --crop_dir data_process\Output\20260311_riding\image\equirect_crop `
  --rtm_dir data_process\Output\20260311_riding\rtm_cache\equirect `
  --bbox_dir data_process\Output\20260311_riding\image\equirect_crop `
  --output_dir data_process\Output\20260311_riding\seg `
  --reviewed data_process\Input\20260311_riding\review.json
'''
