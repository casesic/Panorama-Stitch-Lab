from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from batch_utils import load_reviewed_frame_indices, split_frame_segments
from lib_pano.pipeline.bbox_io import bbox_area, clip_bbox, load_bbox_dict, save_bbox_dict, union_bboxes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smooth equirect-coordinate bboxes with temporal window union.")
    parser.add_argument("--reviewed", type=str, required=True, help="Path to reviewed.json.")
    parser.add_argument("--input_dir", type=str, required=True, help="Input bbox directory from debug040.")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for smoothed bbox json files.")
    parser.add_argument("--image_dir", type=str, default=None, help="Optional equirect_rotate image root for visualization.")
    parser.add_argument("--image_width", type=int, default=5760, help="Image width used for clipping.")
    parser.add_argument("--image_height", type=int, default=2880, help="Image height used for clipping.")
    parser.add_argument("--window", type=int, default=2, help="Sliding window radius in each continuous segment.")
    parser.add_argument("--max_gap", type=int, default=2, help="Maximum frame gap kept in one continuous segment.")
    parser.add_argument("--min_area_ratio", type=float, default=0.65, help="Reserved shrink detection ratio.")
    parser.add_argument(
        "--max_expand_ratio",
        type=float,
        default=2.0,
        help="Reject union bbox when its area is larger than raw bbox area times this ratio.",
    )
    parser.add_argument("--save_vis", action="store_true", help="Save bbox visualization when --image_dir is provided.")
    return parser.parse_args()


def draw_labeled_bbox(image: np.ndarray, bbox: list[int], color: tuple[int, int, int], label: str) -> None:
    x1, y1, x2, y2 = bbox
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 3)
    text_y = max(20, y1 - 8)
    cv2.putText(image, label, (x1, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)


def save_frame_visualization(
    camera_name: str,
    frame_idx: int,
    raw_bbox: list[int] | None,
    smooth_bbox: list[int],
    image_dir: Path,
    output_dir: Path,
) -> None:
    image_path = image_dir / camera_name / f"{frame_idx:05d}.jpg"
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        print(f"Warning: image not found for visualization: {image_path}")
        return

    if raw_bbox is not None:
        draw_labeled_bbox(image, raw_bbox, (0, 255, 0), "raw")
    draw_labeled_bbox(image, smooth_bbox, (0, 0, 255), "smooth")

    vis_dir = output_dir / "vis" / camera_name
    vis_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(vis_dir / f"{frame_idx:05d}.jpg"), image, [cv2.IMWRITE_JPEG_QUALITY, 95])


def process_camera_bboxes(
    camera_name: str,
    frame_indices: list[int],
    raw_bboxes: dict[str, list[int]],
    image_width: int,
    image_height: int,
    window: int,
    max_gap: int,
    max_expand_ratio: float,
    image_dir: Path | None,
    output_dir: Path,
    save_vis: bool,
    progress: tqdm,
) -> dict[str, list[int]]:
    smoothed: dict[str, list[int]] = {}

    for segment in split_frame_segments(frame_indices, gap_threshold=max_gap):
        for idx, frame_idx in enumerate(segment):
            lo = max(0, idx - window)
            hi = min(len(segment), idx + window + 1)
            window_frames = segment[lo:hi]

            raw_bbox = (
                clip_bbox(raw_bboxes.get(str(frame_idx)), image_width, image_height)
                if str(frame_idx) in raw_bboxes
                else None
            )

            window_bboxes = []
            for i in window_frames:
                bbox = raw_bboxes.get(str(i))
                if bbox is None:
                    continue
                bbox = clip_bbox(bbox, image_width, image_height)
                if bbox is not None:
                    window_bboxes.append(bbox)

            union_bbox = union_bboxes(window_bboxes)
            union_bbox = clip_bbox(union_bbox, image_width, image_height) if union_bbox is not None else None

            smooth_bbox = None
            if raw_bbox is None:
                smooth_bbox = union_bbox
            elif union_bbox is None:
                smooth_bbox = raw_bbox
            else:
                raw_area = bbox_area(raw_bbox)
                union_area = bbox_area(union_bbox)
                if raw_area > 0 and union_area > raw_area * max_expand_ratio:
                    print(
                        f"Warning: reject union bbox for frame {frame_idx:05d}: "
                        f"raw_area={raw_area}, union_area={union_area}, ratio={union_area / raw_area:.2f}"
                    )
                    smooth_bbox = raw_bbox
                else:
                    smooth_bbox = union_bbox

            if smooth_bbox is not None:
                smoothed[str(frame_idx)] = smooth_bbox
                if save_vis and image_dir is not None:
                    save_frame_visualization(camera_name, frame_idx, raw_bbox, smooth_bbox, image_dir, output_dir)

            progress.update(1)

    return smoothed


def maybe_infer_image_size(image_dir: Path | None, frame_indices: list[int], image_width: int, image_height: int) -> tuple[int, int]:
    if image_dir is None:
        return image_width, image_height

    for frame_idx in frame_indices:
        for camera_name in ("cam2", "cam4"):
            image_path = image_dir / camera_name / f"{frame_idx:05d}.jpg"
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image is not None:
                height, width = image.shape[:2]
                return width, height
    print("Warning: failed to infer image size from --image_dir, using --image_width/--image_height")
    return image_width, image_height


def main() -> None:
    args = parse_args()

    reviewed_path = Path(args.reviewed)
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    image_dir = Path(args.image_dir) if args.image_dir else None
    output_dir.mkdir(parents=True, exist_ok=True)

    frame_indices = load_reviewed_frame_indices(reviewed_path, require_frame_idx=False)
    image_width, image_height = maybe_infer_image_size(image_dir, frame_indices, args.image_width, args.image_height)

    cam2_raw = load_bbox_dict(input_dir / "cam2_bboxes.json", normalize_values=True)
    cam4_raw = load_bbox_dict(input_dir / "cam4_bboxes.json", normalize_values=True)
    total_steps = len(frame_indices) * 2
    with tqdm(total=total_steps, desc="Smoothing bboxes", unit="frame") as progress:
        cam2_smooth = process_camera_bboxes(
            camera_name="cam2",
            frame_indices=frame_indices,
            raw_bboxes=cam2_raw,
            image_width=image_width,
            image_height=image_height,
            window=max(0, args.window),
            max_gap=args.max_gap,
            max_expand_ratio=args.max_expand_ratio,
            image_dir=image_dir,
            output_dir=output_dir,
            save_vis=args.save_vis,
            progress=progress,
        )
        cam4_smooth = process_camera_bboxes(
            camera_name="cam4",
            frame_indices=frame_indices,
            raw_bboxes=cam4_raw,
            image_width=image_width,
            image_height=image_height,
            window=max(0, args.window),
            max_gap=args.max_gap,
            max_expand_ratio=args.max_expand_ratio,
            image_dir=image_dir,
            output_dir=output_dir,
            save_vis=args.save_vis,
            progress=progress,
        )

    save_bbox_dict(output_dir / "cam2_bboxes.json", cam2_smooth, ensure_ascii=True)
    save_bbox_dict(output_dir / "cam4_bboxes.json", cam4_smooth, ensure_ascii=True)

    print(f"cam2 bboxes: raw={len(cam2_raw)}, smooth={len(cam2_smooth)}")
    print(f"cam4 bboxes: raw={len(cam4_raw)}, smooth={len(cam4_smooth)}")
    print(f"saved smoothed bboxes to: {output_dir}")


if __name__ == "__main__":
    main()


r"""
python src\script\test_function\debug041_smooth_bboxes.py `
  --reviewed data_process\Input\20260320_riding\dataset1\review.json `
  --input_dir data_process\Output\20260320_riding_dataset1\bbox_grounding_dino\equirect_rotate `
  --output_dir data_process\Output\20260320_riding_dataset1\bbox_grounding_dino\equirect_rotate_smooth `
  --image_dir data_process\Output\20260320_riding_dataset1\image\equirect_rotate `
  --window 2 `
  --max_gap 2 `
  --max_expand_ratio 2.0 `
  --save_vis
"""
