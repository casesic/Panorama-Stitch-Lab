from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2

import config_script
from batch_utils import load_reviewed_entries


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a preview video from composite human blend outputs.")
    parser.add_argument("--composite_dir", type=str, required=True, help="composite_human root directory.")
    parser.add_argument("--reviewed", type=str, required=True, help="Path to reviewed.json.")
    parser.add_argument("--output", type=str, required=True, help="Output video path.")
    parser.add_argument("--fps", type=float, default=3.0, help="Video FPS.")
    parser.add_argument(
        "--direction",
        type=str,
        choices=("cam2_to_cam4", "cam4_to_cam2"),
        default="cam2_to_cam4",
        help="Composite direction to use.",
    )
    parser.add_argument(
        "--invalid_json",
        type=str,
        default=None,
        help="Optional json file containing invalid frame indices to skip.",
    )
    return parser.parse_args()


def _extract_frame_indices(obj) -> set[int]:
    frame_indices: set[int] = set()

    if isinstance(obj, int):
        frame_indices.add(int(obj))
        return frame_indices

    if isinstance(obj, list):
        for item in obj:
            frame_indices.update(_extract_frame_indices(item))
        return frame_indices

    if isinstance(obj, dict):
        for key in ("frame_idx", "frame_idxs", "frames", "invalid_frames"):
            if key in obj:
                frame_indices.update(_extract_frame_indices(obj[key]))
        for value in obj.values():
            frame_indices.update(_extract_frame_indices(value))
        return frame_indices

    return frame_indices


def load_invalid_frames(invalid_json_path: Path | None) -> set[int]:
    if invalid_json_path is None:
        return set()
    if not invalid_json_path.exists():
        raise FileNotFoundError(f"invalid_json file not found: {invalid_json_path}")
    with invalid_json_path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    return _extract_frame_indices(data)


def get_frame_path(composite_dir: Path, direction: str, frame_idx: int) -> Path:
    return composite_dir / direction / "blend" / f"{frame_idx:05d}.jpg"


def main() -> int:
    args = parse_args()

    composite_dir = Path(args.composite_dir)
    reviewed_path = Path(args.reviewed)
    output_path = Path(args.output)
    invalid_frames = load_invalid_frames(Path(args.invalid_json) if args.invalid_json else None)

    reviewed_pairs = load_reviewed_entries(reviewed_path)
    frame_indices = [frame_idx for _, frame_idx in reviewed_pairs]

    writer = None
    target_size = None
    written_frames = 0

    for frame_idx in frame_indices:
        if frame_idx in invalid_frames:
            continue

        frame_path = get_frame_path(composite_dir, args.direction, frame_idx)
        if not frame_path.exists():
            continue

        image = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
        if image is None:
            continue

        if writer is None:
            h, w = image.shape[:2]
            target_size = (w, h)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            writer = cv2.VideoWriter(
                str(output_path),
                cv2.VideoWriter_fourcc(*"mp4v"),
                args.fps,
                target_size,
            )

        if image.shape[1] != target_size[0] or image.shape[0] != target_size[1]:
            image = cv2.resize(image, target_size, interpolation=cv2.INTER_LINEAR)

        writer.write(image)
        written_frames += 1

    if writer is None:
        print("[warn] No valid frames were loaded.")
        return 1

    writer.release()
    print(f"Saved video to: {output_path}")
    print(f"Direction: {args.direction}")
    print(f"Invalid frame count: {len(invalid_frames)}")
    print(f"Total frames written: {written_frames}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


r'''
python src\script\test_function\debug015_make_video.py `
  --composite_dir data_process\Output\20260311_riding\composite_human\no_sil `
  --reviewed data_process\Input\20260311_riding\review.json `
  --output data_process\Output\20260311_riding\result_video\preview_video_no_sil.mp4 `
  --direction cam2_to_cam4 `
  --invalid_json data_process\Output\20260311_riding\composite_human\no_sil\invalid.json
'''

r'''
python src\script\test_function\debug015_make_video.py `
  --composite_dir data_process\Output\20260320_riding_dataset1\composite_human\no_sil `
  --reviewed data_process\Input\20260320_riding\dataset1\review.json `
  --output data_process\Output\20260320_riding_dataset1\result_video\preview_video_no_sil.mp4 `
  --direction cam4_to_cam2 `
  --invalid_json data_process\Output\20260320_riding_dataset1\composite_human\no_sil\invalid.json
'''