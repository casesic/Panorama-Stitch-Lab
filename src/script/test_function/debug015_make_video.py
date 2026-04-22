from __future__ import annotations

import argparse
from pathlib import Path

import cv2

import config_script
from batch_utils import load_reviewed_entries


INVALID_FRAMES = {
    710,1217,1895,1878,1898,2958,2960,2962,2965,2966,3650,3653,3767,3770,4623,
    4624,5075,5078,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a preview video from composite human blend outputs.")
    parser.add_argument("--composite_dir", type=str, required=True, help="composite_human root directory.")
    parser.add_argument("--reviewed", type=str, required=True, help="Path to reviewed.json.")
    parser.add_argument("--output", type=str, required=True, help="Output video path.")
    parser.add_argument("--fps", type=float, default=3.0, help="Video FPS.")
    return parser.parse_args()


def get_frame_path(composite_dir: Path, frame_idx: int) -> Path:
    return composite_dir / "cam2_to_cam4" / "blend" / f"{frame_idx:05d}.jpg"


def main() -> int:
    args = parse_args()

    composite_dir = Path(args.composite_dir)
    reviewed_path = Path(args.reviewed)
    output_path = Path(args.output)

    reviewed_pairs = load_reviewed_entries(reviewed_path)
    frame_indices = [frame_idx for _, frame_idx in reviewed_pairs]

    writer = None
    target_size = None
    written_frames = 0

    for frame_idx in frame_indices:
        if frame_idx in INVALID_FRAMES:
            continue

        frame_path = get_frame_path(composite_dir, frame_idx)
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
    print(f"Total frames written: {written_frames}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


r'''
python src\script\test_function\debug015_make_video.py `
  --composite_dir data_process\Output\20260311_riding\composite_human `
  --reviewed data_process\Input\20260311_riding\review.json `
  --output data_process\Output\20260311_riding\preview_video.mp4
'''
