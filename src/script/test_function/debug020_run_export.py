from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for reviewed-frame export."""
    parser = argparse.ArgumentParser(description="Export reviewed fisheye frames for both cameras.")
    parser.add_argument(
        "--reviewed",
        type=str,
        required=True,
        help="Path to reviewed.json.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output root directory.",
    )
    return parser.parse_args()


def _normalize_main_camera_path(video_path: str) -> str | None:
    """Normalize a reviewed video path to the main camera (.0.mp4) path."""
    if video_path.endswith(".0.mp4"):
        return video_path
    if video_path.endswith(".1.mp4"):
        return video_path[:-6] + ".0.mp4"
    return None


def _get_partner_camera_path(video_path: str) -> str | None:
    """Derive the partner camera (.1.mp4) path from a main camera path."""
    if not video_path.endswith(".0.mp4"):
        return None
    return video_path[:-6] + ".1.mp4"


def load_reviewed_entries(reviewed_path: Path) -> list[tuple[str, int]]:
    """Load reviewed.json and return deduplicated, sorted (video_path, frame_idx) pairs."""
    with reviewed_path.open("r", encoding="utf-8") as file:
        reviewed = json.load(file)

    if not isinstance(reviewed, list):
        raise ValueError("reviewed.json must contain a list")

    unique_pairs: set[tuple[str, int]] = set()
    for item in reviewed:
        if not isinstance(item, dict):
            raise ValueError("Each reviewed entry must be a dict")
        video_path = _normalize_main_camera_path(str(item["video_path"]))
        if video_path is None:
            print(f"Warning: unsupported video path pattern, skipping: {item['video_path']}")
            continue
        unique_pairs.add((video_path, int(item["frame_idx"])))

    return sorted(unique_pairs)


def _read_frame_sequential(
    cap: cv2.VideoCapture, current_pos: int, target_idx: int
) -> tuple[Any | None, int]:
    """Read a frame using sequential advance when gap is small, seek otherwise.

    Returns:
        Tuple of (frame_or_None, new_position).
    """
    gap = target_idx - current_pos
    if gap < 0 or gap > 50:
        cap.set(cv2.CAP_PROP_POS_FRAMES, target_idx)
        ok, frame = cap.read()
    else:
        for _ in range(gap - 1):
            cap.grab()
        ok, frame = cap.read()

    new_pos = target_idx + 1 if ok else target_idx
    return (frame if ok else None), new_pos


def _resize_frame(frame: Any) -> Any:
    """Resize one frame to 2880x2880 using area interpolation."""
    return cv2.resize(frame, (2880, 2880), interpolation=cv2.INTER_AREA)


def _encode_jpeg(frame: Any) -> bytes | None:
    """Encode one frame to JPEG bytes with quality 95."""
    ok, encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    if not ok:
        return None
    return encoded.tobytes()


def export_group(
    main_video_path: str,
    frame_indices: list[int],
    cam2_dir: Path,
    cam4_dir: Path,
    progress: tqdm,
) -> None:
    """Export one video's reviewed frames for both cameras."""
    partner_video_path = _get_partner_camera_path(main_video_path)
    if partner_video_path is None:
        for frame_idx in frame_indices:
            print(f"Warning: failed to derive partner camera path for {main_video_path} @ {frame_idx}")
            progress.update(1)
        return

    if not Path(main_video_path).exists():
        for frame_idx in frame_indices:
            print(f"Warning: main camera video not found, skipping: {main_video_path} @ {frame_idx}")
            progress.update(1)
        return

    if not Path(partner_video_path).exists():
        for frame_idx in frame_indices:
            print(f"Warning: partner camera video not found, skipping: {partner_video_path} @ {frame_idx}")
            progress.update(1)
        return

    cap_main = cv2.VideoCapture(main_video_path)
    cap_partner = cv2.VideoCapture(partner_video_path)
    try:
        if not cap_main.isOpened() or not cap_partner.isOpened():
            for frame_idx in frame_indices:
                print(f"Warning: failed to open paired videos, skipping: {main_video_path} @ {frame_idx}")
                progress.update(1)
            return

        pos_main = 0
        pos_partner = 0
        with ThreadPoolExecutor(max_workers=2) as pool:
            for frame_idx in frame_indices:
                future_main = pool.submit(_read_frame_sequential, cap_main, pos_main, frame_idx)
                future_partner = pool.submit(_read_frame_sequential, cap_partner, pos_partner, frame_idx)
                main_frame, pos_main = future_main.result()
                partner_frame, pos_partner = future_partner.result()
                if main_frame is None or partner_frame is None:
                    print(f"Warning: failed to extract paired frames, skipping: {main_video_path} @ {frame_idx}")
                    progress.update(1)
                    continue

                main_output = cam2_dir / f"{frame_idx:05d}.jpg"
                partner_output = cam4_dir / f"{frame_idx:05d}.jpg"
                main_resized = _resize_frame(main_frame)
                partner_resized = _resize_frame(partner_frame)

                main_encoded = _encode_jpeg(main_resized)
                partner_encoded = _encode_jpeg(partner_resized)
                if main_encoded is None or partner_encoded is None:
                    print(f"Warning: failed to save paired frames, skipping: {main_video_path} @ {frame_idx}")
                    progress.update(1)
                    continue

                try:
                    main_output.write_bytes(main_encoded)
                    partner_output.write_bytes(partner_encoded)
                except OSError:
                    print(f"Warning: failed to save paired frames, skipping: {main_video_path} @ {frame_idx}")

                progress.update(1)
    finally:
        cap_main.release()
        cap_partner.release()


def main() -> None:
    """Export reviewed frames for both cameras."""
    args = parse_args()
    reviewed_path = Path(args.reviewed)
    output_dir = Path(args.output_dir)
    cam2_dir = output_dir / "cam2"
    cam4_dir = output_dir / "cam4"
    cam2_dir.mkdir(parents=True, exist_ok=True)
    cam4_dir.mkdir(parents=True, exist_ok=True)

    reviewed_pairs = load_reviewed_entries(reviewed_path)
    grouped_frames: dict[str, list[int]] = defaultdict(list)
    for video_path, frame_idx in reviewed_pairs:
        grouped_frames[video_path].append(frame_idx)

    total_frames = len(reviewed_pairs)
    with tqdm(total=total_frames, desc="Exporting paired frames", unit="frame") as progress:
        for video_path in sorted(grouped_frames):
            export_group(
                video_path,
                grouped_frames[video_path],
                cam2_dir,
                cam4_dir,
                progress,
            )


if __name__ == "__main__":
    main()
