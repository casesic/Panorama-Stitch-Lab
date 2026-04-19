from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_SEG_PATH = Path(r"data_process\Input\20260311_riding\seg.json")
DEFAULT_OUTPUT_PATH = Path(r"data_process\Input\20260311_riding\review.json")

r"""
python src\script\test_function\debug034_generate_review.py `
    --seg data_process\Input\20260311_riding\seg.json `
    --output data_process\Input\20260311_riding\review.json `
    --video_path Circular_Video\00_send\20260311_riding\VID_20181003_034909_00_982.0.mp4
"""


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Expand seg.json ranges into review.json frame entries.")
    parser.add_argument("--seg", type=str, default=str(DEFAULT_SEG_PATH), help="Path to seg.json")
    parser.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT_PATH), help="Path to output review.json")
    parser.add_argument(
        "--video_path",
        type=str,
        default=None,
        help="Video path for all generated review entries. If omitted, try to infer from nearby JSON files.",
    )
    return parser.parse_args()


def load_segments(seg_path: Path) -> list[dict[str, int]]:
    """Load and validate segments from seg.json.

    Args:
        seg_path: Path to seg.json.

    Returns:
        Validated segment list containing integer ``start`` and ``end`` fields.
    """
    with seg_path.open("r", encoding="utf-8") as file:
        payload = json.load(file)

    if not isinstance(payload, list):
        raise ValueError("seg.json must contain a list")

    segments: list[dict[str, int]] = []
    for index, item in enumerate(payload):
        if not isinstance(item, dict):
            raise ValueError(f"Segment at index {index} must be a dict")
        if "start" not in item or "end" not in item:
            raise ValueError(f"Segment at index {index} must contain start and end")

        start = int(item["start"])
        end = int(item["end"])
        if end < start:
            raise ValueError(f"Segment at index {index} has end < start: {start} > {end}")

        segments.append({"start": start, "end": end})

    return segments


def infer_video_path(seg_path: Path) -> str:
    """Infer video_path from nearby JSON files in the same directory.

    Args:
        seg_path: Path to seg.json.

    Returns:
        Inferred video path string.
    """
    parent = seg_path.parent
    candidate_files = [
        parent / "candidates.json",
        parent / "review.json",
    ]
    for candidate_file in candidate_files:
        video_path = extract_video_path_from_json(candidate_file)
        if video_path is not None:
            return video_path

    raise ValueError(
        "Failed to infer video_path from nearby JSON files. "
        "Please pass --video_path explicitly.",
    )


def extract_video_path_from_json(json_path: Path) -> str | None:
    """Extract the first available video_path from a JSON file.

    Args:
        json_path: JSON file to inspect.

    Returns:
        The first discovered video_path, or ``None`` if not found.
    """
    if not json_path.exists():
        return None

    with json_path.open("r", encoding="utf-8") as file:
        payload = json.load(file)

    return find_first_video_path(payload)


def find_first_video_path(payload: Any) -> str | None:
    """Recursively search for the first string-valued video_path field.

    Args:
        payload: Arbitrary JSON-decoded payload.

    Returns:
        The first discovered video_path, or ``None`` if not found.
    """
    if isinstance(payload, dict):
        video_path = payload.get("video_path")
        if isinstance(video_path, str) and video_path:
            return video_path
        for value in payload.values():
            nested = find_first_video_path(value)
            if nested is not None:
                return nested
        return None

    if isinstance(payload, list):
        for item in payload:
            nested = find_first_video_path(item)
            if nested is not None:
                return nested

    return None


def build_review_entries(segments: list[dict[str, int]], video_path: str) -> list[dict[str, Any]]:
    """Expand segments into per-frame review entries.

    Args:
        segments: Segment list with inclusive ``start`` and ``end``.
        video_path: Shared video path for every output entry.

    Returns:
        Sorted review entries.
    """
    entries: list[dict[str, Any]] = []
    for segment in segments:
        for frame_idx in range(segment["start"], segment["end"] + 1):
            entries.append(
                {
                    "video_path": video_path,
                    "frame_idx": frame_idx,
                },
            )

    entries.sort(key=lambda item: int(item["frame_idx"]))
    return entries


def save_review(entries: list[dict[str, Any]], output_path: Path) -> None:
    """Write review entries to review.json.

    Args:
        entries: Review entry list.
        output_path: Output JSON path.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(entries, file, ensure_ascii=False, indent=2)


def main() -> None:
    """Run the review generation script."""
    args = parse_args()

    seg_path = Path(args.seg)
    output_path = Path(args.output)
    video_path = args.video_path or infer_video_path(seg_path)

    segments = load_segments(seg_path)
    entries = build_review_entries(segments, video_path)
    save_review(entries, output_path)

    print(f"Generated {len(entries)} review entries -> {output_path}")


if __name__ == "__main__":
    main()
