from __future__ import annotations

import json
from pathlib import Path


def normalize_main_camera_path(video_path: str) -> str | None:
    if video_path.endswith(".0.mp4"):
        return video_path
    if video_path.endswith(".1.mp4"):
        return video_path[:-6] + ".0.mp4"
    return None


def load_reviewed_entries(reviewed_path: Path) -> list[tuple[str, int]]:
    with reviewed_path.open("r", encoding="utf-8") as file:
        reviewed = json.load(file)

    if not isinstance(reviewed, list):
        raise ValueError("reviewed.json must contain a list")

    unique_pairs: set[tuple[str, int]] = set()
    for item in reviewed:
        if not isinstance(item, dict):
            raise ValueError("Each reviewed entry must be a dict")
        video_path = normalize_main_camera_path(str(item["video_path"]))
        if video_path is None:
            print(f"Warning: unsupported video path pattern, skipping: {item['video_path']}")
            continue
        unique_pairs.add((video_path, int(item["frame_idx"])))

    return sorted(unique_pairs)


def load_reviewed_frame_indices(
    reviewed_path: Path,
    *,
    require_frame_idx: bool = True,
    normalize_video_path: bool = False,
) -> list[int]:
    with reviewed_path.open("r", encoding="utf-8") as file:
        reviewed = json.load(file)

    if not isinstance(reviewed, list):
        raise ValueError("reviewed.json must contain a list")

    frame_indices: set[int] = set()
    for item in reviewed:
        if not isinstance(item, dict):
            if require_frame_idx:
                raise ValueError("Each reviewed entry must be a dict")
            continue
        if "frame_idx" not in item:
            if require_frame_idx:
                raise ValueError("Each reviewed entry must contain frame_idx")
            continue
        if normalize_video_path:
            video_path = normalize_main_camera_path(str(item.get("video_path", "")))
            if video_path is None:
                print(f"Warning: unsupported video path pattern, skipping: {item.get('video_path')}")
                continue
        frame_indices.add(int(item["frame_idx"]))

    return sorted(frame_indices)
