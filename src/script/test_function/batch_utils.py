import json
from pathlib import Path


def _normalize_main_camera_path(video_path: str) -> str | None:
    """Normalize a reviewed video path to the main camera (.0.mp4) path."""
    if video_path.endswith(".0.mp4"):
        return video_path
    if video_path.endswith(".1.mp4"):
        return video_path[:-6] + ".0.mp4"
    return None


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
