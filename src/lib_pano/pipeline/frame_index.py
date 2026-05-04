from __future__ import annotations

import re
from pathlib import Path


_SEGMENT_PATTERN = re.compile(r"^seg_(\d+)_(\d+)_seq\.pkl$")


def parse_frame_idx(path_or_name: str | Path) -> int | None:
    stem = Path(path_or_name).stem
    long_matches = re.findall(r"\d{5,}", stem)
    if long_matches:
        return int(long_matches[0])

    matches = re.findall(r"\d+", stem)
    if not matches:
        return None

    if matches[-1] in ("2", "4") and len(matches) > 1:
        return int(matches[-2])

    return int(matches[-1])


def frame_key(frame_idx: int, width: int = 5) -> str:
    return f"{int(frame_idx):0{width}d}"


def discover_frame_indices(fitting_dir: Path) -> list[int]:
    cam2_dir = fitting_dir / "cam2"
    frame_indices: set[int] = set()
    for pkl_path in cam2_dir.glob("seg_*_*_seq.pkl"):
        match = _SEGMENT_PATTERN.match(pkl_path.name)
        if match is None:
            continue
        start_frame = int(match.group(1))
        end_frame = int(match.group(2))
        frame_indices.update(range(start_frame, end_frame + 1))
    return sorted(frame_indices)


def split_frame_segments(frame_indices: list[int], gap_threshold: int = 2) -> list[list[int]]:
    if not frame_indices:
        return []

    segments: list[list[int]] = [[frame_indices[0]]]
    for frame_idx in frame_indices[1:]:
        if frame_idx - segments[-1][-1] > gap_threshold:
            segments.append([frame_idx])
        else:
            segments[-1].append(frame_idx)
    return segments
