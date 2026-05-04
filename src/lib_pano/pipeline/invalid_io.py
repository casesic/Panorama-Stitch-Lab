from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _extract_frame_indices(obj: Any, *, include_problem_frames: bool = False, recurse_all: bool = True) -> set[int]:
    frame_indices: set[int] = set()

    if isinstance(obj, int):
        frame_indices.add(int(obj))
        return frame_indices

    if isinstance(obj, list):
        for item in obj:
            frame_indices.update(
                _extract_frame_indices(item, include_problem_frames=include_problem_frames, recurse_all=recurse_all)
            )
        return frame_indices

    if isinstance(obj, dict):
        if "frame_idx" in obj:
            frame_indices.update(
                _extract_frame_indices(obj["frame_idx"], include_problem_frames=include_problem_frames, recurse_all=recurse_all)
            )
        if "invalid_frames" in obj:
            frame_indices.update(
                _extract_frame_indices(
                    obj["invalid_frames"], include_problem_frames=include_problem_frames, recurse_all=recurse_all
                )
            )
        if include_problem_frames and "problem_frames" in obj:
            frame_indices.update(
                _extract_frame_indices(
                    obj["problem_frames"], include_problem_frames=include_problem_frames, recurse_all=recurse_all
                )
            )
        for key in ("frame_idxs", "frames", "invalid_detections", "cam2", "cam4"):
            if key in obj:
                frame_indices.update(
                    _extract_frame_indices(obj[key], include_problem_frames=include_problem_frames, recurse_all=recurse_all)
                )
        if recurse_all:
            for key, value in obj.items():
                if key == "problem_frames" and not include_problem_frames:
                    continue
                if key in ("frame_idx", "frame_idxs", "frames", "invalid_frames", "invalid_detections", "cam2", "cam4"):
                    continue
                frame_indices.update(
                    _extract_frame_indices(value, include_problem_frames=include_problem_frames, recurse_all=recurse_all)
                )
        return frame_indices

    return frame_indices


def load_invalid_frames(
    invalid_path: Path | None,
    *,
    label: str = "invalid json",
    missing_ok: bool = False,
    encoding: str = "utf-8",
    recurse_all: bool = True,
) -> set[int]:
    if invalid_path is None:
        return set()
    if not invalid_path.exists():
        if missing_ok:
            return set()
        raise FileNotFoundError(f"{label} file not found: {invalid_path}")

    with invalid_path.open("r", encoding=encoding) as file:
        data = json.load(file)
    if isinstance(data, dict) and "invalid_frames" in data:
        return _extract_frame_indices(data["invalid_frames"], recurse_all=recurse_all)
    return _extract_frame_indices(data, recurse_all=recurse_all)


def load_problem_frames(invalid_path: Path | None, *, label: str = "invalid_json", recurse_all: bool = True) -> set[int]:
    if invalid_path is None:
        return set()
    if not invalid_path.exists():
        raise FileNotFoundError(f"{label} file not found: {invalid_path}")

    with invalid_path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    if isinstance(data, dict) and "problem_frames" in data:
        return _extract_frame_indices(data["problem_frames"], include_problem_frames=True, recurse_all=recurse_all)
    return set()


def merge_invalid_frames(*frame_sets: set[int]) -> set[int]:
    merged: set[int] = set()
    for frames in frame_sets:
        merged.update(frames)
    return merged
