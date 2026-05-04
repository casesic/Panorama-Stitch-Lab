from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def load_bbox_dict(json_path: Path, *, normalize_values: bool = False) -> dict[str, list[int]]:
    with json_path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, dict):
        raise ValueError(f"bbox json must contain a dict: {json_path}")
    if normalize_values:
        return {str(key): [int(round(v)) for v in value] for key, value in data.items()}
    return data


def save_bbox_dict(json_path: Path, bbox_dict: dict[str, list[int]], *, ensure_ascii: bool = False) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with json_path.open("w", encoding="utf-8") as file:
        json.dump(bbox_dict, file, ensure_ascii=ensure_ascii, indent=2)


def clip_bbox(bbox, width: int, height: int) -> list[int] | None:
    if bbox is None:
        return None
    x1, y1, x2, y2 = [int(round(v)) for v in bbox]
    x1 = max(0, min(width, x1))
    y1 = max(0, min(height, y1))
    x2 = max(0, min(width, x2))
    y2 = max(0, min(height, y2))
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


def union_bboxes(bboxes: list[list[int]]) -> list[int] | None:
    if not bboxes:
        return None
    arr = np.asarray(bboxes, dtype=np.float64).reshape(-1, 4)
    return [
        int(np.floor(arr[:, 0].min())),
        int(np.floor(arr[:, 1].min())),
        int(np.ceil(arr[:, 2].max())),
        int(np.ceil(arr[:, 3].max())),
    ]


def bbox_area(bbox: list[int] | None) -> int:
    if bbox is None:
        return 0
    return max(0, bbox[2] - bbox[0]) * max(0, bbox[3] - bbox[1])
