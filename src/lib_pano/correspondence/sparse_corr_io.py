from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np


CORR_FIELDS = (
    "frame_idx",
    "segment_name",
    "image_width",
    "image_height",
    "matched_vertex_ids",
    "pairs",
)


def _npz_scalar(value: Any) -> object:
    arr = np.asarray(value)
    return arr.item() if arr.shape == () else arr.tolist()


def _frame_idx_from_corr_name(path: Path, domain: str | None = None) -> int | None:
    if domain is None:
        match = re.match(r"^(\d+)_(?:erp|fisheye)_corr\.npz$", path.name)
    else:
        match = re.match(rf"^(\d+)_{domain}_corr\.npz$", path.name)
    if match is None:
        return None
    return int(match.group(1))


def load_corr_npz(
    path: Path,
    *,
    fallback_width: int | None = None,
    fallback_height: int | None = None,
    domain: str | None = None,
) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as data:
        corr_data: dict[str, Any] = {
            "matched_vertex_ids": data["matched_vertex_ids"].astype(np.int64),
            "pairs": data["pairs"].astype(np.float32),
        }
        if "frame_idx" in data.files:
            corr_data["frame_idx"] = int(_npz_scalar(data["frame_idx"]))
        else:
            frame_idx = _frame_idx_from_corr_name(path, domain)
            if frame_idx is None:
                raise KeyError(f"frame_idx missing and filename does not match correspondence pattern: {path}")
            corr_data["frame_idx"] = frame_idx
        if "segment_name" in data.files:
            corr_data["segment_name"] = str(_npz_scalar(data["segment_name"]))
        if "image_width" in data.files:
            corr_data["image_width"] = int(_npz_scalar(data["image_width"]))
        elif fallback_width is not None:
            corr_data["image_width"] = int(fallback_width)
        if "image_height" in data.files:
            corr_data["image_height"] = int(_npz_scalar(data["image_height"]))
        elif fallback_height is not None:
            corr_data["image_height"] = int(fallback_height)
    return corr_data


def save_corr_npz(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        frame_idx=np.asarray(data["frame_idx"], dtype=np.int64),
        segment_name=np.asarray(data["segment_name"]),
        image_width=np.asarray(data["image_width"], dtype=np.int64),
        image_height=np.asarray(data["image_height"], dtype=np.int64),
        matched_vertex_ids=np.asarray(data["matched_vertex_ids"], dtype=np.int64),
        pairs=np.asarray(data["pairs"], dtype=np.float32),
    )


def load_corr_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def save_corr_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False)


def corr_npz_to_json_dict(npz_data: dict[str, Any]) -> dict[str, Any]:
    return {
        "frame_idx": int(npz_data["frame_idx"]),
        "segment_name": str(npz_data["segment_name"]),
        "image_width": int(npz_data["image_width"]),
        "image_height": int(npz_data["image_height"]),
        "matched_vertex_ids": np.asarray(npz_data["matched_vertex_ids"]).astype(int).tolist(),
        "pairs": np.asarray(npz_data["pairs"]).astype(float).tolist(),
    }


def convert_corr_npz_to_json(npz_path: Path, json_path: Path) -> None:
    save_corr_json(json_path, corr_npz_to_json_dict(load_corr_npz(npz_path)))


def list_corr_npz_files(
    corr_dir: Path,
    *,
    domain: str | None = None,
    start_frame: int | None = None,
    end_frame: int | None = None,
) -> list[Path]:
    if domain is None:
        input_dirs = [corr_dir / "erp_corr", corr_dir / "fisheye_corr"]
        glob_pattern = "*_corr.npz"
    else:
        input_dirs = [corr_dir / f"{domain}_corr"]
        glob_pattern = f"*_{domain}_corr.npz"

    paths: list[tuple[int, Path]] = []
    for input_dir in input_dirs:
        for npz_path in input_dir.glob(glob_pattern):
            frame_idx = _frame_idx_from_corr_name(npz_path, domain)
            if frame_idx is None:
                continue
            if start_frame is not None and frame_idx < start_frame:
                continue
            if end_frame is not None and frame_idx > end_frame:
                continue
            paths.append((frame_idx, npz_path))
    return [path for _, path in sorted(paths)]
