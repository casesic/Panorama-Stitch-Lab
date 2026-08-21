"""Exact disk cache for correspondence-dependent CPW backward maps."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile
import time
from typing import Any, Callable

import numpy as np


CACHE_SCHEMA = "cpw_backward_map_v1"
CACHE_MODES = ("off", "read_write", "refresh", "read_only")


def _array_digest(array: np.ndarray) -> str:
    canonical = np.ascontiguousarray(array, dtype="<f4")
    return hashlib.sha256(canonical.tobytes()).hexdigest()


def build_cpw_cache_specification(
    *,
    camera_parameters: dict[str, Any],
    erp_size: tuple[int, int],
    source_image_size: tuple[int, int],
    target_image_size: tuple[int, int],
    output_size: tuple[int, int],
    direction: str,
    source_camera: str,
    target_camera: str,
    crop_bbox: tuple[int, int, int, int],
    k: int,
    power: float,
    src_points: np.ndarray,
    dst_points: np.ndarray,
) -> dict[str, Any]:
    """Build a complete key; point digests prevent unsafe cross-frame reuse."""
    return {
        "schema": CACHE_SCHEMA,
        "camera_parameters": camera_parameters,
        "erp_size": [int(erp_size[0]), int(erp_size[1])],
        "source_image_size": [int(source_image_size[0]), int(source_image_size[1])],
        "target_image_size": [int(target_image_size[0]), int(target_image_size[1])],
        "output_size": [int(output_size[0]), int(output_size[1])],
        "direction": direction,
        "source_camera": source_camera,
        "target_camera": target_camera,
        "crop_bbox": [int(value) for value in crop_bbox],
        "k": int(k),
        "power": float(power),
        "point_count": int(len(src_points)),
        "src_points_sha256": _array_digest(src_points),
        "dst_points_sha256": _array_digest(dst_points),
    }


def _cache_key(specification: dict[str, Any]) -> str:
    payload = json.dumps(
        specification, sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def load_or_generate_cpw_backward_map(
    cache_directory: Path,
    specification: dict[str, Any],
    generator: Callable[[], tuple[np.ndarray, dict]],
    *,
    cache_mode: str = "read_write",
) -> tuple[np.ndarray, dict, dict[str, Any]]:
    if cache_mode not in CACHE_MODES:
        raise ValueError(f"invalid CPW cache mode: {cache_mode}")
    key = _cache_key(specification)
    cache_path = cache_directory / key[:2] / f"{key}.npz"
    load_ms = 0.0
    if cache_mode in ("read_write", "read_only") and cache_path.is_file():
        start = time.perf_counter()
        with np.load(cache_path, allow_pickle=False) as cached:
            stored_spec = json.loads(str(cached["specification"].item()))
            if stored_spec == specification:
                map_x = np.ascontiguousarray(cached["map_x"], dtype=np.float32)
                map_y = np.ascontiguousarray(cached["map_y"], dtype=np.float32)
                stats = json.loads(str(cached["stats"].item()))
                load_ms = (time.perf_counter() - start) * 1000.0
                return (
                    np.stack((map_x, map_y), axis=-1), stats,
                    {"cache_hit": True, "cache_key": key,
                     "cache_load_ms": load_ms, "build_ms": 0.0},
                )
    if cache_mode == "read_only":
        raise RuntimeError(f"CPW warp-map cache miss: {key}")

    start = time.perf_counter()
    backward_map, stats = generator()
    build_ms = (time.perf_counter() - start) * 1000.0
    backward_map = np.ascontiguousarray(backward_map, dtype=np.float32)
    if cache_mode != "off":
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            dir=cache_path.parent, prefix=f".{key}.", suffix=".npz",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            np.savez(
                temporary,
                map_x=backward_map[..., 0],
                map_y=backward_map[..., 1],
                specification=json.dumps(specification, sort_keys=True),
                stats=json.dumps(stats, sort_keys=True),
            )
        os.replace(temporary_path, cache_path)
    return (
        backward_map, stats,
        {"cache_hit": False, "cache_key": key,
         "cache_load_ms": load_ms, "build_ms": build_ms},
    )
