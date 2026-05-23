from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

from .timestamp_iterator import TimestampIterator


def _ensure_numpy_pickle_compat() -> None:
    """Allow loading pickles created with NumPy internals from another major version."""
    core_mod = getattr(np, "core", None)
    if core_mod is None:
        return

    sys.modules.setdefault("numpy._core", core_mod)
    for submodule in ("multiarray", "numeric", "umath", "_multiarray_umath"):
        src = f"numpy.core.{submodule}"
        dst = f"numpy._core.{submodule}"
        module = sys.modules.get(src)
        if module is None:
            try:
                module = __import__(src, fromlist=["*"])
            except Exception:
                continue
        sys.modules.setdefault(dst, module)


def load_pickle(path: Path) -> Any:
    _ensure_numpy_pickle_compat()
    with path.open("rb") as f:
        return pickle.load(f)


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _parse_imu_item(item: Any) -> Tuple[float, np.ndarray, np.ndarray]:
    if isinstance(item, dict):
        ts = item.get("timestamp", item.get("ts", item.get("time")))
        position = item.get("position")
        rotation = item.get("rotation")
    elif isinstance(item, (tuple, list)) and len(item) >= 3:
        ts, position, rotation = item[0], item[1], item[2]
    else:
        raise ValueError(f"Unsupported IMU item format: {type(item)}")

    if ts is None or position is None or rotation is None:
        raise ValueError("IMU item misses timestamp/position/rotation.")

    return (
        float(ts),
        np.asarray(position, dtype=np.float32),
        np.asarray(rotation, dtype=np.float32),
    )


def get_sync_frame_window(vr_data_path: Path, sequence_id: int, fps: int = 30) -> Tuple[int, int]:
    sync_path = vr_data_path / f"S{sequence_id}" / "sync.json"
    sync_data = load_json(sync_path)
    start = int(sync_data["camera"] + sync_data["wait_second"] * fps)
    end = start + int(sync_data["len"] * fps)
    return start, end


def load_vr_camera_intrinsic(vr_data_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    data = load_pickle(vr_data_path / "calibrate_room.pkl")
    mtx = np.asarray(data["mtx"], dtype=np.float32)
    dist = np.asarray(data["dist"], dtype=np.float32)
    mtx[0, 0] = mtx[1, 1] = (mtx[0, 0] + mtx[1, 1]) / 2.0
    return mtx, dist


def load_vr_rt_tbias(vr_data_path: Path, sequence_id: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = load_pickle(vr_data_path / f"S{sequence_id}" / "data" / "optimized_calibration.pkl")
    r = np.asarray(data["R"], dtype=np.float32)
    t = np.asarray(data["T"], dtype=np.float32)
    t_bias = np.asarray(data["t_bias"], dtype=np.float32)
    return r, t, t_bias


def load_vr_camera_extrinsic(vr_data_path: Path, sequence_id: int) -> Tuple[np.ndarray, np.ndarray]:
    data = load_pickle(vr_data_path / f"S{sequence_id}" / "camera_extrinsic" / "camera_ext.pkl")
    r = np.asarray(data["R"], dtype=np.float32)
    t = np.asarray(data["tvec"], dtype=np.float32)
    return r, t


def load_vr_imu_aligned_to_video(vr_data_path: Path, sequence_id: int, fps: int = 30) -> List[Dict[str, np.ndarray]]:
    seq_root = vr_data_path / f"S{sequence_id}"
    raw_imu = load_pickle(seq_root / "data" / "imu_data.pkl")
    raw_video_ts = load_pickle(seq_root / "data" / "video_timestamp.pkl")

    prepared_imu = [_parse_imu_item(item) for item in raw_imu]
    iterator = TimestampIterator(prepared_imu)

    frame_start, frame_end = get_sync_frame_window(vr_data_path, sequence_id, fps=fps)
    video_ts_sync = raw_video_ts[frame_start:frame_end]

    imu_data = []
    for video_ts in video_ts_sync:
        imu_timestamp, position, rotation = iterator.find_nearest(float(video_ts))
        imu_data.append(
            {
                "timestamp": np.array([imu_timestamp], dtype=np.float64),
                "position": np.asarray(position, dtype=np.float32),
                "rotation": np.asarray(rotation, dtype=np.float32),
            }
        )
    return imu_data


def transform_plane_world_to_camera(
    r_world_to_cam: np.ndarray,
    t_world_to_cam: np.ndarray,
    plane_world: Tuple[np.ndarray, float] = (np.array([0.0, 0.0, 1.0], dtype=np.float32), -0.08),
) -> Tuple[np.ndarray, float]:
    n_world, d_world = plane_world
    n_cam = r_world_to_cam @ n_world
    d_cam = float(d_world - np.dot(n_cam, t_world_to_cam.reshape(-1)))
    return n_cam.astype(np.float32), d_cam

