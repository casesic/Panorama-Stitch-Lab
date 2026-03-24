#!/usr/bin/env python3
from __future__ import annotations

import argparse
import bisect
import json
import pickle
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

class TimestampIterator:
    """Nearest timestamp lookup with linear interpolation for position."""

    def __init__(self, data: Sequence[Tuple[Any, ...]]):
        self.data = list(data)
        self.timestamps = [abs(float(item[0])) for item in self.data]

    def find_nearest(self, target_timestamp: float) -> Tuple[Any, ...]:
        pos = bisect.bisect_left(self.timestamps, target_timestamp)

        if pos == 0:
            return self.data[0]
        if pos == len(self.timestamps):
            return self.data[-1]

        before = self.data[pos - 1]
        after = self.data[pos]

        if float(after[0]) == target_timestamp:
            return after

        return self._create_interpolated(target_timestamp, before, after)

    @staticmethod
    def _create_interpolated(
        target_ts: float,
        point_a: Tuple[Any, ...],
        point_b: Tuple[Any, ...],
    ) -> Tuple[Any, ...]:
        ts_a, ts_b = float(point_a[0]), float(point_b[0])
        if ts_b == ts_a:
            return point_a

        alpha = (target_ts - ts_a) / (ts_b - ts_a)

        interpolated = [target_ts]
        for idx, (val_a, val_b) in enumerate(zip(point_a[1:], point_b[1:])):
            if idx == 0:
                interpolated_val = val_a + alpha * (val_b - val_a)
            else:
                interpolated_val = val_b if alpha > 0.5 else val_a
            interpolated.append(interpolated_val)
        return tuple(interpolated)


@dataclass
class RunArgs:
    sequence_id: int
    camera_id: int
    index_start: int
    num_frames: int
    vr_data_path: Path
    hmr_cache: Path
    rtm_cache: Path
    output_dir: Path


def load_pickle(path: Path) -> Any:
    with path.open("rb") as f:
        return pickle.load(f)


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def parse_imu_item(item: Any) -> Tuple[float, np.ndarray, np.ndarray]:
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

    prepared_imu = [parse_imu_item(item) for item in raw_imu]
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


def parse_args() -> RunArgs:
    parser = argparse.ArgumentParser(
        description="SingleFitting pipeline adapted from debug004_check_vr_data.py",
    )
    parser.add_argument("--sequence-id", type=int, default=24)
    parser.add_argument("--camera-id", type=int, default=0)
    parser.add_argument("--index-start", type=int, default=300)
    parser.add_argument("--num-frames", type=int, default=300)
    parser.add_argument(
        "--vr-data-path",
        type=Path,
        default=Path("/mnt/ssd2/zl/Data_Process/vrCollect/sync2/"),
    )
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=Path("/mnt/ssd2/zl/Data_Process/sample_output/single_preprocess/debug002_vr"),
    )
    parser.add_argument("--hmr-cache", type=Path, default=None)
    parser.add_argument("--rtm-cache", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "outputs")
    ns = parser.parse_args()

    hmr_cache = ns.hmr_cache or (ns.cache_root / f"hmr_s{ns.sequence_id}_pose.pkl")
    rtm_cache = ns.rtm_cache or (ns.cache_root / f"rtm_joints_cache_s{ns.sequence_id}_c{ns.camera_id}.pkl")

    return RunArgs(
        sequence_id=ns.sequence_id,
        camera_id=ns.camera_id,
        index_start=ns.index_start,
        num_frames=ns.num_frames,
        vr_data_path=ns.vr_data_path,
        hmr_cache=hmr_cache,
        rtm_cache=rtm_cache,
        output_dir=ns.output_dir,
    )


def build_smpl_init_seq(
    smpl_fitter_cls: Any,
    smpl_batch_seq: Any,
    index_seq: List[int],
):
    smpl_seq_init = smpl_fitter_cls.SMPL_Param_Seq(num_batch=len(index_seq))
    for seq_idx, frame_idx in enumerate(index_seq):
        smpl_seq_init.orient[seq_idx] = smpl_batch_seq.orient[frame_idx]
        smpl_seq_init.pose[seq_idx] = smpl_batch_seq.pose[frame_idx]
        smpl_seq_init.transl[seq_idx] = smpl_batch_seq.transl[frame_idx]
    smpl_seq_init.beta = smpl_batch_seq.beta.clone()
    smpl_seq_init.scale = smpl_batch_seq.scale.clone()
    return smpl_seq_init


def check_path_exists(path: Path, tag: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{tag} not found: {path}")


def run() -> Dict[str, Path]:
    args = parse_args()
    from lib_dynamic_hvip.fitting.smpl_seq_forward_module import (  # pylint: disable=import-outside-toplevel
        SMPL_Seq_Forward_Module as SMPLFitter,
    )

    check_path_exists(args.vr_data_path, "VR data root")
    check_path_exists(args.hmr_cache, "HMR cache")
    check_path_exists(args.rtm_cache, "RTM cache")
    
    # TODO: fisheye intrinsic need load
    camera_k, _ = load_vr_camera_intrinsic(args.vr_data_path)
    # r_vr, t_vr, t_bias = load_vr_rt_tbias(args.vr_data_path, args.sequence_id)
    # r_cam, tvec_cam = load_vr_camera_extrinsic(args.vr_data_path, args.sequence_id)
    # n_cam, d_cam = transform_plane_world_to_camera(r_cam, tvec_cam)  # not valid for fisheye, can remove
    # imu_data = load_vr_imu_aligned_to_video(args.vr_data_path, args.sequence_id)
    # TODO: fisheye hmr need get by 4dhumans or other
    smpl_batch = load_pickle(args.hmr_cache)
    if isinstance(smpl_batch, SMPLFitter.SMPL_Param_Batch):
        smpl_batch = SMPLFitter.make_smpl_param_seq_from_avg_smpl_param_batch(smpl_batch)
    if not isinstance(smpl_batch, SMPLFitter.SMPL_Param_Seq):
        raise TypeError(f"Unexpected SMPL cache type: {type(smpl_batch)}")
    # TODO: fisheye 2d detect result need get by rtmlib
    det_results = load_pickle(args.rtm_cache)
    if not hasattr(det_results, "get_result_by_frame_index"):
        raise TypeError(f"Unexpected RTM cache type: {type(det_results)}")

    # n_imu = len(imu_data)
    n_smpl = int(smpl_batch.num_batch)
    n_det = int(getattr(det_results, "N", n_smpl))
    index_end = args.index_start + args.num_frames
    # max_valid = min(n_imu, n_smpl, n_det)
    max_valid = min(n_smpl, n_det)
    if args.index_start < 0 or index_end > max_valid:
        raise ValueError(
            f"Invalid frame range [{args.index_start}, {index_end}). Available max frame count: {max_valid}",
        )
    # Get the detect list and smpl init list
    index_seq = list(range(args.index_start, index_end))
    detection_results_list = [det_results.get_result_by_frame_index(i) for i in index_seq]
    # imu_data_seq = [imu_data[i] for i in index_seq]
    smpl_init_seq = build_smpl_init_seq(SMPLFitter, smpl_batch, index_seq)
    fitter = SMPLFitter()
    fitter.weight_of_losses = SMPLFitter.Weight_of_Losses_Default

    smpl_param_seq = fitter.fit_from_multi_frame_seq(
        detection_results_list=detection_results_list,
        camera=camera_k,
        # cam_ext=(r_cam, tvec_cam),
        # imu_data_seq=imu_data_seq,
        # VR2camera={"R": r_vr, "T": t_vr, "t_bias": t_bias},
        # plane_cam=(n_cam, d_cam),
        smpl_param_init=smpl_init_seq,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)

    path_seq = args.output_dir / (
        f"singlefitting_s{args.sequence_id}_c{args.camera_id}_i{args.index_start}_{index_end - 1}.pkl"
    )
    with path_seq.open("wb") as f:
        pickle.dump(smpl_param_seq, f)

    idx_center = len(index_seq) // 2
    frame_center = index_seq[idx_center]
    center_param = smpl_param_seq.get_smpl_param_at_frame(idx_center)
    path_center = args.output_dir / f"singlefitting_s{args.sequence_id}_c{args.camera_id}_frame{frame_center}.pkl"
    with path_center.open("wb") as f:
        pickle.dump(center_param, f)

    path_meta = args.output_dir / f"singlefitting_s{args.sequence_id}_c{args.camera_id}_meta.json"
    meta = {
        "sequence_id": args.sequence_id,
        "camera_id": args.camera_id,
        "index_start": args.index_start,
        "index_end": index_end,
        "num_frames": len(index_seq),
        "frame_center": frame_center,
        "hmr_cache": str(args.hmr_cache),
        "rtm_cache": str(args.rtm_cache),
        "output_seq": str(path_seq),
        "output_center": str(path_center),
    }
    with path_meta.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"Saved sequence: {path_seq}")
    print(f"Saved center frame param: {path_center}")
    print(f"Saved metadata: {path_meta}")
    return {"seq": path_seq, "center": path_center, "meta": path_meta}


def main() -> int:
    run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
