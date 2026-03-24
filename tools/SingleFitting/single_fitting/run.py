from __future__ import annotations

import argparse
import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List


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


def _parse_args() -> RunArgs:
    parser = argparse.ArgumentParser(
        description="Minimal SingleFitting pipeline adapted from debug004_check_vr_data.py"
    )
    parser.add_argument("--sequence-id", type=int, default=24)
    parser.add_argument("--camera-id", type=int, default=0)
    parser.add_argument("--index-start", type=int, default=300)
    parser.add_argument("--num-frames", type=int, default=300)

    parser.add_argument(
        "--vr-data-path",
        type=Path,
        default=Path("/mnt/ssd2/zl/Data_Process/vrCollect/sync2/"),
        help="Root folder that contains S{sequence_id}/",
    )
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=Path("/mnt/ssd2/zl/Data_Process/sample_output/single_preprocess/debug002_vr"),
    )
    parser.add_argument("--hmr-cache", type=Path, default=None)
    parser.add_argument("--rtm-cache", type=Path, default=None)

    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
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


def _check_path_exists(path: Path, tag: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{tag} not found: {path}")


def _build_smpl_init_seq(
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


def run() -> Dict[str, Path]:
    args = _parse_args()

    from .io_vr_minimal import (  # pylint: disable=import-outside-toplevel
        load_pickle,
        load_vr_camera_extrinsic,
        load_vr_camera_intrinsic,
        load_vr_imu_aligned_to_video,
        load_vr_rt_tbias,
        transform_plane_world_to_camera,
    )

    _check_path_exists(args.vr_data_path, "VR data root")
    _check_path_exists(args.hmr_cache, "HMR cache")
    _check_path_exists(args.rtm_cache, "RTM cache")

    from lib_dynamic_hvip.fitting.smpl_seq_forward_module import (  # pylint: disable=import-error
        SMPL_Seq_Forward_Module as SMPLFitter,
    )

    camera_k, _ = load_vr_camera_intrinsic(args.vr_data_path)
    r_vr, t_vr, t_bias = load_vr_rt_tbias(args.vr_data_path, args.sequence_id)
    r_cam, tvec_cam = load_vr_camera_extrinsic(args.vr_data_path, args.sequence_id)
    n_cam, d_cam = transform_plane_world_to_camera(r_cam, tvec_cam)
    imu_data = load_vr_imu_aligned_to_video(args.vr_data_path, args.sequence_id)

    smpl_batch = load_pickle(args.hmr_cache)
    if isinstance(smpl_batch, SMPLFitter.SMPL_Param_Batch):
        smpl_batch = SMPLFitter.make_smpl_param_seq_from_avg_smpl_param_batch(smpl_batch)
    if not isinstance(smpl_batch, SMPLFitter.SMPL_Param_Seq):
        raise TypeError(f"Unexpected SMPL cache type: {type(smpl_batch)}")

    det_results = load_pickle(args.rtm_cache)
    if not hasattr(det_results, "get_result_by_frame_index"):
        raise TypeError(f"Unexpected RTM cache type: {type(det_results)}")

    n_imu = len(imu_data)
    n_smpl = int(smpl_batch.num_batch)
    n_det = int(getattr(det_results, "N", n_smpl))
    index_end = args.index_start + args.num_frames

    max_valid = min(n_imu, n_smpl, n_det)
    if args.index_start < 0 or index_end > max_valid:
        raise ValueError(
            f"Invalid frame range [{args.index_start}, {index_end}). "
            f"Available max frame count: {max_valid}"
        )

    index_seq = list(range(args.index_start, index_end))
    detection_results_list = [det_results.get_result_by_frame_index(i) for i in index_seq]
    imu_data_seq = [imu_data[i] for i in index_seq]
    smpl_init_seq = _build_smpl_init_seq(SMPLFitter, smpl_batch, index_seq)
    fitter = SMPLFitter()
    fitter.weight_of_losses = SMPLFitter.Weight_of_Losses_Default

    smpl_param_seq = fitter.fit_from_multi_frame_seq(
        detection_results_list=detection_results_list,
        camera=camera_k,
        cam_ext=(r_cam, tvec_cam),
        imu_data_seq=imu_data_seq,
        VR2camera={"R": r_vr, "T": t_vr, "t_bias": t_bias},
        plane_cam=(n_cam, d_cam),
        smpl_param_init=smpl_init_seq,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    path_seq = args.output_dir / (
        f"singlefitting_s{args.sequence_id}_c{args.camera_id}"
        f"_i{args.index_start}_{index_end - 1}.pkl"
    )
    with path_seq.open("wb") as f:
        pickle.dump(smpl_param_seq, f)

    idx_center = len(index_seq) // 2
    frame_center = index_seq[idx_center]
    center_param = smpl_param_seq.get_smpl_param_at_frame(idx_center)
    path_center = args.output_dir / (
        f"singlefitting_s{args.sequence_id}_c{args.camera_id}_frame{frame_center}.pkl"
    )
    with path_center.open("wb") as f:
        pickle.dump(center_param, f)

    path_meta = args.output_dir / (
        f"singlefitting_s{args.sequence_id}_c{args.camera_id}_meta.json"
    )
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
