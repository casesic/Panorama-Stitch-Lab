#!/usr/bin/env python3
"""
Run SMPL-X fitting on fisheye images.

Usage:
  python tools/SingleFitting/scripts/debug008_run_fisheye.py `
    --camera cam2 `
    --smpl54-pkl  data_process/Output/20260311_riding/single_frame_cache/camera2/rtmlib_multi_frame/hmr_s490_529_pose.pkl `
    --rtm-cache data_process/Output/20260311_riding/single_frame_cache/camera2/rtmlib_multi_frame/rtm_joints_cache_s490_529_c2.pkl `
    --index-start 0 --num-frames 30 `
    --output-dir data_process/Output/20260311_riding/fisheye_smplx/cam2
# 490-529  512-529 for cam4
  python tools/SingleFitting/scripts/debug008_run_fisheye.py `
    --camera cam4 `
    --smpl54-pkl  data_process/Output/20260311_riding/single_frame_cache/camera4/rtmlib_multi_frame/hmr_s490_529_pose.pkl `
    --rtm-cache data_process/Output/20260311_riding/single_frame_cache/camera4/rtmlib_multi_frame/rtm_joints_cache_s490_529_c4.pkl `
    --index-start 22 `
    --num-frames 18 `
    --output-dir data_process/Output/20260311_riding/fisheye_smplx/cam4
  
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
# add tools/SingleFitting to python path for imports
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from single_fitting.debug_fit_shared import SingleCamFisheyeFitter, init_smplx_from_smpl54_seq
from single_fitting.debug_shared import DS_PARAMS, load_pickle, load_rtm_cache

SMPLX_Fisheye_Fitter = SingleCamFisheyeFitter


def parse_args():
    p = argparse.ArgumentParser(description="Run SMPL-X fitting on fisheye images.")
    p.add_argument("--camera", type=str, default="cam4", choices=["cam2", "cam4"])
    p.add_argument("--sequence-id", type=int, default=0)
    p.add_argument("--index-start", type=int, default=0)
    p.add_argument("--num-frames", type=int, default=1)
    p.add_argument("--smpl54-pkl", type=Path, default=None)
    p.add_argument("--rtm-cache", type=Path, required=True)
    p.add_argument("--hand-cache", type=Path, default=None)
    p.add_argument("--output-dir", type=Path, default=REPO_ROOT / "outputs" / "fisheye_smplx_cam2")
    p.add_argument("--image-size", type=int, default=2880)
    return p.parse_args()


def run():
    args = parse_args()
    from lib_dynamic_hvip.fitting.smplx_seq_forward_module import SMPLX_Seq_Forward_Module
    from lib_dynamic_hvip.fitting.smpl_seq_forward_module import SMPL_Seq_Forward_Module as SMPLFitter

    index_end = args.index_start + args.num_frames
    index_seq = list(range(args.index_start, index_end))
    det_list, hand_list = load_rtm_cache(args.rtm_cache, index_seq)

    if hand_list is None and args.hand_cache is not None and args.hand_cache.exists():
        hand_raw = load_pickle(args.hand_cache)
        arr = np.array(hand_raw) if isinstance(hand_raw, list) else hand_raw
        hand_list = [arr[i] if i < len(arr) else None for i in index_seq]

    smplx_init = None
    if args.smpl54_pkl is not None and args.smpl54_pkl.exists():
        smpl54 = load_pickle(args.smpl54_pkl)
        if isinstance(smpl54, SMPLFitter.SMPL_Param_Seq):
            smplx_init = init_smplx_from_smpl54_seq(smpl54, index_seq, SMPLFitter, SMPLX_Seq_Forward_Module)

    fitter = SingleCamFisheyeFitter(gender="neutral", camera=args.camera)
    result = fitter.fit(
        detection_results_list=det_list,
        image_size=args.image_size,
        smpl_param_init=smplx_init,
        hand_detection_list=hand_list,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    tag = f"fisheye_{args.camera}_s{args.sequence_id}_i{args.index_start}_{index_end-1}"
    path_seq = args.output_dir / f"{tag}_seq.pkl"
    path_center = args.output_dir / f"{tag}_frame{index_seq[len(index_seq)//2]}.pkl"
    path_meta = args.output_dir / f"{tag}_meta.json"

    with path_seq.open("wb") as f:
        pickle.dump(result, f)
    center = result.get_param_at_frame(len(index_seq) // 2)
    with path_center.open("wb") as f:
        pickle.dump(center, f)
    with path_meta.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "camera": args.camera,
                "sequence_id": args.sequence_id,
                "index_start": args.index_start,
                "index_end": index_end,
                "num_frames": len(index_seq),
                "image_size": args.image_size,
                "ds_params": DS_PARAMS,
                "smpl54_pkl": str(args.smpl54_pkl),
                "rtm_cache": str(args.rtm_cache),
                "output_seq": str(path_seq),
                "output_center": str(path_center),
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    print(f"Saved sequence    : {path_seq}")
    print(f"Saved center frame: {path_center}")
    print(f"Saved metadata    : {path_meta}")


def main():
    run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
