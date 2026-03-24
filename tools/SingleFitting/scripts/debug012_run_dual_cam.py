#!/usr/bin/env python3
"""
Run dual-camera SMPL-X fitting in the shared world coordinate system.

Usage:
  python tools/SingleFitting/scripts/debug012_run_dual_cam.py ^
    --rtm-cam2 tools/SingleFitting/outputs/fisheye/rtm_cam2.pkl ^
    --rtm-cam4 tools/SingleFitting/outputs/fisheye/rtm_cam4.pkl ^
    --smpl54-pkl tools/SingleFitting/outputs/equirect/result_seq.pkl ^
    --index-start 0 --num-frames 30 ^
    --output-dir tools/SingleFitting/outputs/dual_cam
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from single_fitting.debug_fit_shared import DEFAULT_T_CAM4_INIT, DualCamWorldFitter, init_smplx_from_smpl54_seq
from single_fitting.debug_shared import load_pickle, load_rtm_cache

DualCamSMPLXFitter = DualCamWorldFitter


def parse_args():
    p = argparse.ArgumentParser(description="Run dual-camera SMPL-X fitting.")
    p.add_argument("--rtm-cam2", type=Path, required=True)
    p.add_argument("--rtm-cam4", type=Path, required=True)
    p.add_argument("--smpl54-pkl", type=Path, default=None)
    p.add_argument("--index-start", type=int, default=0)
    p.add_argument("--num-frames", type=int, default=1)
    p.add_argument("--image-size", type=int, default=2880)
    p.add_argument("--output-dir", type=Path, default=REPO_ROOT / "outputs" / "dual_cam")
    p.add_argument("--sequence-id", type=int, default=0)
    return p.parse_args()


def run():
    args = parse_args()
    from lib_dynamic_hvip.fitting.smplx_seq_forward_module import SMPLX_Seq_Forward_Module
    from lib_dynamic_hvip.fitting.smpl_seq_forward_module import SMPL_Seq_Forward_Module as SMPLFitter

    index_end = args.index_start + args.num_frames
    index_seq = list(range(args.index_start, index_end))
    det_cam2, hand_cam2, fmt_cam2 = load_rtm_cache(args.rtm_cam2, index_seq, return_kp_fmt=True)
    det_cam4, hand_cam4, fmt_cam4 = load_rtm_cache(args.rtm_cam4, index_seq, return_kp_fmt=True)

    smplx_init = None
    if args.smpl54_pkl is not None and args.smpl54_pkl.exists():
        smpl54 = load_pickle(args.smpl54_pkl)
        if isinstance(smpl54, SMPLFitter.SMPL_Param_Seq):
            smplx_init = init_smplx_from_smpl54_seq(smpl54, index_seq, SMPLFitter, SMPLX_Seq_Forward_Module)

    fitter = DualCamWorldFitter(gender="neutral")
    result = fitter.fit(
        det_list_cam2=det_cam2,
        hand_list_cam2=hand_cam2,
        kp_fmt_cam2=fmt_cam2,
        det_list_cam4=det_cam4,
        hand_list_cam4=hand_cam4,
        kp_fmt_cam4=fmt_cam4,
        image_size=args.image_size,
        smpl_param_init=smplx_init,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    tag = f"dual_s{args.sequence_id}_i{args.index_start}_{index_end-1}"
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
                "sequence_id": args.sequence_id,
                "index_start": args.index_start,
                "index_end": index_end,
                "num_frames": len(index_seq),
                "image_size": args.image_size,
                "rtm_cam2": str(args.rtm_cam2),
                "rtm_cam4": str(args.rtm_cam4),
                "cam4_delta_T": result.delta_T.tolist() if hasattr(result, "delta_T") else DEFAULT_T_CAM4_INIT.tolist(),
                "transl_world": result.transl[len(index_seq) // 2].tolist(),
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
