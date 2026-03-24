#!/usr/bin/env python3
"""
Run SMPL-X fitting on equirectangular images.

Usage:
  python tools/SingleFitting/scripts/debug006_run_equirect_smplx.py ^
    --smpl54-pkl tools/SingleFitting/outputs/equirect/result_seq.pkl ^
    --rtm-cache tools/SingleFitting/outputs/equirect/rtm_cache.pkl ^
    --image-w 5720 --image-h 2880 ^
    --index-start 0 --num-frames 30 ^
    --output-dir tools/SingleFitting/outputs/equirect_smplx
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from single_fitting.debug_shared import build_smpl_init_seq, load_pickle, load_rtm_cache


@dataclass
class RunArgs:
    sequence_id: int
    index_start: int
    num_frames: int
    smpl54_pkl: Optional[Path]
    rtm_cache: Path
    hand_cache: Optional[Path]
    output_dir: Path
    image_w: int
    image_h: int


def parse_args() -> RunArgs:
    p = argparse.ArgumentParser(description="Run SMPL-X fitting on equirectangular images.")
    p.add_argument("--sequence-id", type=int, default=0)
    p.add_argument("--index-start", type=int, default=0)
    p.add_argument("--num-frames", type=int, default=1)
    p.add_argument("--smpl54-pkl", type=Path, default=None, help="Optional SMPL-54 warm start.")
    p.add_argument("--rtm-cache", type=Path, required=True)
    p.add_argument("--hand-cache", type=Path, default=None, help="Optional legacy hand cache.")
    p.add_argument("--output-dir", type=Path, default=REPO_ROOT / "outputs" / "equirect_smplx")
    p.add_argument("--image-w", type=int, default=5720)
    p.add_argument("--image-h", type=int, default=2880)
    ns = p.parse_args()
    return RunArgs(
        sequence_id=ns.sequence_id,
        index_start=ns.index_start,
        num_frames=ns.num_frames,
        smpl54_pkl=ns.smpl54_pkl,
        rtm_cache=ns.rtm_cache,
        hand_cache=ns.hand_cache,
        output_dir=ns.output_dir,
        image_w=ns.image_w,
        image_h=ns.image_h,
    )


def run() -> Dict[str, Path]:
    args = parse_args()

    from lib_dynamic_hvip.fitting.smplx_seq_forward_module import SMPLX_Seq_Forward_Module

    index_end = args.index_start + args.num_frames
    index_seq = list(range(args.index_start, index_end))

    detection_results_list, hand_detection_list = load_rtm_cache(args.rtm_cache, index_seq)

    if hand_detection_list is None and args.hand_cache is not None and args.hand_cache.exists():
        hand_raw = load_pickle(args.hand_cache)
        arr = np.array(hand_raw) if isinstance(hand_raw, list) else hand_raw
        hand_detection_list = [arr[i] if i < len(arr) else None for i in index_seq]

    smplx_init = None
    if args.smpl54_pkl is not None and args.smpl54_pkl.exists():
        from lib_dynamic_hvip.fitting.smpl_seq_forward_module import (
            SMPL_Seq_Forward_Module as SMPLFitter,
        )

        smpl54 = load_pickle(args.smpl54_pkl)
        if isinstance(smpl54, SMPLFitter.SMPL_Param_Seq):
            sub = build_smpl_init_seq(SMPLFitter, smpl54, index_seq)
            smplx_init = SMPLX_Seq_Forward_Module.init_from_smpl54_seq(sub)
        else:
            print(f"[warn] Unexpected smpl54 type: {type(smpl54)}; fallback to zero init.")

    fitter = SMPLX_Seq_Forward_Module(gender="neutral")
    smplx_param_seq = fitter.fit_from_multi_frame_seq_erp(
        detection_results_list=detection_results_list,
        image_wh=(args.image_w, args.image_h),
        smpl_param_init=smplx_init,
        hand_detection_list=hand_detection_list,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    tag = f"smplx_s{args.sequence_id}_i{args.index_start}_{index_end - 1}"
    path_seq = args.output_dir / f"{tag}_seq.pkl"
    path_center = args.output_dir / f"{tag}_frame{index_seq[len(index_seq) // 2]}.pkl"
    path_meta = args.output_dir / f"{tag}_meta.json"

    with path_seq.open("wb") as f:
        pickle.dump(smplx_param_seq, f)

    center = smplx_param_seq.get_param_at_frame(len(index_seq) // 2)
    with path_center.open("wb") as f:
        pickle.dump(center, f)

    meta = {
        "sequence_id": args.sequence_id,
        "index_start": args.index_start,
        "index_end": index_end,
        "num_frames": len(index_seq),
        "image_w": args.image_w,
        "image_h": args.image_h,
        "smpl54_pkl": str(args.smpl54_pkl),
        "rtm_cache": str(args.rtm_cache),
        "hand_cache": str(args.hand_cache),
        "output_seq": str(path_seq),
        "output_center": str(path_center),
    }
    with path_meta.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(f"Saved sequence    : {path_seq}")
    print(f"Saved center frame: {path_center}")
    print(f"Saved metadata    : {path_meta}")
    return {"seq": path_seq, "center": path_center, "meta": path_meta}


def main() -> int:
    run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
