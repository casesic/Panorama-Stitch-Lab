#!/usr/bin/env python3
"""
Run SMPL fitting on equirectangular images.

Usage:
  python tools/SingleFitting/scripts/debug004_run_equirect.py ^
    --hmr-cache tools/SingleFitting/outputs/equirect/hmr_seq.pkl ^
    --rtm-cache tools/SingleFitting/outputs/equirect/rtm_cache.pkl ^
    --image-w 3840 --image-h 1920 ^
    --index-start 0 --num-frames 30 ^
    --output-dir tools/SingleFitting/outputs/equirect
"""
from __future__ import annotations
 
import argparse
import json
import pickle
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict
 
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from single_fitting.debug_shared import (
    build_smpl_init_seq,
    load_pickle,
    normalize_keypoints_format,
)
    
# ---------------------------------------------------------------------------
# Data class
# ---------------------------------------------------------------------------
 
@dataclass
class RunArgs:
    sequence_id: int
    index_start: int
    num_frames:  int
    hmr_cache:   Path
    rtm_cache:   Path
    output_dir:  Path
    image_w:     int
    image_h:     int

def check_path_exists(path: Path, tag: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{tag} not found: {path}")

#----------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
 
def parse_args() -> RunArgs:
    parser = argparse.ArgumentParser(
        description="SMPL fitting on equirectangular panorama sequences.",
    )
    parser.add_argument("--sequence-id", type=int,  default=0)
    parser.add_argument("--index-start", type=int,  default=0)
    parser.add_argument("--num-frames",  type=int,  default=1,
                        help="Number of frames to fit. 1 = single-frame mode.")
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=Path("/mnt/ssd2/zl/Data_Process/sample_output/equirect_preprocess"),
    )
    parser.add_argument("--hmr-cache",  type=Path, default=None,
                        help="Path to 4D-Humans SMPL pkl.")
    parser.add_argument("--rtm-cache",  type=Path, default=None,
                        help="Path to rtmlib keypoints pkl.")
    parser.add_argument("--output-dir", type=Path,
                        default=REPO_ROOT / "outputs" / "equirect")
    parser.add_argument("--image-w",    type=int,  default=3840,
                        help="Equirectangular frame width in pixels.")
    parser.add_argument("--image-h",    type=int,  default=1920,
                        help="Equirectangular frame height in pixels.")
    ns = parser.parse_args()
 
    hmr_cache = ns.hmr_cache or (
        ns.cache_root / f"hmr_s{ns.sequence_id}_pose.pkl"
    )
    rtm_cache = ns.rtm_cache or (
        ns.cache_root / f"rtm_joints_cache_s{ns.sequence_id}.pkl"
    )
    return RunArgs(
        sequence_id=ns.sequence_id,
        index_start=ns.index_start,
        num_frames=ns.num_frames,
        hmr_cache=hmr_cache,
        rtm_cache=rtm_cache,
        output_dir=ns.output_dir,
        image_w=ns.image_w,
        image_h=ns.image_h,
    )

def run() -> Dict[str, Path]:
    args = parse_args()
 
    from lib_dynamic_hvip.fitting.smpl_seq_forward_module import (  # noqa: E402
        SMPL_Seq_Forward_Module as SMPLFitter,
    )
 
    check_path_exists(args.hmr_cache, "HMR cache")
    check_path_exists(args.rtm_cache, "RTM cache")
 
    # ------------------------------------------------------------------
    # Load SMPL init (4D-Humans output, already converted to our format)
    # ------------------------------------------------------------------
    smpl_batch = load_pickle(args.hmr_cache)
    if isinstance(smpl_batch, SMPLFitter.SMPL_Param_Batch):
        smpl_batch = SMPLFitter.make_smpl_param_seq_from_avg_smpl_param_batch(smpl_batch)
    if not isinstance(smpl_batch, SMPLFitter.SMPL_Param_Seq):
        raise TypeError(f"Unexpected SMPL cache type: {type(smpl_batch)}")
 
    # ------------------------------------------------------------------
    # Load RTM keypoints
    # ------------------------------------------------------------------
    det_results = load_pickle(args.rtm_cache)
    if not hasattr(det_results, "get_result_by_frame_index"):
        raise TypeError(f"Unexpected RTM cache type: {type(det_results)}")

    # Normalize common lowercase COCO naming to SMPL naming expected by fitter.
    if hasattr(det_results, "keypoints_format"):
        det_results.keypoints_format = normalize_keypoints_format(det_results.keypoints_format)
 
    # ------------------------------------------------------------------
    # Validate frame range
    # ------------------------------------------------------------------
    n_smpl    = int(smpl_batch.num_batch)
    n_det     = int(getattr(det_results, "N", n_smpl))
    index_end = args.index_start + args.num_frames
    max_valid = min(n_smpl, n_det)
 
    if args.index_start < 0 or index_end > max_valid:
        raise ValueError(
            f"Invalid frame range [{args.index_start}, {index_end}). "
            f"Available: {max_valid} frames."
        )
 
    index_seq              = list(range(args.index_start, index_end))
    detection_results_list = [det_results.get_result_by_frame_index(i) for i in index_seq]
    smpl_init_seq          = build_smpl_init_seq(SMPLFitter, smpl_batch, index_seq)
    
    # ------------------------------------------------------------------
    # Fit  — calls the new ERP method added to smpl_seq_forward_module
    # ------------------------------------------------------------------
    fitter = SMPLFitter()
 
    smpl_param_seq = fitter.fit_from_multi_frame_seq_erp(
        detection_results_list=detection_results_list,
        camera=None,
        camera_model="equirect",
        image_wh=(args.image_w, args.image_h),
        smpl_param_init=smpl_init_seq,
    )
    
    # ------------------------------------------------------------------
    # Save outputs
    # ------------------------------------------------------------------
    args.output_dir.mkdir(parents=True, exist_ok=True)
    tag = f"erp_s{args.sequence_id}_i{args.index_start}_{index_end - 1}"
 
    path_seq    = args.output_dir / f"{tag}_seq.pkl"
    path_center = args.output_dir / f"{tag}_frame{index_seq[len(index_seq)//2]}.pkl"
    path_meta   = args.output_dir / f"{tag}_meta.json"
 
    with path_seq.open("wb") as f:
        pickle.dump(smpl_param_seq, f)
 
    center_param = smpl_param_seq.get_smpl_param_at_frame(len(index_seq) // 2)
    with path_center.open("wb") as f:
        pickle.dump(center_param, f)
 
    meta = {
        "sequence_id":   args.sequence_id,
        "index_start":   args.index_start,
        "index_end":     index_end,
        "num_frames":    len(index_seq),
        "frame_center":  index_seq[len(index_seq) // 2],
        "image_w":       args.image_w,
        "image_h":       args.image_h,
        "hmr_cache":     str(args.hmr_cache),
        "rtm_cache":     str(args.rtm_cache),
        "output_seq":    str(path_seq),
        "output_center": str(path_center),
    }
    with path_meta.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
 
    print(f"Saved sequence    : {path_seq}")
    print(f"Saved center frame: {path_center}")
    print(f"Saved metadata    : {path_meta}")
    return {"seq": path_seq, "center": path_center, "meta": path_meta}

def main() -> int:
    run()
    return 0
 
 
if __name__ == "__main__":
    raise SystemExit(main())
 
