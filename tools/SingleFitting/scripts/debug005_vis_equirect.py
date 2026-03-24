#!/usr/bin/env python3
"""
Visualize SMPL fitting results on equirectangular images.

Usage:
  python tools/SingleFitting/scripts/debug005_vis_equirect.py ^
    --result-pkl tools/SingleFitting/outputs/equirect/result_seq.pkl ^
    --rtm-cache tools/SingleFitting/outputs/equirect/rtm_cache.pkl ^
    --image-dir data/equirect_frames ^
    --image-w 3840 --image-h 1920 ^
    --output-dir tools/SingleFitting/outputs/equirect/vis ^
    --mesh
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from single_fitting.debug_shared import erp_project_np, load_pickle
from single_fitting.debug_vis_erp_shared import COCO_SKELETON_PAIRS, draw_keypoints, render_mesh_simple


def draw_joints_54(
    img: np.ndarray,
    joints_54: np.ndarray,
    width: int,
    height: int,
    smpl_all_54: Dict[str, int],
    radius: int = 5,
    color=(0, 60, 220),
) -> np.ndarray:
    joints_2d = erp_project_np(joints_54, width, height)
    img_h, img_w = img.shape[:2]
    for idx in range(min(54, len(joints_2d))):
        u, v = joints_2d[idx]
        pu, pv = int(round(float(u))), int(round(float(v)))
        if 0 <= pu < img_w and 0 <= pv < img_h:
            cv2.circle(img, (pu, pv), radius, color, -1)
    return img


def visualize_frame(img, smpl_param, det_result, fitter, width, height, smpl_all_54, draw_mesh=False, conf_thresh=0.1):
    import torch

    vis = img.copy()
    if hasattr(det_result, "keypoints") and det_result.keypoints is not None:
        draw_keypoints(vis, np.array(det_result.keypoints, dtype=np.float32), conf_thresh=conf_thresh, radius=7, color=(0, 220, 0), pairs=COCO_SKELETON_PAIRS)

    with torch.no_grad():
        verts_t, joints_54_t = fitter.forward_smpl_param(smpl_param)
    joints_54 = joints_54_t.detach().cpu().numpy()
    verts = verts_t.detach().cpu().numpy()

    draw_joints_54(vis, joints_54, width, height, smpl_all_54)
    if draw_mesh:
        faces = fitter.smpl_models["neutral"].faces
        vis = render_mesh_simple(vis, verts, faces, width, height, alpha=0.4, color=(180, 120, 60))
    return vis


def parse_args():
    p = argparse.ArgumentParser(description="Visualize ERP SMPL fitting results.")
    p.add_argument("--result-pkl", type=Path, required=True)
    p.add_argument("--rtm-cache", type=Path, required=True)
    p.add_argument("--index-start", type=int, default=0)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--image", type=Path, default=None)
    src.add_argument("--image-dir", type=Path, default=None)
    p.add_argument("--image-w", type=int, default=3840)
    p.add_argument("--image-h", type=int, default=1920)
    p.add_argument("--output-dir", type=Path, default=Path("outputs/equirect/vis"))
    p.add_argument("--mesh", action="store_true")
    p.add_argument("--conf-thresh", type=float, default=0.1)
    p.add_argument("--save-video", action="store_true")
    p.add_argument("--fps", type=float, default=30.0)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    from lib_dynamic_hvip.constants_skeleton import SMPL_ALL_54
    from lib_dynamic_hvip.fitting.smpl_seq_forward_module import SMPL_Seq_Forward_Module as SMPLFitter

    result = load_pickle(args.result_pkl)
    if isinstance(result, SMPLFitter.SMPL_Param_Seq):
        smpl_params = [result.get_smpl_param_at_frame(i) for i in range(result.num_batch)]
    elif isinstance(result, SMPLFitter.SMPL_Param):
        smpl_params = [result]
    else:
        raise TypeError(f"Unknown result type: {type(result)}")

    det_results_cache = load_pickle(args.rtm_cache)
    if not hasattr(det_results_cache, "get_result_by_frame_index"):
        raise TypeError(f"RTM cache missing get_result_by_frame_index: {type(det_results_cache)}")
    det_results = [det_results_cache.get_result_by_frame_index(i) for i in range(args.index_start, args.index_start + len(smpl_params))]

    if args.image is not None:
        image_paths = [args.image] * len(smpl_params)
    else:
        exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        image_paths_all = sorted(p for p in args.image_dir.iterdir() if p.suffix.lower() in exts)
        image_paths = image_paths_all[args.index_start : args.index_start + len(smpl_params)]
        smpl_params = smpl_params[: len(image_paths)]
        det_results = det_results[: len(image_paths)]

    fitter = SMPLFitter()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    video_writer = None

    for frame_idx, (smpl_param, det_result, img_path) in enumerate(zip(smpl_params, det_results, image_paths)):
        img = cv2.imread(str(img_path))
        if img is None:
            img = np.zeros((args.image_h, args.image_w, 3), dtype=np.uint8)
        if img.shape[1] != args.image_w or img.shape[0] != args.image_h:
            img = cv2.resize(img, (args.image_w, args.image_h))
        vis = visualize_frame(img, smpl_param, det_result, fitter, args.image_w, args.image_h, SMPL_ALL_54, args.mesh, args.conf_thresh)
        stem = img_path.stem if img_path != args.image else f"frame_{frame_idx:04d}"
        out_path = args.output_dir / f"vis_{stem}.jpg"
        cv2.imwrite(str(out_path), vis, [cv2.IMWRITE_JPEG_QUALITY, 92])
        print(f"[{frame_idx + 1}/{len(smpl_params)}] saved -> {out_path}")
        if args.save_video:
            if video_writer is None:
                video_writer = cv2.VideoWriter(str(args.output_dir / "vis_result.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (args.image_w, args.image_h))
            video_writer.write(vis)

    if video_writer is not None:
        video_writer.release()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
