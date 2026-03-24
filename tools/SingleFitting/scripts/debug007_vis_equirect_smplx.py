#!/usr/bin/env python3
"""
Visualize SMPL-X fitting results on equirectangular images.

Usage:
  python tools/SingleFitting/scripts/debug007_vis_equirect_smplx.py ^
    --result-pkl tools/SingleFitting/outputs/equirect_smplx/result_seq.pkl ^
    --rtm-cache tools/SingleFitting/outputs/equirect/rtm_cache.pkl ^
    --image-dir data/equirect_frames ^
    --image-w 5720 --image-h 2880 ^
    --output-dir tools/SingleFitting/outputs/equirect_smplx/vis ^
    --mesh
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from single_fitting.debug_shared import LHAND_IDX, RHAND_IDX, erp_project_np, load_pickle, normalize_keypoints_format
from single_fitting.debug_vis_erp_shared import draw_keypoints, render_mesh

SMPLX_LHAND = list(range(25, 40))
SMPLX_RHAND = list(range(40, 55))
SMPLX_FACE = [55, 56, 57, 58, 59]


def draw_pts(img, pts2d, conf=None, thresh=0.1, r=5, color=(0, 220, 0)):
    h, w = img.shape[:2]
    for i, (u, v) in enumerate(pts2d):
        if conf is not None and conf[i] < thresh:
            continue
        pu, pv = int(round(float(u))), int(round(float(v)))
        if 0 <= pu < w and 0 <= pv < h:
            cv2.circle(img, (pu, pv), r, color, -1)
    return img


def visualize_frame(
    img,
    smplx_param,
    kp133,
    fitter,
    width,
    height,
    draw_mesh=False,
    thresh=0.1,
    mesh_alpha=0.6,
    kp133_fmt=None,
    smplx_body_map=None,
    use_shader=True,
):
    import torch

    vis = img.copy()
    if kp133 is not None:
        body17 = kp133[:17]
        draw_keypoints(vis, body17, conf_thresh=thresh, radius=6, color=(0, 230, 0))
        if kp133.shape[0] >= 133:
            draw_pts(vis, kp133[LHAND_IDX, :2], conf=kp133[LHAND_IDX, 2], thresh=thresh, r=4, color=(0, 220, 255))
            draw_pts(vis, kp133[RHAND_IDX, :2], conf=kp133[RHAND_IDX, 2], thresh=thresh, r=4, color=(255, 220, 0))

    with torch.no_grad():
        verts_t, joints_t, _ = fitter.forward_param(smplx_param)
    joints_all = joints_t.cpu().numpy()
    verts = verts_t.cpu().numpy()

    if draw_mesh:
        model = fitter._get_model(1)
        faces = model.faces.numpy() if hasattr(model.faces, "numpy") else model.faces
        vis = render_mesh(vis, verts, faces, width, height, alpha=mesh_alpha, use_shader=use_shader)

    if kp133_fmt is not None and smplx_body_map is not None:
        kp_names = set(kp133_fmt.keys())
        matched_smplx_idx = sorted({idx for name, idx in smplx_body_map.items() if name in kp_names and 0 <= idx < joints_all.shape[0]})
        if matched_smplx_idx:
            draw_pts(vis, erp_project_np(joints_all, width, height)[matched_smplx_idx], r=5, color=(0, 140, 255))
    else:
        draw_pts(vis, erp_project_np(joints_all[:22], width, height), r=5, color=(0, 140, 255))

    draw_pts(vis, erp_project_np(joints_all[SMPLX_FACE], width, height), r=4, color=(0, 140, 255))
    draw_pts(vis, erp_project_np(joints_all[SMPLX_LHAND], width, height), r=4, color=(255, 80, 0))
    draw_pts(vis, erp_project_np(joints_all[SMPLX_RHAND], width, height), r=4, color=(200, 0, 255))
    return vis


def parse_args():
    p = argparse.ArgumentParser(description="Visualize ERP SMPL-X fitting results.")
    p.add_argument("--result-pkl", type=Path, required=True)
    p.add_argument("--rtm-cache", type=Path, required=True)
    p.add_argument("--index-start", type=int, default=0)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--image", type=Path, default=None)
    src.add_argument("--image-dir", type=Path, default=None)
    p.add_argument("--image-w", type=int, default=5720)
    p.add_argument("--image-h", type=int, default=2880)
    p.add_argument("--output-dir", type=Path, default=Path("tools/SingleFitting/outputs/equirect_smplx/vis"))
    p.add_argument("--mesh", action="store_true")
    p.add_argument("--mesh-alpha", type=float, default=0.6)
    p.add_argument("--no-shader", action="store_true")
    p.add_argument("--conf-thresh", type=float, default=0.1)
    p.add_argument("--save-video", action="store_true")
    p.add_argument("--fps", type=float, default=30.0)
    return p.parse_args()


def main():
    args = parse_args()
    from lib_dynamic_hvip.fitting.smplx_seq_forward_module import SMPLX_BODY_22, SMPLX_Seq_Forward_Module

    result = load_pickle(args.result_pkl)
    if isinstance(result, SMPLX_Seq_Forward_Module.SMPLX_Param_Seq):
        params = [result.get_param_at_frame(i) for i in range(result.num_batch)]
    elif isinstance(result, SMPLX_Seq_Forward_Module.SMPLX_Param):
        params = [result]
    else:
        raise TypeError(type(result))

    det_raw = load_pickle(args.rtm_cache)
    kp_all = det_raw.array_keypoints if hasattr(det_raw, "array_keypoints") else None
    kp_fmt = normalize_keypoints_format(det_raw.keypoints_format if hasattr(det_raw, "keypoints_format") else {})

    if args.image is not None:
        img_paths = [args.image] * len(params)
    else:
        exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        all_imgs = sorted(p for p in args.image_dir.iterdir() if p.suffix.lower() in exts)
        img_paths = all_imgs[args.index_start : args.index_start + len(params)]
        params = params[: len(img_paths)]

    fitter = SMPLX_Seq_Forward_Module(gender="neutral")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    video_writer = None
    for fi, (param, img_path) in enumerate(zip(params, img_paths)):
        img = cv2.imread(str(img_path))
        if img is None:
            img = np.zeros((args.image_h, args.image_w, 3), dtype=np.uint8)
        if img.shape[1] != args.image_w or img.shape[0] != args.image_h:
            img = cv2.resize(img, (args.image_w, args.image_h))
        ri = args.index_start + fi
        kp133 = kp_all[ri] if (kp_all is not None and ri < len(kp_all)) else None
        vis = visualize_frame(
            img,
            param,
            kp133,
            fitter,
            args.image_w,
            args.image_h,
            draw_mesh=args.mesh,
            thresh=args.conf_thresh,
            mesh_alpha=args.mesh_alpha,
            kp133_fmt=kp_fmt,
            smplx_body_map=SMPLX_BODY_22,
            use_shader=not args.no_shader,
        )
        stem = img_path.stem if args.image is None else f"frame_{fi:04d}"
        out_path = args.output_dir / f"vis_{stem}.jpg"
        cv2.imwrite(str(out_path), vis, [cv2.IMWRITE_JPEG_QUALITY, 92])
        print(f"[{fi + 1}/{len(params)}] -> {out_path}")
        if args.save_video:
            if video_writer is None:
                video_writer = cv2.VideoWriter(str(args.output_dir / "vis_result.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (args.image_w, args.image_h))
            video_writer.write(vis)

    if video_writer is not None:
        video_writer.release()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
