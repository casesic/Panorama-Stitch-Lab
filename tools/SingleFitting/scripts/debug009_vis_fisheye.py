#!/usr/bin/env python3
"""
Visualize SMPL-X fitting results on fisheye images.

Usage:
  python tools/SingleFitting/scripts/debug009_vis_fisheye.py `
    --result-pkl data_process/Output/20260311_riding/fisheye_smplx/cam2/fisheye_cam2_s0_i0_29_seq.pkl `
    --rtm-cache data_process/Output/20260311_riding/single_frame_cache/camera2/rtmlib_multi_frame/rtm_joints_cache_s490_529_c2.pkl `
    --camera cam2 `
    --index 490 `
    --video-path Circular_Video/00_send/20260311_riding/VID_20181003_034909_00_982.0.mp4 `
    --output-dir data_process/Output/20260311_riding/fisheye_smplx/cam2/vis `
    --mesh
    
  python tools/SingleFitting/scripts/debug009_vis_fisheye.py `
    --result-pkl data_process/Output/20260311_riding/fisheye_smplx/cam4/fisheye_cam4_s0_i22_39_seq.pkl `
    --rtm-cache data_process/Output/20260311_riding/single_frame_cache/camera4/rtmlib_multi_frame/rtm_joints_cache_s490_529_c4.pkl `
    --camera cam4 `
    --index 512 `
    --video-path Circular_Video/00_send/20260311_riding/VID_20181003_034909_00_982.1.mp4 `
    --output-dir data_process/Output/20260311_riding/fisheye_smplx/cam4/vis `
    --mesh
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from single_fitting.debug_shared import (
    DS_PARAMS,
    LHAND_IDX,
    RHAND_IDX,
    ds_project_np,
    load_pickle,
    normalize_keypoints_format,
)
from single_fitting.debug_fit_shared import SingleCamFisheyeFitter, forward_param_to_numpy, get_model_faces, unpack_smplx_result
from single_fitting.debug_vis_shared import draw_pts, draw_skel, render_mesh_shading

SMPLX_LHAND = list(range(25, 40))
SMPLX_RHAND = list(range(40, 55))
SMPLX_FACE = [55, 56, 57, 58, 59]

SMPLX_BODY_PAIRS = [
    (0, 3), (3, 6), (6, 9), (9, 12), (12, 15),
    (9, 13), (13, 16), (16, 18), (18, 20),
    (9, 14), (14, 17), (17, 19), (19, 21),
    (0, 1), (1, 4), (4, 7), (7, 10),
    (0, 2), (2, 5), (5, 8), (8, 11),
]
COCO_PAIRS = [
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
]
SMPLX_BODY_22 = {
    "Pelvis": 0, "L_Hip": 1, "R_Hip": 2, "Spine1": 3,
    "L_Knee": 4, "R_Knee": 5, "Spine2": 6,
    "L_Ankle": 7, "R_Ankle": 8, "Spine3": 9,
    "L_Foot": 10, "R_Foot": 11, "Neck": 12,
    "L_Collar": 13, "R_Collar": 14, "Head": 15,
    "L_Shoulder": 16, "R_Shoulder": 17,
    "L_Elbow": 18, "R_Elbow": 19,
    "L_Wrist": 20, "R_Wrist": 21,
    "Nose": 55, "L_Eye": 57, "R_Eye": 56,
    "L_Ear": 59, "R_Ear": 58,
}


def save_obj(path: Path, verts: np.ndarray, faces: np.ndarray) -> None:
    with path.open("w", encoding="utf-8") as f:
        for v in verts:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for face in faces:
            f.write(f"f {face[0] + 1} {face[1] + 1} {face[2] + 1}\n")


def visualize_frame(
    img: np.ndarray,
    smplx_param,
    kp133: Optional[np.ndarray],
    kp_fmt: dict,
    fitter,
    draw_mesh: bool = False,
    thresh: float = 0.1,
    mesh_alpha: float = 0.6,
):
    vis = img.copy()
    if kp133 is not None:
        body17 = kp133[:17]
        draw_skel(vis, body17[:, :2], COCO_PAIRS, conf=body17[:, 2], thresh=thresh, color=(0, 200, 0))
        draw_pts(vis, body17[:, :2], conf=body17[:, 2], thresh=thresh, r=6, color=(0, 230, 0))
        if kp133.shape[0] >= 133:
            draw_pts(vis, kp133[LHAND_IDX, :2], conf=kp133[LHAND_IDX, 2], thresh=thresh, r=4, color=(0, 220, 255))
            draw_pts(vis, kp133[RHAND_IDX, :2], conf=kp133[RHAND_IDX, 2], thresh=thresh, r=4, color=(255, 220, 0))

    verts, joints_all, joints_body = forward_param_to_numpy(fitter, smplx_param)

    faces = None
    if draw_mesh:
        faces = get_model_faces(fitter)
        vis = render_mesh_shading(vis, verts, faces, ds_params=DS_PARAMS, alpha=mesh_alpha)

    kp_names = set(normalize_keypoints_format(kp_fmt).keys())
    matched_idx = [idx for name, idx in SMPLX_BODY_22.items() if name in kp_names and idx < 22]
    if matched_idx:
        matched_pts = joints_all[matched_idx]
        matched_2d = ds_project_np(matched_pts, **DS_PARAMS)
        filtered_pairs = [(i, j) for i, j in SMPLX_BODY_PAIRS if i in matched_idx and j in matched_idx]
        all_body_2d = ds_project_np(joints_body, **DS_PARAMS)
        draw_skel(vis, all_body_2d, filtered_pairs, color=(0, 100, 255), thick=2)
        draw_pts(vis, matched_2d, r=5, color=(0, 140, 255))

    if {"Nose", "L_Eye", "R_Eye", "L_Ear", "R_Ear"} & kp_names:
        draw_pts(vis, ds_project_np(joints_all[SMPLX_FACE], **DS_PARAMS), r=4, color=(0, 255, 180))

    draw_pts(vis, ds_project_np(joints_all[SMPLX_LHAND], **DS_PARAMS), r=4, color=(255, 80, 0))
    draw_pts(vis, ds_project_np(joints_all[SMPLX_RHAND], **DS_PARAMS), r=4, color=(200, 0, 255))
    return vis, verts, faces


def parse_args():
    p = argparse.ArgumentParser(description="Visualize SMPL-X fitting on fisheye images.")
    p.add_argument("--result-pkl", type=Path, required=True)
    p.add_argument("--rtm-cache", type=Path, required=True)
    p.add_argument("--camera", type=str, default="cam2", choices=["cam2", "cam4"])
    p.add_argument("--index-start", type=int, default=490)
    p.add_argument("--image-dir", type=Path, default=None, help="帧图像目录")
    p.add_argument("--video-path", type=Path, default=None, help="视频文件路径，优先级高于 image-dir")
    p.add_argument("--image-filename-pattern", type=str, default="frame_{:05d}.png")
    p.add_argument("--output-dir", type=Path, default=Path("outputs/fisheye_smplx/vis"))
    p.add_argument("--mesh", action="store_true")
    p.add_argument("--mesh-alpha", type=float, default=0.6)
    p.add_argument("--conf-thresh", type=float, default=0.1)
    p.add_argument("--save-video", action="store_true")
    p.add_argument("--save-obj", action="store_true")
    p.add_argument("--fps", type=float, default=30.0)
    return p.parse_args()


def main():
    args = parse_args()

    result = load_pickle(args.result_pkl)
    params = unpack_smplx_result(result)

    det_raw = load_pickle(args.rtm_cache)
    kp_all = det_raw.array_keypoints if hasattr(det_raw, "array_keypoints") else None
    kp_fmt = det_raw.keypoints_format if hasattr(det_raw, "keypoints_format") else {}

    frames = []
    if args.video_path is not None:
        cap = cv2.VideoCapture(str(args.video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {args.video_path}")
        cap.set(cv2.CAP_PROP_POS_FRAMES, args.index_start)
        for _ in range(len(params)):
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            frames.append(frame)
        cap.release()
    else:
        if args.image_dir is None:
            raise ValueError("Either --video-path or --image-dir must be provided.")
        for i in range(len(params)):
            frame_idx = args.index_start + i
            img_path = args.image_dir / args.image_filename_pattern.format(frame_idx)
            img = cv2.imread(str(img_path))
            if img is None:
                print(f"[warn] Failed to read image: {img_path}")
                continue
            frames.append(img)

    fitter = SingleCamFisheyeFitter(gender="neutral", camera=args.camera)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    video_writer = None

    for fi, (param, img) in enumerate(zip(params, frames)):

        ri = args.index_start + fi
        kp133 = kp_all[ri] if (kp_all is not None and ri < len(kp_all)) else None
        vis, verts_out, faces_out = visualize_frame(
            img=img,
            smplx_param=param,
            kp133=kp133,
            kp_fmt=kp_fmt,
            fitter=fitter,
            draw_mesh=args.mesh,
            thresh=args.conf_thresh,
            mesh_alpha=args.mesh_alpha,
        )

        stem = f"frame_{args.index_start + fi:05d}"
        out_path = args.output_dir / f"vis_{args.camera}_{stem}.jpg"
        cv2.imwrite(str(out_path), vis, [cv2.IMWRITE_JPEG_QUALITY, 92])
        print(f"[{fi + 1}/{len(params)}] -> {out_path}")

        if args.save_obj:
            if faces_out is None:
                verts_out, _, _ = forward_param_to_numpy(fitter, param)
                faces_out = get_model_faces(fitter)
            save_obj(args.output_dir / f"mesh_{args.camera}_{stem}.obj", verts_out, faces_out)

        if args.save_video:
            if video_writer is None:
                h, w = img.shape[:2]
                video_writer = cv2.VideoWriter(
                    str(args.output_dir / f"vis_{args.camera}.mp4"),
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    args.fps,
                    (w, h),
                )
            video_writer.write(vis)

    if video_writer is not None:
        video_writer.release()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
