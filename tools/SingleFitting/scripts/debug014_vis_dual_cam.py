#!/usr/bin/env python3
"""
Visualize a dual-camera SMPL-X result on cam2 and cam4 images.

Usage:
  python tools/SingleFitting/scripts/debug014_vis_dual_cam.py ^
    --result-pkl tools/SingleFitting/outputs/dual_cam/dual_s0_i0_29_frame15.pkl ^
    --rtm-cam2 tools/SingleFitting/outputs/fisheye/rtm_cam2.pkl ^
    --rtm-cam4 tools/SingleFitting/outputs/fisheye/rtm_cam4.pkl ^
    --image-cam2 data/fisheye_cam2_frames/frame0015.jpg ^
    --image-cam4 data/fisheye_cam4_frames/frame0015.jpg ^
    --output-dir tools/SingleFitting/outputs/dual_cam_vis ^
    --mesh
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, Optional

import cv2
import numpy as np

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from single_fitting.debug_shared import (
    DEFAULT_DELTA_T,
    DS_PARAMS,
    LHAND_IDX,
    RHAND_IDX,
    load_pickle,
    normalize_keypoints_format,
    world_to_cam4,
)
from single_fitting.debug_fit_shared import SingleCamFisheyeFitter, forward_param_to_numpy, get_model_faces
from single_fitting.debug_vis_shared import draw_pts, draw_skel, render_mesh_shading

SMPLX_LHAND = list(range(25, 40))
SMPLX_RHAND = list(range(40, 55))
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
}


def ds_project_np(pts: np.ndarray) -> np.ndarray:
    from single_fitting.debug_shared import ds_project_np as _proj

    return _proj(pts, **DS_PARAMS)


def get_rtm_keypoints(rtm_cache_path: Path, frame_idx: int):
    det = load_pickle(rtm_cache_path)
    if hasattr(det, "array_keypoints"):
        kp = det.array_keypoints[frame_idx]
        fmt = normalize_keypoints_format(det.keypoints_format if hasattr(det, "keypoints_format") else {})
        j = kp.shape[0]
        return kp[:17], (kp[LHAND_IDX] if j >= 133 else None), (kp[RHAND_IDX] if j >= 133 else None), fmt
    if hasattr(det, "get_result_by_frame_index"):
        dr = det.get_result_by_frame_index(frame_idx)
        kp = dr.keypoints
        fmt = normalize_keypoints_format(dr.keypoints_format)
        j = kp.shape[0]
        return kp[:17], (kp[LHAND_IDX] if j >= 133 else None), (kp[RHAND_IDX] if j >= 133 else None), fmt
    raise TypeError(f"RTM cache format not supported: {type(det)}")


def project_world(pts_world: np.ndarray, camera: str, delta_t: np.ndarray) -> np.ndarray:
    pts_cam = pts_world if camera == "cam2" else world_to_cam4(pts_world, delta_t)
    return ds_project_np(pts_cam)


def _resize_h(img: np.ndarray, target_h: int) -> np.ndarray:
    if target_h >= img.shape[0]:
        return img
    scale = target_h / img.shape[0]
    return cv2.resize(img, (int(img.shape[1] * scale), target_h))


def visualize_one_camera(
    img: np.ndarray,
    verts_world: np.ndarray,
    joints_world: np.ndarray,
    faces: np.ndarray,
    kp_body: np.ndarray,
    kp_fmt: Dict[str, int],
    camera: str,
    delta_t: np.ndarray,
    draw_mesh: bool = True,
    resize_h: int = 720,
    label: str = "",
) -> np.ndarray:
    verts_cam = verts_world if camera == "cam2" else world_to_cam4(verts_world, delta_t)
    joints_cam = joints_world if camera == "cam2" else world_to_cam4(joints_world, delta_t)
    canvas = img.copy()

    if draw_mesh:
        canvas = render_mesh_shading(canvas, verts_cam, faces, ds_params=DS_PARAMS, alpha=0.55)

    draw_skel(canvas, kp_body[:, :2], COCO_PAIRS, conf=kp_body[:, 2], thresh=0.1, color=(0, 180, 0), thick=2)
    draw_pts(canvas, kp_body[:, :2], conf=kp_body[:, 2], thresh=0.1, r=max(5, canvas.shape[0] // 576), color=(0, 220, 0))

    matched_smplx_idx = []
    matched_kp_idx = []
    for name, kp_idx in kp_fmt.items():
        if name in SMPLX_BODY_22 and kp_idx < kp_body.shape[0]:
            matched_smplx_idx.append(SMPLX_BODY_22[name])
            matched_kp_idx.append(kp_idx)

    if matched_smplx_idx:
        smplx_2d = ds_project_np(joints_cam[matched_smplx_idx])
        conf = kp_body[matched_kp_idx, 2]
        draw_pts(canvas, smplx_2d, conf=conf, thresh=0.1, r=max(5, canvas.shape[0] // 576), color=(30, 140, 255))

        matched_set = set(matched_smplx_idx)
        body_2d = ds_project_np(joints_cam[:22])
        filtered_pairs = [(i, j) for i, j in SMPLX_BODY_PAIRS if i in matched_set and j in matched_set]
        draw_skel(canvas, body_2d, filtered_pairs, color=(30, 140, 255), thick=2)

    draw_pts(canvas, ds_project_np(joints_cam[SMPLX_LHAND]), r=3, color=(255, 80, 0))
    draw_pts(canvas, ds_project_np(joints_cam[SMPLX_RHAND]), r=3, color=(200, 0, 255))

    canvas = _resize_h(canvas, resize_h)
    if label:
        h = canvas.shape[0]
        cv2.putText(canvas, label, (20, int(h * 0.04)), cv2.FONT_HERSHEY_SIMPLEX, max(1.0, h / 960), (0, 0, 0), max(3, h // 400))
        cv2.putText(canvas, label, (20, int(h * 0.04)), cv2.FONT_HERSHEY_SIMPLEX, max(1.0, h / 960), (255, 255, 255), max(1, h // 800))
    return canvas


def run(
    result_pkl: Path,
    rtm_cam2: Path,
    rtm_cam4: Path,
    image_cam2: Path,
    image_cam4: Path,
    output_dir: Path,
    frame_idx: int = 0,
    draw_mesh: bool = True,
    resize_h: int = 2880,
    delta_t: Optional[np.ndarray] = None,
):
    from lib_dynamic_hvip.fitting.smplx_seq_forward_module import SMPLX_Seq_Forward_Module

    output_dir.mkdir(parents=True, exist_ok=True)
    dt = DEFAULT_DELTA_T if delta_t is None else delta_t

    param = load_pickle(result_pkl)
    if isinstance(param, SMPLX_Seq_Forward_Module.SMPLX_Param_Seq):
        param = param.get_param_at_frame(frame_idx)

    fitter = SingleCamFisheyeFitter(gender="neutral", camera="cam2")
    verts_world, joints_world, _ = forward_param_to_numpy(fitter, param)
    faces = get_model_faces(fitter)

    img_cam2 = cv2.imread(str(image_cam2))
    img_cam4 = cv2.imread(str(image_cam4))
    if img_cam2 is None:
        raise FileNotFoundError(image_cam2)
    if img_cam4 is None:
        raise FileNotFoundError(image_cam4)

    kp2_body, _, _, fmt2 = get_rtm_keypoints(rtm_cam2, frame_idx)
    kp4_body, _, _, fmt4 = get_rtm_keypoints(rtm_cam4, frame_idx)

    vis_cam2 = visualize_one_camera(img_cam2, verts_world, joints_world, faces, kp2_body, fmt2, "cam2", dt, draw_mesh, resize_h, "cam2")
    vis_cam4 = visualize_one_camera(img_cam4, verts_world, joints_world, faces, kp4_body, fmt4, "cam4", dt, draw_mesh, resize_h, "cam4")

    h = min(vis_cam2.shape[0], vis_cam4.shape[0])
    vis_cam2 = _resize_h(vis_cam2, h)
    vis_cam4 = _resize_h(vis_cam4, h)
    divider = np.ones((h, 8, 3), dtype=np.uint8) * 200
    combined = np.concatenate([vis_cam2, divider, vis_cam4], axis=1)

    stem = result_pkl.stem
    out_all = output_dir / f"dual_vis_{stem}.jpg"
    out_c2 = output_dir / f"dual_vis_{stem}_cam2.jpg"
    out_c4 = output_dir / f"dual_vis_{stem}_cam4.jpg"
    cv2.imwrite(str(out_all), combined, [cv2.IMWRITE_JPEG_QUALITY, 92])
    cv2.imwrite(str(out_c2), vis_cam2, [cv2.IMWRITE_JPEG_QUALITY, 92])
    cv2.imwrite(str(out_c4), vis_cam4, [cv2.IMWRITE_JPEG_QUALITY, 92])
    print(f"Saved combined : {out_all}")
    print(f"Saved cam2     : {out_c2}")
    print(f"Saved cam4     : {out_c4}")
    return out_all


def parse_args():
    p = argparse.ArgumentParser(description="Visualize dual-camera SMPL-X result.")
    p.add_argument("--result-pkl", type=Path, required=True)
    p.add_argument("--rtm-cam2", type=Path, required=True)
    p.add_argument("--rtm-cam4", type=Path, required=True)
    p.add_argument("--image-cam2", type=Path, required=True)
    p.add_argument("--image-cam4", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, default=Path("outputs/dual_cam_vis"))
    p.add_argument("--frame-idx", type=int, default=0)
    p.add_argument("--mesh", action="store_true")
    p.add_argument("--resize-h", type=int, default=2880)
    p.add_argument("--delta-t", type=float, nargs=3, default=None, metavar=("TX", "TY", "TZ"))
    return p.parse_args()


def main():
    args = parse_args()
    run(
        result_pkl=args.result_pkl,
        rtm_cam2=args.rtm_cam2,
        rtm_cam4=args.rtm_cam4,
        image_cam2=args.image_cam2,
        image_cam4=args.image_cam4,
        output_dir=args.output_dir,
        frame_idx=args.frame_idx,
        draw_mesh=args.mesh,
        resize_h=args.resize_h,
        delta_t=np.array(args.delta_t, dtype=np.float32) if args.delta_t is not None else None,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
