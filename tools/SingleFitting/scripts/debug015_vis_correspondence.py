#!/usr/bin/env python3
"""
Visualize sparse and dense correspondences between cam2 and cam4.

Usage:
  python tools/SingleFitting/scripts/debug015_vis_correspondence.py ^
    --smpl-cam2 tools/SingleFitting/outputs/fisheye_smplx_cam2/fisheye_cam2_s0_i0_29_frame15.pkl ^
    --smpl-cam4 tools/SingleFitting/outputs/fisheye_smplx_cam4/fisheye_cam4_s0_i0_29_frame15.pkl ^
    --rtm-cam2 tools/SingleFitting/outputs/fisheye/rtm_cam2.pkl ^
    --rtm-cam4 tools/SingleFitting/outputs/fisheye/rtm_cam4.pkl ^
    --image-cam2 data/fisheye_cam2_frames/frame0015.jpg ^
    --image-cam4 data/fisheye_cam4_frames/frame0015.jpg ^
    --output-dir tools/SingleFitting/outputs/correspondence_vis
"""
from __future__ import annotations

import argparse
import os
import random
import sys
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from single_fitting.debug_shared import (
    DEFAULT_DELTA_T,
    DS_PARAMS,
    load_pickle,
    normalize_keypoints_format,
)
from single_fitting.debug_fit_shared import SingleCamFisheyeFitter, forward_param_to_numpy, get_model_faces
from single_fitting.debug_vis_shared import render_mesh_shading

COCO_PAIRS = [
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
]


def ds_project_np(pts: np.ndarray) -> np.ndarray:
    from single_fitting.debug_shared import ds_project_np as _proj

    return _proj(pts, **DS_PARAMS)


def in_img(pt, h, w):
    return 0 <= int(pt[0]) < w and 0 <= int(pt[1]) < h


def compute_visibility(verts: np.ndarray, faces: np.ndarray) -> np.ndarray:
    try:
        import trimesh
    except ImportError:
        print("[warn] trimesh not installed; fallback to z>0 visibility")
        return verts[:, 2] > 0

    mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    dirs = -verts.copy()
    dirs = dirs / np.linalg.norm(dirs, axis=1, keepdims=True).clip(1e-8)
    origins = verts + dirs * 1e-4
    _, index_ray, _ = mesh.ray.intersects_location(ray_origins=origins, ray_directions=dirs, multiple_hits=False)
    occluded = np.zeros(len(verts), dtype=bool)
    occluded[index_ray] = True
    return ~occluded


def get_kp(rtm_path: Path, frame_idx: int = 0):
    det = load_pickle(rtm_path)
    if hasattr(det, "array_keypoints"):
        kp = det.array_keypoints[frame_idx]
        fmt = normalize_keypoints_format(det.keypoints_format if hasattr(det, "keypoints_format") else {})
    elif hasattr(det, "get_result_by_frame_index"):
        dr = det.get_result_by_frame_index(frame_idx)
        kp, fmt = dr.keypoints, normalize_keypoints_format(dr.keypoints_format)
    else:
        raise TypeError(type(det))
    return kp, fmt


def get_smplx(pkl_path: Path, fitter):
    from lib_dynamic_hvip.fitting.smplx_seq_forward_module import SMPLX_Seq_Forward_Module

    param = load_pickle(pkl_path)
    if isinstance(param, SMPLX_Seq_Forward_Module.SMPLX_Param_Seq):
        param = param.get_param_at_frame(param.num_batch // 2)
    verts, _, _ = forward_param_to_numpy(fitter, param)
    faces = get_model_faces(fitter)
    return verts, faces


def _resize_h(img: np.ndarray, h: int):
    s = h / img.shape[0]
    return cv2.resize(img, (int(img.shape[1] * s), h)), s


def fig1(img2, img4, kp2, kp4, fmt2, fmt4, thresh=0.1):
    c2, c4 = img2.copy(), img4.copy()
    h2, w2 = c2.shape[:2]
    h4, w4 = c4.shape[:2]

    for img_c, kp, h, w in [(c2, kp2, h2, w2), (c4, kp4, h4, w4)]:
        for i, j in COCO_PAIRS:
            if i >= len(kp) or j >= len(kp) or kp[i, 2] < thresh or kp[j, 2] < thresh:
                continue
            p1, p2 = (int(kp[i, 0]), int(kp[i, 1])), (int(kp[j, 0]), int(kp[j, 1]))
            if in_img(p1, h, w) and in_img(p2, h, w):
                cv2.line(img_c, p1, p2, (0, 160, 0), 2)

    coco_names = {
        "Nose", "L_Eye", "R_Eye", "L_Ear", "R_Ear",
        "L_Shoulder", "R_Shoulder", "L_Elbow", "R_Elbow",
        "L_Wrist", "R_Wrist", "L_Hip", "R_Hip",
        "L_Knee", "R_Knee", "L_Ankle", "R_Ankle",
    }
    common = sorted(set(fmt2) & set(fmt4) & coco_names)
    matched_pts2, matched_pts4 = [], []
    for name in common:
        i2, i4 = fmt2[name], fmt4[name]
        if i2 >= len(kp2) or i4 >= len(kp4) or kp2[i2, 2] < thresh or kp4[i4, 2] < thresh:
            continue
        p2, p4 = (int(kp2[i2, 0]), int(kp2[i2, 1])), (int(kp4[i4, 0]), int(kp4[i4, 1]))
        if not (in_img(p2, h2, w2) and in_img(p4, h4, w4)):
            continue
        matched_pts2.append(p2)
        matched_pts4.append(p4)
        cv2.circle(c2, p2, 6, (50, 80, 220), -1)
        cv2.circle(c4, p4, 6, (30, 180, 50), -1)

    ht = min(h2, h4)
    c2r, s2 = _resize_h(c2, ht)
    c4r, s4 = _resize_h(c4, ht)
    offset = c2r.shape[1] + 6
    combined = np.concatenate([c2r, np.ones((ht, 6, 3), np.uint8) * 180, c4r], axis=1)
    for p2, p4 in zip(matched_pts2, matched_pts4):
        lp2 = (int(round(p2[0] * s2)), int(round(p2[1] * s2)))
        lp4 = (int(round(p4[0] * s4 + offset)), int(round(p4[1] * s4)))
        cv2.line(combined, lp2, lp4, (0, 220, 200), 1, cv2.LINE_AA)
    return c2, c4, combined


def fig2(img2, img4, verts_cam2, verts_cam4, faces, vis2, vis4):
    c2 = render_mesh_shading(img2.copy(), verts_cam2, faces, ds_params=DS_PARAMS, alpha=0.55)
    c4 = render_mesh_shading(img4.copy(), verts_cam4, faces, ds_params=DS_PARAMS, alpha=0.55)
    pts2, pts4 = ds_project_np(verts_cam2), ds_project_np(verts_cam4)
    for vi in range(len(verts_cam2)):
        p2, p4 = pts2[vi], pts4[vi]
        if vis2[vi] and in_img(p2, c2.shape[0], c2.shape[1]):
            cv2.circle(c2, (int(p2[0]), int(p2[1])), 3, (0, 0, 220), -1)
        if vis4[vi] and in_img(p4, c4.shape[0], c4.shape[1]):
            cv2.circle(c4, (int(p4[0]), int(p4[1])), 3, (20, 180, 50), -1)
    return c2, c4


def fig3(img2, img4, verts_cam2, verts_cam4, faces, vis2, vis4, n_sample=50, seed=42):
    c2 = render_mesh_shading(img2.copy(), verts_cam2, faces, ds_params=DS_PARAMS, alpha=0.55)
    c4 = render_mesh_shading(img4.copy(), verts_cam4, faces, ds_params=DS_PARAMS, alpha=0.55)
    h2, w2 = c2.shape[:2]
    h4, w4 = c4.shape[:2]
    pts2, pts4 = ds_project_np(verts_cam2), ds_project_np(verts_cam4)
    both = [vi for vi in range(len(verts_cam2)) if vis2[vi] and vis4[vi] and in_img(pts2[vi], h2, w2) and in_img(pts4[vi], h4, w4)]
    rng = random.Random(seed)
    sampled = rng.sample(both, min(n_sample, len(both)))

    ht = min(h2, h4)
    c2r, s2 = _resize_h(c2, ht)
    c4r, s4 = _resize_h(c4, ht)
    offset = c2r.shape[1] + 6
    combined = np.concatenate([c2r, np.ones((ht, 6, 3), np.uint8) * 180, c4r], axis=1)
    np.random.seed(seed)
    colors = [tuple(int(c) for c in np.random.randint(80, 255, 3)) for _ in range(len(sampled))]
    for color, vi in zip(colors, sampled):
        p2 = (int(round(pts2[vi, 0] * s2)), int(round(pts2[vi, 1] * s2)))
        p4 = (int(round(pts4[vi, 0] * s4 + offset)), int(round(pts4[vi, 1] * s4)))
        cv2.line(combined, p2, p4, color, 1, cv2.LINE_AA)
        cv2.circle(combined, p2, 4, color, -1)
        cv2.circle(combined, p4, 4, color, -1)
    return combined


def run(
    smpl_cam2_pkl,
    smpl_cam4_pkl,
    rtm_cam2_path,
    rtm_cam4_path,
    image_cam2_path,
    image_cam4_path,
    output_dir,
    frame_idx=0,
    conf_thresh=0.1,
    n_sample=50,
    delta_t=None,
    use_raycast=True,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    _ = DEFAULT_DELTA_T if delta_t is None else delta_t

    img2 = cv2.imread(str(image_cam2_path))
    img4 = cv2.imread(str(image_cam4_path))
    if img2 is None:
        raise FileNotFoundError(image_cam2_path)
    if img4 is None:
        raise FileNotFoundError(image_cam4_path)

    kp2, fmt2 = get_kp(rtm_cam2_path, frame_idx)
    kp4, fmt4 = get_kp(rtm_cam4_path, frame_idx)

    fitter2 = SingleCamFisheyeFitter(gender="neutral", camera="cam2")
    fitter4 = SingleCamFisheyeFitter(gender="neutral", camera="cam4")
    verts_cam2, faces = get_smplx(smpl_cam2_pkl, fitter2)
    verts_cam4, _ = get_smplx(smpl_cam4_pkl, fitter4)

    if use_raycast:
        vis2 = compute_visibility(verts_cam2, faces)
        vis4 = compute_visibility(verts_cam4, faces)
    else:
        vis2 = verts_cam2[:, 2] > 0
        vis4 = verts_cam4[:, 2] > 0

    f1c2, f1c4, f1 = fig1(img2, img4, kp2, kp4, fmt2, fmt4, conf_thresh)
    f2c2, f2c4 = fig2(img2, img4, verts_cam2, verts_cam4, faces, vis2, vis4)
    f3 = fig3(img2, img4, verts_cam2, verts_cam4, faces, vis2, vis4, n_sample=n_sample)

    cv2.imwrite(str(output_dir / "fig1_sparse_cam2.jpg"), f1c2, [cv2.IMWRITE_JPEG_QUALITY, 92])
    cv2.imwrite(str(output_dir / "fig1_sparse_cam4.jpg"), f1c4, [cv2.IMWRITE_JPEG_QUALITY, 92])
    cv2.imwrite(str(output_dir / "fig1_sparse_combined.jpg"), f1, [cv2.IMWRITE_JPEG_QUALITY, 92])
    cv2.imwrite(str(output_dir / "fig2_dense_cam2.jpg"), f2c2, [cv2.IMWRITE_JPEG_QUALITY, 92])
    cv2.imwrite(str(output_dir / "fig2_dense_cam4.jpg"), f2c4, [cv2.IMWRITE_JPEG_QUALITY, 92])
    cv2.imwrite(str(output_dir / "fig3_correspondences.jpg"), f3, [cv2.IMWRITE_JPEG_QUALITY, 92])
    print(f"Done -> {output_dir}")


def parse_args():
    p = argparse.ArgumentParser(description="Visualize sparse and dense fisheye correspondences.")
    p.add_argument("--smpl-cam2", type=Path, required=True)
    p.add_argument("--smpl-cam4", type=Path, required=True)
    p.add_argument("--rtm-cam2", type=Path, required=True)
    p.add_argument("--rtm-cam4", type=Path, required=True)
    p.add_argument("--image-cam2", type=Path, required=True)
    p.add_argument("--image-cam4", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, default=Path("outputs/correspondence_vis"))
    p.add_argument("--frame-idx", type=int, default=0)
    p.add_argument("--n-sample", type=int, default=50)
    p.add_argument("--conf-thresh", type=float, default=0.1)
    p.add_argument("--no-raycast", action="store_true")
    p.add_argument("--delta-t", type=float, nargs=3, default=None, metavar=("TX", "TY", "TZ"))
    return p.parse_args()


def main():
    args = parse_args()
    run(
        smpl_cam2_pkl=args.smpl_cam2,
        smpl_cam4_pkl=args.smpl_cam4,
        rtm_cam2_path=args.rtm_cam2,
        rtm_cam4_path=args.rtm_cam4,
        image_cam2_path=args.image_cam2,
        image_cam4_path=args.image_cam4,
        output_dir=args.output_dir,
        frame_idx=args.frame_idx,
        conf_thresh=args.conf_thresh,
        n_sample=args.n_sample,
        delta_t=np.array(args.delta_t, dtype=np.float32) if args.delta_t else None,
        use_raycast=not args.no_raycast,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
