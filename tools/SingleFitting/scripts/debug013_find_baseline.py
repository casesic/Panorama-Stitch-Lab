#!/usr/bin/env python3
"""
Estimate the cam4 translation correction from reprojection error.

Usage:
  python tools/SingleFitting/scripts/debug013_find_baseline.py ^
    --cam4-pkl tools/SingleFitting/outputs/fisheye_smplx_cam4/fisheye_cam4_s0_i0_29_frame15.pkl ^
    --rtm-cam2 tools/SingleFitting/outputs/fisheye/rtm_cam2.pkl ^
    --rtm-cam4 tools/SingleFitting/outputs/fisheye/rtm_cam4.pkl ^
    --image-cam2 data/fisheye_cam2_frames/frame0015.jpg ^
    --output-dir tools/SingleFitting/outputs/extrinsic_estimation
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from scipy.optimize import minimize

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from single_fitting.debug_fit_shared import SingleCamFisheyeFitter, forward_param_to_numpy
from single_fitting.debug_shared import DS_PARAMS, cam4_to_world, ds_project_np, load_pickle, normalize_keypoints_format

SMPLX_BODY_NAMES = [
    "Pelvis", "L_Hip", "R_Hip", "Spine1",
    "L_Knee", "R_Knee", "Spine2",
    "L_Ankle", "R_Ankle", "Spine3",
    "L_Foot", "R_Foot", "Neck",
    "L_Collar", "R_Collar", "Head",
    "L_Shoulder", "R_Shoulder",
    "L_Elbow", "R_Elbow",
    "L_Wrist", "R_Wrist",
]
BODY22 = {name: idx for idx, name in enumerate(SMPLX_BODY_NAMES)}


def get_joints(param, fitter) -> np.ndarray:
    _, _, joints_body = forward_param_to_numpy(fitter, param)
    return joints_body


def get_rtm_kp_and_conf(rtm_cache_path, frame_idx):
    det = load_pickle(Path(rtm_cache_path))
    if hasattr(det, "array_keypoints"):
        kp = det.array_keypoints[frame_idx]
        raw_fmt = det.keypoints_format if hasattr(det, "keypoints_format") else {}
    elif hasattr(det, "get_result_by_frame_index"):
        dr = det.get_result_by_frame_index(frame_idx)
        kp = dr.keypoints
        raw_fmt = dr.keypoints_format
    else:
        raise TypeError(f"RTM cache format not supported: {type(det)}")

    fmt = normalize_keypoints_format(raw_fmt)
    j_kp = kp.shape[0]
    kp_xy = np.zeros((22, 2), dtype=np.float32)
    conf = np.zeros(22, dtype=np.float32)
    for name, si in BODY22.items():
        if name in fmt and fmt[name] < j_kp:
            ki = fmt[name]
            kp_xy[si] = kp[ki, :2]
            conf[si] = float(kp[ki, 2])
    return kp_xy, conf


def _build_weights(conf_cam2, conf_cam4):
    w = np.sqrt(conf_cam2 * conf_cam4).astype(np.float64) if conf_cam4 is not None else conf_cam2.astype(np.float64)
    w[w < 0.05] = 0.0
    if w.sum() < 1e-8:
        w = np.ones(22)
    return w / (w.sum() + 1e-8)


def _loss_2d(t, j4, gt, w):
    proj_2d = ds_project_np(cam4_to_world(j4, np.asarray(t)), **DS_PARAMS)
    return float((np.linalg.norm(proj_2d - gt, axis=1) * w).sum())


def _per_joint_err(t, j4, gt):
    proj_2d = ds_project_np(cam4_to_world(j4, np.asarray(t)), **DS_PARAMS)
    return np.linalg.norm(proj_2d - gt, axis=1)


def find_optimal_baseline(
    joints_cam4: np.ndarray,
    rtm_cam2_xy: np.ndarray,
    conf_cam2: np.ndarray,
    conf_cam4: Optional[np.ndarray],
) -> dict:
    w = _build_weights(conf_cam2, conf_cam4)
    j4 = joints_cam4.astype(np.float64)
    gt = rtm_cam2_xy.astype(np.float64)

    loss_nom = _loss_2d(np.zeros(3), j4, gt, w)
    res = minimize(lambda t: _loss_2d(t, j4, gt, w), x0=np.zeros(3), method="Powell", bounds=[(-0.5, 0.5)] * 3)
    t_opt = np.asarray(res.x, dtype=np.float64)
    err_nom = _per_joint_err(np.zeros(3), j4, gt)
    err_opt = _per_joint_err(t_opt, j4, gt)
    valid = conf_cam2 > 0.05
    rmse_nom = float(np.sqrt((err_nom[valid] ** 2).mean())) if valid.any() else float("nan")
    rmse_opt = float(np.sqrt((err_opt[valid] ** 2).mean())) if valid.any() else float("nan")
    improvement = 100.0 * (rmse_nom - rmse_opt) / max(rmse_nom, 1e-8)
    dominant_axis = ["x", "y", "z"][int(np.argmax(np.abs(t_opt)))]
    return {
        "T_optimal": t_opt,
        "T_optimal_cm": t_opt * 100.0,
        "x_optimal_cm": float(np.linalg.norm(t_opt) * 100.0),
        "baseline_axis": dominant_axis,
        "loss_nom": loss_nom,
        "rmse_nom_px": rmse_nom,
        "rmse_px": rmse_opt,
        "improvement_pct": improvement,
        "per_joint_nom_px": err_nom,
        "per_joint_err_px": err_opt,
        "w": w,
    }


def _vis_projection(joints_cam4, t_before, t_after, rtm_cam2_xy, conf_cam2, image_path: Path, save_dir: Path):
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"[warn] Failed to read image: {image_path}")
        return
    h, w = img.shape[:2]
    r = max(8, h // 360)
    thick = max(2, h // 720)
    j_before = ds_project_np(cam4_to_world(joints_cam4, t_before), **DS_PARAMS)
    j_after = ds_project_np(cam4_to_world(joints_cam4, t_after), **DS_PARAMS)

    def _in_img(pt):
        return 0 <= int(pt[0]) < w and 0 <= int(pt[1]) < h

    for tag, j_proj, color_proj in [("before", j_before, (0, 0, 220)), ("after", j_after, (220, 100, 0))]:
        canvas = img.copy()
        for i in range(22):
            if conf_cam2[i] < 0.05:
                continue
            gt = (int(round(float(rtm_cam2_xy[i, 0]))), int(round(float(rtm_cam2_xy[i, 1]))))
            pr = (int(round(float(j_proj[i, 0]))), int(round(float(j_proj[i, 1]))))
            if _in_img(gt) and _in_img(pr):
                cv2.line(canvas, gt, pr, (200, 200, 200), max(1, thick - 1))
            if _in_img(gt):
                cv2.circle(canvas, gt, r, (0, 220, 0), -1)
            if _in_img(pr):
                cv2.circle(canvas, pr, r, color_proj, -1)
        out_path = save_dir / f"proj_vis_{tag}.jpg"
        cv2.imwrite(str(out_path), canvas, [cv2.IMWRITE_JPEG_QUALITY, 92])
        print(f"Visualization ({tag}) -> {out_path}")


def run(cam4_pkl: Path, output_dir: Path, rtm_cam2_path: Path, rtm_cam4_path: Optional[Path] = None, frame_idx: int = 0, image_cam2_path: Optional[Path] = None):
    from lib_dynamic_hvip.fitting.smplx_seq_forward_module import SMPLX_Seq_Forward_Module

    output_dir.mkdir(parents=True, exist_ok=True)
    cam4_param = load_pickle(cam4_pkl)
    fitter = SingleCamFisheyeFitter(gender="neutral", camera="cam4")
    if isinstance(cam4_param, SMPLX_Seq_Forward_Module.SMPLX_Param_Seq):
        cam4_param = cam4_param.get_param_at_frame(frame_idx)
    joints_cam4 = get_joints(cam4_param, fitter)

    rtm_cam2_xy, conf_cam2 = get_rtm_kp_and_conf(rtm_cam2_path, frame_idx)
    conf_cam4 = None
    if rtm_cam4_path is not None:
        _, conf_cam4 = get_rtm_kp_and_conf(rtm_cam4_path, frame_idx)

    best_result = find_optimal_baseline(joints_cam4, rtm_cam2_xy, conf_cam2, conf_cam4)
    print(f"T = {best_result['T_optimal_cm']} cm, RMSE {best_result['rmse_nom_px']:.1f}px -> {best_result['rmse_px']:.1f}px")

    output = {
        "optimal_baseline_cm": best_result["x_optimal_cm"],
        "T_optimal_cm": best_result["T_optimal_cm"].tolist(),
        "optimal_axis": best_result["baseline_axis"],
        "rmse_before_px": best_result["rmse_nom_px"],
        "rmse_after_px": best_result["rmse_px"],
        "improvement_pct": best_result["improvement_pct"],
        "T_correction": best_result["T_optimal"].tolist(),
        "per_joint": [
            {
                "name": SMPLX_BODY_NAMES[i],
                "err_before_px": float(best_result["per_joint_nom_px"][i]),
                "err_after_px": float(best_result["per_joint_err_px"][i]),
                "conf_cam2": float(conf_cam2[i]),
                "conf_cam4": float(conf_cam4[i]) if conf_cam4 is not None else 0.0,
                "weight": float(best_result["w"][i]),
            }
            for i in range(22)
            if conf_cam2[i] > 0.05
        ],
    }
    meta_path = output_dir / "baseline_estimation.json"
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"Results -> {meta_path}")

    if image_cam2_path is not None and image_cam2_path.exists():
        _vis_projection(joints_cam4, np.zeros(3), best_result["T_optimal"], rtm_cam2_xy, conf_cam2, image_cam2_path, output_dir)
    return output


def parse_args():
    p = argparse.ArgumentParser(description="Estimate cam4 baseline correction from 2D reprojection error.")
    p.add_argument("--cam4-pkl", type=Path, required=True)
    p.add_argument("--rtm-cam2", type=Path, required=True)
    p.add_argument("--rtm-cam4", type=Path, default=None)
    p.add_argument("--output-dir", type=Path, default=Path("outputs/extrinsic_estimation"))
    p.add_argument("--frame-idx", type=int, default=0)
    p.add_argument("--image-cam2", type=Path, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    run(
        cam4_pkl=args.cam4_pkl,
        output_dir=args.output_dir,
        rtm_cam2_path=args.rtm_cam2,
        rtm_cam4_path=args.rtm_cam4,
        frame_idx=args.frame_idx,
        image_cam2_path=args.image_cam2,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
