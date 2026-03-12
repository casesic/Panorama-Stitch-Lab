# SMPL 鱼眼相机人体对齐 Pipeline — 主入口 + Step 4 导出
# Step0 YOLO分割 → Step1 RTMPose检测 → Step2 刚体拟合 → Step3 VPoser优化 → Step4 导出OBJ

import sys, os, time
import logging
import numpy as np
import torch
import cv2
import config_script

import lib_pano.local_setting as ls
from lib_pano.io.params_read       import load_single_K_and_distortion
from lib_pano.human.smpl_model     import SmplFull, export_obj, crop_upper_body
from lib_pano.human.vposer         import load_vposer
from lib_pano.human.detect         import (step0_segment_yolo, step1_detect_keypoints,
                                           SMPL_ALIGN)
from lib_pano.human.rigid_fit      import step2_rigid_fit
from lib_pano.human.pose_optimize  import step3_vposer_fit
from lib_pano.vis.visual_human           import (save_thumb, rigid_fit_render_silhouette, make_comparison, step4_export,
                                            C_GREEN, C_ORANGE, C_CYAN, C_WHITE, C_GRAY)
from lib_pano.logging_config import configure_logging

# ── 配置 ────────────────────────────────────────────────────────
SMPL_PKL    = os.path.join(ls.PROJECT_ROOT, 'tools', '4D-Humans', 'data',
                            'basicmodel_neutral_lbs_10_207_0_v1.1.0.pkl')
OBJ_PATH    = os.path.join(ls.HUMAN_OBJ_DIR, 'frame_00001_rotate_0.obj')
VPOSER_DIR  = os.path.join(ls.PROJECT_ROOT, 'tools', 'V02_05')
CALIB_PATH  = ls.CAMERA_CALIB_FILE
CAMERA_NAME = 'camera6'
FISHEYE_IMG = os.path.join(ls.IMAGE_ORIGIN6_DIR, 'frame_00001_rotate.png')
OUTPUT_DIR  = os.path.join(ls.DATA_OUTPUT_DIR, 'test_human', 'pipeline')
os.makedirs(OUTPUT_DIR, exist_ok=True)

DEVICE        = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
YOLO_SEG_PATH = os.path.join(config_script.PROJECT_ROOT, 'yolo11x-seg.pt')

configure_logging(level='INFO')
logger = logging.getLogger(__name__)

# ── 主入口 ──────────────────────────────────────────────────────

def main():
    t0 = time.time()
    logger.info("=" * 62)
    logger.info("SMPL 鱼眼人体对齐 Pipeline")
    logger.info(f"输出目录: {OUTPUT_DIR}")
    logger.info("=" * 62)

    # ── 加载相机参数 ─────────────────────────────
    K, dist, xi_val, _ = load_single_K_and_distortion(CALIB_PATH, CAMERA_NAME)
    cam = dict(fx=float(K[0,0]), fy=float(K[1,1]),
               cx=float(K[0,2]), cy=float(K[1,2]),
               k1=float(dist[0]), k2=float(dist[1]),
               p1=float(dist[2]), p2=float(dist[3]),
               xi=float(xi_val))
    logger.info(f"相机 {CAMERA_NAME}: fx={cam['fx']:.0f}  xi={cam['xi']:.4f}")

    # ── 加载 SMPL PKL 和 J_regressor ────────────────
    logger.info("加载 SMPL PKL...")
    smpl  = SmplFull(SMPL_PKL).to(DEVICE)
    J_reg = smpl.J_regressor.cpu().numpy()   # (24,6890)

    # ── 加载 VPoser ───────────────────────────────
    logger.info("加载 VPoser...")
    enc, dec = load_vposer(VPOSER_DIR)

    # ── Step 0: YOLO 人体分割（silhouette loss 用）────
    logger.info("Step 0: YOLO-seg 人体分割")
    seg_mask = step0_segment_yolo(FISHEYE_IMG, YOLO_SEG_PATH, OUTPUT_DIR)

    # ── Step 1: 关节检测 ─────────────────────────
    s1 = step1_detect_keypoints(FISHEYE_IMG, OUTPUT_DIR)

    # ── Step 2: 刚体拟合 ─────────────────────────
    s2 = step2_rigid_fit(s1, J_reg, cam, OBJ_PATH, OUTPUT_DIR)
    rigid_fit_render_silhouette(s1, s2, cam, OBJ_PATH, OUTPUT_DIR)

    # ── Step 3: VPoser 优化 ──────────────────────
    s3 = step3_vposer_fit(s1, s2, smpl, enc, dec, cam,
                          DEVICE, OBJ_PATH, OUTPUT_DIR, seg_mask=seg_mask)

    # ── Step 4: 导出 OBJ ─────────────────────────
    step4_export(s1, s3, smpl, cam, OUTPUT_DIR)

    # ── 汇总对比图 ─────────────────────────────
    make_comparison(OUTPUT_DIR)

    logger.info("=" * 62)
    logger.info(f"Pipeline 完成！耗时 {time.time()-t0:.1f}s")
    logger.info("=" * 62)


if __name__ == '__main__':
    main()
