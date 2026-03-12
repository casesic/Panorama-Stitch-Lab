"""YOLO-Seg 人体分割 + RTMPose 关节检测（Step 0 / Step 1）"""

import os
import logging
import numpy as np
import cv2
from ultralytics import YOLO
from rtmlib import Body

from lib_pano.vis.visual_human import save_thumb, save_joint_overlay

# COCO↔SMPL 上半身关节对应（7 个对齐关节）
# 分别对应: [鼻子/头, 左肩, 右肩, 左肘, 右肘, 左腕, 右腕]
COCO_IDX   = [0,  5,  6,  7,  8,  9,  10]  # COCO 17关键点索引
SMPL_ALIGN = [15, 16, 17, 18, 19, 20, 21]  # SMPL 24关节对应索引

logger = logging.getLogger(__name__)


# ── Step 0: YOLO-Seg ────────────────────────────────────────────

def step0_segment_yolo(img_path: str, yolo_path: str,
                       output_dir: str) -> np.ndarray:
    """YOLO-seg 人体分割，返回 bool (H,W) mask 或 None"""
    img = cv2.imread(img_path)
    if img is None: raise FileNotFoundError(img_path)
    H, W = img.shape[:2]
    logger.info(f"  YOLO-seg: {os.path.basename(yolo_path)}")
    model   = YOLO(yolo_path)
    results = model(img, classes=[0], verbose=False)   # 0 = person
    if not results or results[0].masks is None:
        logger.warning("  ⚠ YOLO-seg 未检测到人，silhouette loss 将跳过")
        return None
    masks_raw = results[0].masks.data.cpu().numpy()    # (N, H', W')
    areas     = masks_raw.sum(axis=(1, 2))
    best_i    = int(areas.argmax())
    mask_small = masks_raw[best_i]                     # (H', W') float32 [0,1]
    mask_full  = cv2.resize(mask_small.astype(np.uint8), (W, H),
                            interpolation=cv2.INTER_NEAREST).astype(bool)
    logger.info(f"  选第 {best_i} 人 mask，有效像素 {mask_full.sum()} / {H*W}, H:{H}, W:{W}")
    # 可视化：蓝色叠加
    vis = img.copy()
    vis[mask_full] = (vis[mask_full].astype(np.float32) * 0.45 +
                      np.array([200, 80, 30], np.float32) * 0.55).astype(np.uint8)
    save_thumb(vis, os.path.join(output_dir, 'step0_segmentation.png'))
    return mask_full


# ── Step 1: RTMPose ─────────────────────────────────────────────

def step1_detect_keypoints(img_path: str, output_dir: str) -> dict:
    """RTMPose 2D 关节检测，返回 img / target_uv / valid_mask 等"""
    logger.info("Step 1: RTMPose 2D 关节检测")
    img = cv2.imread(img_path)
    if img is None: raise FileNotFoundError(img_path)
    H, W = img.shape[:2]
    body = Body(mode='balanced', to_openpose=False,
                backend='onnxruntime', device='cpu')
    # RTMPose may detect many people, select the score highest one
    all_kps, all_sc = body(img)
    if len(all_kps) == 0: raise RuntimeError("RTMPose 未检测到人")
    best_i = int(np.argmax([sc[:11].mean() for sc in all_sc]))
    kps17, sc17 = all_kps[best_i], all_sc[best_i]
    target_uv  = kps17[COCO_IDX].astype(np.float32)   # (7,2)
    valid_mask = sc17[COCO_IDX] > 0.3                  # (7,)
    logger.info(f"  选第 {best_i} 人，有效关节 {valid_mask.sum()}/7")
    save_joint_overlay(img, target_uv, valid_mask,
                       "Step1: RTMPose Detection (green)",
                       'step1_keypoints.png', output_dir, font_scale=0.5)
    return {'img': img, 'H': H, 'W': W,
            'target_uv': target_uv, 'valid_mask': valid_mask,
            'kps17': kps17, 'sc17': sc17}
