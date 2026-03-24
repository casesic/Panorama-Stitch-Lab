# -*- coding: utf-8 -*-

import os
import argparse
import numpy as np
import cv2
from typing import Optional, Tuple


def load_matches(npz_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    data = np.load(npz_path, allow_pickle=True)
    kpts0 = data['keypoints0']
    kpts1 = data['keypoints1']
    matches = data['matches']
    conf = data['match_confidence'] if 'match_confidence' in data else np.zeros_like(matches, dtype=float)
    valid_mask = matches > -1
    return kpts0, kpts1, matches, conf, valid_mask


def load_mask(mask_path: Optional[str]):
    m = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    return (m > 0).astype(np.uint8)


def coords_to_int_xy(pts: np.ndarray, img_w: int, img_h: int):
    # pts shape (N,2) float. Clip to image bounds and to integer indices
    xs = np.round(pts[:, 0]).astype(int)
    ys = np.round(pts[:, 1]).astype(int)
    # if xs < 0: xs = 0 ; if xs >= img_w: xs = img_w -1 
    xs = np.clip(xs, 0, img_w - 1)
    ys = np.clip(ys, 0, img_h - 1)
    return xs, ys


def _filter_mask_on_keypoints(mask: np.ndarray, kpts: np.ndarray, valid_mask: np.ndarray):
    """将 `valid_mask` 中对应的 keypoints 按 `mask` 过滤（同索引空间）。
    """
    h, w = mask.shape
    idx = np.where(valid_mask)[0]
    if idx.size > 0:
        pts = kpts[idx]
        xs, ys = coords_to_int_xy(pts, w, h)
        inside = mask[ys, xs].astype(bool)
        valid_mask[idx[~inside]] = False
    return valid_mask


def _filter_mask_on_matched(mask: np.ndarray, kpts1: np.ndarray, matches: np.ndarray, valid_mask: np.ndarray):
    """按 matches 映射到第二张图的关键点，然后按 mask 过滤。
    """
    h, w = mask.shape
    idx = np.where(valid_mask)[0]
    if idx.size > 0:
        pts = kpts1[matches[idx].astype(int)]
        xs, ys = coords_to_int_xy(pts, w, h)
        inside = mask[ys, xs].astype(bool)
        valid_mask[idx[~inside]] = False
    return valid_mask


def filter_matches_by_masks(npz_path: str,
                            mask0_path: Optional[str],
                            mask1_path: Optional[str],
                            out_path: Optional[str] = None,
                            conf_thresh: float = 0.0,
                            ):
    """过滤并保存结果。
    """
    kpts0, kpts1, matches, conf, valid_mask = load_matches(npz_path)
    print(f"Loaded {len(matches)} matches from {npz_path}")

    mask0 = load_mask(mask0_path) if mask0_path else None
    mask1 = load_mask(mask1_path) if mask1_path else None

    # 根据置信度剔除
    if conf is not None:
        valid_mask = np.logical_and(valid_mask, conf >= conf_thresh)

    # 如果提供了 mask，则检查对应点是否在 mask 内
    if mask0 is not None:
        valid_mask = _filter_mask_on_keypoints(mask0, kpts0, valid_mask)

    if mask1 is not None:
        valid_mask = _filter_mask_on_matched(mask1, kpts1, matches, valid_mask)

    # 生成输出
    matches_filtered = matches.copy()
    matches_filtered[~valid_mask] = -1
    print(f"Filtered matches count: {np.sum(valid_mask)} / {len(matches)}")
    
    np.savez(out_path, keypoints0=kpts0, keypoints1=kpts1,
                 matches=matches_filtered, match_confidence=conf)

    return out_path

def feature_filter(npz_path: str, mask_paths: list[str], out_path: str, image_paths: list[str], vis_out: str):
    
    Conf = 0.0
    Prune_unmatched = False
    out = filter_matches_by_masks(npz_path, mask_paths[0], mask_paths[1], out_path, Conf)
    print('Saved filtered matches to', out)
    
    Max_draw = 200
    vis_path = visualize_filtered(out, image_paths[0], image_paths[1], vis_out, max_draw=Max_draw)
    print('Saved visualization to', vis_path)
    



def visualize_filtered(npz_path: str, img0_path: str, img1_path: str, out_img_path: str, max_draw: int = 2000):
    """可视化并保存过滤后的匹配结果。
     - 标准 SuperGlue 输出含 `keypoints0`,`keypoints1`,`matches`（会使用 matches > -1 的匹配）

    返回保存的图片路径。
    """
    # 读取图像
    img0 = cv2.imread(img0_path)
    img1 = cv2.imread(img1_path)

    kpts0_arr, kpts1_arr, matches_arr, conf, valid_mask = load_matches(npz_path)
    valid_idx = np.where(valid_mask)[0]
    # 创建全部 keypoints（drawMatches 期望完整 kp 列表）
    kps0 = [cv2.KeyPoint(float(x), float(y), 1) for x, y in kpts0_arr]
    kps1 = [cv2.KeyPoint(float(x), float(y), 1) for x, y in kpts1_arr]
    matches = [cv2.DMatch(int(i), int(matches_arr[i]), 0) for i in valid_idx]

    # 限制绘制数量
    print(len(matches))
    if len(matches) > max_draw:
        import random
        random.seed(0)
        sel = random.sample(range(len(matches)), max_draw)
        matches = [matches[i] for i in sel]

    # 绘制并保存
    vis = cv2.drawMatches(img0, kps0, img1, kps1, matches, None, flags=2)
    cv2.imwrite(out_img_path, vis)
    return out_img_path




