from __future__ import annotations

import cv2
import numpy as np


def draw_mask_contour(image_bgr: np.ndarray, mask: np.ndarray, color: tuple[int, int, int], label: str) -> np.ndarray:
    vis = image_bgr.copy()
    if mask.shape[:2] != vis.shape[:2]:
        mask = cv2.resize(mask, (vis.shape[1], vis.shape[0]), interpolation=cv2.INTER_NEAREST)
    contours, _ = cv2.findContours((mask > 0).astype(np.uint8) * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(vis, contours, -1, color, 2)
    cv2.putText(vis, label, (16, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(vis, label, (16, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2, cv2.LINE_AA)
    return vis


def make_fusion_vis(
    image_bgr: np.ndarray,
    original_crop: np.ndarray,
    warped_crop: np.ndarray,
    completed_crop: np.ndarray,
) -> np.ndarray:
    panels = [
        draw_mask_contour(image_bgr, original_crop, (0, 255, 255), "original SAM3"),
        draw_mask_contour(image_bgr, warped_crop, (255, 0, 255), "warped support"),
        draw_mask_contour(image_bgr, completed_crop, (0, 255, 0), "completed support"),
    ]
    return cv2.hconcat(panels)


def draw_matches_vis(
    image0_bgr: np.ndarray,
    image1_bgr: np.ndarray,
    pts0: np.ndarray,
    pts1: np.ndarray,
    confidence: np.ndarray,
    inlier_mask: np.ndarray,
    max_draw: int = 200,
) -> np.ndarray:
    if len(pts0) == 0:
        height = max(image0_bgr.shape[0], image1_bgr.shape[0])
        width = image0_bgr.shape[1] + image1_bgr.shape[1]
        canvas = np.zeros((height, width, 3), dtype=np.uint8)
        canvas[: image0_bgr.shape[0], : image0_bgr.shape[1]] = image0_bgr
        canvas[: image1_bgr.shape[0], image0_bgr.shape[1] :] = image1_bgr
        cv2.putText(canvas, "no matches", (16, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2, cv2.LINE_AA)
        return canvas

    order = np.argsort(-confidence) if confidence.size == len(pts0) else np.arange(len(pts0))
    selected = order[: min(max_draw, len(order))]
    kps0 = [cv2.KeyPoint(float(pts0[i, 0]), float(pts0[i, 1]), 1) for i in selected]
    kps1 = [cv2.KeyPoint(float(pts1[i, 0]), float(pts1[i, 1]), 1) for i in selected]
    matches = [cv2.DMatch(_queryIdx=i, _trainIdx=i, _distance=0.0) for i in range(len(selected))]
    vis = cv2.drawMatches(
        image0_bgr,
        kps0,
        image1_bgr,
        kps1,
        matches,
        None,
        matchColor=(0, 255, 0),
        singlePointColor=(180, 180, 180),
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )
    offset_x = image0_bgr.shape[1]
    for local_idx, source_idx in enumerate(selected):
        if source_idx >= len(inlier_mask) or bool(inlier_mask[source_idx]):
            continue
        p0 = tuple(np.round(kps0[local_idx].pt).astype(int))
        p1_raw = np.round(kps1[local_idx].pt).astype(int)
        p1 = (int(p1_raw[0] + offset_x), int(p1_raw[1]))
        cv2.line(vis, p0, p1, (0, 0, 255), 1, cv2.LINE_AA)
    return vis
