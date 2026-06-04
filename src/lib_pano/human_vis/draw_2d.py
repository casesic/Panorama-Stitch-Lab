from __future__ import annotations

from typing import Optional, Sequence, Tuple

import cv2
import numpy as np


def draw_pts(
    img: np.ndarray,
    pts2d: np.ndarray,
    conf: Optional[np.ndarray] = None,
    thresh: float = 0.1,
    r: int = 5,
    color: Tuple[int, int, int] = (0, 220, 0),
) -> np.ndarray:
    h, w = img.shape[:2]
    for i, (u, v) in enumerate(pts2d):
        if conf is not None and float(conf[i]) < thresh:
            continue
        pu, pv = int(round(float(u))), int(round(float(v)))
        if 0 <= pu < w and 0 <= pv < h:
            cv2.circle(img, (pu, pv), r, color, -1)
    return img


def draw_skel(
    img: np.ndarray,
    pts2d: np.ndarray,
    pairs: Sequence[Tuple[int, int]],
    conf: Optional[np.ndarray] = None,
    thresh: float = 0.1,
    color: Tuple[int, int, int] = (0, 180, 0),
    thick: int = 2,
) -> np.ndarray:
    h, w = img.shape[:2]
    for i, j in pairs:
        if i >= len(pts2d) or j >= len(pts2d):
            continue
        if conf is not None and (float(conf[i]) < thresh or float(conf[j]) < thresh):
            continue
        u0, v0 = int(round(float(pts2d[i, 0]))), int(round(float(pts2d[i, 1])))
        u1, v1 = int(round(float(pts2d[j, 0]))), int(round(float(pts2d[j, 1])))
        if 0 <= u0 < w and 0 <= v0 < h and 0 <= u1 < w and 0 <= v1 < h:
            cv2.line(img, (u0, v0), (u1, v1), color, thick)
    return img
