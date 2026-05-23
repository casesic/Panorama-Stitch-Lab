from __future__ import annotations

from typing import Optional, Sequence, Tuple

import cv2
import numpy as np

from .debug_shared import DS_PARAMS, ds_project_np


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


def render_mesh_shading(
    img: np.ndarray,
    verts_cam: np.ndarray,
    faces: np.ndarray,
    *,
    alpha: float = 0.6,
    base_color: Tuple[int, int, int] = (165, 190, 220),
    light_dir: Tuple[float, float, float] = (0.5, -1.0, 0.8),
    ambient: float = 0.35,
    ds_params=None,
) -> np.ndarray:
    img_h, img_w = img.shape[:2]
    overlay = img.copy().astype(np.float32)
    v2d = ds_project_np(verts_cam, **(DS_PARAMS if ds_params is None else ds_params))

    ld = np.array(light_dir, dtype=np.float32)
    ld = ld / (np.linalg.norm(ld) + 1e-8)

    v0 = verts_cam[faces[:, 0]]
    v1 = verts_cam[faces[:, 1]]
    v2 = verts_cam[faces[:, 2]]
    normals = np.cross(v1 - v0, v2 - v0)
    normals = normals / np.linalg.norm(normals, axis=1, keepdims=True).clip(1e-8)

    diffuse = np.abs(np.dot(normals, ld)).clip(0, 1)
    shade = ambient + (1 - ambient) * diffuse

    depths = (
        np.linalg.norm(v0, axis=1)
        + np.linalg.norm(v1, axis=1)
        + np.linalg.norm(v2, axis=1)
    ) / 3.0
    order = np.argsort(-depths)

    bc = np.array(base_color, dtype=np.float32)
    mask = np.zeros((img_h, img_w), dtype=np.uint8)

    for fi in order:
        face = faces[fi]
        pts = v2d[face].astype(np.int32)
        if pts[:, 0].max() < 0 or pts[:, 0].min() >= img_w:
            continue
        if pts[:, 1].max() < 0 or pts[:, 1].min() >= img_h:
            continue
        color = (bc * float(shade[fi])).clip(0, 255).astype(np.int32).tolist()
        cv2.fillPoly(overlay, [pts], color)
        cv2.fillPoly(mask, [pts], 255)

    result = (alpha * overlay + (1 - alpha) * img.astype(np.float32)).clip(0, 255).astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(result, contours, -1, (220, 230, 255), 1)
    return result
