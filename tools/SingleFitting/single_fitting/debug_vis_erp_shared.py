from __future__ import annotations

from typing import Optional, Sequence, Tuple

import cv2
import numpy as np

from .debug_shared import erp_project_np


COCO_SKELETON_PAIRS = [
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 6),
    (5, 7), (7, 9),
    (6, 8), (8, 10),
    (5, 11), (6, 12),
    (11, 12),
    (11, 13), (13, 15),
    (12, 14), (14, 16),
]


def draw_keypoints(
    img: np.ndarray,
    kps: np.ndarray,
    conf_thresh: float = 0.1,
    radius: int = 6,
    color: Tuple[int, int, int] = (0, 220, 0),
    pairs: Optional[Sequence[Tuple[int, int]]] = None,
    line_color: Tuple[int, int, int] = (0, 180, 0),
    line_thickness: int = 2,
) -> np.ndarray:
    h, w = img.shape[:2]
    for u, v, c in kps:
        if c < conf_thresh:
            continue
        pu, pv = int(round(float(u))), int(round(float(v)))
        if 0 <= pu < w and 0 <= pv < h:
            cv2.circle(img, (pu, pv), radius, color, -1)

    if pairs is not None:
        for i, j in pairs:
            if i >= len(kps) or j >= len(kps):
                continue
            if kps[i, 2] < conf_thresh or kps[j, 2] < conf_thresh:
                continue
            u0, v0 = int(round(float(kps[i, 0]))), int(round(float(kps[i, 1])))
            u1, v1 = int(round(float(kps[j, 0]))), int(round(float(kps[j, 1])))
            if 0 <= u0 < w and 0 <= v0 < h and 0 <= u1 < w and 0 <= v1 < h:
                cv2.line(img, (u0, v0), (u1, v1), line_color, line_thickness)
    return img


def render_mesh_simple(
    img: np.ndarray,
    verts: np.ndarray,
    faces: np.ndarray,
    width: int,
    height: int,
    alpha: float = 0.35,
    color: Tuple[int, int, int] = (60, 120, 200),
) -> np.ndarray:
    overlay = img.copy()
    v2d = erp_project_np(verts, width, height).astype(np.int32)
    img_h, img_w = img.shape[:2]
    for face in faces:
        pts = v2d[face]
        if pts[:, 0].max() - pts[:, 0].min() > img_w // 2:
            continue
        if pts[:, 0].max() < 0 or pts[:, 0].min() >= img_w:
            continue
        if pts[:, 1].max() < 0 or pts[:, 1].min() >= img_h:
            continue
        cv2.fillPoly(overlay, [pts], color)
    return cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)


def render_mesh_numpy_shading(
    img: np.ndarray,
    verts: np.ndarray,
    faces: np.ndarray,
    width: int,
    height: int,
    alpha: float = 0.6,
    base_color: Tuple[int, int, int] = (165, 190, 220),
    light_dir: Tuple[float, float, float] = (0.5, -1.0, 0.8),
    ambient: float = 0.35,
) -> np.ndarray:
    img_h, img_w = img.shape[:2]
    overlay = img.copy().astype(np.float32)
    v2d = erp_project_np(verts, width, height)
    ld = np.array(light_dir, dtype=np.float32)
    ld = ld / (np.linalg.norm(ld) + 1e-8)

    v0 = verts[faces[:, 0]]
    v1 = verts[faces[:, 1]]
    v2 = verts[faces[:, 2]]
    normals = np.cross(v1 - v0, v2 - v0)
    normals = normals / np.linalg.norm(normals, axis=1, keepdims=True).clip(1e-8)
    shade = ambient + (1 - ambient) * np.abs(normals @ ld).clip(0, 1)
    depths = (np.linalg.norm(v0, axis=1) + np.linalg.norm(v1, axis=1) + np.linalg.norm(v2, axis=1)) / 3.0
    order = np.argsort(-depths)
    bc = np.array(base_color, dtype=np.float32)
    mask = np.zeros((img_h, img_w), dtype=np.uint8)

    for fi in order:
        face = faces[fi]
        pts = v2d[face].astype(np.int32)
        if pts[:, 0].max() - pts[:, 0].min() > img_w // 2:
            continue
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


def render_mesh(
    img: np.ndarray,
    verts: np.ndarray,
    faces: np.ndarray,
    width: int,
    height: int,
    alpha: float = 0.6,
    use_shader: bool = True,
) -> np.ndarray:
    if use_shader:
        return render_mesh_numpy_shading(img, verts, faces, width, height, alpha=alpha)
    return render_mesh_simple(img, verts, faces, width, height, alpha=alpha)
