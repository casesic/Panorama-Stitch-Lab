"""网格渲染：将 SMPL-X 网格投影到 ERP 并以法向着色或纯色半透明叠加到图像上。"""

from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np


def _erp_project_np(points_3d: np.ndarray, width: int, height: int) -> np.ndarray:
    """将 3D 点投影为 ERP 像素坐标 (u,v)（numpy 版）。"""
    x = points_3d[..., 0]
    y = points_3d[..., 1]
    z = points_3d[..., 2]
    r = np.linalg.norm(points_3d, axis=-1).clip(1e-6, None)
    lon = np.arctan2(x, z)
    lat = np.arcsin((-y / r).clip(-1 + 1e-6, 1 - 1e-6))
    u = (lon / (2.0 * np.pi) + 0.5) * width
    v = (0.5 - lat / np.pi) * height
    return np.stack([u, v], axis=-1)


def _render_mesh_numpy_shading(
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
    """按面片法向做朗伯着色、深度排序后绘制并半透明叠加，附带轮廓线。"""
    img_h, img_w = img.shape[:2]
    overlay = img.copy().astype(np.float32)
    v2d = _erp_project_np(verts, width, height)
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


def _render_mesh_simple(
    img: np.ndarray,
    verts: np.ndarray,
    faces: np.ndarray,
    width: int,
    height: int,
    alpha: float = 0.35,
    color: Tuple[int, int, int] = (60, 120, 200),
) -> np.ndarray:
    """将网格面片以单一颜色填充并半透明叠加（不做着色，跳过跨经度/出界面片）。"""
    overlay = img.copy()
    v2d = _erp_project_np(verts, width, height).astype(np.int32)
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


def render_mesh(
    img: np.ndarray,
    verts: np.ndarray,
    faces: np.ndarray,
    width: int,
    height: int,
    alpha: float = 0.6,
    use_shader: bool = True,
) -> np.ndarray:
    """网格渲染入口：use_shader 时走法向着色版，否则走纯色叠加版。"""
    if use_shader:
        return _render_mesh_numpy_shading(img, verts, faces, width, height, alpha=alpha)
    return _render_mesh_simple(img, verts, faces, width, height, alpha=alpha)
