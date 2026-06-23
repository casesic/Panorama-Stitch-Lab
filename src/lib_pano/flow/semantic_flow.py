"""语义光流计算：由双视角 SMPL 顶点在 ERP 上投影生成稠密对应光流，含 bbox 估计与可见性处理。"""

from __future__ import annotations

import numpy as np


def compute_flow_bbox(pts_list, erp_h: int, erp_w: int, padding: int = 30) -> tuple[int, int, int, int]:
    """由多组投影点的并集求外接矩形并外扩 padding，限制在 ERP 范围内；无有效点则返回全图。"""
    valid_pts = []
    for pts in pts_list:
        if pts is None or len(pts) == 0:
            continue
        pts = np.asarray(pts, dtype=np.float64)
        finite_mask = np.isfinite(pts).all(axis=1)
        if np.any(finite_mask):
            valid_pts.append(pts[finite_mask])

    if not valid_pts:
        return (0, 0, erp_w, erp_h)

    all_pts = np.concatenate(valid_pts, axis=0)
    x1 = max(0, int(np.floor(all_pts[:, 0].min())) - padding)
    y1 = max(0, int(np.floor(all_pts[:, 1].min())) - padding)
    x2 = min(erp_w, int(np.ceil(all_pts[:, 0].max())) + padding)
    y2 = min(erp_h, int(np.ceil(all_pts[:, 1].max())) + padding)
    return (x1, y1, x2, y2)


def compute_flow_bbox_from_crop_bboxes(
    frame_idx: int,
    cam2_bboxes: dict[str, list[int]] | None,
    cam4_bboxes: dict[str, list[int]] | None,
    erp_h: int,
    erp_w: int,
    padding: int,
) -> tuple[int, int, int, int] | None:
    """由 cam2/cam4 该帧的裁剪框并集求外扩 bbox；无可用框或退化则返回 None。"""
    if cam2_bboxes is None and cam4_bboxes is None:
        return None

    bbox_items = []
    key = str(frame_idx)
    if cam2_bboxes is not None and key in cam2_bboxes:
        bbox_items.append(cam2_bboxes[key])
    if cam4_bboxes is not None and key in cam4_bboxes:
        bbox_items.append(cam4_bboxes[key])
    if not bbox_items:
        return None

    bboxes = np.asarray(bbox_items, dtype=np.float64).reshape(-1, 4)
    x1 = max(0, int(np.floor(bboxes[:, 0].min())) - padding)
    y1 = max(0, int(np.floor(bboxes[:, 1].min())) - padding)
    x2 = min(erp_w, int(np.ceil(bboxes[:, 2].max())) + padding)
    y2 = min(erp_h, int(np.ceil(bboxes[:, 3].max())) + padding)
    if x2 <= x1 or y2 <= y1:
        return None
    return (x1, y1, x2, y2)


def compute_valid_flow_mask(confidence_roi: np.ndarray) -> np.ndarray:
    """由置信度图生成有效光流 mask（置信度 > 0 处为有效）。"""
    return np.asarray(confidence_roi) > 0


def build_flow_from_projected_vertices(
    verts_world_a: np.ndarray,
    verts_world_b: np.ndarray,
    faces: np.ndarray,
    erp_h: int,
    erp_w: int,
    rotation_matrix: np.ndarray,
    roi_bbox: tuple[int, int, int, int],
    visible_mask_b: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """由 A/B 两组世界系顶点和面片，在 ERP 上计算 B→A 的稠密语义光流与置信度。"""
    from lib_pano.human3d.visibility import compute_semantic_flow

    return compute_semantic_flow(
        verts_world_A=verts_world_a,
        verts_world_B=verts_world_b,
        faces=faces,
        erp_h=erp_h,
        erp_w=erp_w,
        rotation_matrix=rotation_matrix,
        roi_bbox=roi_bbox,
        visible_mask_B=visible_mask_b,
    )


def compute_semantic_flow_frame(
    verts_world_a: np.ndarray,
    verts_world_b: np.ndarray,
    verts_visible_source: np.ndarray,
    faces: np.ndarray,
    erp_h: int,
    erp_w: int,
    rotation_matrix: np.ndarray,
    roi_bbox: tuple[int, int, int, int],
) -> tuple[np.ndarray, np.ndarray]:
    """单帧单方向语义光流：先求 source 顶点在 ERP 的可见性，再生成 B→A 的光流。"""
    from lib_pano.human3d.visibility import get_visible_vertices_erp

    _, visible_mask = get_visible_vertices_erp(
        verts_world=verts_visible_source,
        faces=faces,
        rotation_matrix=rotation_matrix,
        erp_h=erp_h,
        erp_w=erp_w,
        roi_bbox=roi_bbox,
    )
    return build_flow_from_projected_vertices(
        verts_world_a,
        verts_world_b,
        faces,
        erp_h,
        erp_w,
        rotation_matrix,
        roi_bbox,
        visible_mask,
    )


def compute_semantic_flow_bidirectional(
    verts_unified_c2: np.ndarray,
    verts_unified_c4: np.ndarray,
    faces: np.ndarray,
    erp_h: int,
    erp_w: int,
    rotation_matrix: np.ndarray,
    roi_bbox: tuple[int, int, int, int],
    directions: str,
) -> dict[str, tuple[np.ndarray, np.ndarray] | None]:
    """按 directions 计算 c2↔c4 双向语义光流，返回 {"c2_to_c4", "c4_to_c2"} 字典。"""
    result: dict[str, tuple[np.ndarray, np.ndarray] | None] = {
        "c2_to_c4": None,
        "c4_to_c2": None,
    }

    if directions in ("both", "c2_to_c4"):
        result["c2_to_c4"] = compute_semantic_flow_frame(
            verts_world_a=verts_unified_c4,
            verts_world_b=verts_unified_c2,
            verts_visible_source=verts_unified_c2,
            faces=faces,
            erp_h=erp_h,
            erp_w=erp_w,
            rotation_matrix=rotation_matrix,
            roi_bbox=roi_bbox,
        )

    if directions in ("both", "c4_to_c2"):
        result["c4_to_c2"] = compute_semantic_flow_frame(
            verts_world_a=verts_unified_c2,
            verts_world_b=verts_unified_c4,
            verts_visible_source=verts_unified_c4,
            faces=faces,
            erp_h=erp_h,
            erp_w=erp_w,
            rotation_matrix=rotation_matrix,
            roi_bbox=roi_bbox,
        )

    return result


def project_vertices_to_erp(
    verts_world: np.ndarray,
    rotation_matrix: np.ndarray,
    erp_h: int,
    erp_w: int,
) -> np.ndarray:
    """将世界系顶点经旋转后投影到 ERP，返回像素坐标。"""
    from lib_pano.human3d.visibility import project_verts_to_rotated_erp

    return project_verts_to_rotated_erp(verts_world, rotation_matrix, erp_h, erp_w)
