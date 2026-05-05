from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt


def load_semantic_flow_pair(frame_flow_dir: Path, flow_name: str) -> tuple[np.ndarray, np.ndarray] | None:
    frame_flow_dir = Path(frame_flow_dir)
    flow_path = frame_flow_dir / flow_name
    confidence_name = flow_name.replace(".npy", "").replace("semantic_flow", "semantic_flow_confidence") + ".npy"
    confidence_path = frame_flow_dir / confidence_name
    if not flow_path.exists() or not confidence_path.exists():
        return None
    return np.load(flow_path), np.load(confidence_path)


def build_full_flow_from_human_bbox(
    flow: np.ndarray,
    confidence: np.ndarray,
    human_bbox: list[int],
    erp_h: int,
    erp_w: int,
) -> tuple[np.ndarray, np.ndarray] | None:
    hx1, hy1, _, _ = [int(v) for v in human_bbox]
    roi_y1, roi_x1 = hy1, hx1
    roi_h, roi_w = flow.shape[:2]
    roi_y2 = roi_y1 + roi_h
    roi_x2 = roi_x1 + roi_w
    clip_y1 = max(0, roi_y1)
    clip_x1 = max(0, roi_x1)
    clip_y2 = min(erp_h, roi_y2)
    clip_x2 = min(erp_w, roi_x2)
    if clip_y1 >= clip_y2 or clip_x1 >= clip_x2:
        return None

    flow_y1 = clip_y1 - roi_y1
    flow_x1 = clip_x1 - roi_x1
    flow_y2 = flow_y1 + (clip_y2 - clip_y1)
    flow_x2 = flow_x1 + (clip_x2 - clip_x1)

    full_mask = np.zeros((erp_h, erp_w), dtype=bool)
    full_flow = np.zeros((erp_h, erp_w, 2), dtype=np.float32)
    full_mask[clip_y1:clip_y2, clip_x1:clip_x2] = confidence[flow_y1:flow_y2, flow_x1:flow_x2] > 0
    full_flow[clip_y1:clip_y2, clip_x1:clip_x2] = flow[flow_y1:flow_y2, flow_x1:flow_x2].astype(np.float32)
    return full_flow, full_mask


def propagate_flow_with_decay(
    full_flow: np.ndarray,
    full_mask: np.ndarray,
    decay_radius: float = 400.0,
    blur_kernel: tuple[int, int] = (51, 51),
) -> np.ndarray:
    erp_h, erp_w = full_mask.shape[:2]
    pad = int(decay_radius) + 1
    ys, xs = np.where(full_mask)
    if len(ys) == 0:
        return np.zeros((erp_h, erp_w, 2), dtype=np.float32)

    roi_y1_edt = max(0, int(ys.min()) - pad)
    roi_y2_edt = min(erp_h, int(ys.max()) + pad + 1)
    roi_x1_edt = max(0, int(xs.min()) - pad)
    roi_x2_edt = min(erp_w, int(xs.max()) + pad + 1)

    mask_roi = full_mask[roi_y1_edt:roi_y2_edt, roi_x1_edt:roi_x2_edt]
    flow_roi = full_flow[roi_y1_edt:roi_y2_edt, roi_x1_edt:roi_x2_edt]

    dist_roi, nearest_indices_roi = distance_transform_edt(
        ~mask_roi, return_distances=True, return_indices=True
    )
    nearest_y_roi = nearest_indices_roi[0]
    nearest_x_roi = nearest_indices_roi[1]
    propagated_flow_roi = flow_roi[nearest_y_roi, nearest_x_roi]
    weight_roi = np.clip(1.0 - dist_roi / decay_radius, 0.0, 1.0).astype(np.float32)
    flow_with_decay_roi = propagated_flow_roi * weight_roi[..., None]
    flow_with_decay_roi[..., 0] = cv2.GaussianBlur(flow_with_decay_roi[..., 0], blur_kernel, 0)
    flow_with_decay_roi[..., 1] = cv2.GaussianBlur(flow_with_decay_roi[..., 1], blur_kernel, 0)
    flow_with_decay_roi[mask_roi] = flow_roi[mask_roi]

    flow_with_decay = np.zeros((erp_h, erp_w, 2), dtype=np.float32)
    flow_with_decay[roi_y1_edt:roi_y2_edt, roi_x1_edt:roi_x2_edt] = flow_with_decay_roi
    return flow_with_decay


def warp_rgba_with_flow(
    source_rgba: np.ndarray,
    source_bbox: list[int],
    flow_with_decay: np.ndarray,
    erp_h: int,
    erp_w: int,
) -> np.ndarray | None:
    source_x1, source_y1, _, _ = [int(v) for v in source_bbox]
    src_roi_y1, src_roi_x1 = source_y1, source_x1
    src_roi_h, src_roi_w = source_rgba.shape[:2]
    src_roi_y2 = src_roi_y1 + src_roi_h
    src_roi_x2 = src_roi_x1 + src_roi_w
    src_clip_y1 = max(0, src_roi_y1)
    src_clip_x1 = max(0, src_roi_x1)
    src_clip_y2 = min(erp_h, src_roi_y2)
    src_clip_x2 = min(erp_w, src_roi_x2)
    if src_clip_y1 >= src_clip_y2 or src_clip_x1 >= src_clip_x2:
        return None

    src_y1 = src_clip_y1 - src_roi_y1
    src_x1 = src_clip_x1 - src_roi_x1
    src_y2 = src_y1 + (src_clip_y2 - src_clip_y1)
    src_x2 = src_x1 + (src_clip_x2 - src_clip_x1)

    source_rgba_full = np.zeros((erp_h, erp_w, 4), dtype=np.uint8)
    source_rgba_full[src_clip_y1:src_clip_y2, src_clip_x1:src_clip_x2] = source_rgba[src_y1:src_y2, src_x1:src_x2]

    grid_x, grid_y = np.meshgrid(np.arange(erp_w, dtype=np.float32), np.arange(erp_h, dtype=np.float32))
    map_x = np.clip(grid_x + flow_with_decay[..., 0], 0.0, float(erp_w - 1))
    map_y = np.clip(grid_y + flow_with_decay[..., 1], 0.0, float(erp_h - 1))
    warped_rgba = cv2.remap(
        source_rgba_full,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0, 0),
    )

    result_bgr = np.zeros((erp_h, erp_w, 3), dtype=np.uint8)
    warped_alpha = warped_rgba[..., 3] > 0
    result_bgr[warped_alpha] = warped_rgba[..., :3][warped_alpha]
    return result_bgr


def composite_direction(
    frame_idx: int,
    source_camera: str,
    target_camera: str,
    flow_name: str,
    seg_dir: Path,
    flow_dir: Path,
    source_bboxes: dict[str, list[int]],
    target_bboxes: dict[str, list[int]],
    human_bbox: list[int],
    erp_h: int,
    erp_w: int,
) -> np.ndarray | None:
    bbox_key = str(frame_idx)
    source_bbox = source_bboxes.get(bbox_key)
    target_bbox = target_bboxes.get(bbox_key)
    if source_bbox is None or target_bbox is None:
        print(f"Warning: missing bbox, skipping frame {frame_idx:05d} for {source_camera}_to_{target_camera}")
        return None

    frame_flow_dir = Path(flow_dir) / f"frame_{frame_idx:05d}"
    flow_pair = load_semantic_flow_pair(frame_flow_dir, flow_name)
    rgba_path = Path(seg_dir) / source_camera / "rgba" / f"{frame_idx:05d}.png"

    if flow_pair is None:
        print(f"Warning: missing flow files, skipping frame {frame_idx:05d} for {source_camera}_to_{target_camera}")
        return None
    flow, confidence = flow_pair
    source_rgba = cv2.imread(str(rgba_path), cv2.IMREAD_UNCHANGED)

    if source_rgba is None:
        print(f"Warning: source rgba not found, skipping {source_camera} frame {frame_idx:05d}")
        return None
    if source_rgba.ndim != 3 or source_rgba.shape[2] != 4:
        print(f"Warning: source rgba is not BGRA, skipping {source_camera} frame {frame_idx:05d}")
        return None
    flow_h, flow_w = flow.shape[:2]
    conf_h, conf_w = confidence.shape[:2]
    if flow_h <= 0 or flow_w <= 0:
        print(f"Warning: empty flow, skipping frame {frame_idx:05d} for {source_camera}_to_{target_camera}")
        return None

    if (conf_h, conf_w) != (flow_h, flow_w):
        print(
            f"Warning: confidence shape mismatch with flow, skipping frame {frame_idx:05d}: "
            f"flow={flow.shape}, confidence={confidence.shape}"
        )
        return None

    source_x1, source_y1, source_x2, source_y2 = [int(v) for v in source_bbox]
    bbox_w = source_x2 - source_x1
    bbox_h = source_y2 - source_y1
    rgba_h, rgba_w = source_rgba.shape[:2]

    if (rgba_w, rgba_h) != (bbox_w, bbox_h):
        print(
            f"Warning: source rgba size mismatch with bbox for {source_camera} frame {frame_idx:05d}: "
            f"rgba=({rgba_w}, {rgba_h}), bbox_size=({bbox_w}, {bbox_h}), bbox={source_bbox}"
        )
        return None

    full_flow_pair = build_full_flow_from_human_bbox(flow, confidence, human_bbox, erp_h, erp_w)
    if full_flow_pair is None:
        print(f"Warning: ROI out of image bounds, skipping frame {frame_idx:05d} for {source_camera}_to_{target_camera}")
        return None
    full_flow, full_mask = full_flow_pair

    if not np.any(full_mask):
        return np.zeros((erp_h, erp_w, 3), dtype=np.uint8)

    flow_with_decay = propagate_flow_with_decay(full_flow, full_mask, decay_radius=400.0, blur_kernel=(51, 51))
    result_bgr = warp_rgba_with_flow(source_rgba, source_bbox, flow_with_decay, erp_h, erp_w)
    if result_bgr is None:
        print(f"Warning: source RGBA ROI out of image bounds, skipping frame {frame_idx:05d} for {source_camera}_to_{target_camera}")
        return None
    return result_bgr
