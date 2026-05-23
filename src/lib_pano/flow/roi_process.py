import cv2
import numpy as np
from typing import Optional, Tuple


def _ensure_binary_mask(mask: np.ndarray) -> np.ndarray:
    return (mask > 0).astype(np.uint8) * 255


def soft_overlap_from_masks(
    mask_1: np.ndarray,
    mask_2: np.ndarray,
    extend_y: int = 0,
) -> np.ndarray:
    mask_1 = _ensure_binary_mask(mask_1)
    mask_2 = _ensure_binary_mask(mask_2)

    overlap = cv2.bitwise_and(mask_1, mask_2)

    overlap_rows = np.any(overlap > 0, axis=1)
    if not np.any(overlap_rows):
        return np.zeros_like(overlap)

    row_min = np.argmax(overlap_rows)
    row_max = len(overlap_rows) - np.argmax(overlap_rows[::-1]) - 1

    row_min_ext = max(0, row_min - extend_y)
    row_max_ext = min(mask_1.shape[0] - 1, row_max + extend_y)

    result = np.zeros_like(overlap)
    result[row_min_ext:row_max_ext + 1, :] = 1

    return result


def extract_overlap_roi(
    image_1: np.ndarray,
    image_2: np.ndarray,
    overlap_mask: np.ndarray
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[Tuple[int, int, int, int]]]:
    y_indices, x_indices = np.where(overlap_mask > 0)
    if len(y_indices) == 0:
        return None, None, None

    y_min, y_max = np.min(y_indices), np.max(y_indices)
    x_min, x_max = np.min(x_indices), np.max(x_indices)
    y_max_exclusive = y_max + 1
    x_max_exclusive = x_max + 1

    crop_1 = image_1[y_min:y_max_exclusive, x_min:x_max_exclusive].copy()
    crop_2 = image_2[y_min:y_max_exclusive, x_min:x_max_exclusive].copy()

    roi_bbox = (x_min, y_min, x_max_exclusive, y_max_exclusive)

    return crop_1, crop_2, roi_bbox


def crop_mask_by_bbox(mask: np.ndarray, roi_bbox: Tuple[int, int, int, int]) -> np.ndarray:
    x_min, y_min, x_max, y_max = roi_bbox
    return mask[y_min:y_max, x_min:x_max].copy()


def crop_bbox_dense_semantic_to_roi(semantic_prior, roi_bbox):
    if semantic_prior is None:
        return None, None

    flow = semantic_prior["flow"]
    conf = semantic_prior["confidence"]
    sx1, sy1, sx2, sy2 = semantic_prior["bbox"]
    rx1, ry1, rx2, ry2 = roi_bbox

    ix1 = max(sx1, rx1)
    iy1 = max(sy1, ry1)
    ix2 = min(sx2, rx2)
    iy2 = min(sy2, ry2)

    roi_h = ry2 - ry1
    roi_w = rx2 - rx1

    semantic_flow_roi = np.zeros((roi_h, roi_w, 2), dtype=np.float32)
    semantic_conf_roi = np.zeros((roi_h, roi_w), dtype=np.float32)

    if ix2 <= ix1 or iy2 <= iy1:
        return semantic_flow_roi, semantic_conf_roi

    src_x1 = ix1 - sx1
    src_y1 = iy1 - sy1
    src_x2 = ix2 - sx1
    src_y2 = iy2 - sy1

    dst_x1 = ix1 - rx1
    dst_y1 = iy1 - ry1
    dst_x2 = ix2 - rx1
    dst_y2 = iy2 - ry1

    semantic_flow_roi[dst_y1:dst_y2, dst_x1:dst_x2] = flow[src_y1:src_y2, src_x1:src_x2]
    semantic_conf_roi[dst_y1:dst_y2, dst_x1:dst_x2] = conf[src_y1:src_y2, src_x1:src_x2]

    return semantic_flow_roi, semantic_conf_roi


def build_semantic_weight_map(
    semantic_confidence_roi,
    semantic_weight_scale: float = 1.0,
    semantic_conf_thresh: float = 0.05,
):
    if semantic_confidence_roi is None:
        return None

    conf = np.clip(semantic_confidence_roi.astype(np.float32) / 2.0, 0.0, 1.0)
    conf[conf < semantic_conf_thresh] = 0.0
    return (conf * float(semantic_weight_scale)).astype(np.float32)


def combine_flow_with_semantic_prior(
    raft_flow_roi,
    raft_weight_map,
    semantic_flow_roi=None,
    semantic_weight_map=None,
):
    if semantic_flow_roi is None or semantic_weight_map is None:
        return raft_flow_roi, raft_weight_map


    w_raft = raft_weight_map.astype(np.float32)
    w_sem = semantic_weight_map.astype(np.float32)

    denom = w_raft + w_sem
    valid = denom > 1e-6

    combined_flow = raft_flow_roi.astype(np.float32).copy()
    combined_flow[valid] = (
        raft_flow_roi[valid] * w_raft[valid, None]
        + semantic_flow_roi[valid] * w_sem[valid, None]
    ) / denom[valid, None]

    combined_weight = np.maximum(w_raft, w_sem).astype(np.float32)

    return combined_flow, combined_weight


def compute_landed_source_valid_mask(
    flow_roi: np.ndarray,
    source_valid_roi: np.ndarray,
) -> np.ndarray:
    h, w = flow_roi.shape[:2]
    grid_y, grid_x = np.mgrid[0:h, 0:w].astype(np.float32)
    map_x = grid_x + flow_roi[..., 0].astype(np.float32)
    map_y = grid_y + flow_roi[..., 1].astype(np.float32)

    source_valid_at_landed = cv2.remap(
        _ensure_binary_mask(source_valid_roi),
        map_x,
        map_y,
        interpolation=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    return _ensure_binary_mask(source_valid_at_landed)


def compute_photometric_mask(
    crop_source: np.ndarray,
    crop_target: np.ndarray,
    flow_roi: np.ndarray,
    threshold: int = 30,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    h, w = flow_roi.shape[:2]
    grid_y, grid_x = np.mgrid[0:h, 0:w].astype(np.float32)
    map_x = grid_x + flow_roi[..., 0].astype(np.float32)
    map_y = grid_y + flow_roi[..., 1].astype(np.float32)

    warped_source = cv2.remap(
        crop_source,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    err = np.mean(
        np.abs(warped_source.astype(np.float32) - crop_target.astype(np.float32)),
        axis=2,
    )
    photometric_mask = (err < threshold).astype(np.uint8) * 255
    return photometric_mask, warped_source, err


def build_flow_seed_mask(
    target_valid_roi: np.ndarray,
    source_valid_at_landed: np.ndarray,
    raft_valid_mask: Optional[np.ndarray] = None,
    photometric_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    flow_seed_mask = (target_valid_roi > 0) & (source_valid_at_landed > 0)

    if raft_valid_mask is not None:
        flow_seed_mask &= raft_valid_mask > 0

    if photometric_mask is not None:
        flow_seed_mask &= photometric_mask > 0

    return flow_seed_mask.astype(np.uint8) * 255


def build_flow_weight_map(
    target_valid_roi: np.ndarray,
    source_valid_at_landed: np.ndarray,
    raft_valid_mask: Optional[np.ndarray] = None,
    photometric_error: Optional[np.ndarray] = None,
    photometric_mask: Optional[np.ndarray] = None,
    support_roi_roi: Optional[np.ndarray] = None,
    photo_sigma: float = 60.0,
    landed_invalid_multiplier: float = 0.10,
    raft_invalid_multiplier: float = 0.35,
    min_weight: float = 0.0,
) -> np.ndarray:
    weight = (target_valid_roi > 0).astype(np.float32)

    if support_roi_roi is not None:
        weight *= (support_roi_roi > 0).astype(np.float32)

    weight *= np.where(source_valid_at_landed > 0, 1.0, landed_invalid_multiplier).astype(np.float32)

    if raft_valid_mask is not None:
        weight *= np.where(raft_valid_mask > 0, 1.0, raft_invalid_multiplier).astype(np.float32)

    if photometric_error is not None:
        sigma = max(float(photo_sigma), 1e-6)
        photo_weight = np.exp(-photometric_error.astype(np.float32) / sigma)
        weight *= photo_weight.astype(np.float32)
    elif photometric_mask is not None:
        weight *= np.where(photometric_mask > 0, 1.0, 0.2).astype(np.float32)

    weight = np.clip(weight, 0.0, 1.0)
    if min_weight > 0:
        weight[weight < min_weight] = 0.0
    return weight.astype(np.float32)


def blend_priority(
    image_source: np.ndarray,
    image_target: np.ndarray,
    threshold: int = 10,
    erode_kernel_size: int = 0,
    source_valid_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    使用灰度值判断, 避免单通道为0的问题
    :erode_kernel_size: 腐蚀Mask的核大小, 用于去除边缘插值产生的杂色. 默认为0.
    :source_valid_mask: 若提供则直接使用几何mask, 否则从灰度值推断.
    """
    image_to_process = image_source

    if source_valid_mask is not None:
        mask = (source_valid_mask > 0).astype(np.uint8) * 255
    else:
        gray = cv2.cvtColor(image_to_process, cv2.COLOR_BGR2GRAY)
        mask = (gray > threshold).astype(np.uint8) * 255

    if erode_kernel_size > 0:
        if source_valid_mask is not None:
            kernel = np.ones((1, erode_kernel_size), np.uint8)
        else:
            kernel = np.ones((erode_kernel_size, erode_kernel_size), np.uint8)
        mask = cv2.erode(mask, kernel, iterations=1)

    result = image_to_process.copy()
    result[mask == 0] = image_target[mask == 0]

    return result
