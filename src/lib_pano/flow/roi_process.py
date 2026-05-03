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
) -> Tuple[np.ndarray, np.ndarray]:
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
    return photometric_mask, warped_source


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
