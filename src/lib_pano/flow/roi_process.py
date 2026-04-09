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
