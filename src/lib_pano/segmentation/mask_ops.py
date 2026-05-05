from __future__ import annotations

import cv2
import numpy as np

from lib_pano.pipeline.bbox_io import clip_bbox


def place_crop_mask_on_full(mask_crop: np.ndarray, bbox: list[int], width: int, height: int) -> np.ndarray:
    clipped = clip_bbox(bbox, width, height)
    full = np.zeros((height, width), dtype=np.uint8)
    if clipped is None:
        return full
    x1, y1, x2, y2 = clipped
    target_w = x2 - x1
    target_h = y2 - y1
    if mask_crop.shape[:2] != (target_h, target_w):
        mask_crop = cv2.resize(mask_crop, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
    full[y1:y2, x1:x2] = (mask_crop > 0).astype(np.uint8) * 255
    return full


def translate_mask(mask: np.ndarray, dx: float, dy: float) -> np.ndarray:
    height, width = mask.shape[:2]
    matrix = np.array([[1.0, 0.0, dx], [0.0, 1.0, dy]], dtype=np.float32)
    return cv2.warpAffine(mask, matrix, (width, height), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def postprocess_mask(mask: np.ndarray, min_component_area: int = 0, close_kernel: int = 0) -> np.ndarray:
    binary = (mask > 0).astype(np.uint8) * 255
    if close_kernel and close_kernel > 1:
        kernel_size = int(close_kernel)
        if kernel_size % 2 == 0:
            kernel_size += 1
        kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    if min_component_area and min_component_area > 0:
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats((binary > 0).astype(np.uint8), connectivity=8)
        cleaned = np.zeros_like(binary)
        for label in range(1, num_labels):
            if int(stats[label, cv2.CC_STAT_AREA]) >= min_component_area:
                cleaned[labels == label] = 255
        binary = cleaned
    return binary


def crop_full_mask(full_mask: np.ndarray, bbox: list[int], width: int, height: int) -> np.ndarray:
    clipped = clip_bbox(bbox, width, height)
    if clipped is None:
        return np.zeros((1, 1), dtype=np.uint8)
    x1, y1, x2, y2 = clipped
    return full_mask[y1:y2, x1:x2].copy()


def fuse_dual_masks_with_translation(
    mask2_crop: np.ndarray,
    mask4_crop: np.ndarray,
    cam2_bbox: list[int],
    cam4_bbox: list[int],
    translation: dict,
    width: int,
    height: int,
    min_component_area: int = 0,
    close_kernel: int = 0,
) -> dict[str, np.ndarray]:
    full2_original = place_crop_mask_on_full(mask2_crop, cam2_bbox, width, height)
    full4_original = place_crop_mask_on_full(mask4_crop, cam4_bbox, width, height)

    if translation["valid_translation"]:
        dx = float(translation["dx_cam2_to_cam4"])
        dy = float(translation["dy_cam2_to_cam4"])
        full2_to_4 = translate_mask(full2_original, dx, dy)
        full4_to_2 = translate_mask(full4_original, -dx, -dy)
        full2_completed = cv2.bitwise_or(full2_original, full4_to_2)
        full4_completed = cv2.bitwise_or(full4_original, full2_to_4)
    else:
        full2_to_4 = np.zeros_like(full2_original)
        full4_to_2 = np.zeros_like(full4_original)
        full2_completed = full2_original.copy()
        full4_completed = full4_original.copy()

    full2_completed = postprocess_mask(full2_completed, min_component_area, close_kernel)
    full4_completed = postprocess_mask(full4_completed, min_component_area, close_kernel)
    return {
        "full2_original": full2_original,
        "full4_original": full4_original,
        "full2_to_4": full2_to_4,
        "full4_to_2": full4_to_2,
        "full2_completed": full2_completed,
        "full4_completed": full4_completed,
        "mask2_completed_crop": crop_full_mask(full2_completed, cam2_bbox, width, height),
        "mask4_completed_crop": crop_full_mask(full4_completed, cam4_bbox, width, height),
        "warped4_to_2_crop": crop_full_mask(full4_to_2, cam2_bbox, width, height),
        "warped2_to_4_crop": crop_full_mask(full2_to_4, cam4_bbox, width, height),
    }
