import cv2
import numpy as np


def repair_seam(
    image: np.ndarray,
    source_mask: np.ndarray,
    target_mask: np.ndarray,
    seam_band_radius: int = 15,
    inpaint_radius: int = 3,
) -> np.ndarray:
    source_mask = np.where(source_mask > 0, 255, 0).astype(np.uint8)
    target_mask = np.where(target_mask > 0, 255, 0).astype(np.uint8)

    boundary = cv2.morphologyEx(source_mask, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8))
    seam_band = cv2.dilate(boundary, np.ones((seam_band_radius, seam_band_radius), np.uint8))

    overlap = cv2.bitwise_and(source_mask, target_mask)
    seam_band = cv2.bitwise_and(seam_band, overlap)

    problem = cv2.bitwise_and(seam_band, cv2.bitwise_not(overlap))

    if cv2.countNonZero(problem) == 0:
        return image

    inpaint_mask = cv2.dilate(problem, np.ones((3, 3), np.uint8), iterations=2)
    return cv2.inpaint(image, inpaint_mask, inpaint_radius, cv2.INPAINT_TELEA)
