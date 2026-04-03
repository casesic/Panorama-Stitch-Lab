
#!/usr/bin/env python3
"""
Apply a binary human mask to a fisheye image and fill the background with black.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def apply_human_mask(img_path, mask_path, output_path):
    img_path = Path(img_path)
    mask_path = Path(mask_path)
    output_path = Path(output_path)

    image = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Failed to read image: {img_path}")

    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Failed to read mask: {mask_path}")

    if image.shape[:2] != mask.shape[:2]:
        raise ValueError(
            f"Image and mask size mismatch: image={image.shape[:2]}, mask={mask.shape[:2]}"
        )

    human_mask = mask > 127
    segmented = np.zeros_like(image)
    segmented[human_mask] = image[human_mask]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    success = cv2.imwrite(str(output_path), segmented, [cv2.IMWRITE_JPEG_QUALITY, 95])
    if not success:
        raise IOError(f"Failed to write output image: {output_path}")

    pixel_count = int(np.count_nonzero(human_mask))
    pixel_ratio = pixel_count / float(human_mask.size)
    return pixel_count, pixel_ratio


def main():
    img_path = Path(r"data_process\Input\20260311_riding\camera4\frame_00515.png")
    mask_path = Path(r"tools\SingleFitting\outputs\segmentation\cam4\mask\mask_frame_00515.png")
    output_path = Path(r"tools\SingleFitting\outputs\segmentation\cam4\seg\segmented_frame_00515.jpg")

    pixel_count, pixel_ratio = apply_human_mask(img_path, mask_path, output_path)
    print(f"Saved segmented image to: {output_path}")
    print(f"Mask covered pixels: {pixel_count}")
    print(f"Mask covered ratio : {pixel_ratio:.6f} ({pixel_ratio * 100:.2f}%)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
