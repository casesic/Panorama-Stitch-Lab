
#!/usr/bin/env python3
"""
Apply a binary human mask to a fisheye image and fill the background with black.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

VIDEO_PATH = Path(r"Circular_Video\00_send\20260311_riding\VID_20181003_034909_00_982.1.mp4")
MASK_DIR = Path(r"data_process\Output\20260311_riding\segmentation\cam4\mask")
OUTPUT_DIR = Path(r"data_process\Output\20260311_riding\segmentation\cam4\seg")
INDEX_START = 512
NUM_FRAMES = 18
MASK_FILENAME_PATTERN = "mask_frame_{:05d}.png"
OUTPUT_FILENAME_PATTERN = "segmented_frame_{:05d}.jpg"

def apply_human_mask(image: np.ndarray, mask_path, output_path):
    mask_path = Path(mask_path)
    output_path = Path(output_path)

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
    cap = cv2.VideoCapture(str(VIDEO_PATH))
    cap.set(cv2.CAP_PROP_POS_FRAMES, INDEX_START)

    processed_count = 0
    for i in range(NUM_FRAMES):
        ok, frame = cap.read()
        if not ok or frame is None:
            print(f"[warn] Failed to read video frame: {INDEX_START + i}")
            break

        frame_index = INDEX_START + i
        mask_path = MASK_DIR / MASK_FILENAME_PATTERN.format(frame_index)
        output_path = OUTPUT_DIR / OUTPUT_FILENAME_PATTERN.format(frame_index)

        if not mask_path.exists():
            print(f"[warn] Missing mask: {mask_path}")
            continue

        pixel_count, pixel_ratio = apply_human_mask(frame, mask_path, output_path)
        processed_count += 1
        print(
            f"[{i + 1}/{NUM_FRAMES}] saved {output_path}  "
            f"pixels={pixel_count}  ratio={pixel_ratio:.6f} ({pixel_ratio * 100:.2f}%)"
        )

    cap.release()
    print(f"Processed {processed_count} frame(s) in total.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
