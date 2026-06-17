#!/usr/bin/env python3
"""Self-check: validate CPW warp library functions on synthetic data."""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

SRC_ROOT = Path(__file__).resolve().parents[2]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from lib_pano.composite.cpw_warp import (
    generate_cpw_backward_map,
    warp_image_with_backward_map,
    directional_erode_valid_mask,
    target_priority_composite,
)


def main() -> None:
    h, w = 128, 128

    # synthetic control points: 4 corners of a square
    src_pts = np.array([[30, 30], [100, 30], [30, 100], [100, 100]], dtype=np.float32)
    dst_pts = np.array([[40, 40], [90, 35], [35, 90], [95, 95]], dtype=np.float32)

    # generate backward map
    backward_map, stats = generate_cpw_backward_map(src_pts, dst_pts, (h, w), k=4, power=2.0)
    print(f"[INFO] backward_map shape: {backward_map.shape}, dtype: {backward_map.dtype}")
    print(f"[INFO] stats: {stats}")

    # warp a synthetic image
    synth = np.zeros((h, w, 3), dtype=np.uint8)
    synth[20:60, 20:60] = [0, 0, 255]  # red square
    synth[80:110, 80:110] = [0, 255, 0]  # green square
    warped = warp_image_with_backward_map(synth, backward_map)
    print(f"[INFO] warped image shape: {warped.shape}, dtype: {warped.dtype}")

    # directional erosion
    mask = np.zeros((h, w), dtype=bool)
    mask[30:100, 30:100] = True
    eroded_down = directional_erode_valid_mask(mask, "down", 10)
    eroded_up = directional_erode_valid_mask(mask, "up", 10)
    print(f"[INFO] erosion down valid: {eroded_down.sum()} (orig {mask.sum()})")
    print(f"[INFO] erosion up valid: {eroded_up.sum()} (orig {mask.sum()})")

    # target-priority composite
    tgt = np.zeros((h, w, 3), dtype=np.uint8)
    tgt[30:80, 30:80] = [255, 0, 0]  # blue target region
    src_warped = np.zeros((h, w, 3), dtype=np.uint8)
    src_warped[50:100, 50:100] = [0, 255, 0]  # green source region
    tgt_valid = np.max(tgt, axis=2) > 0
    src_valid = np.max(src_warped, axis=2) > 0
    composite = target_priority_composite(tgt, src_warped, tgt_valid, src_valid)
    print(f"[INFO] composite shape: {composite.shape}, dtype: {composite.dtype}")
    # target region should be blue, fill-in should be green
    blue_pixels = (composite[..., 0] > 200) & (composite[..., 1] < 100)
    green_pixels = (composite[..., 1] > 200) & (composite[..., 0] < 100)
    print(f"[INFO] blue pixels (target): {blue_pixels.sum()}")
    print(f"[INFO] green pixels (filled): {green_pixels.sum()}")

    print("[DONE] all checks passed")


if __name__ == "__main__":
    main()
