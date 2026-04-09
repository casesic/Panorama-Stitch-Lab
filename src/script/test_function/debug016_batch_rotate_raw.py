import os
from pathlib import Path

import cv2
import numpy as np
import config_script
import lib_pano.local_setting as ls
from lib_pano.calibration.rotate_equirect import rotate_equirectangular_image

INPUT_DIR = r"data_process/Output/20260311_riding/stitch_raw"
INDEX_START = 500
NUM_FRAMES = 30
ROTATION_MATRIX = np.array([1, 0, 0, 0, 0, 1, 0, -1, 0], dtype=np.float64).reshape(3, 3)


def process_frame(frame_idx):
    frame_dir = Path(INPUT_DIR) / f"frame_{frame_idx:05d}"
    paths = {
        "cam0": frame_dir / "debug_sampled_cam_0.png",
        "cam1": frame_dir / "debug_sampled_cam_1.png",
        "mask_cam0": frame_dir / "debug_valid_mask_cam_0.png",
        "mask_cam1": frame_dir / "debug_valid_mask_cam_1.png",
    }

    if any(not path.exists() for path in paths.values()):
        print(f"[warn] Missing input files for frame {frame_idx:05d}")
        return False

    cam0 = cv2.imread(str(paths["cam0"]), cv2.IMREAD_COLOR)
    cam1 = cv2.imread(str(paths["cam1"]), cv2.IMREAD_COLOR)
    mask_cam0 = cv2.imread(str(paths["mask_cam0"]), cv2.IMREAD_GRAYSCALE)
    mask_cam1 = cv2.imread(str(paths["mask_cam1"]), cv2.IMREAD_GRAYSCALE)

    if cam0 is None or cam1 is None or mask_cam0 is None or mask_cam1 is None:
        print(f"[warn] Failed to read input files for frame {frame_idx:05d}")
        return False

    rotated_cam0 = rotate_equirectangular_image(
        input_img=cam0,
        rotation_matrix=ROTATION_MATRIX,
        interpolation=cv2.INTER_LINEAR,
        border_value=0,
    )
    rotated_cam1 = rotate_equirectangular_image(
        input_img=cam1,
        rotation_matrix=ROTATION_MATRIX,
        interpolation=cv2.INTER_LINEAR,
        border_value=0,
    )
    rotated_mask_cam0 = rotate_equirectangular_image(
        input_img=mask_cam0,
        rotation_matrix=ROTATION_MATRIX,
        interpolation=cv2.INTER_NEAREST,
        border_value=0,
    )
    rotated_mask_cam1 = rotate_equirectangular_image(
        input_img=mask_cam1,
        rotation_matrix=ROTATION_MATRIX,
        interpolation=cv2.INTER_NEAREST,
        border_value=0,
    )

    cv2.imwrite(str(frame_dir / f"rotated_cam2_frame_{frame_idx:05d}.png"), rotated_cam0)
    cv2.imwrite(str(frame_dir / f"rotated_cam4_frame_{frame_idx:05d}.png"), rotated_cam1)
    cv2.imwrite(str(frame_dir / f"rotated_valid_cam2_frame_{frame_idx:05d}.png"), rotated_mask_cam0)
    cv2.imwrite(str(frame_dir / f"rotated_valid_cam4_frame_{frame_idx:05d}.png"), rotated_mask_cam1)
    return True


def main():
    success_count = 0
    for frame_idx in range(INDEX_START, INDEX_START + NUM_FRAMES):
        ok = process_frame(frame_idx)
        if ok:
            success_count += 1
            print(f"[frame {frame_idx:05d}] done")
        else:
            print(f"[frame {frame_idx:05d}] skipped")

    print(f"successful frames: {success_count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
