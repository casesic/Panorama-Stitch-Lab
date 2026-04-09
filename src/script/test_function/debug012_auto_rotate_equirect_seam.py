import os

import cv2
import numpy as np

import config_script
import lib_pano.local_setting as ls
from lib_pano.calibration.rotate_equirect import (
    rotate_equirectangular_image,
)


def main():
    preview_dir = os.path.join(ls.DATA_OUTPUT_DIR, "20260311_riding","segmentation", "preview")
    output_dir = os.path.join(preview_dir, "auto_rotated")
    os.makedirs(output_dir, exist_ok=True)

    sampled_0_path = os.path.join(preview_dir, "debug_sampled_cam_0.png")
    sampled_1_path = os.path.join(preview_dir, "debug_sampled_cam_1.png")
    valid_mask_0_path = os.path.join(preview_dir, "debug_valid_mask_cam_0.png")
    valid_mask_1_path = os.path.join(preview_dir, "debug_valid_mask_cam_1.png")

    sampled_0 = cv2.imread(sampled_0_path, cv2.IMREAD_COLOR)
    sampled_1 = cv2.imread(sampled_1_path, cv2.IMREAD_COLOR)
    valid_mask_0 = cv2.imread(valid_mask_0_path, cv2.IMREAD_GRAYSCALE)
    valid_mask_1 = cv2.imread(valid_mask_1_path, cv2.IMREAD_GRAYSCALE)

    if sampled_0 is None or sampled_1 is None or valid_mask_0 is None or valid_mask_1 is None:
        raise FileNotFoundError("Missing debug_sampled or debug_valid_mask input files")

    rotation_matrix = np.array([1, 0, 0, 0, 0, 1, 0, -1, 0]).reshape(3, 3) 
    print("Rotation matrix:")
    print(np.array2string(rotation_matrix, precision=6, suppress_small=True))

    for idx, sampled in enumerate([sampled_0, sampled_1]):
        rotated = rotate_equirectangular_image(
            input_img=sampled,
            rotation_matrix=rotation_matrix,
            interpolation=cv2.INTER_LINEAR,
            border_value=0,
        )
        cv2.imwrite(os.path.join(output_dir, f"debug_sampled_cam_{idx}_auto_rot.png"), rotated)

    for idx, mask in enumerate([valid_mask_0, valid_mask_1]):
        rotated_mask = rotate_equirectangular_image(
            input_img=mask,
            rotation_matrix=rotation_matrix,
            interpolation=cv2.INTER_NEAREST,
            border_value=0,
        )
        cv2.imwrite(os.path.join(output_dir, f"debug_valid_mask_cam_{idx}_auto_rot.png"), rotated_mask)


if __name__ == "__main__":
    main()
