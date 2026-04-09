import os

import cv2
import numpy as np

import config_script
import lib_pano.local_setting as ls
from lib_pano.calibration.rotate_equirect import (
    estimate_seam_rotation_from_masks,
    rotate_equirectangular_image,
)

BOUNDARY_BAND_PX = 75
SAMPLE_STRIDE = 12


def main():
    preview_dir = os.path.join(ls.DATA_OUTPUT_DIR, "20260311_riding", "preview")
    result_dir = os.path.join(ls.DATA_OUTPUT_DIR, "20260311_riding", "test_raft_result")

    mask_0_path = os.path.join(preview_dir, "debug_valid_mask_cam_0.png")
    mask_1_path = os.path.join(preview_dir, "debug_valid_mask_cam_1.png")
    pano_rot_path = os.path.join(result_dir, "step1_final_composed_filled.png")
    pano_mask_path = os.path.join(result_dir, "step1_final_valid_mask.png")

    mask_0 = cv2.imread(mask_0_path, cv2.IMREAD_GRAYSCALE)
    mask_1 = cv2.imread(mask_1_path, cv2.IMREAD_GRAYSCALE)
    pano_rot = cv2.imread(pano_rot_path, cv2.IMREAD_COLOR)
    pano_mask = cv2.imread(pano_mask_path, cv2.IMREAD_GRAYSCALE)

    if mask_0 is None or mask_1 is None:
        raise FileNotFoundError("Missing original valid masks for seam rotation estimation")
    if pano_rot is None:
        raise FileNotFoundError(f"Missing rotated panorama: {pano_rot_path}")

    rotation_matrix = np.array([1, 0, 0, 0, 0, 1, 0, -1, 0]).reshape(3, 3)
    rotation_inv = rotation_matrix.T

    pano_unrot = rotate_equirectangular_image(
        input_img=pano_rot,
        rotation_matrix=rotation_inv,
        interpolation=cv2.INTER_LINEAR,
        border_value=0,
    )
    mask_rot = rotate_equirectangular_image(
        input_img=pano_mask,
        rotation_matrix=rotation_inv,
        interpolation=cv2.INTER_NEAREST,
        border_value=0,
    )
    impaint_mask = (mask_rot == 0).astype(np.uint8) * 255
    impaint_mask_dilated = cv2.dilate(impaint_mask, np.ones((3,3), np.uint8), iterations=2)
    if cv2.countNonZero(impaint_mask_dilated) > 0:
        pano_unrot = cv2.inpaint(pano_unrot, impaint_mask_dilated, inpaintRadius=5, flags=cv2.INPAINT_TELEA)

    out_path = os.path.join(result_dir, "step1_final_composed_result_unrotated.png")
    cv2.imwrite(out_path, pano_unrot)

    print("Saved unrotated panorama:", out_path)
    print("Forward rotation matrix:")
    print(np.array2string(rotation_matrix, precision=6, suppress_small=True))


if __name__ == "__main__":
    main()
