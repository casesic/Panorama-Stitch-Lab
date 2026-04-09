import os
from pathlib import Path

os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')

import config_script
import numpy as np
import cv2
from lib_pano import local_setting as ls
from lib_pano.stitch.one_step_stitch import remap_and_accumulate, generate_equirectangular_dirs, load_images_for_names, load_valid_masks_for_names
from lib_pano.io.params_read import load_all_K_and_distortion, load_extrinsics_from_opencv_yaml
from lib_pano.calibration.rotate_equirect import rotate_equirectangular_image

IMAGE_DIR_CAM2 = r"data_process/Output/20260311_riding/segmentation/cam2/seg"
IMAGE_DIR_CAM4 = r"data_process/Output/20260311_riding/segmentation/cam4/seg"
MASK_DIR_CAM2 = r"data_process/Output/20260311_riding/segmentation/cam2/mask"
MASK_DIR_CAM4 = r"data_process/Output/20260311_riding/segmentation/cam4/mask"
CALIB_PATH_CAM2 = ls.CALIB_20260311_YAML_FILE
CALIB_PATH_CAM4 = ls.CALIB_20260311_YAML_FILE
OUTPUT_DIR = r"data_process/Output/20260311_riding/stitch_batch"
PANO_SIZE = (5760, 2880)
SHIFT_RIGHT_PX = 0
INDEX_START = 490
NUM_FRAMES = 40
IMAGE_FILENAME_PATTERN = "segmented_frame_{:05d}.jpg"
MASK_FILENAME_PATTERN = "mask_frame_{:05d}.png"

ROTATION_MATRIX = np.array([1, 0, 0, 0, 0, 1, 0, -1, 0], dtype=np.float64).reshape(3, 3)


def _resolve_camera_index(names, cam_id: str) -> int:
    candidates = [cam_id]
    if cam_id.startswith("cam"):
        candidates.append("camera" + cam_id[len("cam"):])
    elif cam_id.startswith("camera"):
        candidates.append("cam" + cam_id[len("camera"):])

    for candidate in candidates:
        if candidate in names:
            return names.index(candidate)
    raise ValueError(f"Camera {cam_id} not found in calibration names: {names}")


def _cleanup_unused_debug_files(frame_debug_dir: Path):
    for name in [
        "debug_final_valid.png",
        "debug_best_idx.png",
        "debug_circle_cam_0.png",
        "debug_circle_cam_1.png",
        "debug_sample_cam_0.png",
        "debug_sample_cam_1.png"
    ]:
        path = frame_debug_dir / name
        if path.exists():
            path.unlink()


def process_single_cam(idx, cam_id, image_dir, mask_dir, calib_path, frame_debug_dir):
    image_path = Path(image_dir) / IMAGE_FILENAME_PATTERN.format(idx)
    mask_path = Path(mask_dir) / MASK_FILENAME_PATTERN.format(idx)
    if not image_path.exists():
        return None

    names, K_list, dist_list, xi_list, model_list = load_all_K_and_distortion(calib_path)
    R_list, T_list = load_extrinsics_from_opencv_yaml(calib_path, names)
    cam_index = _resolve_camera_index(names, cam_id)

    images = load_images_for_names([str(image_path)])
    valid_masks = load_valid_masks_for_names([str(mask_path) if mask_path.exists() else None])

    pano_dirs = generate_equirectangular_dirs(PANO_SIZE[1], PANO_SIZE[0])
    pano = remap_and_accumulate(
        pano_dirs,
        images,
        [K_list[cam_index]],
        [dist_list[cam_index]],
        [xi_list[cam_index]],
        [R_list[cam_index]],
        [T_list[cam_index]],
        [model_list[cam_index]],
        source_valid_masks=valid_masks,
        output_size=PANO_SIZE,
        debug_dir=str(frame_debug_dir),
    )
    _cleanup_unused_debug_files(frame_debug_dir)

    pano_rot = rotate_equirectangular_image(pano, rotation_matrix=ROTATION_MATRIX)
    rotated_path = frame_debug_dir / f"rotated_{cam_id}_frame_{idx:05d}.png"
    cv2.imwrite(str(rotated_path), pano_rot)
    valid_mask_path = frame_debug_dir / "debug_valid_mask_cam_0.png"
    valid_mask = cv2.imread(str(valid_mask_path), cv2.IMREAD_GRAYSCALE) if valid_mask_path.exists() else None
    if valid_mask is not None:
        valid_mask_rot = rotate_equirectangular_image(valid_mask, rotation_matrix=ROTATION_MATRIX)
        rotated_valid_path = frame_debug_dir / f"rotated_valid_{cam_id}_frame_{idx:05d}.png"
        cv2.imwrite(str(rotated_valid_path), valid_mask_rot)
    return pano_rot


def main():
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    for idx in range(INDEX_START, INDEX_START + NUM_FRAMES):
        frame_debug_dir = output_dir / f"frame_{idx:05d}"
        frame_debug_dir.mkdir(parents=True, exist_ok=True)

        result_cam2 = process_single_cam(idx, "cam2", IMAGE_DIR_CAM2, MASK_DIR_CAM2, CALIB_PATH_CAM2, frame_debug_dir)
        result_cam4 = process_single_cam(idx, "cam4", IMAGE_DIR_CAM4, MASK_DIR_CAM4, CALIB_PATH_CAM4, frame_debug_dir)

        if result_cam2 is not None and result_cam4 is not None:
            merged = result_cam2.copy()
            cam4_valid = np.any(result_cam4 > 0, axis=-1)
            cam2_empty = np.all(result_cam2 == 0, axis=-1)
            fill_mask = cam4_valid & cam2_empty
            merged[fill_mask] = result_cam4[fill_mask]
        elif result_cam2 is not None:
            merged = result_cam2
        elif result_cam4 is not None:
            merged = result_cam4
        else:
            print(f"[warn] No valid panorama source for frame {idx:05d}")
            continue

        if SHIFT_RIGHT_PX != 0:
            merged = np.roll(merged, shift=SHIFT_RIGHT_PX, axis=1)

        output_path = frame_debug_dir / f"pano_frame_{idx:05d}.png"
        cv2.imwrite(str(output_path), merged)
        print(f"[done] frame {idx:05d} saved to {output_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
