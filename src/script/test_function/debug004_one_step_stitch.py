import os

# Windows 下部分依赖会重复加载 OpenMP 运行时，先设置兜底以避免进程直接退出。
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')

import config_script
from lib_pano import local_setting as ls
from lib_pano.stitch.one_step_stitch import one_step_stitch
import numpy as np


def run_ds_two_cam_test(frame_idx: int = 515, pano_size=(1920, 960), shift_right_px: int = 200):
    """两相机(cam2/cam4) DS 标定的一步合图测试。"""
    frame_name = f"frame_{frame_idx:05d}.png"
    image_paths = [
        os.path.join(ls.IMAGE_20260311_DIR, 'camera2', frame_name),
        os.path.join(ls.IMAGE_20260311_DIR, 'camera4', frame_name),
    ]
    image_paths = [
        os.path.join(ls.DATA_OUTPUT_DIR, '20260311_riding', 'segmentation', 'cam2', 'segmented_frame_00515.jpg'),
        os.path.join(ls.DATA_OUTPUT_DIR, '20260311_riding', 'segmentation', 'cam4', 'segmented_frame_00515.jpg'),
    ]


    calib_path = ls.CALIB_20260311_YAML_FILE

    output_dir = os.path.join(ls.DATA_ROOT, 'Output', '20260311_riding', 'segmentation', 'preview')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f'frame_{frame_idx:05d}_cam2_cam4_stitched.png')

    # 对 YAML 路径而言，第三个参数当前不会被使用，传同一路径即可。
    one_step_stitch(
    	image_paths,
    	calib_path,
    	calib_path,
    	output_path,
    	pano_size=pano_size,
    	shift_right_px=shift_right_px,
    )


    print(f"[OK] panorama saved to: {output_path}")


if __name__ == '__main__':
    run_ds_two_cam_test(frame_idx=515, pano_size=(5760, 2880), shift_right_px=0)

