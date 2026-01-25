import config_script
import lib_pano.local_setting as ls
import os
from lib_pano.calibration.refine_near_region_by_flow import refine_near_region_with_optical_flow

img_ref_path = os.path.join(ls.DATA_OUTPUT_DIR, 'preview', 'debug_sampled_cam_0.png')

# 源图像和mask列表
img_src_paths = [
    os.path.join(ls.DATA_OUTPUT_DIR, 'preview', 'debug_sampled_cam_1.png'),
    os.path.join(ls.DATA_OUTPUT_DIR, 'preview', 'debug_sampled_cam_2.png'),
]

near_mask_paths = [
    os.path.join(ls.DATA_OUTPUT_DIR, 'preview', 'debug_overlap_cam_0_1.png'),
    os.path.join(ls.DATA_OUTPUT_DIR, 'preview', 'debug_overlap_cam_0_2.png'),
]

panorama, refined_images = refine_near_region_with_optical_flow(img_ref_path, img_src_paths, near_mask_paths)