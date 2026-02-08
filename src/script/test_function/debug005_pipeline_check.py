import config_script
from lib_pano import local_setting as ls
import os
import numpy as np
from lib_pano.io.video_process import extract_video_frames
from lib_pano.io.params_read import show_all_params, load_R_and_T, load_R_and_T_2, load_all_K_and_distortion
from lib_pano.vis.visual_RT import visualize_cameras, circle_center_images, divide_by_y_scopes
from lib_pano.stitch.omni_radtan_to_sphere import omni_radtan_to_sphere_main

path_video = os.path.join(ls.RAW_VIDEO_DIR, 'origin_5_20251027_110506.mp4')
output_dir = os.path.join(ls.IMAGE_ORIGIN5_DIR)
extract_video_frames(path_video, output_dir, frame_index=1)
R_list, T_list = load_R_and_T_2(ls.CAMERA_CALIB_FILE)
camera_list, K_list, dist_list, xi_list = load_all_K_and_distortion(ls.CAMERA_CALIB_FILE)
save_path = os.path.join(ls.DATA_OUTPUT_DIR, 'extrinsics_visualization.png')


# visualize_cameras(K_list, R_list, T_list, world_align=world_align)
# show_all_params(ls.CAMERA_CALIB_FILE)
preview_path = os.path.join(ls.DATA_OUTPUT_DIR, 'preview', 'preview.png')
output_path = os.path.join(ls.DATA_OUTPUT_DIR, 'preview', 'preview_with_centers.png')
# project_center_to_preview(ls.CAMERA_CALIB_FILE, preview_path, output_path)  
image_paths = [os.path.join(ls.IMAGE_ORIGIN2_DIR, 'frame_00001_rotate.png'),
               os.path.join(ls.IMAGE_ORIGIN4_DIR, 'frame_00001_rotate.png'),
               os.path.join(ls.IMAGE_ORIGIN6_DIR, 'frame_00001_rotate.png')]
output_paths = [os.path.join(ls.DATA_OUTPUT_DIR, 'origin_2', 'frame_00001_cxcy.png'),
                os.path.join(ls.DATA_OUTPUT_DIR, 'origin_4', 'frame_00001_cxcy.png'),
                os.path.join(ls.DATA_OUTPUT_DIR, 'origin_6', 'frame_00001_cxcy.png')]
# circle_center_images(image_paths, output_paths, ls.CAMERA_CALIB_FILE)
# omni_radtan_to_sphere_main(image_paths[0], ls.CAMERA_CALIB_FILE, 'camera2')
