import config_script 
from lib_pano import local_setting as ls
import os
from lib_pano.io.feature_filter import feature_filter 

npz_path = os.path.join(ls.FEATURE_ORIGIN_DIR, 'frame_00001_equirect_cam4_frame_00001_equirect_cam6_matches.npz')
mask_paths = [os.path.join(ls.DATA_OUTPUT_DIR, 'origin_4', 'frame_00001_equirect_ground_mask.png'),
               os.path.join(ls.DATA_OUTPUT_DIR, 'origin_6', 'frame_00001_equirect_ground_mask.png')]
out_path = os.path.join(ls.FEATURE_GROUND_FILTER_DIR, 'frame_00001_equirect_cam4_cam6_ground_matches_filtered.npz')
image_paths = [os.path.join(ls.DATA_OUTPUT_DIR,'origin_4', 'frame_00001_equirect.png'),
               os.path.join(ls.DATA_OUTPUT_DIR, 'origin_6', 'frame_00001_equirect.png')]
vis_out = os.path.join(ls.FEATURE_GROUND_FILTER_DIR, 'frame_00001_equirect_cam4_cam6_ground_matches_filtered.png')
feature_filter(npz_path,mask_paths,out_path=out_path, image_paths=image_paths,vis_out=vis_out)
