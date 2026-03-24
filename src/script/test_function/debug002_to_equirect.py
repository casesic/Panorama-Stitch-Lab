import config_script
from lib_pano import local_setting as ls
import os
import cv2 as cv

from lib_pano.calibration.project_to_equirect import test_main

# path_img = os.path.join(ls.IMAGE_ORIGIN1_DIR, 'frame_00001_rotate.png')
# output_path = os.path.join(ls.DATA_OUTPUT_DIR, 'origin_1', 'frame_00001_equirect.png')
FRAME_INDEX = 515
# test 500
path_img_0 = os.path.join(ls.IMAGE_20260311_DIR, 'camera2', f'frame_00{FRAME_INDEX}.png')
output_path_0 = os.path.join(ls.DATA_OUTPUT_DIR, '20260311_riding', 'camera2', f'frame_00{FRAME_INDEX}_equirect.png')
test_main(ls.CALIB_20260311_YAML_FILE, 'camera2', path_img_0, output_path_0, True)
path_img = os.path.join(ls.IMAGE_20260311_DIR, 'camera4', f'frame_00{FRAME_INDEX}.png')
output_path = os.path.join(ls.DATA_OUTPUT_DIR, '20260311_riding', 'camera4', f'frame_00{FRAME_INDEX}_equirect.png')
test_main(ls.CALIB_20260311_YAML_FILE, 'camera4', path_img, output_path, True)