import config_script
from lib_pano import local_setting as ls
import os
import cv2 as cv

from lib_pano.calibration.project_to_equirect import test_main

path_img = os.path.join(ls.IMAGE_ORIGIN6_DIR, 'frame_00001_rotate.png')

output_path = os.path.join(ls.DATA_OUTPUT_DIR, 'origin_6', 'frame_00001_equirect.png')
test_main(ls.CAMERA_CALIB_FILE, 'camera6', path_img, output_path, True)