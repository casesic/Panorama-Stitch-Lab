import os
import config_script

from lib_pano import local_setting as ls
from lib_pano.calibration.undistor import undistor
from lib_pano.calibration.undistor import undistort_r1_quick, undistort_r2_quick
from lib_pano.calibration.project_to_equirect import test_main

image_path = os.path.join(ls.IMAGE_ORIGIN6_DIR, 'frame_00001_rotate.png')
output_path = os.path.join(ls.DATA_OUTPUT_DIR, 'origin_6', 'frame_00001.undist_de.png') 

# undistor(image_path, output_path, ls.INTRINSIC_FILE, method='r1')
# undistort_r1_quick(image_path, output_path, ls.INTRINSIC_FILE)

# undistort_r2_quick(ls.CAMERA_CALIB_FILE, 'camera4', image_path, output_path)
test_main(ls.CAMERA_CALIB_FILE, 'camera6', image_path, output_path, True)