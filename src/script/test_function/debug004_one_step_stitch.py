import config_script
from lib_pano import local_setting as ls
import os
import numpy as np
from lib_pano.stitch.one_step_stitch import one_step_stitch
image_paths = [
    os.path.join(ls.IMAGE_ORIGIN2_DIR, 'frame_00001_rotate.png'),
    os.path.join(ls.IMAGE_ORIGIN4_DIR, 'frame_00001_rotate.png'),
    os.path.join(ls.IMAGE_ORIGIN6_DIR, 'frame_00001_rotate.png'),
]
output_path = os.path.join(ls.DATA_OUTPUT_DIR, 'preview','frame_00001_stitched.png')
one_step_stitch(image_paths, ls.CAMERA_CALIB_FILE, ls.CALIB_OPT_FILE, output_path)

