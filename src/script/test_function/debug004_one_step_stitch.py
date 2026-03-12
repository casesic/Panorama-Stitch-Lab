import config_script
from lib_pano import local_setting as ls
import os
import numpy as np
from lib_pano.stitch.one_step_stitch import one_step_stitch
image246_paths = [
    os.path.join(ls.IMAGE_ORIGIN2_DIR, 'frame_00001_rotate.png'),
    os.path.join(ls.IMAGE_ORIGIN4_DIR, 'frame_00001_rotate.png'),
    os.path.join(ls.IMAGE_ORIGIN6_DIR, 'frame_00001_rotate.png'),
]
image135_paths = [
    os.path.join(ls.IMAGE_ORIGIN1_DIR, 'frame_00001_rotate.png'),
    os.path.join(ls.IMAGE_ORIGIN3_DIR, 'frame_00001_rotate.png'),
    os.path.join(ls.IMAGE_ORIGIN5_DIR, 'frame_00001_rotate.png'),
]
# output_path = os.path.join(ls.DATA_OUTPUT_DIR, 'preview','frame_135_00001_stitched.png')
# using 135 config
# Pass 135 calib as both arguments (second one is ignored by logic but must be valid str)
# one_step_stitch(image135_paths, ls.CAMERA_135_CALIB_FILE, ls.CAMERA_135_CALIB_FILE, output_path)

output_path = os.path.join(ls.DATA_OUTPUT_DIR, 'preview','frame_135_00001_stitched.png')
one_step_stitch(image135_paths, ls.CAMERA_135_CALIB_FILE, ls.CALIB_OPT_FILE, output_path)

