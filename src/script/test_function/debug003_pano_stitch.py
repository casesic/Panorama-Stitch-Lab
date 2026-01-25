import config_script
from lib_pano import local_setting as ls
import os
from lib_pano.stitch.pano_stitch import stitch_pano

equirect_images_path = [
    os.path.join(ls.DATA_OUTPUT_DIR, 'origin_2', 'frame_00001_equirect.png'),
    os.path.join(ls.DATA_OUTPUT_DIR, 'origin_4', 'frame_00001_equirect.png'),
    os.path.join(ls.DATA_OUTPUT_DIR, 'origin_6', 'frame_00001_equirect.png'),]

output_path = os.path.join(ls.DATA_OUTPUT_DIR, 'preview', 'stitched.png')

stitch_pano(equirect_images_path, ls.CALIB_OPT_FILE, ls.CALIB_GROUND_FILE, output_path, (1920, 960))