import config_script 
from lib_pano import local_setting as ls
from lib_pano.calibration.optimize_R import global_optimize_R, multi_local_optimize_R
from lib_pano.calibration.optimize_ground import optimize_ground
import os


npz_distant_paths = [os.path.join(ls.FEATURE_DISTANT_FILTER_DIR, 'frame_00001_equirect_cam2_cam4_matches_filtered.npz'),
             os.path.join(ls.FEATURE_DISTANT_FILTER_DIR, 'frame_00001_equirect_cam2_cam6_matches_filtered.npz'),
             os.path.join(ls.FEATURE_DISTANT_FILTER_DIR, 'frame_00001_equirect_cam4_cam6_matches_filtered.npz')]     

# global_optimize_R(ls.CAMERA_CALIB_FILE, npz_distant_paths)
npz_origin_paths = [os.path.join(ls.FEATURE_ORIGIN_DIR, 'frame_00001_equirect_cam2_frame_00001_equirect_cam4_matches.npz'),
                    os.path.join(ls.FEATURE_ORIGIN_DIR, 'frame_00001_equirect_cam2_frame_00001_equirect_cam6_matches.npz'),
                    os.path.join(ls.FEATURE_ORIGIN_DIR, 'frame_00001_equirect_cam4_frame_00001_equirect_cam6_matches.npz')]
y_scopes = [(520, 680), (680, 800), (800, 960)]
# multi_local_optimize_R(ls.CAMERA_CALIB_FILE, npz_origin_paths, y_scopes=y_scopes)

npz_ground_paths = [os.path.join(ls.FEATURE_GROUND_FILTER_DIR, 'frame_00001_equirect_cam2_cam4_ground_matches_filtered.npz'),
                    os.path.join(ls.FEATURE_GROUND_FILTER_DIR, 'frame_00001_equirect_cam2_cam6_ground_matches_filtered.npz'),
                    os.path.join(ls.FEATURE_GROUND_FILTER_DIR, 'frame_00001_equirect_cam4_cam6_ground_matches_filtered.npz')]

save_G_path = os.path.join(ls.CALIBPARAM_DIR, 'ground_params.json')
optimize_ground(npz_ground_paths, ls.CALIB_OPT_FILE, save_G_path)