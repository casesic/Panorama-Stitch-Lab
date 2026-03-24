import os

try:
    from script.test_function import config_script
    PROJECT_ROOT = config_script.PROJECT_ROOT
except (ImportError, AttributeError):
    print("警告: 未能从 config_script 导入 PROJECT_ROOT，将自行计算。")
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

RAW_VIDEO_DIR = os.path.join(PROJECT_ROOT, 'Circular_Video', '00_send', '20251125_calibration_rlt', 'VID_20251027_110506')
RAW_20260311_RIDING_VIDEO_DIR = os.path.join(PROJECT_ROOT, 'Circular_Video', '00_send', '20260311_riding')
RAW_PANO_IMAGE_20260309_DIR = os.path.join(PROJECT_ROOT, 'Circular_Video', '00_send', '20260309', 'human')

DATA_ROOT = os.path.join(PROJECT_ROOT, 'data_process')

SRC_ROOT = os.path.join(PROJECT_ROOT, 'src')

DATA_INPUT_DIR = os.path.join(DATA_ROOT, 'input')
DATA_OUTPUT_DIR = os.path.join(DATA_ROOT, 'output')
DATA_VIS_DIR = os.path.join(DATA_ROOT, 'vis')

CALIBPARAM_DIR = os.path.join(DATA_INPUT_DIR, 'calib_params')
INTRINSIC_FILE = os.path.join(CALIBPARAM_DIR, 'intrinsic.json')
EXTRINSIC_FILE = os.path.join(CALIBPARAM_DIR, 'extrinsic.json')
CAMERA_CALIB_FILE = os.path.join(CALIBPARAM_DIR, 'camera_calib.json')
CAMERA_135_CALIB_FILE = os.path.join(CALIBPARAM_DIR, 'calib_135_custom.json')
CALIB_OPT_FILE = os.path.join(CALIBPARAM_DIR, 'opt_camera_calib.json')
CALIB_GROUND_FILE = os.path.join(CALIBPARAM_DIR, 'ground_params.json')
CALIB_20260311_YAML_FILE = os.path.join(CALIBPARAM_DIR, 'calibration_opencv_ds.yaml')

IMAGE_DIR = os.path.join(DATA_INPUT_DIR, 'images')
IMAGE_ORIGIN1_DIR = os.path.join(IMAGE_DIR, 'origin_1')
IMAGE_ORIGIN3_DIR = os.path.join(IMAGE_DIR, 'origin_3')
IMAGE_ORIGIN5_DIR = os.path.join(IMAGE_DIR, 'origin_5')
IMAGE_ORIGIN2_DIR = os.path.join(IMAGE_DIR, 'origin_2')
IMAGE_ORIGIN4_DIR = os.path.join(IMAGE_DIR, 'origin_4')
IMAGE_ORIGIN6_DIR = os.path.join(IMAGE_DIR, 'origin_6')

IMAGE_20260311_DIR = os.path.join(DATA_INPUT_DIR, '20260311_riding')

FEATURE_MATCH_DIR = os.path.join(DATA_INPUT_DIR, 'feature_match')
FEATURE_ORIGIN_DIR = os.path.join(FEATURE_MATCH_DIR, 'origin')
FEATURE_FILTER_DIR = os.path.join(FEATURE_MATCH_DIR, 'filter')
FEATURE_DISTANT_FILTER_DIR = os.path.join(FEATURE_FILTER_DIR, 'distant_view')
FEATURE_GROUND_FILTER_DIR = os.path.join(FEATURE_FILTER_DIR, 'ground')

HUMAN_OBJ_DIR = os.path.join(DATA_INPUT_DIR, 'human_obj')