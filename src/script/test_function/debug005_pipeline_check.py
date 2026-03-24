import config_script
from lib_pano import local_setting as ls
import os
import numpy as np
import cv2 as cv
from lib_pano.io.video_process import extract_video_frames, extract_video_frames_range
from lib_pano.human3d.keypoints_detector import PoseDetector, draw_keypoints

# 500-600
START_FRAME = 517

path_video_0 = os.path.join(ls.RAW_20260311_RIDING_VIDEO_DIR, 'VID_20181003_034909_00_982.0.mp4')
output_dir_0 = os.path.join(ls.DATA_INPUT_DIR, '20260311_riding', 'camera2')
extract_video_frames_range(path_video_0, output_dir_0, start_frame=START_FRAME, num_frames=1, rotate_left_90=False)

path_video = os.path.join(ls.RAW_20260311_RIDING_VIDEO_DIR, 'VID_20181003_034909_00_982.1.mp4')
output_dir = os.path.join(ls.DATA_INPUT_DIR, '20260311_riding', 'camera4')
extract_video_frames_range(path_video, output_dir, start_frame=START_FRAME, num_frames=1, rotate_left_90=False)

# path_image_0 = os.path.join(ls.DATA_INPUT_DIR,'20260311_riding', 'camera2', 'frame_00515.png')
# path_image_1 = os.path.join(ls.DATA_INPUT_DIR,'20260311_riding', 'camera4', 'frame_00515.png')
# detector = PoseDetector()
# img = cv.imread(path_image_0)
# keypoints = detector.detect_single_image(img)
# img_kp = draw_keypoints(img, keypoints)
# path_output_0 = os.path.join(ls.DATA_VIS_DIR, '20260311_riding', '00515_cam2_keypoints.jpg')
# cv.imwrite(path_output_0, img_kp)

# img = cv.imread(path_image_1)
# keypoints = detector.detect_single_image(img)
# img_kp = draw_keypoints(img, keypoints)
# path_output_1 = os.path.join(ls.DATA_VIS_DIR, '20260311_riding', '00515_cam4_keypoints.jpg')
# cv.imwrite(path_output_1, img_kp)

