import os

os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')

import config_script
import cv2
import numpy as np
from pathlib import Path
from lib_pano import local_setting as ls
from lib_pano.stitch.one_step_stitch import one_step_stitch

VIDEO_PATH_CAM2 = r"Circular_Video\00_send\20260311_riding\VID_20181003_034909_00_982.0.mp4"
VIDEO_PATH_CAM4 = r"Circular_Video\00_send\20260311_riding\VID_20181003_034909_00_982.1.mp4"
CALIB_PATH = ls.CALIB_20260311_YAML_FILE
OUTPUT_DIR = r"data_process/Output/20260311_riding/stitch_raw"
PANO_SIZE = (5760, 2880)
SHIFT_RIGHT_PX = 0
INDEX_START = 500
NUM_FRAMES = 30


def extract_frame(cap, frame_idx):
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    if not ok:
        return None
    return frame


def main():
    cap_cam2 = cv2.VideoCapture(VIDEO_PATH_CAM2)
    cap_cam4 = cv2.VideoCapture(VIDEO_PATH_CAM4)
    if not cap_cam2.isOpened() or not cap_cam4.isOpened():
        if cap_cam2.isOpened():
            cap_cam2.release()
        if cap_cam4.isOpened():
            cap_cam4.release()
        raise RuntimeError("Failed to open cam2 or cam4 video")

    tmp_dir = Path("/tmp")
    tmp_dir.mkdir(parents=True, exist_ok=True)
    cam2_tmp_path = tmp_dir / "cam2_tmp.png"
    cam4_tmp_path = tmp_dir / "cam4_tmp.png"

    try:
        for idx in range(INDEX_START, INDEX_START + NUM_FRAMES):
            frame_cam2 = extract_frame(cap_cam2, idx)
            frame_cam4 = extract_frame(cap_cam4, idx)
            if frame_cam2 is None or frame_cam4 is None:
                print(f"[warn] Failed to read frame {idx:05d}")
                continue

            cv2.imwrite(str(cam2_tmp_path), frame_cam2)
            cv2.imwrite(str(cam4_tmp_path), frame_cam4)

            output_dir = Path(OUTPUT_DIR) / f"frame_{idx:05d}"
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / f"pano_raw_frame_{idx:05d}.png"

            one_step_stitch(
                [str(cam2_tmp_path), str(cam4_tmp_path)],
                CALIB_PATH,
                CALIB_PATH,
                str(output_path),
                pano_size=PANO_SIZE,
                shift_right_px=SHIFT_RIGHT_PX,
            )

            print(f"[frame {idx:05d}] saved to: {output_path}")
    finally:
        cap_cam2.release()
        cap_cam4.release()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
