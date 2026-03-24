import config_script
import importlib
import pickle
import sys
from pathlib import Path
import os
import cv2 as cv
import numpy as np
import lib_pano.local_setting as ls

from lib_pano.human3d.keypoints_detector import PoseDetector, WholePoseDetector

IMAGE_PATH = os.path.join(ls.DATA_INPUT_DIR,'20260311_riding', 'camera4','frame_00515.png')
OUTPUT_DIR = os.path.join(ls.DATA_OUTPUT_DIR, "20260311_riding", "single_frame_cache", 'camera4')
HMR_SOURCE_PKL = os.path.join(ls.DATA_OUTPUT_DIR, "20260311_riding", "4d_humans",'camera4', "hmr_s2_pose.pkl")
SEQUENCE_ID = 2
CAMERA_ID = 4   # 0: panorama; 1-6: camera1-6

DEVICE = "cpu"
SAVE_VIS = True
VIS_CONF_THRESH = 0.5

def _ensure_paths() -> Path:
    repo_root = Path(__file__).resolve().parents[3]
    sf_root = repo_root / "tools" / "SingleFitting"
    if str(sf_root) not in sys.path:
        sys.path.insert(0, str(sf_root))
    return repo_root


def _build_single_frame_det_sequence(det_result):
    base_engine_mod = importlib.import_module("lib_dynamic_hvip.detection.engine.base_engine")
    Base_Detect_Engine = base_engine_mod.Base_Detect_Engine

    if not det_result.flag_valid or det_result.keypoints is None:
        raise RuntimeError("关键点检测失败，无法生成 single fitting 输入。")

    bbox = np.asarray(det_result.bbox, dtype=np.int32).reshape(1, 4)
    score = np.asarray([float(det_result.bbox_score)], dtype=np.float32)
    kps = np.asarray(det_result.keypoints, dtype=np.float32).reshape(1, -1, 3)
    valid = np.asarray([True], dtype=np.bool_)

    return Base_Detect_Engine.Result_Single_Human_Sequence(
        array_valid=valid,
        array_bbox_h1h2w1w2=bbox,
        array_bbox_score=score,
        array_keypoints=kps,
        keypoints_format=det_result.keypoints_format,
        flag_with_keypoints=True,
    )


def _build_single_frame_smpl_init(repo_root: Path):
    smpl_mod = importlib.import_module("lib_dynamic_hvip.fitting.smpl_seq_forward_module")
    SMPLFitter = smpl_mod.SMPL_Seq_Forward_Module

    src_path = repo_root / HMR_SOURCE_PKL
    with src_path.open("rb") as f:
        src_obj = pickle.load(f)

    if isinstance(src_obj, SMPLFitter.SMPL_Param_Seq):
        return src_obj

    if isinstance(src_obj, dict):
        num_batch = int(src_obj["num_batch"])
        dim_shape = int(src_obj["beta"].shape[0])
        gender = str(src_obj.get("gender", "neutral"))
        smpl_init = SMPLFitter.SMPL_Param_Seq(dim_shape=dim_shape, gender=gender, num_batch=num_batch)
        smpl_init.orient = src_obj["orient"].clone()
        smpl_init.transl = src_obj["transl"].clone()
        smpl_init.pose = src_obj["pose"].clone()
        smpl_init.beta = src_obj["beta"].clone()
        smpl_init.scale = src_obj["scale"].clone()
        return smpl_init

    raise TypeError(f"不支持的 hmr pkl 类型: {type(src_obj)}")


def main():
    repo_root = _ensure_paths()
    image_path = repo_root / IMAGE_PATH
    output_dir = repo_root / OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    image = cv.imread(str(image_path), cv.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"无法读取图像: {image_path}")

    detector = WholePoseDetector(device=DEVICE)
    det_result = detector.detect_single_image(image)

    det_seq = _build_single_frame_det_sequence(det_result)
    smpl_init = _build_single_frame_smpl_init(repo_root)

    path_rtm = output_dir / f"rtm_joints_cache_s{SEQUENCE_ID}_c{CAMERA_ID}.pkl"
    path_hmr = output_dir / f"hmr_s{SEQUENCE_ID}_pose.pkl"

    with path_rtm.open("wb") as f:
        pickle.dump(det_seq, f)
    with path_hmr.open("wb") as f:
        pickle.dump(smpl_init, f)

    if SAVE_VIS:
        from lib_pano.human3d.keypoints_detector import draw_keypoints

        vis = draw_keypoints(
            image,
            det_result,
            conf_thresh=VIS_CONF_THRESH,
            draw_face=True,
            draw_hands=True,
        )
        cv.imwrite(str(output_dir / "single_frame_keypoints_vis.jpg"), vis)

    print(f"saved: {path_rtm}")
    print(f"saved: {path_hmr}")
    print(f"keypoints: {det_result.keypoints.shape if det_result.keypoints is not None else None}")


if __name__ == "__main__":
    main()
