import config_script
import importlib
import pickle
import sys
from pathlib import Path
import os
import cv2 as cv
import numpy as np
import torch
import lib_pano.local_setting as ls

from lib_pano.human3d.keypoints_detector import PoseDetector, WholePoseDetector
# camera4 512-529  17 帧
# camera2 490-519  30 帧
IMAGE_DIR = os.path.join(ls.DATA_INPUT_DIR, '20260311_riding', 'camera4')
VIDEO_PATH = 'Circular_Video\\00_send\\20260311_riding\\VID_20181003_034909_00_982.1.mp4'  # 如果设置了视频路径则优先使用视频，例如 'xxx/camera4.mp4'
START = 490  # hmr pkl 中的起始帧编号
INDEX_START = 490  # 要处理的起始帧编号
NUM_FRAMES = 40    # 要处理的帧数
OUTPUT_DIR = os.path.join(ls.DATA_OUTPUT_DIR, "20260311_riding", "single_frame_cache", 'camera4', 'rtmlib_multi_frame')
HMR_SOURCE_PKL = os.path.join(ls.DATA_OUTPUT_DIR, "20260311_riding", "4d_humans",'camera4', "VID_20181003_1_seq_490_529_pose.pkl")
SEQUENCE_ID = "490_529"
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


def _build_multi_frame_det_sequence(det_results: list):
    base_engine_mod = importlib.import_module("lib_dynamic_hvip.detection.engine.base_engine")
    Base_Detect_Engine = base_engine_mod.Base_Detect_Engine

    if not det_results:
        raise RuntimeError("det_results 为空，无法生成 single fitting 输入。")

    num_frames = len(det_results)
    bbox = np.zeros((num_frames, 4), dtype=np.int32)
    score = np.zeros((num_frames,), dtype=np.float32)
    kps = np.zeros((num_frames, 133, 3), dtype=np.float32)
    valid = np.zeros((num_frames,), dtype=np.bool_)

    keypoints_format = None
    for i, det_result in enumerate(det_results):
        is_valid = bool(getattr(det_result, "flag_valid", False)) and getattr(det_result, "keypoints", None) is not None
        valid[i] = is_valid
        if not is_valid:
            continue

        if keypoints_format is None:
            keypoints_format = det_result.keypoints_format

        bbox_i = np.asarray(det_result.bbox, dtype=np.int32).reshape(-1)
        if bbox_i.size >= 4:
            bbox[i] = bbox_i[:4]
        score[i] = float(det_result.bbox_score)

        kps_i = np.asarray(det_result.keypoints, dtype=np.float32)
        if kps_i.ndim == 2:
            n = min(133, kps_i.shape[0])
            c = min(3, kps_i.shape[1])
            kps[i, :n, :c] = kps_i[:n, :c]

    if keypoints_format is None:
        raise RuntimeError("所有帧检测都失败，无法确定 keypoints_format。")

    return Base_Detect_Engine.Result_Single_Human_Sequence(
        array_valid=valid,
        array_bbox_h1h2w1w2=bbox,
        array_bbox_score=score,
        array_keypoints=kps,
        keypoints_format=keypoints_format,
        flag_with_keypoints=True,
    )


def _build_single_frame_smpl_init():
    smpl_mod = importlib.import_module("lib_dynamic_hvip.fitting.smpl_seq_forward_module")
    SMPLFitter = smpl_mod.SMPL_Seq_Forward_Module

    src_path = Path(HMR_SOURCE_PKL)
    with src_path.open("rb") as f:
        src_obj = pickle.load(f)

    if isinstance(src_obj, SMPLFitter.SMPL_Param_Seq):
        smpl_init = src_obj

    if isinstance(src_obj, dict):
        num_batch = int(src_obj["num_batch"])
        dim_shape = int(src_obj["beta"].shape[-1])
        gender = str(src_obj.get("gender", "neutral"))
        smpl_init = SMPLFitter.SMPL_Param_Seq(dim_shape=dim_shape, gender=gender, num_batch=num_batch)
        smpl_init.orient = src_obj["orient"].clone()
        smpl_init.transl = src_obj["transl"].clone()
        smpl_init.pose = src_obj["pose"].clone()
        smpl_init.beta = src_obj["beta"].clone()
        smpl_init.scale = src_obj["scale"].clone()
    elif not isinstance(src_obj, SMPLFitter.SMPL_Param_Seq):
        raise TypeError(f"不支持的 hmr pkl 类型: {type(src_obj)}")

    dim_shape = int(smpl_init.beta.shape[-1])
    gender = str(getattr(smpl_init, "gender", "neutral"))
    local_start = INDEX_START - START
    local_end = INDEX_START - START + NUM_FRAMES

    valid_mask_local = None
    if isinstance(src_obj, dict) and "valid" in src_obj:
        valid_mask_local = src_obj["valid"][local_start:local_end]
        if not isinstance(valid_mask_local, torch.Tensor):
            valid_mask_local = torch.as_tensor(valid_mask_local, dtype=torch.bool)
        else:
            valid_mask_local = valid_mask_local.to(dtype=torch.bool)

    smpl_sliced = SMPLFitter.SMPL_Param_Seq(dim_shape=dim_shape, gender=gender, num_batch=NUM_FRAMES)
    smpl_sliced.orient = smpl_init.orient[local_start:local_end].clone()
    smpl_sliced.transl = smpl_init.transl[local_start:local_end].clone()
    smpl_sliced.pose = smpl_init.pose[local_start:local_end].clone()

    beta_src = smpl_init.beta
    if beta_src.ndim > 1:
        beta_slice = beta_src[local_start:local_end]
        if valid_mask_local is not None and beta_slice.shape[0] == valid_mask_local.shape[0] and bool(valid_mask_local.any().item()):
            beta_src = beta_slice[valid_mask_local].mean(dim=0)
        else:
            beta_src = beta_slice.mean(dim=0)
    smpl_sliced.beta = beta_src.clone()

    scale_src = smpl_init.scale
    if scale_src.ndim > 1:
        scale_slice = scale_src[local_start:local_end]
        if valid_mask_local is not None and scale_slice.shape[0] == valid_mask_local.shape[0] and bool(valid_mask_local.any().item()):
            scale_src = scale_slice[valid_mask_local].mean(dim=0)
        else:
            scale_src = scale_slice.mean(dim=0)
    if scale_src.numel() > 1:
        scale_src = scale_src.mean().reshape(1)
    smpl_sliced.scale = scale_src.clone()

    if valid_mask_local is not None and valid_mask_local.shape[0] == NUM_FRAMES:
        smpl_sliced.valid = valid_mask_local.clone()
    else:
        smpl_sliced.valid = torch.ones(NUM_FRAMES, dtype=torch.bool)

    return smpl_sliced


def main():
    repo_root = _ensure_paths()
    output_dir = Path(OUTPUT_DIR)
    print(f"output_dir: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    frames = []
    frame_indices = []

    if VIDEO_PATH is not None:
        video_path = Path(VIDEO_PATH)
        cap = cv.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"无法打开视频: {video_path}")

        cap.set(cv.CAP_PROP_POS_FRAMES, INDEX_START)
        for i in range(NUM_FRAMES):
            ok, frame = cap.read()
            if not ok or frame is None:
                print(f"video stop at frame index {INDEX_START + i}")
                break
            frames.append(frame)
            frame_indices.append(INDEX_START + i)
        cap.release()
    else:
        image_dir = repo_root / IMAGE_DIR
        for i in range(NUM_FRAMES):
            idx = INDEX_START + i
            image_path = image_dir / f"frame_{idx:05d}.png"
            image = cv.imread(str(image_path), cv.IMREAD_COLOR)
            if image is None:
                raise RuntimeError(f"无法读取图像: {image_path}")
            frames.append(image)
            frame_indices.append(idx)

    if not frames:
        raise RuntimeError("未读取到任何图像帧。")

    detector = WholePoseDetector(device=DEVICE)
    det_results = []
    for idx, image in zip(frame_indices, frames):
        det_result = detector.detect_single_image(image)
        det_results.append(det_result)

        kp_shape = det_result.keypoints.shape if det_result.keypoints is not None else None
        print(f"frame {idx:05d}: valid={det_result.flag_valid}, keypoints_shape={kp_shape}")

        if SAVE_VIS:
            from lib_pano.human3d.keypoints_detector import draw_keypoints

            vis = draw_keypoints(
                image,
                det_result,
                conf_thresh=VIS_CONF_THRESH,
                draw_face=True,
                draw_hands=True,
            )
            cv.imwrite(str(output_dir / f"vis_frame_{idx:05d}.jpg"), vis)

    det_seq = _build_multi_frame_det_sequence(det_results)
    smpl_init = _build_single_frame_smpl_init()

    path_rtm = output_dir / f"rtm_joints_cache_s{SEQUENCE_ID}_c{CAMERA_ID}.pkl"
    path_hmr = output_dir / f"hmr_s{SEQUENCE_ID}_pose.pkl"

    with path_rtm.open("wb") as f:
        pickle.dump(det_seq, f)
    with path_hmr.open("wb") as f:
        pickle.dump(smpl_init, f)

    print(f"saved: {path_rtm}")
    print(f"saved: {path_hmr}")
    print(f"det_seq_keypoints_shape: {det_seq.array_keypoints.shape}")


if __name__ == "__main__":
    main()
