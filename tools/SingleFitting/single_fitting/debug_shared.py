from __future__ import annotations

import math
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

DS_PARAMS = {
    "xi": 0.02541467,
    "alpha": 0.58707823,
    "fx": 819.659819122500,
    "fy": 819.389361330000,
    "cx": 1455.78851018250,
    "cy": 1435.04276513250,
}

R_CAM4 = np.array(
    [[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]],
    dtype=np.float32,
)

DEFAULT_DELTA_T = np.array([0.04, 0.039, -0.175], dtype=np.float32)
LHAND_IDX = list(range(91, 112))
RHAND_IDX = list(range(112, 133))

COCO_NAME_MAP = {
    "nose": "Nose",
    "left_eye": "L_Eye",
    "right_eye": "R_Eye",
    "left_ear": "L_Ear",
    "right_ear": "R_Ear",
    "left_shoulder": "L_Shoulder",
    "right_shoulder": "R_Shoulder",
    "left_elbow": "L_Elbow",
    "right_elbow": "R_Elbow",
    "left_wrist": "L_Wrist",
    "right_wrist": "R_Wrist",
    "left_hip": "L_Hip",
    "right_hip": "R_Hip",
    "left_knee": "L_Knee",
    "right_knee": "R_Knee",
    "left_ankle": "L_Ankle",
    "right_ankle": "R_Ankle",
}


class SimpleFrame:
    def __init__(self, keypoints: np.ndarray, keypoints_format: Dict[str, int]):
        self.keypoints = keypoints
        self.keypoints_format = keypoints_format
        self.flag_valid = True


def load_pickle(path: Path) -> Any:
    with path.open("rb") as f:
        return pickle.load(f)


def normalize_keypoints_format(kp_fmt: Dict[str, int]) -> Dict[str, int]:
    return {COCO_NAME_MAP.get(k, k): v for k, v in kp_fmt.items()}


def build_smpl_init_seq(smpl_fitter_cls: Any, smpl_batch_seq: Any, index_seq: Sequence[int]):
    smpl_seq_init = smpl_fitter_cls.SMPL_Param_Seq(num_batch=len(index_seq))
    for seq_idx, frame_idx in enumerate(index_seq):
        smpl_seq_init.orient[seq_idx] = smpl_batch_seq.orient[frame_idx]
        smpl_seq_init.pose[seq_idx] = smpl_batch_seq.pose[frame_idx]
        smpl_seq_init.transl[seq_idx] = smpl_batch_seq.transl[frame_idx]
    smpl_seq_init.beta = smpl_batch_seq.beta.clone()
    smpl_seq_init.scale = smpl_batch_seq.scale.clone()
    return smpl_seq_init


def extract_hand_keypoints(kp: np.ndarray) -> Optional[np.ndarray]:
    if kp.shape[0] < 133:
        return None
    hand = np.concatenate([kp[LHAND_IDX, :], kp[RHAND_IDX, :]], axis=0)
    return hand if hand[:, 2].mean() > 0.1 else None


def load_rtm_cache(
    rtm_cache: Path,
    index_seq: Sequence[int],
    *,
    return_kp_fmt: bool = False,
):
    det = load_pickle(rtm_cache)

    if hasattr(det, "array_keypoints") and isinstance(det.array_keypoints, np.ndarray):
        kp_all = det.array_keypoints
        kp_fmt = normalize_keypoints_format(
            det.keypoints_format if hasattr(det, "keypoints_format") else {}
        )
        detection_results_list = []
        hand_detection_list = []
        for i in index_seq:
            kp = kp_all[i]
            detection_results_list.append(SimpleFrame(kp, kp_fmt))
            hand_detection_list.append(extract_hand_keypoints(kp))
        if not any(h is not None for h in hand_detection_list):
            hand_detection_list = None
        if return_kp_fmt:
            return detection_results_list, hand_detection_list, kp_fmt
        return detection_results_list, hand_detection_list

    if hasattr(det, "get_result_by_frame_index"):
        if hasattr(det, "keypoints_format"):
            det.keypoints_format = normalize_keypoints_format(det.keypoints_format)
            kp_fmt = det.keypoints_format
        else:
            kp_fmt = {}
        detection_results_list = [det.get_result_by_frame_index(i) for i in index_seq]
        if return_kp_fmt:
            return detection_results_list, None, kp_fmt
        return detection_results_list, None

    raise TypeError(f"Unsupported RTM cache format: {type(det)}")


def ds_project_np(
    points_3d: np.ndarray,
    xi: float,
    alpha: float,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
) -> np.ndarray:
    x = points_3d[..., 0]
    y = points_3d[..., 1]
    z = points_3d[..., 2]
    r2 = x**2 + y**2
    d1 = np.sqrt(r2 + z**2).clip(1e-6)
    d2 = np.sqrt(r2 + (xi * d1 + z) ** 2).clip(1e-6)
    w1 = (alpha * d2 + (1.0 - alpha) * (xi * d1 + z)).clip(1e-6)
    return np.stack([fx * x / w1 + cx, fy * y / w1 + cy], axis=-1)


def erp_project_np(points_3d: np.ndarray, width: int, height: int) -> np.ndarray:
    x = points_3d[..., 0]
    y = points_3d[..., 1]
    z = points_3d[..., 2]
    r = np.linalg.norm(points_3d, axis=-1).clip(1e-6, None)
    lon = np.arctan2(x, z)
    lat = np.arcsin((-y / r).clip(-1 + 1e-6, 1 - 1e-6))
    u = (lon / (2.0 * math.pi) + 0.5) * width
    v = (0.5 - lat / math.pi) * height
    return np.stack([u, v], axis=-1)

# 专门针对光流和one_step_stitch里做的
def erp_project_consistent(
    points_3d: np.ndarray,
    width: int,
    height: int,
    w_offset: float = 0.0,
) -> np.ndarray:
    x = points_3d[..., 0]
    y = points_3d[..., 1]
    z = points_3d[..., 2]
    r = np.linalg.norm(points_3d, axis=-1).clip(1e-8)
    
    phi = np.arcsin((y / r).clip(-1 + 1e-7, 1 - 1e-7))
    theta = np.arctan2(z, x)
    
    us = (theta / np.pi + 1.0 - w_offset) / 2.0
    us = us % 1.0
    vs = phi / np.pi + 0.5
    
    u_pixel = us * width
    v_pixel = vs * height
    return np.stack([u_pixel, v_pixel], axis=-1)

def world_to_cam4(pts_world: np.ndarray, delta_t: np.ndarray = DEFAULT_DELTA_T) -> np.ndarray:
    return pts_world @ R_CAM4.T - delta_t


def cam4_to_world(pts_cam4: np.ndarray, t_cam4: np.ndarray) -> np.ndarray:
    return pts_cam4 @ R_CAM4 + t_cam4


def project_world_to_fisheye(
    pts_world: np.ndarray,
    camera: str,
    *,
    delta_t: Optional[np.ndarray] = None,
    ds_params: Optional[Dict[str, float]] = None,
) -> np.ndarray:
    pts_cam = pts_world if camera == "cam2" else world_to_cam4(
        pts_world,
        DEFAULT_DELTA_T if delta_t is None else delta_t,
    )
    return ds_project_np(pts_cam, **(DS_PARAMS if ds_params is None else ds_params))
