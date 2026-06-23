"""RTM/RTMW 关键点检测结果读取：从 pkl 加载关键点、bbox 和格式信息。"""

import pickle
from pathlib import Path

import numpy as np

from lib_pano.human3d.keypoints_detector import Result_Single_Human_Single_Frame


def infer_keypoints_format_and_dim(frame_indices: list[int], rtm_dir: Path, cam_id: str) -> tuple[dict[str, int], int]:
    """扫描 RTM pkl 序列，推断关键点格式（名称→索引映射）和数量。"""
    cam_suffix = "c2" if cam_id == "cam2" else "c4"
    for frame_idx in frame_indices:
        pkl_path = rtm_dir / cam_id / f"rtm_s{frame_idx:05d}_{cam_suffix}.pkl"
        if not pkl_path.exists():
            continue
        with pkl_path.open("rb") as file:
            seq = pickle.load(file)
        keypoints_format = dict(seq.keypoints_format)
        num_keypoints = int(seq.array_keypoints.shape[1])
        return keypoints_format, num_keypoints
    raise RuntimeError(f"Failed to infer keypoints_format from RTM pkls for {cam_id} reviewed frames")


def rtm_pkl_path(frame_idx: int, rtm_dir: Path, cam_id: str) -> Path:
    """构造 RTM pkl 文件的路径。"""
    cam_suffix = "c2" if cam_id == "cam2" else "c4"
    return rtm_dir / cam_id / f"rtm_s{frame_idx:05d}_{cam_suffix}.pkl"


def is_rtm_det_valid(frame_idx: int, rtm_dir: Path, invalid_frames: set[int], cam_id: str) -> bool:
    """判断该帧的 RTM 关键点检测是否有效（pkl 存在且 array_valid[0] 为 True）。"""
    if frame_idx in invalid_frames:
        return False
    pkl_path = rtm_pkl_path(frame_idx, rtm_dir, cam_id)
    if not pkl_path.exists():
        return False
    try:
        with pkl_path.open("rb") as file:
            seq = pickle.load(file)
        return bool(seq.array_valid[0])
    except Exception:
        return False


def load_det_result(
    frame_idx: int,
    rtm_dir: Path,
    invalid_frames: set[int],
    default_keypoints_format: dict[str, int],
    num_keypoints: int,
    cam_id: str,
) -> Result_Single_Human_Single_Frame:
    """加载单帧 RTM 检测结果，返回统一格式的关键点、bbox 和置信度。"""
    zero_keypoints = None
    if num_keypoints > 0:
        zero_keypoints = np.zeros((num_keypoints, 3), dtype=np.float32)

    if frame_idx in invalid_frames:
        return Result_Single_Human_Single_Frame(
            bbox=None,
            bbox_score=0.0,
            keypoints=zero_keypoints,
            keypoints_format=default_keypoints_format,
        )

    pkl_path = rtm_pkl_path(frame_idx, rtm_dir, cam_id)
    if not pkl_path.exists():
        return Result_Single_Human_Single_Frame(
            bbox=None,
            bbox_score=0.0,
            keypoints=zero_keypoints,
            keypoints_format=default_keypoints_format,
        )

    try:
        with pkl_path.open("rb") as file:
            seq = pickle.load(file)
    except Exception:
        return Result_Single_Human_Single_Frame(
            bbox=None,
            bbox_score=0.0,
            keypoints=zero_keypoints,
            keypoints_format=default_keypoints_format,
        )

    is_valid = bool(seq.array_valid[0])
    det = Result_Single_Human_Single_Frame(
        bbox=tuple(seq.array_bbox_h1h2w1w2[0].tolist()) if is_valid else None,
        bbox_score=float(seq.array_bbox_score[0]),
        keypoints=seq.array_keypoints[0].copy() if is_valid else zero_keypoints.copy(),
        keypoints_format=dict(seq.keypoints_format),
    )
    return det
