"""SMPL/SMPL-X 初始值构建：从 4D-Humans pkl 加载 HMR 结果，转换为 SMPL-X 并做帧间插值。"""

import importlib
import pickle
from pathlib import Path
from typing import Any

import torch

from lib_pano.human_fitting.smplx_sequence import SMPLX_Seq_Forward_Module
from lib_pano.human_fitting import init_smplx_from_smpl54_seq, slice_smplx_seq


def _build_smpl_init_from_single_frame_pkl(pkl_path: Path):
    """从单个 4D-Humans SMPL pkl 文件构建 SMPl_Param_Seq 对象。"""
    smpl_mod = importlib.import_module("lib_dynamic_hvip.fitting.smpl_seq_forward_module")
    SMPLFitter = smpl_mod.SMPL_Seq_Forward_Module

    with pkl_path.open("rb") as file:
        src = pickle.load(file)

    if isinstance(src, SMPLFitter.SMPL_Param_Seq):
        return src
    if not isinstance(src, dict):
        raise TypeError(f"Unsupported HMR pkl object type: {type(src)}")

    num_batch = int(src["num_batch"])
    dim_shape = int(src["beta"].shape[-1])
    gender = str(src.get("gender", "neutral"))

    smpl_init = SMPLFitter.SMPL_Param_Seq(dim_shape=dim_shape, gender=gender, num_batch=num_batch)
    smpl_init.orient = src["orient"].clone()
    smpl_init.transl = src["transl"].clone()
    smpl_init.pose = src["pose"].clone()
    smpl_init.beta = src["beta"].clone().reshape(-1)
    smpl_init.scale = src["scale"].clone().reshape(-1)[:1]
    smpl_init.valid = src.get("valid", torch.ones(num_batch, dtype=torch.bool)).clone()
    return smpl_init


def load_smplx_init(frame_idx: int, hmr_dir: Path, cam_id: str):
    """加载单帧 HMR 结果并转换为 SMPL-X 初始值。"""
    cam_suffix = "c2" if cam_id == "cam2" else "c4"
    pkl_path = hmr_dir / cam_id / f"hmr_s{frame_idx:05d}_{cam_suffix}.pkl"
    if not pkl_path.exists():
        return None

    smpl_seq = _build_smpl_init_from_single_frame_pkl(pkl_path)
    return init_smplx_from_smpl54_seq(smpl_seq, [0])


def try_load_smplx_init(frame_idx: int, hmr_dir: Path, hmr_invalid_frames: set[int], cam_id: str):
    """加载 SMPL-X 初始值，若帧在 invalid 集中或加载失败则返回 None。"""
    if frame_idx in hmr_invalid_frames:
        return None
    try:
        return load_smplx_init(frame_idx, hmr_dir, cam_id)
    except Exception:
        return None


def blend_smplx_init(left, right, alpha: float):
    """对两帧 SMPL-X 参数做线性插值，alpha=0 取 left，alpha=1 取 right。"""
    if left is None:
        return right
    if right is None:
        return left

    alpha = float(max(0.0, min(1.0, alpha)))
    out = slice_smplx_seq(left, [0])
    for attr in ("orient", "transl", "body_pose", "left_hand_pose", "right_hand_pose"):
        if hasattr(out, attr) and hasattr(left, attr) and hasattr(right, attr):
            getattr(out, attr)[0] = (1.0 - alpha) * getattr(left, attr)[0] + alpha * getattr(right, attr)[0]
    for attr in ("beta", "scale"):
        if hasattr(out, attr) and hasattr(left, attr) and hasattr(right, attr):
            setattr(out, attr, (1.0 - alpha) * getattr(left, attr).clone() + alpha * getattr(right, attr).clone())
    return out


def build_hmr_init_for_frame(frame_idx: int, original_segment: list[int], direct_by_frame: dict[int, Any]):
    """为指定帧构建 SMPL-X 初始值：优先直接 HMR 结果，其次相邻帧插值，否则 T-pose。"""
    direct = direct_by_frame.get(frame_idx)
    if direct is not None:
        return direct, "hmr"

    left_frames = [idx for idx in original_segment if idx < frame_idx and direct_by_frame.get(idx) is not None]
    right_frames = [idx for idx in original_segment if idx > frame_idx and direct_by_frame.get(idx) is not None]
    left_idx = left_frames[-1] if left_frames else None
    right_idx = right_frames[0] if right_frames else None

    if left_idx is not None and right_idx is not None:
        denom = max(1, right_idx - left_idx)
        alpha = (frame_idx - left_idx) / denom
        return blend_smplx_init(direct_by_frame[left_idx], direct_by_frame[right_idx], alpha), "interpolated"
    if left_idx is not None:
        return slice_smplx_seq(direct_by_frame[left_idx], [0]), "neighbor"
    if right_idx is not None:
        return slice_smplx_seq(direct_by_frame[right_idx], [0]), "neighbor"
    return None, "tpose"


def build_segment_smplx_init(smplx_list_c2, smplx_list_c4):
    """将 cam2/cam4 的 SMPL-X 参数列表拼装成 SMPLX_Param_Seq 序列，缺失帧用 fallback 填充。"""
    def _build_single(smplx_list):
        fallback = next((item for item in smplx_list if item is not None), None)
        if fallback is None:
            return None

        if len(smplx_list) == 1 and smplx_list[0] is not None:
            return slice_smplx_seq(smplx_list[0], [0])

        n = len(smplx_list)
        out = SMPLX_Seq_Forward_Module.SMPLX_Param_Seq(num_batch=n, gender="neutral")
        for i, smplx_item in enumerate(smplx_list):
            src = smplx_item if smplx_item is not None else fallback
            out.orient[i] = src.orient[0]
            out.transl[i] = src.transl[0]
            out.body_pose[i] = src.body_pose[0]
            out.left_hand_pose[i] = src.left_hand_pose[0]
            out.right_hand_pose[i] = src.right_hand_pose[0]
        out.beta = fallback.beta.clone()
        out.scale = fallback.scale.clone()
        return out

    return _build_single(smplx_list_c2), _build_single(smplx_list_c4)
