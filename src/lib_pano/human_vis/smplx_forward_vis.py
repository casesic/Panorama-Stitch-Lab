"""SMPL-X 前向可视化辅助：RTM pkl 读取与 numpy 兼容、关键点命名归一化、SMPL-X 前向求顶点/关节。"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch

from lib_pano.human_fitting.smplx_sequence import SMPLX_Seq_Forward_Module

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


def _ensure_numpy_pickle_compat() -> None:
    """注册 numpy._core → numpy.core 的模块别名，兼容旧 numpy 版本保存的 pickle。"""
    core_mod = getattr(np, "core", None)
    if core_mod is None:
        return
    sys.modules.setdefault("numpy._core", core_mod)
    for submodule in ("multiarray", "numeric", "umath", "_multiarray_umath"):
        src = f"numpy.core.{submodule}"
        dst = f"numpy._core.{submodule}"
        module = sys.modules.get(src)
        if module is None:
            try:
                module = __import__(src, fromlist=["*"])
            except Exception:
                continue
        sys.modules.setdefault(dst, module)


def _load_pickle(path: Path) -> Any:
    """加载 pickle 文件（先做 numpy 兼容处理）。"""
    _ensure_numpy_pickle_compat()
    with path.open("rb") as f:
        return pickle.load(f)


def normalize_keypoints_format(kp_fmt: Dict[str, int]) -> Dict[str, int]:
    """将 COCO 关键点名映射为 SMPL-X 关节命名规范。"""
    return {COCO_NAME_MAP.get(k, k): v for k, v in kp_fmt.items()}


def unpack_smplx_result(result) -> list:
    """将 SMPL-X 序列或单帧结果统一拆为逐帧参数列表。"""
    if isinstance(result, SMPLX_Seq_Forward_Module.SMPLX_Param_Seq):
        return [result.get_param_at_frame(i) for i in range(result.num_batch)]
    if isinstance(result, SMPLX_Seq_Forward_Module.SMPLX_Param):
        return [result]
    raise TypeError(type(result))


class SMPLXForwarder:
    """SMPL-X 前向封装：持有前向模块，提供模型获取、参数前向到 numpy 与面片读取。"""
    def __init__(self, gender: str = "neutral"):
        """初始化指定性别的 SMPL-X 前向模块。"""
        self._fitter = SMPLX_Seq_Forward_Module(gender=gender)

    def _get_model(self):
        """返回底层 SMPL-X 模型实例。"""
        return self._fitter._get_model()

    def forward_param_to_numpy(self, smplx_param):
        """前向求顶点/关节并转为 numpy，返回 (verts, joints, joints_body)。"""
        with torch.no_grad():
            verts_t, joints_t, joints_body_t = self._fitter.forward_param(smplx_param)
        return verts_t.cpu().numpy(), joints_t.cpu().numpy(), joints_body_t.cpu().numpy()

    def get_model_faces(self):
        """返回 SMPL-X 模型的面片数组（numpy）。"""
        model = self._get_model()
        return model.faces.numpy() if hasattr(model.faces, "numpy") else model.faces
