"""
smpl_runner.py
==============
独立实现的 SMPL 前向推理模块，替代 lib_dynamic_hvip 私有库依赖。
依赖仅需：torch、numpy、smplx（pip install smplx）。

使用前需要设置 SMPL 模型文件路径，有两种方式：
  方式A（推荐）：设置环境变量
      $env:LIB_SMPL_MODEL_DIR = "D:/path/to/smpl_full"
  方式B：直接在下方修改 _DEFAULT_SMPL_DIR

smpl_full 目录结构（与 lib_dynamic_hvip 一致）：
  smpl_full/
    SMPL_NEUTRAL.pkl
    SMPL_MALE.pkl
    SMPL_FEMALE.pkl
    J_regressor_extra.npy   ← 强烈建议提供（9 x 6890），否则用近似值
"""

from __future__ import annotations

import os
import os.path as osp
from typing import Dict, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import smplx

# =============================================================================
# 配置：SMPL 模型路径
#   可以是单个 .pkl 文件路径，也可以是含 SMPL_NEUTRAL.pkl 等文件的目录。
#   查找优先级：
#     1. 环境变量 LIB_SMPL_MODEL_FILE（直接 pkl 文件路径）
#     2. 环境变量 LIB_SMPL_MODEL_DIR（目录）
#     3. 本文件下方的 _DEFAULT_SMPL_DIR 硬编码值
#     4. 自动探测：项目根目录下的 tools/smpl/*.pkl
# =============================================================================
_DEFAULT_SMPL_DIR: str = ""   # 可改为具体路径，例如 r"D:\models\smpl_full"


def _get_smpl_model_path() -> str:
    """按优先级返回 SMPL 模型路径（pkl 文件或目录）。"""
    for env_key in ("LIB_SMPL_MODEL_FILE", "LIB_SMPL_MODEL_DIR"):
        val = os.environ.get(env_key, "").strip()
        if val:
            return val
    if _DEFAULT_SMPL_DIR:
        return _DEFAULT_SMPL_DIR
    # 自动探测：相对于本文件位置的 ../../../tools/smpl/ 目录
    _here = osp.dirname(osp.abspath(__file__))
    auto_dir = osp.abspath(osp.join(_here, '..', '..', '..', 'tools', 'smpl'))
    if osp.isdir(auto_dir):
        # 优先找 neutral pkl
        for f in sorted(os.listdir(auto_dir)):
            if f.lower().endswith('.pkl') and 'neutral' in f.lower():
                return osp.join(auto_dir, f)
        # 退而求其次：找任意 pkl
        for f in sorted(os.listdir(auto_dir)):
            if f.lower().endswith('.pkl'):
                return osp.join(auto_dir, f)
    return ""


# 兼容旧接口（部分地方调用 _get_smpl_dir）
_get_smpl_dir = _get_smpl_model_path


# =============================================================================
# 旋转表示转换工具（来源参考 HMR2/4D-Humans geometry.py，作了泛化处理）
# =============================================================================

def quat_to_rotmat(quat: torch.Tensor) -> torch.Tensor:
    """
    四元数 [w, x, y, z] -> 旋转矩阵。
    quat: (..., 4) -> (..., 3, 3)
    """
    shape = quat.shape[:-1]
    q = F.normalize(quat.reshape(-1, 4), p=2, dim=1)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    w2, x2, y2, z2 = w**2, x**2, y**2, z**2
    R = torch.stack([
        w2 + x2 - y2 - z2,  2*(x*y - w*z),       2*(x*z + w*y),
        2*(x*y + w*z),       w2 - x2 + y2 - z2,  2*(y*z - w*x),
        2*(x*z - w*y),       2*(y*z + w*x),       w2 - x2 - y2 + z2,
    ], dim=1).reshape(-1, 3, 3)
    return R.reshape(*shape, 3, 3)


def aa_to_rotmat(aa: torch.Tensor) -> torch.Tensor:
    """
    轴角 -> 旋转矩阵（经由四元数，避免 arcsin 奇点）。
    aa: (..., 3) -> (..., 3, 3)
    """
    shape = aa.shape[:-1]
    theta = aa.reshape(-1, 3)
    norm = torch.norm(theta + 1e-8, p=2, dim=1, keepdim=True)
    half = norm * 0.5
    quat = torch.cat([torch.cos(half), torch.sin(half) * (theta / norm)], dim=1)
    return quat_to_rotmat(quat).reshape(*shape, 3, 3)


def rotmat_to_aa(R: torch.Tensor) -> torch.Tensor:
    """
    旋转矩阵 -> 轴角（可微分，对近似单位矩阵数值稳定）。
    R: (..., 3, 3) -> (..., 3)
    """
    shape = R.shape[:-2]
    R_flat = R.reshape(-1, 3, 3)

    # skew 的各分量 = sin(θ) * axis_component
    skew_x = (R_flat[:, 2, 1] - R_flat[:, 1, 2]) * 0.5
    skew_y = (R_flat[:, 0, 2] - R_flat[:, 2, 0]) * 0.5
    skew_z = (R_flat[:, 1, 0] - R_flat[:, 0, 1]) * 0.5
    skew = torch.stack([skew_x, skew_y, skew_z], dim=1)  # (N, 3)

    trace = R_flat[:, 0, 0] + R_flat[:, 1, 1] + R_flat[:, 2, 2]
    cos_a = ((trace - 1.0) * 0.5).clamp(-1 + 1e-7, 1 - 1e-7)
    angle = torch.acos(cos_a)                    # (N,)
    sin_a = torch.sin(angle).clamp(min=1e-7)

    # angle/sin(angle): 当 angle→0 时极限为 1
    coeff = torch.where(angle.abs() < 1e-5,
                        torch.ones_like(angle),
                        angle / sin_a)
    return (skew * coeff.unsqueeze(1)).reshape(*shape, 3)


def rot6d_to_rotmat(x: torch.Tensor) -> torch.Tensor:
    """
    6D 旋转表示 -> 旋转矩阵（Zhou et al. CVPR 2019）。
    x: (..., 6) -> (..., 3, 3)
    """
    shape = x.shape[:-1]
    x = x.reshape(-1, 2, 3).permute(0, 2, 1)   # (N, 3, 2)
    a1, a2 = x[:, :, 0], x[:, :, 1]
    b1 = F.normalize(a1, dim=1)
    b2 = F.normalize(a2 - (b1 * a2).sum(dim=1, keepdim=True) * b1, dim=1)
    b3 = torch.cross(b1, b2, dim=1)
    return torch.stack([b1, b2, b3], dim=-1).reshape(*shape, 3, 3)


# =============================================================================
# SMPL_Full：输出 54 个关节点的 SMPL 模型封装
#   [0  - 23]: 标准 SMPL 24 关节（与 SMPL_24 字典对齐）
#   [24 - 44]: 面部 / 脚 / 手指关节（从 mesh 顶点采样，近似值）
#   [45 - 53]: LSP/H36M 风格关节（来自 J_regressor_extra.npy，或近似）
# =============================================================================

# SMPL mesh 上 21 个额外关节的顶点索引（近似值，与 SMPL_EXTRA_30 前 21 项对应）
# 若发现关节位置偏差大，请用 lib_dynamic_hvip 中的精确索引替换
_EXTRA_21_VERT_IDS: list[int] = [
    332,   # 24: Nose
    6260,  # 25: R_Eye
    2800,  # 26: L_Eye
    4071,  # 27: R_Ear
    583,   # 28: L_Ear
    3216,  # 29: L_BigToe
    3226,  # 30: L_SmallToe
    3387,  # 31: L_Heel
    6617,  # 32: R_BigToe
    6624,  # 33: R_SmallToe
    6787,  # 34: R_Heel
    2746,  # 35: L_Hand_thumb
    2319,  # 36: L_Hand_index
    2445,  # 37: L_Hand_middle
    2556,  # 38: L_Hand_ring
    2673,  # 39: L_Hand_pinky
    6191,  # 40: R_Hand_thumb
    5782,  # 41: R_Hand_index
    5905,  # 42: R_Hand_middle
    6016,  # 43: R_Hand_ring
    6133,  # 44: R_Hand_pinky
]
assert len(_EXTRA_21_VERT_IDS) == 21


class SMPL_Full(nn.Module):
    """
    SMPL 模型封装，对外接口与 lib_dynamic_hvip.smpl.smpl_full_54.SMPL_Full 相同：
        forward(poses, betas, pose2rot=True) -> {'verts': (B,6890,3), 'j3d': (B,54,3)}
    """

    def __init__(
        self,
        model_path: str,
        J_reg_extra9_path: str = "",
        str_version: str = "v100",
        gender: str = "neutral",
    ) -> None:
        super().__init__()
        self.gender = gender
        # v100 对应 10 个 beta，v110 对应 300 个（需对应模型文件支持）
        self.num_betas = 10 if str_version == "v100" else 300
        num_betas_smplx = min(self.num_betas, 10)   # smplx.SMPL 通常最多支持 10

        is_file = osp.isfile(model_path) and model_path.lower().endswith('.pkl')
        if is_file:
            # 直接传入 pkl 文件，smplx 会忽略 gender 参数直接加载
            if gender != 'neutral':
                print(f"[smpl_runner] 单文件模式：忽略 gender='{gender}'，直接加载 {osp.basename(model_path)}")
            self.smpl = smplx.SMPL(
                model_path=model_path,
                num_betas=num_betas_smplx,
            )
        elif osp.isdir(model_path):
            self.smpl = smplx.SMPL(
                model_path=model_path,
                gender=gender,
                num_betas=num_betas_smplx,
            )
        else:
            raise FileNotFoundError(
                f"SMPL 模型路径不存在：{model_path}\n"
                "请提供 pkl 文件路径或含 SMPL_NEUTRAL.pkl 的目录。"
            )

        # 额外 21 个关节：顶点索引采样
        self.register_buffer(
            "_extra21_vids",
            torch.tensor(_EXTRA_21_VERT_IDS, dtype=torch.long),
        )

        # 额外 9 个关节：J_regressor_extra.npy（shape: 9×6890）
        self._has_jreg9 = False
        if J_reg_extra9_path and osp.isfile(J_reg_extra9_path):
            jreg = np.load(J_reg_extra9_path)
            if jreg.shape != (9, 6890):
                raise ValueError(
                    f"J_regressor_extra.npy 形状应为 (9, 6890)，实际为 {jreg.shape}"
                )
            self.register_buffer("_jreg9", torch.from_numpy(jreg).float())
            self._has_jreg9 = True
        else:
            print(
                "[smpl_runner] 未找到 J_regressor_extra.npy，"
                "关节 45-53（R_Hip, L_Hip, Neck_LSP...）将使用近似值。"
            )

    # ------------------------------------------------------------------
    def forward(
        self,
        poses: torch.Tensor,
        betas: torch.Tensor,
        pose2rot: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            poses   : (B, 72)   轴角，当 pose2rot=True；
                      (B, 216)  展平旋转矩阵，当 pose2rot=False
            betas   : (B, num_betas) 形状参数
            pose2rot: True -> poses 为轴角；False -> poses 为旋转矩阵
        Returns:
            {'verts': (B, 6890, 3), 'j3d': (B, 54, 3)}
        """
        B = poses.shape[0]

        if not pose2rot:
            # 将旋转矩阵转回轴角，再调用 smplx（避免 smplx 版本兼容问题）
            R = poses.reshape(B, 24, 3, 3)
            aa = rotmat_to_aa(R.reshape(-1, 3, 3)).reshape(B, 24, 3)
            global_orient = aa[:, 0, :]        # (B, 3)
            body_pose = aa[:, 1:, :].reshape(B, 69)  # (B, 69)
        else:
            assert poses.shape == (B, 72), f"当 pose2rot=True 时 poses 应为 (B,72)，实际 {poses.shape}"
            global_orient = poses[:, :3]       # (B, 3)
            body_pose = poses[:, 3:]           # (B, 69)

        smpl_out = self.smpl(
            global_orient=global_orient,
            body_pose=body_pose,
            betas=betas[:, :10],
        )

        verts = smpl_out.vertices               # (B, 6890, 3)
        joints_24 = smpl_out.joints[:, :24, :]  # (B, 24, 3) — 标准 SMPL 24 关节

        # 关节 24-44：顶点采样
        joints_21 = verts[:, self._extra21_vids, :]  # (B, 21, 3)

        # 关节 45-53：J_regressor_extra 或近似
        if self._has_jreg9:
            joints_9 = torch.einsum("jv,bvd->bjd", self._jreg9, verts)  # (B, 9, 3)
        else:
            joints_9 = self._approx_lsp_joints(joints_24)

        joints_54 = torch.cat([joints_24, joints_21, joints_9], dim=1)  # (B, 54, 3)
        assert joints_54.shape == (B, 54, 3), joints_54.shape

        return {"verts": verts, "j3d": joints_54}

    def _approx_lsp_joints(self, j24: torch.Tensor) -> torch.Tensor:
        """
        从标准 24 关节近似推算 9 个 LSP/H36M 关节（索引 45-53）。
        SMPL_24: Pelvis=0, L_Hip_SMPL=1, R_Hip_SMPL=2, Thorax_SMPL=6, Neck=12, Jaw=15
        """
        dev = j24.device
        offset_head = torch.tensor([[[0.0, 0.12, 0.0]]], device=dev)

        r_hip     = j24[:, 2:3]                                    # 45: R_Hip
        l_hip     = j24[:, 1:2]                                    # 46: L_Hip
        neck_lsp  = j24[:, 12:13]                                  # 47: Neck_LSP
        head_top  = j24[:, 15:16] + offset_head                   # 48: Head_top（Jaw 上方）
        pelvis    = j24[:, 0:1]                                    # 49: Pelvis
        thorax    = j24[:, 6:7]                                    # 50: Thorax_MPII
        spine     = (j24[:, 0:1] + j24[:, 6:7]) * 0.5            # 51: Spine_H36M
        jaw       = j24[:, 15:16]                                  # 52: Jaw_H36M
        head      = (j24[:, 15:16] + head_top) * 0.5              # 53: Head

        return torch.cat(
            [r_hip, l_hip, neck_lsp, head_top, pelvis, thorax, spine, jaw, head], dim=1
        )  # (B, 9, 3)


# =============================================================================
# 公开接口：与 smpl_lib.py 中同名函数保持一致
# =============================================================================

def get_smpl_full(version: str = "v100", gender: str = "neutral") -> SMPL_Full:
    """
    加载并返回 SMPL_Full 模型实例。

    路径查找（按优先级）：
      1. 环境变量 LIB_SMPL_MODEL_FILE  —— 直接 pkl 文件路径
      2. 环境变量 LIB_SMPL_MODEL_DIR   —— 含 SMPL_NEUTRAL.pkl 等的目录
      3. smpl_runner.py 顶部的 _DEFAULT_SMPL_DIR
      4. 自动探测：项目根目录 tools/smpl/*.pkl（已自动检测到当前工作区）

    可选：在模型同一目录放 J_regressor_extra.npy（9×6890）可提升关节精度。
    """
    model_path = _get_smpl_model_path()
    if not model_path or (not osp.isfile(model_path) and not osp.isdir(model_path)):
        _here = osp.dirname(osp.abspath(__file__))
        auto_dir = osp.abspath(osp.join(_here, '..', '..', '..', 'tools', 'smpl'))
        raise RuntimeError(
            f"找不到 SMPL 模型：'{model_path}'\n"
            f"自动探测路径：{auto_dir}\n"
            "解决方式（任选一）：\n"
            "  A. 将模型 pkl 放入上述目录（已自动探测）\n"
            "  B. $env:LIB_SMPL_MODEL_FILE = 'D:/path/to/model.pkl'\n"
            "  C. 修改 smpl_runner.py 顶部的 _DEFAULT_SMPL_DIR"
        )

    print(f"[smpl_runner] 加载 SMPL 模型：{model_path}")
    # J_regressor_extra.npy 与 pkl 文件同目录
    model_dir = osp.dirname(model_path) if osp.isfile(model_path) else model_path
    j_reg_path = osp.join(model_dir, "J_regressor_extra.npy")
    return SMPL_Full(
        model_path=model_path,
        J_reg_extra9_path=j_reg_path,
        str_version=version,
        gender=gender,
    )


def forward_smpl_full_tensor(
    smpl_model: SMPL_Full,
    poses: torch.Tensor,
    shapes: torch.Tensor,
    trans: torch.Tensor = torch.zeros(3),
    scale: Union[torch.Tensor, float] = 1.0,
    ret_joints_54: bool = False,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """
    单帧可微分 SMPL 前向推理。

    Args:
        smpl_model : SMPL_Full 实例
        poses      : (72,) 轴角
        shapes     : (10,) 或 (num_betas,) 形状参数
        trans      : (3,) 平移
        scale      : 缩放标量
        ret_joints_54: 是否同时返回 54 个关节点
    Returns:
        verts (6890,3)，或 (verts, joints_54) 当 ret_joints_54=True
    """
    assert poses.shape == (72,), poses.shape
    assert trans.shape == (3,), trans.shape

    if isinstance(scale, float):
        scale_t = torch.tensor([scale], dtype=torch.float32, device=poses.device)
    else:
        scale_t = scale.reshape(1)

    out = smpl_model.forward(poses=poses[None], betas=shapes[None, :10])
    verts    = out["verts"][0] * scale_t + trans    # (6890, 3)
    joints54 = out["j3d"][0]   * scale_t + trans    # (54, 3)

    return (verts, joints54) if ret_joints_54 else verts


def forward_smpl_full_tensor_batch(
    smpl_model: SMPL_Full,
    poses: torch.Tensor,
    shapes: torch.Tensor,
    trans: torch.Tensor,
    scale: Optional[torch.Tensor] = None,
    ret_joints_54: bool = False,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """
    批量可微分 SMPL 前向推理（支持梯度回传，用于 LBFGS 优化）。

    Args:
        smpl_model : SMPL_Full 实例
        poses      : (B, 72) 轴角，或 (B, 24,3,3) / (B, 216) 旋转矩阵
        shapes     : (B, 10)
        trans      : (B, 3) 平移
        scale      : (B,) 可选缩放；None 时不缩放
        ret_joints_54: 是否同时返回 54 个关节点
    Returns:
        verts (B,6890,3)，或 (verts, joints_54) 当 ret_joints_54=True
    """
    B = poses.shape[0]

    if poses.shape == (B, 72):
        pose2rot = True
    elif poses.shape == (B, 24, 3, 3):
        poses = poses.reshape(B, 216)
        pose2rot = False
    elif poses.shape == (B, 216):
        pose2rot = False
    else:
        raise ValueError(f"不支持的 poses 形状：{poses.shape}")

    assert shapes.shape == (B, 10) or shapes.shape[0] == B, shapes.shape
    assert trans.shape == (B, 3), trans.shape

    out = smpl_model.forward(poses=poses, betas=shapes[:, :10], pose2rot=pose2rot)
    verts    = out["verts"]    # (B, 6890, 3)
    joints54 = out["j3d"]     # (B, 54, 3)

    if scale is not None:
        s = scale.reshape(B, 1, 1)
        verts    = verts    * s
        joints54 = joints54 * s

    t = trans.reshape(B, 1, 3)
    verts    = verts    + t
    joints54 = joints54 + t

    return (verts, joints54) if ret_joints_54 else verts
