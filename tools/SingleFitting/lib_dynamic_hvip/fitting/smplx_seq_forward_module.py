"""
smplx_seq_forward_module.py
----------------------------
SMPL-X 版本的多帧序列拟合模块，支持手指关节。

与 smpl_seq_forward_module.py 的区别：
  - 使用 SMPLXLayer 代替 SMPL_Full
  - 参数结构新增 left_hand_pose / right_hand_pose (各 45 维)
  - 2D loss 支持 COCO-133 手部关键点
  - 投影模型支持 equirect（全景图）

依赖：
  - smpl_lib.get_smplx_batch
  - smpl_lib.forward_smplx_tensor_batch
  - smpl_seq_forward_module.SMPL_Seq_Forward_Module.erp_project（复用）
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F

from lib_dynamic_hvip.smpl.smpl_lib import (
    get_smplx_batch,
    forward_smplx_tensor_batch,
)
from lib_dynamic_hvip.detection.engine.base_engine import Base_Detect_Engine



# SMPL-X 关节名 → 索引（前 22 个身体关节 + 手指）
# joints 输出顺序参考 smplx 官方文档
SMPLX_BODY_22 = {
    'Pelvis': 0, 'L_Hip': 1, 'R_Hip': 2, 'Spine1': 3,
    'L_Knee': 4, 'R_Knee': 5, 'Spine2': 6,
    'L_Ankle': 7, 'R_Ankle': 8, 'Spine3': 9,
    'L_Foot': 10, 'R_Foot': 11, 'Neck': 12,
    'L_Collar': 13, 'R_Collar': 14, 'Head': 15,
    'L_Shoulder': 16, 'R_Shoulder': 17,
    'L_Elbow': 18, 'R_Elbow': 19,
    'L_Wrist': 20, 'R_Wrist': 21,
    'Nose':  55,
    'L_Eye': 57,
    'R_Eye': 56,
    'L_Ear': 59,
    'R_Ear': 58,
}

SMPLX_JOINT_MAP = {
    # 身体 22 个
    'Pelvis': 0, 
    'L_Hip': 1, 'R_Hip': 2, 'Spine1': 3,
    'L_Knee': 4, 'R_Knee': 5, 'Spine2': 6,
    'L_Ankle': 7, 'R_Ankle': 8, 'Spine3': 9,
    'L_Foot': 10, 'R_Foot': 11, 'Neck': 12,
    'L_Collar': 13, 'R_Collar': 14, 'Head': 15,
    'L_Shoulder': 16, 'R_Shoulder': 17,
    'L_Elbow': 18, 'R_Elbow': 19,
    'L_Wrist': 20, 'R_Wrist': 21,
    # 面部 5 个（对应 COCO Nose/Eye/Ear）
    'Nose':  55,
    'R_Eye': 56,
    'L_Eye': 57,
    'R_Ear': 58,
    'L_Ear': 59,
    # 脚部 6 个（对应 COCO-133 脚部关键点）
    'L_BigToe':   60, 'L_SmallToe': 61, 'L_Heel': 62,
    'R_BigToe':   63, 'R_SmallToe': 64, 'R_Heel': 65,
}
# 左手 15 关节（index 25~39 in smplx joints）
SMPLX_LHAND_15 = {
    'L_Index1': 25, 'L_Index2': 26, 'L_Index3': 27,
    'L_Middle1': 28, 'L_Middle2': 29, 'L_Middle3': 30,
    'L_Pinky1': 31, 'L_Pinky2': 32, 'L_Pinky3': 33,
    'L_Ring1': 34, 'L_Ring2': 35, 'L_Ring3': 36,
    'L_Thumb1': 37, 'L_Thumb2': 38, 'L_Thumb3': 39,
}

# 右手 15 关节（index 40~54 in smplx joints）
SMPLX_RHAND_15 = {
    'R_Index1': 40, 'R_Index2': 41, 'R_Index3': 42,
    'R_Middle1': 43, 'R_Middle2': 44, 'R_Middle3': 45,
    'R_Pinky1': 46, 'R_Pinky2': 47, 'R_Pinky3': 48,
    'R_Ring1': 49, 'R_Ring2': 50, 'R_Ring3': 51,
    'R_Thumb1': 52, 'R_Thumb2': 53, 'R_Thumb3': 54,
}

SMPLX_ALL = {**SMPLX_JOINT_MAP, **SMPLX_LHAND_15, **SMPLX_RHAND_15}

# COCO-133 手部关键点名称 → 索引（rtmlib 输出格式）
# 左手：index 91~111，右手：112~132（COCO-133 官方顺序）
# 这里给出关节名到 COCO-133 索引的映射
COCO133_LHAND = {
    'L_Hand_wrist':   91,
    'L_Hand_thumb1':  92, 'L_Hand_thumb2':  93, 'L_Hand_thumb3':  94, 'L_Hand_thumb4':  95,
    'L_Hand_index1':  96, 'L_Hand_index2':  97, 'L_Hand_index3':  98, 'L_Hand_index4':  99,
    'L_Hand_middle1': 100,'L_Hand_middle2': 101,'L_Hand_middle3': 102,'L_Hand_middle4': 103,
    'L_Hand_ring1':   104,'L_Hand_ring2':   105,'L_Hand_ring3':   106,'L_Hand_ring4':   107,
    'L_Hand_pinky1':  108,'L_Hand_pinky2':  109,'L_Hand_pinky3':  110,'L_Hand_pinky4':  111,
}
COCO133_RHAND = {
    'R_Hand_wrist':   112,
    'R_Hand_thumb1':  113,'R_Hand_thumb2':  114,'R_Hand_thumb3':  115,'R_Hand_thumb4':  116,
    'R_Hand_index1':  117,'R_Hand_index2':  118,'R_Hand_index3':  119,'R_Hand_index4':  120,
    'R_Hand_middle1': 121,'R_Hand_middle2': 122,'R_Hand_middle3': 123,'R_Hand_middle4': 124,
    'R_Hand_ring1':   125,'R_Hand_ring2':   126,'R_Hand_ring3':   127,'R_Hand_ring4':   128,
    'R_Hand_pinky1':  129,'R_Hand_pinky2':  130,'R_Hand_pinky3':  131,'R_Hand_pinky4':  132,
}


class SMPLX_Seq_Forward_Module:
    """
    SMPL-X 多帧序列拟合模块（ERP 全景图版本）。

    核心接口：
        fit_from_multi_frame_seq_erp(...)  → SMPLX_Param_Seq
    """

    # ------------------------------------------------------------------
    # 参数数据类
    # ------------------------------------------------------------------

    class SMPLX_Param:
        """单帧 SMPL-X 参数。"""
        def __init__(self, gender: str = 'neutral'):
            self.orient          = torch.zeros(3)         # 全局朝向 axis-angle
            self.transl          = torch.zeros(3)         # 全局位移
            self.body_pose       = torch.zeros(63)        # 21 × 3 身体关节
            self.left_hand_pose  = torch.zeros(45)        # 15 × 3 左手
            self.right_hand_pose = torch.zeros(45)        # 15 × 3 右手
            self.beta            = torch.zeros(10)        # 体型参数
            self.scale           = torch.ones(1)
            self.gender          = gender

    class SMPLX_Param_Seq:
        """多帧 SMPL-X 参数序列。beta/scale 共享，pose/transl/orient 每帧独立。"""
        def __init__(self, num_batch: int = 1, gender: str = 'neutral'):
            self.num_batch       = num_batch
            self.orient          = torch.zeros(num_batch, 3)
            self.transl          = torch.zeros(num_batch, 3)
            self.body_pose       = torch.zeros(num_batch, 63)
            self.left_hand_pose  = torch.zeros(num_batch, 45)
            self.right_hand_pose = torch.zeros(num_batch, 45)
            self.beta            = torch.zeros(10)        # 序列共享
            self.scale           = torch.ones(1)          # 序列共享
            self.gender          = gender

        def get_param_at_frame(self, idx: int) -> 'SMPLX_Seq_Forward_Module.SMPLX_Param':
            p = SMPLX_Seq_Forward_Module.SMPLX_Param(gender=self.gender)
            p.orient          = self.orient[idx]
            p.transl          = self.transl[idx]
            p.body_pose       = self.body_pose[idx]
            p.left_hand_pose  = self.left_hand_pose[idx]
            p.right_hand_pose = self.right_hand_pose[idx]
            p.beta            = self.beta
            p.scale           = self.scale
            return p

    # ------------------------------------------------------------------
    # 初始化
    # ------------------------------------------------------------------

    def __init__(self, gender: str = 'neutral'):
        self.gender = gender
        # batch_size 在 fit 时动态重建，这里先加载 batch=1 用于单帧 forward
        self._model_cache: Dict[int, object] = {}

    def _get_model(self, batch_size: int):
        """按 batch_size 缓存 SMPLXLayer，避免重复加载。"""
        if batch_size not in self._model_cache:
            if 'model' not in self._model_cache:
                self._model_cache['model'] = get_smplx_batch(gender=self.gender)
            return self._model_cache['model']
        return self._model_cache[batch_size]

    # ------------------------------------------------------------------
    # ERP 球面投影（复用 SMPL_Seq_Forward_Module 里的逻辑）
    # ------------------------------------------------------------------

    @staticmethod
    def erp_project(points_3d: torch.Tensor, W: int, H: int) -> torch.Tensor:
        """
        等距柱状投影。
        points_3d: (..., 3)，SMPL 坐标系（Y朝上）
        返回: (..., 2) 像素坐标 (u, v)，图像 Y 朝下
        """
        X = points_3d[..., 0]
        Y = points_3d[..., 1]
        Z = points_3d[..., 2]
        r = torch.norm(points_3d, dim=-1).clamp_min(1e-6)
        lon = torch.atan2(X, Z)
        lat = torch.asin((-Y / r).clamp(-1 + 1e-6, 1 - 1e-6))  # -Y：3D Y朝上→图像Y朝下
        u = (lon / (2.0 * math.pi) + 0.5) * W
        v = (0.5 - lat / math.pi) * H
        return torch.stack([u, v], dim=-1)

    # ------------------------------------------------------------------
    # 单帧 forward（用于可视化）
    # ------------------------------------------------------------------

    def forward_param(self, param: 'SMPLX_Seq_Forward_Module.SMPLX_Param'):
        """单帧 forward，返回 (verts, joints_144, joints_body_22)。"""
        model = self._get_model(batch_size=1)
        verts, joints, joints_body = forward_smplx_tensor_batch(
            smplx_model=model,
            global_orient=param.orient.unsqueeze(0),
            body_pose=param.body_pose.unsqueeze(0),
            left_hand_pose=param.left_hand_pose.unsqueeze(0),
            right_hand_pose=param.right_hand_pose.unsqueeze(0),
            betas=param.beta.unsqueeze(0),
            transl=param.transl.unsqueeze(0),
            scale=param.scale.unsqueeze(0),
        )
        return verts[0], joints[0], joints_body[0]

    # ------------------------------------------------------------------
    # 2D loss（ERP）
    # ------------------------------------------------------------------

    def compute_loss_2d_erp(
        self,
        W: int,
        H: int,
        joints_all: torch.Tensor,        # (B, 144, 3)
        keypoints_detected: torch.Tensor, # (B, J, 3) (u,v,conf)
        keypoints_format: Dict[str, int],
        keypoints_weight: Optional[Dict[str, float]] = None,
        hand_keypoints_detected: Optional[torch.Tensor] = None,  # (B, 42, 3)
        hand_weight: float = 2.0,
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """
        计算 ERP 2D 重投影 loss，分身体和手部两部分。

        keypoints_format: 已 normalize 的 COCO 关节名 → 检测数组索引
        hand_keypoints_detected: (B, 42, 3)，前21个左手，后21个右手
                                  如果 None 则跳过手部 loss
        """
        if keypoints_weight is None:
            keypoints_weight = {}

        B = joints_all.shape[0]
        # J 用实际 tensor 的列数，不用 len(keypoints_format)
        # 因为 COCO-133 的 format 里面部关键点可能没有全部命名
        J_actual = keypoints_detected.shape[1]
        assert joints_all.shape[0] == B and joints_all.shape[2] == 3, joints_all.shape
        assert keypoints_detected.shape == (B, J_actual, 3), keypoints_detected.shape

        # ---- 投影所有关节 ----
        joints_2d_all = self.erp_project(joints_all, W, H)  # (B, 144, 2)

        losses = {}

        # ---- 身体关节 loss（COCO-17 对应 SMPLX_BODY_22）----
        idx_smplx, idx_kp, w_list = [], [], []
        for name, kp_idx in keypoints_format.items():
            if name in SMPLX_JOINT_MAP:
                idx_smplx.append(SMPLX_JOINT_MAP[name])
                idx_kp.append(kp_idx)
                w_list.append(keypoints_weight.get(name, 1.0))

        if len(idx_smplx) > 0:
            dev = joints_all.device
            idx_smplx_t = torch.tensor(idx_smplx, dtype=torch.long, device=dev)
            idx_kp_t    = torch.tensor(idx_kp,    dtype=torch.long, device=dev)
            w_t         = torch.tensor(w_list,    dtype=torch.float32, device=dev)
            w_t         = w_t / (w_t.sum() + 1e-8)

            proj_body = joints_2d_all[:, idx_smplx_t, :]   # (B, J_body, 2)
            det_body  = keypoints_detected[:, idx_kp_t, :]  # (B, J_body, 3)

            loss_body = torch.norm(proj_body - det_body[:, :, :2], dim=2)  # (B, J_body)
            loss_body = (loss_body * det_body[:, :, 2] * w_t.unsqueeze(0)).sum(dim=1).mean()
            losses['body_2d_det'] = loss_body

        # ---- 手部关节 loss ----
        if hand_keypoints_detected is not None:
            assert hand_keypoints_detected.shape == (B, 42, 3), hand_keypoints_detected.shape

            # 左手：SMPLX index 25~39，检测 0~20
            lhand_smplx_idx = list(range(25, 40))  # 15个
            rhand_smplx_idx = list(range(40, 55))  # 15个

            # 检测点：每根手指4个点（wrist + 3节），对应 SMPL-X 的3节
            # 简单策略：用指尖（每4个点的最后一个）对应 SMPL-X 末端关节
            # 左手指尖 index（在 COCO-133 中去掉 wrist 后）：
            # thumb:3, index:7, middle:11, ring:15, pinky:19（0-based，wrist在0）
            FINGER_TIP_IDX = [4, 8, 12, 16, 20]  # 在 21 点中的指尖位置

            # 对应 SMPLX 手指末端关节（每根手指3个关节，取第3个）
            # L: Thumb3=39, Index3=27, Middle3=30, Ring3=36, Pinky3=33
            LHAND_TIP_SMPLX = [39, 27, 30, 36, 33]
            RHAND_TIP_SMPLX = [54, 42, 45, 51, 48]

            dev = joints_all.device
            lhand_det = hand_keypoints_detected[:, :21, :]   # (B, 21, 3)
            rhand_det = hand_keypoints_detected[:, 21:, :]   # (B, 21, 3)

            l_tip_smplx = torch.tensor(LHAND_TIP_SMPLX, dtype=torch.long, device=dev)
            r_tip_smplx = torch.tensor(RHAND_TIP_SMPLX, dtype=torch.long, device=dev)
            tip_det_idx = torch.tensor(FINGER_TIP_IDX,  dtype=torch.long, device=dev)

            proj_l = joints_2d_all[:, l_tip_smplx, :]       # (B, 5, 2)
            proj_r = joints_2d_all[:, r_tip_smplx, :]       # (B, 5, 2)
            det_l  = lhand_det[:, tip_det_idx, :]            # (B, 5, 3)
            det_r  = rhand_det[:, tip_det_idx, :]            # (B, 5, 3)

            loss_lhand = (torch.norm(proj_l - det_l[:, :, :2], dim=2) * det_l[:, :, 2]).mean()
            loss_rhand = (torch.norm(proj_r - det_r[:, :, :2], dim=2) * det_r[:, :, 2]).mean()
            losses['hand_2d_det'] = (loss_lhand + loss_rhand) * hand_weight

        return losses, joints_2d_all

    # ------------------------------------------------------------------
    # 主优化函数
    # ------------------------------------------------------------------

    def fit_from_multi_frame_seq_erp(
        self,
        detection_results_list: List[Base_Detect_Engine.Result_Single_Human_Single_Frame],
        image_wh: Tuple[int, int],
        smpl_param_init: Optional['SMPLX_Seq_Forward_Module.SMPLX_Param_Seq'] = None,
        hand_detection_list: Optional[List[np.ndarray]] = None,  # List of (42, 3) or None
    ) -> 'SMPLX_Seq_Forward_Module.SMPLX_Param_Seq':
        """
        多帧 SMPL-X ERP 拟合。

        detection_results_list : 每帧的身体检测结果（COCO-17 或更多）
        image_wh               : (W, H) 全景图分辨率
        smpl_param_init        : 可选的初始参数（可从 SMPL-54 结果转换过来）
        hand_detection_list    : 每帧的手部检测，每个元素 shape=(42,3)，
                                 前21列左手，后21列右手。None 表示该帧无手部数据。
        """
        assert len(detection_results_list) > 0
        n_frame = len(detection_results_list)
        erp_W, erp_H = image_wh

        # ---- 加载模型 ----
        model = self._get_model(batch_size=n_frame)

        # ---- 准备 keypoints tensor ----
        keypoints_list, key_fmt = [], None
        for dr in detection_results_list:
            kp = torch.from_numpy(dr.keypoints).float()
            keypoints_list.append(kp)
            if key_fmt is None:
                key_fmt = dr.keypoints_format
        keypoints_tensor = torch.stack(keypoints_list, dim=0)  # (N, J, 3)

        # ---- 准备手部 keypoints tensor ----
        has_hand = hand_detection_list is not None and any(
            h is not None for h in hand_detection_list
        )
        if has_hand:
            hand_tensors = []
            for h in hand_detection_list:
                if h is not None:
                    hand_tensors.append(torch.from_numpy(h).float())
                else:
                    hand_tensors.append(torch.zeros(42, 3))
            hand_tensor = torch.stack(hand_tensors, dim=0)  # (N, 42, 3)
        else:
            hand_tensor = None

        # ---- 初始化参数 ----
        if smpl_param_init is not None:
            smpl_param = smpl_param_init
        else:
            smpl_param = SMPLX_Seq_Forward_Module.SMPLX_Param_Seq(
                num_batch=n_frame, gender=self.gender
            )

        # shape anchor（防止 beta 漂移太远）
        shape_anchor = smpl_param.beta.clone().detach()

        # 设置可优化变量
        smpl_param.orient          = smpl_param.orient.clone().detach().requires_grad_(True)
        smpl_param.transl          = smpl_param.transl.clone().detach().requires_grad_(True)
        smpl_param.body_pose       = smpl_param.body_pose.clone().detach().requires_grad_(True)
        smpl_param.left_hand_pose  = smpl_param.left_hand_pose.clone().detach().requires_grad_(True)
        smpl_param.right_hand_pose = smpl_param.right_hand_pose.clone().detach().requires_grad_(True)
        smpl_param.beta            = smpl_param.beta.clone().detach().requires_grad_(True)
        smpl_param.scale           = smpl_param.scale.clone().detach()

        # ---- 超参 ----
        lr_lbfgs             = 0.3
        max_iter_per_closure = 60
        fps                  = 30.0
        max_speed_mps        = 3.0
        stop_ratio           = 0.001
        shape_soft_bound     = 0.35
        shape_hard_clip      = 0.55
        shape_norm_bound     = 0.9

        # ---- stage-1：用 bbox 初始化 transl ----
        print("stage-1 transl init from bbox...")
        bbox_xyxy = torch.zeros((n_frame, 4), dtype=keypoints_tensor.dtype)
        for i in range(n_frame):
            kp = keypoints_tensor[i]
            mask = kp[:, 2] > 0.05
            xy = kp[mask][:, :2] if mask.any() else kp[:, :2]
            bbox_xyxy[i] = torch.stack([
                xy[:, 0].min(), xy[:, 1].min(),
                xy[:, 0].max(), xy[:, 1].max()
            ])

        with torch.no_grad():
            u_c = (bbox_xyxy[:, 0] + bbox_xyxy[:, 2]) / 2.0
            v_c = (bbox_xyxy[:, 1] + bbox_xyxy[:, 3]) / 2.0
            lon = (u_c / erp_W - 0.5) * (2.0 * math.pi)
            lat = (v_c / erp_H - 0.5) * math.pi
            dir_x = torch.sin(lon) * torch.cos(lat)
            dir_y = -torch.sin(lat)
            dir_z = torch.cos(lon) * torch.cos(lat)
            bbox_h_px = (bbox_xyxy[:, 3] - bbox_xyxy[:, 1]).clamp_min(1.0)
            fov_h = bbox_h_px / erp_H * math.pi
            dist  = (1.7 / (2.0 * torch.tan(fov_h / 2.0).clamp_min(1e-3))).clamp(0.5, 15.0)
            smpl_param.transl.data[:, 0] = dir_x * dist
            smpl_param.transl.data[:, 1] = dir_y * dist
            smpl_param.transl.data[:, 2] = dir_z * dist
        print(f"[init] transl: {smpl_param.transl.data[0].tolist()}")

        # ---- 损失权重 ----
        base_weights = {
            'body_2d_det':       n_frame * 1.0,
            'hand_2d_det':       n_frame * 0.5,   # 手部权重可调
            'loss_pose_prior':   n_frame * 0.05,
            'loss_hand_prior':   n_frame * 0.02,   # 手部 prior（防止手指乱转）
            'loss_shape_prior':  n_frame * 6.0,
            'loss_shape_bound':  n_frame * 20.0,
            'loss_shape_anchor': n_frame * 30.0,
            'loss_shape_norm':   n_frame * 15.0,
            'loss_transl_diff':  n_frame * 0.0,
            'loss_pose_diff':    n_frame * 0.0,
        }

        latest_loss   = float('inf')
        latest_losses: Dict[str, float] = {}
        current_shape_soft_bound = shape_soft_bound

        # ---- 优化阶段定义 ----
        optimization_stages = [
            # stage1.5: 只优化 transl+orient，固定 pose，避免初值被破坏
            (
                "stage2_transl_orient",
                lambda: [smpl_param.orient, smpl_param.transl],
                40, stop_ratio,
                {
                    'body_2d_det': 1.0, 'hand_2d_det': 0.0,
                    'loss_pose_prior': 0.0, 'loss_hand_prior': 0.0,
                    'loss_shape_prior': 0.0, 'loss_shape_bound': 0.0,
                    'loss_shape_anchor': 1.0, 'loss_shape_norm': 0.0,
                    'loss_transl_diff': 1.0, 'loss_pose_diff': 0.0,
                },
            ),
            # stage2: 再放开 pose+beta
            (
                "stage2_body",
                lambda: [smpl_param.orient, smpl_param.transl,
                        smpl_param.body_pose, smpl_param.beta],
                80, stop_ratio,
                {
                    'body_2d_det': 1.0, 'hand_2d_det': 0.0,
                    'loss_pose_prior': 1.0, 'loss_hand_prior': 0.0,
                    'loss_shape_prior': 3.0, 'loss_shape_bound': 12.0,
                    'loss_shape_anchor': 16.0, 'loss_shape_norm': 8.0,
                    'loss_transl_diff': 1.0, 'loss_pose_diff': 1.0,
                },
            ),
            # stage3: 手部
            (
                "stage3_hand",
                lambda: [smpl_param.orient, smpl_param.transl,
                        smpl_param.body_pose,
                        smpl_param.left_hand_pose, smpl_param.right_hand_pose],
                60, stop_ratio * 0.7,
                {
                    'body_2d_det': 1.5, 'hand_2d_det': 1.0,
                    'loss_pose_prior': 0.5, 'loss_hand_prior': 1.0,
                    'loss_shape_prior': 0.0, 'loss_shape_bound': 0.0,
                    'loss_shape_anchor': 0.0, 'loss_shape_norm': 0.0,
                    'loss_transl_diff': 1.0, 'loss_pose_diff': 0.8,
                },
            ),
        ]

        for (stage_name, get_params, max_iter,
             stage_stop, weight_mult) in optimization_stages:

            stage_weights = {
                k: base_weights[k] * weight_mult.get(k, 1.0)
                for k in base_weights
            }
            params = get_params()
            optimizer = torch.optim.LBFGS(
                params,
                line_search_fn='strong_wolfe',
                lr=lr_lbfgs,
                max_iter=max_iter_per_closure,
            )

            def make_closure(opt):
                def closure():
                    nonlocal latest_loss, latest_losses
                    for p in params:
                        p.grad = None
                    opt.zero_grad()

                    # forward
                    verts, joints_all, joints_body = forward_smplx_tensor_batch(
                        smplx_model=model,
                        global_orient=smpl_param.orient,
                        body_pose=smpl_param.body_pose,
                        left_hand_pose=smpl_param.left_hand_pose,
                        right_hand_pose=smpl_param.right_hand_pose,
                        betas=smpl_param.beta.unsqueeze(0).expand(n_frame, -1),
                        transl=smpl_param.transl,
                        scale=smpl_param.scale.expand(n_frame),
                    )

                    losses_all = {}

                    # 2D loss
                    losses_2d, _ = self.compute_loss_2d_erp(
                        W=erp_W, H=erp_H,
                        joints_all=joints_all,
                        keypoints_detected=keypoints_tensor,
                        keypoints_format=key_fmt,
                        hand_keypoints_detected=hand_tensor,
                    )
                    losses_all.update(losses_2d)

                    # 时序平滑
                    if n_frame > 1:
                        t_diff  = smpl_param.transl[1:] - smpl_param.transl[:-1]
                        t_speed = torch.norm(t_diff, dim=1) * fps
                        losses_all['loss_transl_diff'] = torch.relu(t_speed - max_speed_mps).mean()
                        p_diff = (smpl_param.body_pose[1:] - smpl_param.body_pose[:-1]).reshape(-1, 21, 3)
                        losses_all['loss_pose_diff'] = torch.norm(p_diff, dim=2).mean()
                    else:
                        z = torch.zeros((), dtype=smpl_param.body_pose.dtype)
                        losses_all['loss_transl_diff'] = z
                        losses_all['loss_pose_diff']   = z

                    # prior
                    losses_all['loss_pose_prior']  = torch.norm(smpl_param.body_pose, dim=1).mean()
                    losses_all['loss_hand_prior']  = (
                        torch.norm(smpl_param.left_hand_pose,  dim=1).mean() +
                        torch.norm(smpl_param.right_hand_pose, dim=1).mean()
                    )
                    losses_all['loss_shape_prior']  = torch.mean(smpl_param.beta ** 2)
                    losses_all['loss_shape_bound']  = torch.mean(
                        torch.relu(torch.abs(smpl_param.beta) - current_shape_soft_bound) ** 2
                    )
                    losses_all['loss_shape_anchor'] = torch.mean((smpl_param.beta - shape_anchor) ** 2)
                    losses_all['loss_shape_norm']   = (
                        torch.relu(torch.norm(smpl_param.beta) - shape_norm_bound) ** 2
                    )

                    # 加权求和
                    loss = sum(
                        losses_all.get(k, torch.zeros(1)) * stage_weights[k]
                        for k in stage_weights
                    )
                    latest_loss   = loss.detach().cpu().item()
                    latest_losses = {k: v.detach().cpu().item() if isinstance(v, torch.Tensor)
                                     else float(v) for k, v in losses_all.items()}
                    loss.backward()
                    return loss
                return closure

            closure_fn = make_closure(optimizer)
            print(f"\nopt ({stage_name})...")
            loss_prev = 1e9
            for it in range(max_iter):
                optimizer.step(closure_fn)
                with torch.no_grad():
                    smpl_param.beta.clamp_(-shape_hard_clip, shape_hard_clip)
                loss = latest_loss
                s = ", ".join(f"{k}={v:.4f}" for k, v in sorted(latest_losses.items()))
                print(f"[{stage_name}] iter={it+1:03d}/{max_iter} total={loss:.4f} | {s}")
                if it > 0 and loss_prev - loss < loss_prev * stage_stop:
                    break
                loss_prev = loss

        # 关闭梯度
        for attr in ['orient', 'transl', 'body_pose',
                     'left_hand_pose', 'right_hand_pose', 'beta']:
            getattr(smpl_param, attr).requires_grad_(False)

        return smpl_param

    # ------------------------------------------------------------------
    # 从 SMPL-54 结果初始化 SMPL-X 参数（用于热启动）
    # ------------------------------------------------------------------

    @staticmethod
    def init_from_smpl54_seq(smpl54_seq) -> 'SMPLX_Seq_Forward_Module.SMPLX_Param_Seq':
        """
        把现有 SMPL-54 的拟合结果转成 SMPLX_Param_Seq 作为初始值。
        body_pose 前 63 维直接复用，手部 pose 初始化为零。

        smpl54_seq: SMPL_Seq_Forward_Module.SMPL_Param_Seq
        """
        n = smpl54_seq.num_batch
        seq = SMPLX_Seq_Forward_Module.SMPLX_Param_Seq(num_batch=n, gender=smpl54_seq.gender)
        seq.orient = smpl54_seq.orient.clone()
        seq.transl = smpl54_seq.transl.clone()
        # smpl54 的 pose 是 69 维（23关节×3），smplx body_pose 是 63 维（21关节×3）
        # 前 21 个关节对应关系基本一致，直接取前 63 维
        seq.body_pose = smpl54_seq.pose[:, :63].clone()
        seq.beta  = smpl54_seq.beta.clone()
        seq.scale = smpl54_seq.scale.clone()
        # 手部默认零（T-pose 手型）
        seq.left_hand_pose  = torch.zeros(n, 45)
        seq.right_hand_pose = torch.zeros(n, 45)
        return seq