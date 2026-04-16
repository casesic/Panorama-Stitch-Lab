from __future__ import annotations

import math
from typing import Dict, Optional

import torch
from lib_dynamic_hvip.fitting.smplx_seq_forward_module import SMPLX_Seq_Forward_Module

from .base import BaseSMPLXFitter


class ERPFitter(BaseSMPLXFitter):
    def __init__(self, gender: str = "neutral"):
        super().__init__(gender=gender)
        self.fps = 30.0
        self.max_speed_mps = 3.0
        self.shape_norm_bound = 0.9
        self.shape_hard_clip = 0.55
        self.current_shape_soft_bound = 0.35
        self.lr_lbfgs = 0.3
        self.max_iter_closure = 60
        self.stop_ratio = 1e-3
        self.hand_overall_weight = 2.0

    @staticmethod
    def _project(points_3d: torch.Tensor, W: int, H: int) -> torch.Tensor:
        x = points_3d[..., 0]
        y = points_3d[..., 1]
        z = points_3d[..., 2]
        r = torch.norm(points_3d, dim=-1).clamp_min(1e-6)
        lon = torch.atan2(x, z)
        lat = torch.asin((-y / r).clamp(-1 + 1e-6, 1 - 1e-6))
        u = (lon / (2.0 * math.pi) + 0.5) * W
        v = (0.5 - lat / math.pi) * H
        return torch.stack([u, v], dim=-1)

    def _init_transl_from_bbox(
        self,
        keypoints_tensor: torch.Tensor,
        image_size,
    ) -> Optional[torch.Tensor]:
        erp_w, erp_h = image_size
        n_frame = keypoints_tensor.shape[0]
        bbox_xyxy = torch.zeros((n_frame, 4), dtype=keypoints_tensor.dtype, device=keypoints_tensor.device)
        for i in range(n_frame):
            kp = keypoints_tensor[i]
            mask = kp[:, 2] > 0.05
            xy = kp[mask][:, :2] if mask.any() else kp[:, :2]
            bbox_xyxy[i] = torch.stack([xy[:, 0].min(), xy[:, 1].min(), xy[:, 0].max(), xy[:, 1].max()])

        u_c = (bbox_xyxy[:, 0] + bbox_xyxy[:, 2]) / 2.0
        v_c = (bbox_xyxy[:, 1] + bbox_xyxy[:, 3]) / 2.0
        lon = (u_c / erp_w - 0.5) * (2.0 * math.pi)
        lat = (v_c / erp_h - 0.5) * math.pi
        dir_x = torch.sin(lon) * torch.cos(lat)
        dir_y = -torch.sin(lat)
        dir_z = torch.cos(lon) * torch.cos(lat)
        bbox_h_px = (bbox_xyxy[:, 3] - bbox_xyxy[:, 1]).clamp_min(1.0)
        fov_h = bbox_h_px / erp_h * math.pi
        dist = (1.7 / (2.0 * torch.tan(fov_h / 2.0).clamp_min(1e-3))).clamp(0.5, 15.0)
        return torch.stack([dir_x * dist, dir_y * dist, dir_z * dist], dim=1)

    def compute_loss_2d_erp(
        self,
        W: int,
        H: int,
        joints_all: torch.Tensor,
        keypoints_detected: torch.Tensor,
        keypoints_format: Dict[str, int],
        keypoints_weight: Optional[Dict[str, float]] = None,
        hand_keypoints_detected: Optional[torch.Tensor] = None,
    ):
        from lib_dynamic_hvip.fitting.smplx_seq_forward_module import SMPLX_JOINT_MAP

        if keypoints_weight is None:
            keypoints_weight = {}

        joints_2d_all = self._project(joints_all, W, H)
        losses = {}
        idx_smplx, idx_kp, w_list = [], [], []
        j_actual = keypoints_detected.shape[1]
        for name, kp_idx in keypoints_format.items():
            if name in SMPLX_JOINT_MAP and kp_idx < j_actual:
                idx_smplx.append(SMPLX_JOINT_MAP[name])
                idx_kp.append(kp_idx)
                w_list.append(keypoints_weight.get(name, 1.0))

        if idx_smplx:
            dev = joints_all.device
            idx_s = torch.tensor(idx_smplx, dtype=torch.long, device=dev)
            idx_k = torch.tensor(idx_kp, dtype=torch.long, device=dev)
            w_t = torch.tensor(w_list, dtype=torch.float32, device=dev)
            w_t = w_t / (w_t.sum() + 1e-8)
            proj = joints_2d_all[:, idx_s, :]
            det = keypoints_detected[:, idx_k, :]
            loss_body = torch.norm(proj - det[:, :, :2], dim=2)
            losses["body_2d_det"] = (loss_body * det[:, :, 2] * w_t.unsqueeze(0)).sum(dim=1).mean()

        if hand_keypoints_detected is not None:
            losses["hand_2d_det"] = self._hand_joint_loss(joints_2d_all, hand_keypoints_detected)
        return losses, joints_2d_all

    def fit(
        self,
        detection_results_list: list,
        image_size: tuple,
        smpl_param_init=None,
        hand_detection_list=None,
    ) -> SMPLX_Seq_Forward_Module.SMPLX_Param_Seq:
        from lib_dynamic_hvip.smpl.smpl_lib import forward_smplx_tensor_batch

        assert len(detection_results_list) > 0
        n_frame = len(detection_results_list)
        erp_w, erp_h = image_size
        model = self._get_model()
        key_fmt = detection_results_list[0].keypoints_format
        kp_tensor = self._stack_keypoints(detection_results_list)
        hand_tensor = self._stack_hand(hand_detection_list)
        smpl_param = smpl_param_init if smpl_param_init is not None else self._build_default_smplx(n_frame)
        transl_init = self._init_transl_from_bbox(kp_tensor, image_size)
        if transl_init is not None:
            smpl_param.transl.data.copy_(transl_init)
        smpl_param = self._prepare_smplx_param_for_optimization(smpl_param)
        shape_anchor = smpl_param.beta.clone().detach()

        base_weights = {
            "body_2d_det": n_frame * 1.0,
            "hand_2d_det": n_frame * 0.5,
            "loss_pose_prior": n_frame * 0.05,
            "loss_hand_prior": n_frame * 0.02,
            "loss_shape_prior": n_frame * 6.0,
            "loss_shape_bound": n_frame * 20.0,
            "loss_shape_anchor": n_frame * 30.0,
            "loss_shape_norm": n_frame * 15.0,
            "loss_transl_diff": n_frame * 0.0,
            "loss_pose_diff": n_frame * 0.0,
        }
        optimization_stages = [
            (
                "stage1_transl_orient",
                lambda: [smpl_param.orient, smpl_param.transl],
                40,
                self.stop_ratio,
                {
                    "body_2d_det": 1.0,
                    "hand_2d_det": 0.0,
                    "loss_pose_prior": 0.0,
                    "loss_hand_prior": 0.0,
                    "loss_shape_prior": 0.0,
                    "loss_shape_bound": 0.0,
                    "loss_shape_anchor": 1.0,
                    "loss_shape_norm": 0.0,
                    "loss_transl_diff": 1.0,
                    "loss_pose_diff": 0.0,
                },
            ),
            (
                "stage2_body",
                lambda: [smpl_param.orient, smpl_param.transl, smpl_param.body_pose, smpl_param.beta],
                80,
                self.stop_ratio,
                {
                    "body_2d_det": 1.0,
                    "hand_2d_det": 0.0,
                    "loss_pose_prior": 1.0,
                    "loss_hand_prior": 0.0,
                    "loss_shape_prior": 3.0,
                    "loss_shape_bound": 12.0,
                    "loss_shape_anchor": 16.0,
                    "loss_shape_norm": 8.0,
                    "loss_transl_diff": 1.0,
                    "loss_pose_diff": 1.0,
                },
            ),
            (
                "stage3_hand",
                lambda: [
                    smpl_param.orient,
                    smpl_param.transl,
                    smpl_param.body_pose,
                    smpl_param.left_hand_pose,
                    smpl_param.right_hand_pose,
                ],
                60,
                self.stop_ratio * 0.7,
                {
                    "body_2d_det": 1.5,
                    "hand_2d_det": 1.0,
                    "loss_pose_prior": 0.5,
                    "loss_hand_prior": 1.0,
                    "loss_shape_prior": 0.0,
                    "loss_shape_bound": 0.0,
                    "loss_shape_anchor": 0.0,
                    "loss_shape_norm": 0.0,
                    "loss_transl_diff": 1.0,
                    "loss_pose_diff": 0.8,
                },
            ),
        ]

        def make_loss_fn():
            _, joints_all, _ = forward_smplx_tensor_batch(
                smplx_model=model,
                global_orient=smpl_param.orient,
                body_pose=smpl_param.body_pose,
                left_hand_pose=smpl_param.left_hand_pose,
                right_hand_pose=smpl_param.right_hand_pose,
                betas=smpl_param.beta.unsqueeze(0).expand(n_frame, -1),
                transl=smpl_param.transl,
                scale=smpl_param.scale.expand(n_frame),
            )
            losses_all, _ = self.compute_loss_2d_erp(
                W=erp_w,
                H=erp_h,
                joints_all=joints_all,
                keypoints_detected=kp_tensor,
                keypoints_format=key_fmt,
                hand_keypoints_detected=hand_tensor,
            )
            losses_all.update(self._common_regularization_losses(smpl_param, shape_anchor, n_frame))
            return losses_all

        return self._run_optimization(
            smpl_param,
            base_weights,
            optimization_stages,
            make_loss_fn,
            after_step=lambda: smpl_param.beta.data.clamp_(-self.shape_hard_clip, self.shape_hard_clip),
        )
