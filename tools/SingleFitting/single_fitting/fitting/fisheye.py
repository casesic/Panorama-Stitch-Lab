from __future__ import annotations

from typing import Dict, Optional

import torch

from .base import BaseSMPLXFitter, resolve_delta_t
from .projection import (
    DEFAULT_T_CAM4_INIT,
    ds_project_torch,
    project_world_to_cam2_torch,
    project_world_to_cam4_torch,
)


class SingleCamFisheyeFitter(BaseSMPLXFitter):
    def __init__(self, gender: str = "neutral", camera: str = "cam2"):
        super().__init__(gender=gender)
        self.camera = camera
        # Keep single-camera behavior closer to the original fitter.
        self.fps = 30.0
        self.max_speed_mps = 3.0
        self.shape_norm_bound = 0.9
        self.shape_hard_clip = 0.55
        self.current_shape_soft_bound = 0.35
        self.lr_lbfgs = 0.3
        self.max_iter_closure = 60
        self.stop_ratio = 1e-3
        self.hand_overall_weight = 2.0
        self.body_keypoint_weight = {
           # "L_Hip": 1.5,
           # "R_Hip": 1.5,
           # "R_Wrist": 1.5,
        }

    def _ds_project_batch(self, points_3d: torch.Tensor) -> torch.Tensor:
        return ds_project_torch(points_3d)

    def compute_loss_2d_ds(
        self,
        joints_all: torch.Tensor,
        keypoints_detected: torch.Tensor,
        keypoints_format: Dict[str, int],
        keypoints_weight: Optional[Dict[str, float]] = None,
        hand_keypoints_detected: Optional[torch.Tensor] = None,
        hand_weight: float = 2.0,
    ):
        from lib_dynamic_hvip.fitting.smplx_seq_forward_module import SMPLX_BODY_22

        if keypoints_weight is None:
            keypoints_weight = {}

        joints_2d_all = self._ds_project_batch(joints_all)
        losses = {}
        idx_smplx, idx_kp, w_list = [], [], []
        j_actual = keypoints_detected.shape[1]
        for name, kp_idx in keypoints_format.items():
            if name in SMPLX_BODY_22 and kp_idx < j_actual:
                idx_smplx.append(SMPLX_BODY_22[name])
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
            loss_body = (loss_body * det[:, :, 2] * w_t.unsqueeze(0)).sum(dim=1).mean()
            losses["body_2d_det"] = loss_body

        if hand_keypoints_detected is not None:
            losses["hand_2d_det"] = self._hand_joint_loss(joints_2d_all, hand_keypoints_detected)
        return losses, joints_2d_all

    def fit(self, detection_results_list: list, image_size: int = 2880, smpl_param_init=None, hand_detection_list=None):
        from lib_dynamic_hvip.smpl.smpl_lib import forward_smplx_tensor_batch

        n_frame = len(detection_results_list)
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
                "stage1_global",
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
                lambda: [smpl_param.orient, smpl_param.transl, smpl_param.body_pose, smpl_param.left_hand_pose, smpl_param.right_hand_pose],
                80,
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
            losses_all, _ = self.compute_loss_2d_ds(
                joints_all,
                kp_tensor,
                key_fmt,
                keypoints_weight=self.body_keypoint_weight,
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


class DualCamWorldFitter(BaseSMPLXFitter):
    def compute_loss_2d(
        self,
        joints_world: torch.Tensor,
        kp_tensor: torch.Tensor,
        kp_fmt: Dict[str, int],
        camera: str,
        delta_t: Optional[torch.Tensor] = None,
        kp_weight: Optional[Dict[str, float]] = None,
    ) -> Dict[str, torch.Tensor]:
        from lib_dynamic_hvip.fitting.smplx_seq_forward_module import SMPLX_BODY_22

        if kp_weight is None:
            kp_weight = {}
        joints_2d = project_world_to_cam2_torch(joints_world) if camera == "cam2" else project_world_to_cam4_torch(joints_world, resolve_delta_t(delta_t, joints_world))

        idx_smplx, idx_kp, w_list = [], [], []
        j_actual = kp_tensor.shape[1]
        for name, kp_idx in kp_fmt.items():
            if name in SMPLX_BODY_22 and kp_idx < j_actual:
                idx_smplx.append(SMPLX_BODY_22[name])
                idx_kp.append(kp_idx)
                w_list.append(kp_weight.get(name, 1.0))

        if not idx_smplx:
            return {"body_2d_det": torch.zeros(1, device=joints_world.device)}

        dev = joints_world.device
        idx_s = torch.tensor(idx_smplx, dtype=torch.long, device=dev)
        idx_k = torch.tensor(idx_kp, dtype=torch.long, device=dev)
        w_t = torch.tensor(w_list, dtype=torch.float32, device=dev)
        w_t = w_t / (w_t.sum() + 1e-8)
        proj = joints_2d[:, idx_s, :]
        det = kp_tensor[:, idx_k, :]
        loss = torch.norm(proj - det[:, :, :2], dim=2)
        loss = (loss * det[:, :, 2] * w_t.unsqueeze(0)).sum(dim=1).mean()
        return {"body_2d_det": loss}

    def fit(
        self,
        det_list_cam2: list,
        hand_list_cam2,
        kp_fmt_cam2: Dict[str, int],
        det_list_cam4: list,
        hand_list_cam4,
        kp_fmt_cam4: Dict[str, int],
        image_size: int = 2880,
        smpl_param_init=None,
    ):
        from lib_dynamic_hvip.smpl.smpl_lib import forward_smplx_tensor_batch

        assert len(det_list_cam2) == len(det_list_cam4)
        n_frame = len(det_list_cam2)
        model = self._get_model()
        kp_cam2 = self._stack_keypoints(det_list_cam2)
        kp_cam4 = self._stack_keypoints(det_list_cam4)
        hand_cam2 = self._stack_hand(hand_list_cam2)
        hand_cam4 = self._stack_hand(hand_list_cam4)
        smpl_param = smpl_param_init if smpl_param_init is not None else self._build_default_smplx(n_frame)
        transl_init = self._init_transl_from_bbox(kp_cam2, image_size)
        if transl_init is not None:
            smpl_param.transl.data.copy_(transl_init)
        smpl_param = self._prepare_smplx_param_for_optimization(smpl_param)
        shape_anchor = smpl_param.beta.clone().detach()
        delta_t = DEFAULT_T_CAM4_INIT.clone().detach().requires_grad_(True)
        delta_t_anchor = delta_t.clone().detach()

        base_weights = {
            "body_2d_cam2": 4.0,
            "body_2d_cam4": 4.0,
            "hand_2d_cam2": 1.0,
            "hand_2d_cam4": 1.0,
            "loss_pose_prior": 0.3,
            "loss_hand_prior": 0.5,
            "loss_shape_prior": 0.01,
            "loss_shape_bound": 0.3,
            "loss_shape_anchor": 0.03,
            "loss_shape_norm": 0.01,
            "loss_transl_diff": 2.0,
            "loss_pose_diff": 1.0,
            "loss_delta_t": 10.0,
        }
        optimization_stages = [
            (
                "stage1_global",
                lambda: [smpl_param.orient, smpl_param.transl, delta_t],
                25,
                self.stop_ratio,
                {
                    "body_2d_cam2": 1.0,
                    "body_2d_cam4": 1.0,
                    "loss_pose_prior": 0.0,
                    "loss_hand_prior": 0.0,
                    "loss_shape_prior": 0.0,
                    "loss_shape_bound": 0.0,
                    "loss_shape_anchor": 0.0,
                    "loss_shape_norm": 0.0,
                    "loss_transl_diff": 0.0,
                    "loss_pose_diff": 0.0,
                    "loss_delta_t": 1.0,
                },
            ),
            (
                "stage2_body",
                lambda: [smpl_param.orient, smpl_param.transl, smpl_param.body_pose, delta_t],
                50,
                self.stop_ratio * 0.7,
                {
                    "body_2d_cam2": 1.2,
                    "body_2d_cam4": 1.2,
                    "loss_pose_prior": 0.5,
                    "loss_hand_prior": 0.0,
                    "loss_shape_prior": 0.0,
                    "loss_shape_bound": 0.0,
                    "loss_shape_anchor": 0.0,
                    "loss_shape_norm": 0.0,
                    "loss_transl_diff": 1.0,
                    "loss_pose_diff": 0.8,
                    "loss_delta_t": 1.0,
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
                    delta_t,
                ],
                60,
                self.stop_ratio * 0.7,
                {
                    "body_2d_cam2": 1.5,
                    "body_2d_cam4": 1.5,
                    "hand_2d_cam2": 1.0,
                    "hand_2d_cam4": 1.0,
                    "loss_pose_prior": 0.5,
                    "loss_hand_prior": 1.0,
                    "loss_shape_prior": 0.0,
                    "loss_shape_bound": 0.0,
                    "loss_shape_anchor": 0.0,
                    "loss_shape_norm": 0.0,
                    "loss_transl_diff": 1.0,
                    "loss_pose_diff": 0.8,
                    "loss_delta_t": 1.0,
                },
            ),
        ]

        def make_loss_fn():
            _, joints_world, _ = forward_smplx_tensor_batch(
                smplx_model=model,
                global_orient=smpl_param.orient,
                body_pose=smpl_param.body_pose,
                left_hand_pose=smpl_param.left_hand_pose,
                right_hand_pose=smpl_param.right_hand_pose,
                betas=smpl_param.beta.unsqueeze(0).expand(n_frame, -1),
                transl=smpl_param.transl,
                scale=smpl_param.scale.expand(n_frame),
            )
            joints_2d_cam2 = project_world_to_cam2_torch(joints_world)
            joints_2d_cam4 = project_world_to_cam4_torch(joints_world, delta_t)
            losses_all = self._common_regularization_losses(smpl_param, shape_anchor, n_frame)
            losses_all["body_2d_cam2"] = self.compute_loss_2d(joints_world, kp_cam2, kp_fmt_cam2, camera="cam2", delta_t=delta_t)["body_2d_det"]
            losses_all["body_2d_cam4"] = self.compute_loss_2d(joints_world, kp_cam4, kp_fmt_cam4, camera="cam4", delta_t=delta_t)["body_2d_det"]
            losses_all["hand_2d_cam2"] = self._hand_joint_loss(joints_2d_cam2, hand_cam2)
            losses_all["hand_2d_cam4"] = self._hand_joint_loss(joints_2d_cam4, hand_cam4)
            losses_all["loss_delta_t"] = torch.norm(delta_t - delta_t_anchor) ** 2
            return losses_all

        smpl_param = self._run_optimization(
            smpl_param,
            base_weights,
            optimization_stages,
            make_loss_fn,
            after_step=lambda: smpl_param.beta.data.clamp_(-self.shape_hard_clip, self.shape_hard_clip),
        )
        smpl_param.delta_T = delta_t.detach().clone()
        return smpl_param
