from __future__ import annotations

import math
from types import SimpleNamespace
from typing import Callable, Dict

import torch
from torch import nn

from .base import BaseSMPLXFitter


ROTATION_MATRIX = torch.tensor(
    [1, 0, 0, 0, 0, 1, 0, -1, 0], dtype=torch.float32
).reshape(3, 3)

_SHAPE_LOSS_KEYS = (
    "loss_shape_prior",
    "loss_shape_bound",
    "loss_shape_anchor",
    "loss_shape_norm",
)

_NON_SHAPE_LOSS_KEYS = (
    "loss_transl_diff",
    "loss_pose_diff",
    "loss_pose_prior",
    "loss_hand_prior",
)


class DualCamERPFitter(BaseSMPLXFitter):
    def __init__(self, gender: str = "neutral", R_cam4: torch.Tensor | None = None):
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
        if R_cam4 is None:
            raise ValueError("R_cam4 must be provided for DualCamERPFitter")
        self.R_cam4 = R_cam4.clone().detach().float()

    def _init_transl_from_bbox(
        self,
        keypoints_tensor: torch.Tensor,
        image_size,
    ):
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

    @staticmethod
    def _project_erp(points_3d: torch.Tensor, W: int, H: int) -> torch.Tensor:
        x = points_3d[..., 0]
        y = points_3d[..., 1]
        z = points_3d[..., 2]
        r = torch.norm(points_3d, dim=-1).clamp_min(1e-6)
        lon = torch.atan2(x, z)
        lat = torch.asin((-y / r).clamp(-1 + 1e-6, 1 - 1e-6))
        u = (lon / (2.0 * math.pi) + 0.5) * W
        v = (0.5 - lat / math.pi) * H
        return torch.stack([u, v], dim=-1)

    @staticmethod
    def _project_cam2(points_3d: torch.Tensor, W: int, H: int) -> torch.Tensor:
        R = ROTATION_MATRIX.to(points_3d.device, dtype=points_3d.dtype)
        return DualCamERPFitter._project_erp(points_3d @ R.T, W, H)

    def _project_cam4(self, points_3d: torch.Tensor, W: int, H: int) -> torch.Tensor:
        R_cam4 = self.R_cam4.to(points_3d.device, dtype=points_3d.dtype)
        R_rot = ROTATION_MATRIX.to(points_3d.device, dtype=points_3d.dtype)
        pts_cam4 = points_3d @ R_cam4.T
        return self._project_erp(pts_cam4 @ R_rot.T, W, H)

    def _init_params(
        self,
        smplx_init_c2,
        smplx_init_c4,
        kp_tensor_c2: torch.Tensor,
        kp_tensor_c4: torch.Tensor,
        image_size,
    ):
        n_frame = kp_tensor_c2.shape[0]
        smpl_c2 = smplx_init_c2 if smplx_init_c2 is not None else self._build_default_smplx(n_frame)
        smpl_c4 = smplx_init_c4 if smplx_init_c4 is not None else self._build_default_smplx(n_frame)

        transl_init_c2 = self._init_transl_from_bbox(kp_tensor_c2, image_size)
        transl_init_c4 = self._init_transl_from_bbox(kp_tensor_c4, image_size)
        if transl_init_c2 is not None:
            smpl_c2.transl.data.copy_(transl_init_c2)
        if transl_init_c4 is not None:
            smpl_c4.transl.data.copy_(transl_init_c4)

        smpl_c2 = self._prepare_smplx_param_for_optimization(smpl_c2)
        smpl_c4 = self._prepare_smplx_param_for_optimization(smpl_c4)

        beta_c2 = getattr(smpl_c2, "beta", None)
        beta_c4 = getattr(smpl_c4, "beta", None)
        if beta_c2 is None and beta_c4 is None:
            beta_init = torch.zeros(10, dtype=kp_tensor_c2.dtype, device=kp_tensor_c2.device)
        elif beta_c2 is None:
            beta_init = beta_c4.clone().detach()
        elif beta_c4 is None:
            beta_init = beta_c2.clone().detach()
        else:
            beta_init = 0.5 * (beta_c2.clone().detach() + beta_c4.clone().detach())
        shared_beta = nn.Parameter(beta_init)
        return smpl_c2, smpl_c4, shared_beta

    def _compute_loss_2d_with_proj(
        self,
        joints_all: torch.Tensor,
        dets_tensor: torch.Tensor,
        keypoints_format: Dict[str, int],
        project_fn: Callable[[torch.Tensor, int, int], torch.Tensor],
        W: int,
        H: int,
    ):
        from lib_dynamic_hvip.fitting.smplx_seq_forward_module import SMPLX_JOINT_MAP

        joints_2d_all = project_fn(joints_all, W, H)
        losses = {}
        idx_smplx, idx_kp, w_list = [], [], []
        j_actual = dets_tensor.shape[1]
        for name, kp_idx in keypoints_format.items():
            if name in SMPLX_JOINT_MAP and kp_idx < j_actual:
                idx_smplx.append(SMPLX_JOINT_MAP[name])
                idx_kp.append(kp_idx)
                w_list.append(1.0)

        if idx_smplx:
            dev = joints_all.device
            idx_s = torch.tensor(idx_smplx, dtype=torch.long, device=dev)
            idx_k = torch.tensor(idx_kp, dtype=torch.long, device=dev)
            w_t = torch.tensor(w_list, dtype=torch.float32, device=dev)
            w_t = w_t / (w_t.sum() + 1e-8)
            proj = joints_2d_all[:, idx_s, :]
            det = dets_tensor[:, idx_k, :]
            loss_body = torch.norm(proj - det[:, :, :2], dim=2)
            losses["body_2d_det"] = (loss_body * det[:, :, 2] * w_t.unsqueeze(0)).sum(dim=1).mean()

        return losses, joints_2d_all

    def _shape_regularization_losses(
        self,
        shared_beta: torch.Tensor,
        shape_anchor: torch.Tensor,
        ref_param,
        n_frame: int,
    ):
        proxy = SimpleNamespace(
            transl=ref_param.transl,
            body_pose=ref_param.body_pose,
            left_hand_pose=ref_param.left_hand_pose,
            right_hand_pose=ref_param.right_hand_pose,
            beta=shared_beta,
        )
        losses = self._common_regularization_losses(proxy, shape_anchor, n_frame)
        return {key: losses[key] for key in _SHAPE_LOSS_KEYS}

    @staticmethod
    def _suffix_non_shape_losses(losses_all: Dict[str, torch.Tensor], suffix: str) -> Dict[str, torch.Tensor]:
        return {f"{key}_{suffix}": losses_all[key] for key in _NON_SHAPE_LOSS_KEYS}

    def fit(
        self,
        cam2_dets,
        cam4_dets,
        image_size,
        smplx_init_c2=None,
        smplx_init_c4=None,
    ):
        from lib_dynamic_hvip.smpl.smpl_lib import forward_smplx_tensor_batch

        assert len(cam2_dets) > 0
        assert len(cam2_dets) == len(cam4_dets)

        n_frame = len(cam2_dets)
        erp_w, erp_h = image_size
        model = self._get_model()
        kp_tensor_c2 = self._stack_keypoints(cam2_dets)
        kp_tensor_c4 = self._stack_keypoints(cam4_dets)
        key_fmt_c2 = cam2_dets[0].keypoints_format
        key_fmt_c4 = cam4_dets[0].keypoints_format

        smpl_c2, smpl_c4, shared_beta = self._init_params(
            smplx_init_c2=smplx_init_c2,
            smplx_init_c4=smplx_init_c4,
            kp_tensor_c2=kp_tensor_c2,
            kp_tensor_c4=kp_tensor_c4,
            image_size=image_size,
        )
        shape_anchor = shared_beta.clone().detach()

        base_weights = {
            "body_2d_det_c2": n_frame * 1.0,
            "body_2d_det_c4": n_frame * 1.0,
            "loss_pose_prior_c2": n_frame * 0.05,
            "loss_pose_prior_c4": n_frame * 0.05,
            "loss_hand_prior_c2": n_frame * 0.02,
            "loss_hand_prior_c4": n_frame * 0.02,
            "loss_transl_diff_c2": n_frame * 0.0,
            "loss_transl_diff_c4": n_frame * 0.0,
            "loss_pose_diff_c2": n_frame * 0.0,
            "loss_pose_diff_c4": n_frame * 0.0,
            "loss_shape_prior": n_frame * 6.0,
            "loss_shape_bound": n_frame * 20.0,
            "loss_shape_anchor": n_frame * 30.0,
            "loss_shape_norm": n_frame * 15.0,
        }
        optimization_stages = [
            (
                "stage1_transl_orient",
                lambda: [smpl_c2.orient, smpl_c2.transl, smpl_c4.orient, smpl_c4.transl],
                40,
                self.stop_ratio,
                {
                    "body_2d_det_c2": 1.0,
                    "body_2d_det_c4": 1.0,
                    "loss_pose_prior_c2": 0.0,
                    "loss_pose_prior_c4": 0.0,
                    "loss_hand_prior_c2": 0.0,
                    "loss_hand_prior_c4": 0.0,
                    "loss_transl_diff_c2": 1.0,
                    "loss_transl_diff_c4": 1.0,
                    "loss_pose_diff_c2": 0.0,
                    "loss_pose_diff_c4": 0.0,
                    "loss_shape_prior": 0.0,
                    "loss_shape_bound": 0.0,
                    "loss_shape_anchor": 1.0,
                    "loss_shape_norm": 0.0,
                },
            ),
            (
                "stage2_body",
                lambda: [
                    smpl_c2.orient,
                    smpl_c2.transl,
                    smpl_c2.body_pose,
                    smpl_c4.orient,
                    smpl_c4.transl,
                    smpl_c4.body_pose,
                    shared_beta,
                ],
                60,
                self.stop_ratio,
                {
                    "body_2d_det_c2": 1.0,
                    "body_2d_det_c4": 1.0,
                    "loss_pose_prior_c2": 1.0,
                    "loss_pose_prior_c4": 1.0,
                    "loss_hand_prior_c2": 0.0,
                    "loss_hand_prior_c4": 0.0,
                    "loss_transl_diff_c2": 1.0,
                    "loss_transl_diff_c4": 1.0,
                    "loss_pose_diff_c2": 1.0,
                    "loss_pose_diff_c4": 1.0,
                    "loss_shape_prior": 3.0,
                    "loss_shape_bound": 12.0,
                    "loss_shape_anchor": 16.0,
                    "loss_shape_norm": 8.0,
                },
            ),
            (
                "stage3_hand",
                lambda: [
                    smpl_c2.orient,
                    smpl_c2.transl,
                    smpl_c2.body_pose,
                    smpl_c2.left_hand_pose,
                    smpl_c2.right_hand_pose,
                    smpl_c4.orient,
                    smpl_c4.transl,
                    smpl_c4.body_pose,
                    smpl_c4.left_hand_pose,
                    smpl_c4.right_hand_pose,
                    shared_beta,
                ],
                60,
                self.stop_ratio * 0.7,
                {
                    "body_2d_det_c2": 1.5,
                    "body_2d_det_c4": 1.5,
                    "loss_pose_prior_c2": 0.5,
                    "loss_pose_prior_c4": 0.5,
                    "loss_hand_prior_c2": 1.0,
                    "loss_hand_prior_c4": 1.0,
                    "loss_transl_diff_c2": 1.0,
                    "loss_transl_diff_c4": 1.0,
                    "loss_pose_diff_c2": 0.8,
                    "loss_pose_diff_c4": 0.8,
                    "loss_shape_prior": 0.0,
                    "loss_shape_bound": 0.0,
                    "loss_shape_anchor": 0.0,
                    "loss_shape_norm": 0.0,
                },
            ),
        ]

        def make_loss_fn():
            _, joints_c2, _ = forward_smplx_tensor_batch(
                smplx_model=model,
                global_orient=smpl_c2.orient,
                body_pose=smpl_c2.body_pose,
                left_hand_pose=smpl_c2.left_hand_pose,
                right_hand_pose=smpl_c2.right_hand_pose,
                betas=shared_beta.unsqueeze(0).expand(n_frame, -1),
                transl=smpl_c2.transl,
                scale=smpl_c2.scale.expand(n_frame),
            )
            _, joints_c4, _ = forward_smplx_tensor_batch(
                smplx_model=model,
                global_orient=smpl_c4.orient,
                body_pose=smpl_c4.body_pose,
                left_hand_pose=smpl_c4.left_hand_pose,
                right_hand_pose=smpl_c4.right_hand_pose,
                betas=shared_beta.unsqueeze(0).expand(n_frame, -1),
                transl=smpl_c4.transl,
                scale=smpl_c4.scale.expand(n_frame),
            )

            losses_c2, _ = self._compute_loss_2d_with_proj(
                joints_all=joints_c2,
                dets_tensor=kp_tensor_c2,
                keypoints_format=key_fmt_c2,
                project_fn=self._project_cam2,
                W=erp_w,
                H=erp_h,
            )
            losses_c4, _ = self._compute_loss_2d_with_proj(
                joints_all=joints_c4,
                dets_tensor=kp_tensor_c4,
                keypoints_format=key_fmt_c4,
                project_fn=self._project_cam4,
                W=erp_w,
                H=erp_h,
            )

            reg_c2 = self._common_regularization_losses(smpl_c2, shape_anchor, n_frame)
            reg_c4 = self._common_regularization_losses(smpl_c4, shape_anchor, n_frame)
            losses_all = {
                "body_2d_det_c2": losses_c2.get("body_2d_det", joints_c2.new_zeros(())),
                "body_2d_det_c4": losses_c4.get("body_2d_det", joints_c4.new_zeros(())),
            }
            losses_all.update(self._suffix_non_shape_losses(reg_c2, "c2"))
            losses_all.update(self._suffix_non_shape_losses(reg_c4, "c4"))
            losses_all.update(self._shape_regularization_losses(shared_beta, shape_anchor, smpl_c2, n_frame))
            return losses_all

        latest_loss = 0.0
        latest_losses_detail = {}

        for stage_name, get_params, max_iter, stage_stop, weight_mult in optimization_stages:
            stage_weights = {k: base_weights[k] * weight_mult.get(k, 1.0) for k in base_weights}
            params = get_params()
            optimizer = torch.optim.LBFGS(
                params,
                line_search_fn="strong_wolfe",
                lr=self.lr_lbfgs,
                max_iter=self.max_iter_closure,
            )

            def closure():
                nonlocal latest_loss, latest_losses_detail
                for p in params:
                    p.grad = None
                optimizer.zero_grad()
                losses_all = make_loss_fn()
                zero = next(iter(losses_all.values())).new_zeros(1)
                loss = sum(losses_all.get(k, zero) * stage_weights[k] for k in stage_weights)
                latest_loss = float(loss.detach().cpu().item())
                latest_losses_detail = {k: float(v.detach().cpu().item()) for k, v in losses_all.items()}
                loss.backward()
                return loss

            print(f"\nopt ({stage_name})...")
            loss_prev = 1e9
            for it in range(max_iter):
                optimizer.step(closure)
                shared_beta.data.clamp_(-self.shape_hard_clip, self.shape_hard_clip)
                if it > 0 and loss_prev - latest_loss < loss_prev * stage_stop:
                    break
                loss_prev = latest_loss
                detail_str = "  ".join(
                    f"{k}={latest_losses_detail.get(k, 0.0) * stage_weights.get(k, 1.0):.4f}"
                    for k in stage_weights
                    if stage_weights.get(k, 0.0) > 0
                )
                print(f"[{stage_name}] iter={it+1:03d}/{max_iter} total={latest_loss:.4f}  |  {detail_str}")

        for smpl_param in (smpl_c2, smpl_c4):
            for attr in ["orient", "transl", "body_pose", "left_hand_pose", "right_hand_pose", "beta"]:
                getattr(smpl_param, attr).requires_grad_(False)
        shared_beta.requires_grad_(False)

        shared_beta_final = shared_beta.detach().clone()
        smpl_c2.beta = shared_beta_final.clone()
        smpl_c4.beta = shared_beta_final.clone()
        return smpl_c2, smpl_c4, shared_beta_final
