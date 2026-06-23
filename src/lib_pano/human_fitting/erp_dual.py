"""双相机 ERP SMPL-X 拟合器：用 cam2/cam4 检测共同约束一套体型/姿态，分两阶段优化。"""

from __future__ import annotations

import math
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn

from .base import BaseSMPLXFitter
from .constraints import (
    DenseKPLoss,
    compute_keypoint_2d_loss_with_proj,
    compute_shape_regularization_losses,
    compute_silhouette_loss,
    suffix_non_shape_losses,
)
from .loss_config import DualCamLossWeights, build_base_weights, build_optimization_stages
from .projection import project_cam2, project_cam4, project_erp
from .constraints.sam3d_dense import (
    compute_sam3d_dense_sequence_reprojection_loss,
    load_default_sam3d_sample_ids,
)




class DualCamERPFitter(BaseSMPLXFitter):
    """两阶段双相机 ERP 优化器：cam2/cam4 共享 beta 与 body_pose，覆盖优化超参。"""
    def __init__(
        self,
        gender: str = "neutral",
        R_cam4: torch.Tensor | None = None,
        use_silhouette_loss: bool = True,
        densekp_filter_path: str | None = None,
        sam3d_dense_num_points: int = 1024,
    ):
        """配置双相机拟合超参、损失权重、稠密关键点过滤与 cam4 外参；可选配置 SAM dense surface 固定采样点数。"""
        super().__init__(gender=gender)
        self.conf_thresh = 0.57
        self.fps = 30.0
        self.max_speed_mps = 3.0
        self.shape_norm_bound = 0.9
        self.shape_hard_clip = 0.55
        self.current_shape_soft_bound = 0.35
        self.lr_lbfgs = 0.3
        self.max_iter_closure = 60
        self.stop_ratio = 1e-3
        self.hand_overall_weight = 2.0
        self.use_silhouette_loss = use_silhouette_loss
        self.sam3d_dense_num_points = int(sam3d_dense_num_points)
        self.loss_weights = DualCamLossWeights()
        project_root = Path(__file__).resolve().parents[3]
        data_dir = project_root / "tools" / "CameraHMR" / "data" / "train-eval-utils"
        id_mask = None
        if densekp_filter_path is not None:
            from lib_pano.fitting_io.densekp_filter import load_densekp_filter

            id_mask = load_densekp_filter(densekp_filter_path)
        self.densekp_loss = DenseKPLoss(data_dir=data_dir, weight=self.loss_weights.densekp, id_mask=id_mask)
        if R_cam4 is None:
            raise ValueError("R_cam4 must be provided for DualCamERPFitter")
        self.R_cam4 = R_cam4.clone().detach().float()

    def _init_transl_from_bbox(
        self,
        keypoints_tensor: torch.Tensor,
        image_size,
    ):
        """由关键点外接框中心反算 ERP 经纬方向、按框高估计深度，初始化每帧平移向量。"""
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
        """裸 ERP 投影（不施加相机旋转）。"""
        return project_erp(points_3d, W, H)

    @staticmethod
    def _project_cam2(points_3d: torch.Tensor, W: int, H: int) -> torch.Tensor:
        """带 cam2 旋转的 ERP 投影。"""
        return project_cam2(points_3d, W, H)

    def _project_cam4(self, points_3d: torch.Tensor, W: int, H: int) -> torch.Tensor:
        """带 cam4 外参旋转的 ERP 投影。"""
        return project_cam4(points_3d, W, H, self.R_cam4)

    def _init_params(
        self,
        smplx_init_c2,
        smplx_init_c4,
        kp_tensor_c2: torch.Tensor,
        kp_tensor_c4: torch.Tensor,
        image_size,
    ):
        """初始化两相机 SMPL-X 参数：bbox 估平移、置为可优化、共享 beta（两相机均值）与 body_pose。"""
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
        smpl_c2.scale = smpl_c2.scale.reshape(-1)[:1].clone().detach()
        smpl_c4.scale = smpl_c4.scale.reshape(-1)[:1].clone().detach()

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
        beta_init = beta_init.reshape(-1)
        shared_beta = nn.Parameter(beta_init)

        bp_c2 = smpl_c2.body_pose.clone().detach()
        bp_c4 = smpl_c4.body_pose.clone().detach()
        shared_body_pose_init = 0.5 * (bp_c2 + bp_c4)
        shared_body_pose = nn.Parameter(shared_body_pose_init)
        return smpl_c2, smpl_c4, shared_beta, shared_body_pose

    def fit(
        self,
        cam2_dets,
        cam4_dets,
        image_size,
        smplx_init_c2=None,
        smplx_init_c4=None,
        boundary_pts_list_c2: list[torch.Tensor | None] | None = None,
        boundary_pts_list_c4: list[torch.Tensor | None] | None = None,
        bbox_list_c2: list[list[int] | None] | None = None,
        bbox_list_c4: list[list[int] | None] | None = None,
        densekp_list_c2: list | None = None,
        densekp_list_c4: list | None = None,
        sam3d_dense_list_c2: list | None = None,
        sam3d_dense_list_c4: list | None = None,
    ):
        """两阶段双相机联合拟合主流程：初始化→Stage 1（orient/transl）→Stage 2（加入共享 body_pose/beta 与各约束）的 LBFGS 优化，返回两相机参数与共享 beta。

        sam3d_dense_list_c2/c4：每帧一个 SAM3DDenseResult 或 None，其 points_2d 已在 equirect_rotate
        全图坐标，cam2/cam4 均用 project_cam2 投影，仅在现有 Stage 2 启用。
        """
        from .smplx_sequence import forward_smplx_tensor_batch

        assert len(cam2_dets) > 0
        assert len(cam2_dets) == len(cam4_dets)

        n_frame = len(cam2_dets)
        erp_w, erp_h = image_size
        model = self._get_model()
        kp_tensor_c2 = self._stack_keypoints(cam2_dets)
        kp_tensor_c4 = self._stack_keypoints(cam4_dets)
        key_fmt_c2 = cam2_dets[0].keypoints_format
        key_fmt_c4 = cam4_dets[0].keypoints_format

        smpl_c2, smpl_c4, shared_beta, shared_body_pose = self._init_params(
            smplx_init_c2=smplx_init_c2,
            smplx_init_c4=smplx_init_c4,
            kp_tensor_c2=kp_tensor_c2,
            kp_tensor_c4=kp_tensor_c4,
            image_size=image_size,
        )
        shape_anchor = shared_beta.clone().detach()
        pose_anchor = shared_body_pose.clone().detach()

        base_weights = build_base_weights(
            n_frame=n_frame,
            use_silhouette_loss=self.use_silhouette_loss,
            weights=self.loss_weights,
        )
        optimization_stages = build_optimization_stages(
            stop_ratio=self.stop_ratio,
            smpl_c2=smpl_c2,
            smpl_c4=smpl_c4,
            shared_body_pose=shared_body_pose,
            shared_beta=shared_beta,
        )

        has_sam3d_dense_c2 = (
            sam3d_dense_list_c2 is not None
            and any(obs is not None for obs in sam3d_dense_list_c2)
        )
        has_sam3d_dense_c4 = (
            sam3d_dense_list_c4 is not None
            and any(obs is not None for obs in sam3d_dense_list_c4)
        )
        sam3d_dense_sample_ids = (
            load_default_sam3d_sample_ids(num_points=self.sam3d_dense_num_points)
            if has_sam3d_dense_c2 or has_sam3d_dense_c4
            else None
        )

        def make_loss_fn(
            compute_sil_c2: bool,
            compute_sil_c4: bool,
            compute_densekp_c2: bool,
            compute_densekp_c4: bool,
            compute_sam3d_dense_c2: bool,
            compute_sam3d_dense_c4: bool,
        ):
            verts_c2, joints_c2, _ = forward_smplx_tensor_batch(
                smplx_model=model,
                global_orient=smpl_c2.orient,
                body_pose=shared_body_pose.unsqueeze(0).expand(n_frame, -1)
                if shared_body_pose.dim() == 1
                else shared_body_pose,
                left_hand_pose=smpl_c2.left_hand_pose,
                right_hand_pose=smpl_c2.right_hand_pose,
                betas=shared_beta.unsqueeze(0).expand(n_frame, -1),
                transl=smpl_c2.transl,
                scale=smpl_c2.scale.expand(n_frame),
            )
            verts_c4, joints_c4, _ = forward_smplx_tensor_batch(
                smplx_model=model,
                global_orient=smpl_c4.orient,
                body_pose=shared_body_pose.unsqueeze(0).expand(n_frame, -1)
                if shared_body_pose.dim() == 1
                else shared_body_pose,
                left_hand_pose=smpl_c4.left_hand_pose,
                right_hand_pose=smpl_c4.right_hand_pose,
                betas=shared_beta.unsqueeze(0).expand(n_frame, -1),
                transl=smpl_c4.transl,
                scale=smpl_c4.scale.expand(n_frame),
            )

            losses_c2, _ = compute_keypoint_2d_loss_with_proj(
                joints_all=joints_c2,
                dets_tensor=kp_tensor_c2,
                keypoints_format=key_fmt_c2,
                project_fn=self._project_cam2,
                W=erp_w,
                H=erp_h,
                conf_thresh=self.conf_thresh,
            )
            losses_c4, _ = compute_keypoint_2d_loss_with_proj(
                joints_all=joints_c4,
                dets_tensor=kp_tensor_c4,
                keypoints_format=key_fmt_c4,
                project_fn=self._project_cam2,
                W=erp_w,
                H=erp_h,
                conf_thresh=self.conf_thresh,
            )

            if compute_sil_c2:
                sil_losses_c2 = []
                for i in range(n_frame):
                    bpts = boundary_pts_list_c2[i] if boundary_pts_list_c2 else None
                    bbox = bbox_list_c2[i] if bbox_list_c2 else None
                    if bpts is not None and bbox is not None:
                        sil_losses_c2.append(
                            compute_silhouette_loss(
                                verts_c2[i], bpts, bbox, self._project_cam2, erp_w, erp_h
                            )
                        )
                loss_sil_c2 = torch.stack(sil_losses_c2).mean() if sil_losses_c2 else verts_c2.new_zeros(())
            else:
                loss_sil_c2 = verts_c2.new_zeros(())

            if compute_sil_c4:
                sil_losses_c4 = []
                for i in range(n_frame):
                    bpts = boundary_pts_list_c4[i] if boundary_pts_list_c4 else None
                    bbox = bbox_list_c4[i] if bbox_list_c4 else None
                    if bpts is not None and bbox is not None:
                        sil_losses_c4.append(
                            compute_silhouette_loss(
                                verts_c4[i], bpts, bbox, self._project_cam2, erp_w, erp_h
                            )
                        )
                loss_sil_c4 = torch.stack(sil_losses_c4).mean() if sil_losses_c4 else verts_c4.new_zeros(())
            else:
                loss_sil_c4 = verts_c4.new_zeros(())

            proxy_c2 = SimpleNamespace(
                transl=smpl_c2.transl,
                body_pose=shared_body_pose,
                left_hand_pose=smpl_c2.left_hand_pose,
                right_hand_pose=smpl_c2.right_hand_pose,
                beta=shared_beta,
            )
            proxy_c4 = SimpleNamespace(
                transl=smpl_c4.transl,
                body_pose=shared_body_pose,
                left_hand_pose=smpl_c4.left_hand_pose,
                right_hand_pose=smpl_c4.right_hand_pose,
                beta=shared_beta,
            )
            reg_c2 = self._common_regularization_losses(
                proxy_c2, shape_anchor, n_frame, pose_anchor=pose_anchor
            )
            reg_c4 = self._common_regularization_losses(
                proxy_c4, shape_anchor, n_frame, pose_anchor=pose_anchor
            )

            if compute_sam3d_dense_c2:
                projected_sam_c2 = self._project_cam2(verts_c2, erp_w, erp_h)
                loss_sam3d_dense_c2, _ = compute_sam3d_dense_sequence_reprojection_loss(
                    projected_points_2d=projected_sam_c2,
                    obs_list=sam3d_dense_list_c2,
                    sample_ids=sam3d_dense_sample_ids,
                    num_points=self.sam3d_dense_num_points,
                    compute_stats=False,
                )
            else:
                loss_sam3d_dense_c2 = verts_c2.new_zeros(())

            if compute_sam3d_dense_c4:
                projected_sam_c4 = self._project_cam2(verts_c4, erp_w, erp_h)
                loss_sam3d_dense_c4, _ = compute_sam3d_dense_sequence_reprojection_loss(
                    projected_points_2d=projected_sam_c4,
                    obs_list=sam3d_dense_list_c4,
                    sample_ids=sam3d_dense_sample_ids,
                    num_points=self.sam3d_dense_num_points,
                    compute_stats=False,
                )
            else:
                loss_sam3d_dense_c4 = verts_c4.new_zeros(())

            losses_all = {
                "body_2d_det_c2": losses_c2.get("body_2d_det", joints_c2.new_zeros(())),
                "body_2d_det_c4": losses_c4.get("body_2d_det", joints_c4.new_zeros(())),
                "loss_sil_c2": loss_sil_c2,
                "loss_sil_c4": loss_sil_c4,
                "loss_densekp_c2": self.densekp_loss.compute(verts_c2, densekp_list_c2, self._project_cam2, erp_w, erp_h) if compute_densekp_c2 else verts_c2.new_zeros(()),
                "loss_densekp_c4": self.densekp_loss.compute(verts_c4, densekp_list_c4, self._project_cam2, erp_w, erp_h) if compute_densekp_c4 else verts_c4.new_zeros(()),
                "loss_sam3d_dense_c2": loss_sam3d_dense_c2,
                "loss_sam3d_dense_c4": loss_sam3d_dense_c4,
            }
            losses_all.update(suffix_non_shape_losses(reg_c2, "c2"))
            losses_all.update(suffix_non_shape_losses(reg_c4, "c4"))
            losses_all.update(
                compute_shape_regularization_losses(
                    common_regularization_fn=self._common_regularization_losses,
                    shared_beta=shared_beta,
                    shared_body_pose=shared_body_pose,
                    shape_anchor=shape_anchor,
                    ref_param=smpl_c2,
                    n_frame=n_frame,
                )
            )
            return losses_all

        latest_loss = 0.0
        latest_losses_detail = {}

        for stage_name, get_params, max_iter, stage_stop, weight_mult in optimization_stages:
            stage_weights = {k: base_weights[k] * weight_mult.get(k, 1.0) for k in base_weights}
            compute_sil_c2 = stage_weights.get("loss_sil_c2", 0.0) > 0
            compute_sil_c4 = stage_weights.get("loss_sil_c4", 0.0) > 0
            compute_densekp_c2 = stage_weights.get("loss_densekp_c2", 0.0) > 0
            compute_densekp_c4 = stage_weights.get("loss_densekp_c4", 0.0) > 0
            compute_sam3d_dense_c2 = (
                stage_weights.get("loss_sam3d_dense_c2", 0.0) > 0
                and has_sam3d_dense_c2
            )
            compute_sam3d_dense_c4 = (
                stage_weights.get("loss_sam3d_dense_c4", 0.0) > 0
                and has_sam3d_dense_c4
            )
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
                losses_all = make_loss_fn(
                    compute_sil_c2=compute_sil_c2,
                    compute_sil_c4=compute_sil_c4,
                    compute_densekp_c2=compute_densekp_c2,
                    compute_densekp_c4=compute_densekp_c4,
                    compute_sam3d_dense_c2=compute_sam3d_dense_c2,
                    compute_sam3d_dense_c4=compute_sam3d_dense_c4,
                )
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
            for attr in ["orient", "transl", "left_hand_pose", "right_hand_pose", "beta"]:
                getattr(smpl_param, attr).requires_grad_(False)
        shared_beta.requires_grad_(False)
        shared_body_pose.requires_grad_(False)

        shared_beta_final = shared_beta.detach().clone()
        smpl_c2.beta = shared_beta_final.clone()
        smpl_c4.beta = shared_beta_final.clone()
        smpl_c2.body_pose = shared_body_pose.detach().clone()
        smpl_c4.body_pose = shared_body_pose.detach().clone()
        return smpl_c2, smpl_c4, shared_beta_final
