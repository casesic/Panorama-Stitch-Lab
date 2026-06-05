from __future__ import annotations

import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn

from .base import BaseSMPLXFitter
from .constraints import (
    DenseKPLoss,
    MaskFillLoss,
    compute_keypoint_2d_loss_with_proj,
    compute_shape_regularization_losses,
    compute_silhouette_loss,
    prepare_mask_fill_samples_for_sequence,
    suffix_non_shape_losses,
)
from .loss_config import DualCamLossWeights, build_base_weights, build_optimization_stages
from .projection import project_cam2, project_cam4, project_erp




class DualCamERPFitter(BaseSMPLXFitter):
    def __init__(
        self,
        gender: str = "neutral",
        R_cam4: torch.Tensor | None = None,
        use_silhouette_loss: bool = True,
        densekp_filter_path: str | None = None,
        enable_mask_fill_stage3: bool = False,
        image_root: str | None = None,
        seg_dir: str | None = None,
        bbox_dir: str | None = None,
    ):
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
        self.loss_weights = DualCamLossWeights()
        project_root = Path(__file__).resolve().parents[3]
        data_dir = project_root / "tools" / "CameraHMR" / "data" / "train-eval-utils"
        id_mask = None
        if densekp_filter_path is not None:
            from lib_pano.fitting.densekp_filter import load_densekp_filter

            id_mask = load_densekp_filter(densekp_filter_path)
        self.densekp_loss = DenseKPLoss(data_dir=data_dir, weight=self.loss_weights.densekp, id_mask=id_mask)
        if R_cam4 is None:
            raise ValueError("R_cam4 must be provided for DualCamERPFitter")
        self.R_cam4 = R_cam4.clone().detach().float()

        self.enable_mask_fill_stage3 = enable_mask_fill_stage3
        self.image_root = Path(image_root) if image_root else None
        self.seg_dir = Path(seg_dir) if seg_dir else None
        self.bbox_dir = Path(bbox_dir) if bbox_dir else None

        if self.enable_mask_fill_stage3:
            if self.image_root is None or self.seg_dir is None or self.bbox_dir is None:
                raise ValueError(
                    "mask_fill_stage3 requires --image_root, --seg_dir, --bbox_dir"
                )

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
        return project_erp(points_3d, W, H)

    @staticmethod
    def _project_cam2(points_3d: torch.Tensor, W: int, H: int) -> torch.Tensor:
        return project_cam2(points_3d, W, H)

    def _project_cam4(self, points_3d: torch.Tensor, W: int, H: int) -> torch.Tensor:
        return project_cam4(points_3d, W, H, self.R_cam4)

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
        output_dir: str | None = None,
        frame_indices: list[int] | None = None,
    ):
        from .smplx_backend import forward_smplx_tensor_batch

        assert len(cam2_dets) > 0
        assert len(cam2_dets) == len(cam4_dets)

        n_frame = len(cam2_dets)
        if frame_indices is None:
            frame_indices = list(range(n_frame))
        else:
            assert len(frame_indices) == n_frame
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

        def make_loss_fn(compute_sil_c2: bool, compute_sil_c4: bool, compute_densekp_c2: bool, compute_densekp_c4: bool):
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
            losses_all = {
                "body_2d_det_c2": losses_c2.get("body_2d_det", joints_c2.new_zeros(())),
                "body_2d_det_c4": losses_c4.get("body_2d_det", joints_c4.new_zeros(())),
                "loss_sil_c2": loss_sil_c2,
                "loss_sil_c4": loss_sil_c4,
                "loss_densekp_c2": self.densekp_loss.compute(verts_c2, densekp_list_c2, self._project_cam2, erp_w, erp_h) if compute_densekp_c2 else verts_c2.new_zeros(()),
                "loss_densekp_c4": self.densekp_loss.compute(verts_c4, densekp_list_c4, self._project_cam2, erp_w, erp_h) if compute_densekp_c4 else verts_c4.new_zeros(()),
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

        # ==================================================================
        # Stage 3 (optional): mask fill refine
        # ==================================================================
        if self.enable_mask_fill_stage3:
            import json

            from lib_pano.pipeline.bbox_io import load_bbox_dict

            print("\n=== Stage 3: mask fill refine ===")
            # Get current SMPL-X verts for all frames
            with torch.no_grad():
                verts_c2_s3, _, _ = forward_smplx_tensor_batch(
                    smplx_model=model,
                    global_orient=smpl_c2.orient,
                    body_pose=shared_body_pose.unsqueeze(0).expand(n_frame, -1)
                    if shared_body_pose.dim() == 1 else shared_body_pose,
                    left_hand_pose=smpl_c2.left_hand_pose,
                    right_hand_pose=smpl_c2.right_hand_pose,
                    betas=shared_beta.unsqueeze(0).expand(n_frame, -1),
                    transl=smpl_c2.transl,
                    scale=smpl_c2.scale.expand(n_frame),
                )
                verts_c4_s3, _, _ = forward_smplx_tensor_batch(
                    smplx_model=model,
                    global_orient=smpl_c4.orient,
                    body_pose=shared_body_pose.unsqueeze(0).expand(n_frame, -1)
                    if shared_body_pose.dim() == 1 else shared_body_pose,
                    left_hand_pose=smpl_c4.left_hand_pose,
                    right_hand_pose=smpl_c4.right_hand_pose,
                    betas=shared_beta.unsqueeze(0).expand(n_frame, -1),
                    transl=smpl_c4.transl,
                    scale=smpl_c4.scale.expand(n_frame),
                )

            verts_c2_np = [verts_c2_s3[i].cpu().numpy() for i in range(n_frame)]
            verts_c4_np = [verts_c4_s3[i].cpu().numpy() for i in range(n_frame)]

            bbox_c2 = load_bbox_dict(self.bbox_dir / "cam2_bboxes.json")
            bbox_c4 = load_bbox_dict(self.bbox_dir / "cam4_bboxes.json")

            rng_s3 = np.random.RandomState(0)

            print("  generating underfill samples for cam2 ...")
            samples_c2, stats_c2 = prepare_mask_fill_samples_for_sequence(
                frame_indices=frame_indices,
                verts_list=verts_c2_np,
                image_root=self.image_root,
                seg_dir=self.seg_dir,
                bbox_dict=bbox_c2,
                faces=model.faces.numpy() if hasattr(model.faces, "numpy") else model.faces,
                erp_w=erp_w,
                erp_h=erp_h,
                cam="cam2",
                rng=rng_s3,
            )
            print("  generating underfill samples for cam4 ...")
            samples_c4, stats_c4 = prepare_mask_fill_samples_for_sequence(
                frame_indices=frame_indices,
                verts_list=verts_c4_np,
                image_root=self.image_root,
                seg_dir=self.seg_dir,
                bbox_dict=bbox_c4,
                faces=model.faces.numpy() if hasattr(model.faces, "numpy") else model.faces,
                erp_w=erp_w,
                erp_h=erp_h,
                cam="cam4",
                rng=rng_s3,
            )

            # Cam weights based on mask area
            total_mask_c2 = sum(s.get("mask_area", 0) for s in stats_c2 if "mask_area" in s)
            total_mask_c4 = sum(s.get("mask_area", 0) for s in stats_c4 if "mask_area" in s)
            sum_mask = total_mask_c2 + total_mask_c4
            w_cam_c2 = np.clip(total_mask_c2 / max(sum_mask, 1), 0.3, 0.7) if sum_mask > 0 else 0.5
            w_cam_c4 = 1.0 - w_cam_c2
            print(f"  mask area: c2={total_mask_c2}, c4={total_mask_c4} -> w_c2={w_cam_c2:.3f}, w_c4={w_cam_c4:.3f}")

            # Save stage3 summary early (before optimization, so we have a record)
            summary = {
                "cam2": {str(frame_indices[i]): s for i, s in enumerate(stats_c2)},
                "cam4": {str(frame_indices[i]): s for i, s in enumerate(stats_c4)},
                "cam_weights": {"cam2": float(w_cam_c2), "cam4": float(w_cam_c4)},
            }

            mask_fill_loss_fn = MaskFillLoss(erp_w, erp_h, robust_scale=100.0)

            # Stage 3 V2: optimize shared_beta + shared_body_pose, strong pose anchor
            stage3_params = [shared_beta, shared_body_pose]
            stage3_optimizer = torch.optim.LBFGS(
                stage3_params,
                line_search_fn="strong_wolfe",
                lr=self.lr_lbfgs * 0.5,
                max_iter=self.max_iter_closure,
            )

            # V2 weights — strong mask_fill, moderate keypoint/densekp, strong anchors
            s3_mask_fill_weight = 80.0
            s3_keypoint_weight = n_frame * self.loss_weights.body_2d * 0.15
            s3_densekp_weight = n_frame * self.loss_weights.densekp * 0.15
            s3_shape_anchor_weight = n_frame * self.loss_weights.shape_anchor * 2.0
            s3_shape_norm_weight = n_frame * self.loss_weights.shape_norm * 0.5
            s3_shape_prior_weight = n_frame * self.loss_weights.shape_prior * 0.5
            s3_shape_bound_weight = n_frame * self.loss_weights.shape_bound * 1.0
            s3_pose_anchor_weight = n_frame * self.loss_weights.pose_anchor * 20.0

            s3_max_iter = 30

            beta_before_s3 = shared_beta.detach().clone()
            pose_before_s3 = shared_body_pose.detach().clone()
            print(f"  beta before S3: {beta_before_s3.cpu().numpy().tolist()}")

            def stage3_closure():
                for p in stage3_params:
                    p.grad = None
                stage3_optimizer.zero_grad()

                verts_c2_s3l, joints_c2_s3l, _ = forward_smplx_tensor_batch(
                    smplx_model=model,
                    global_orient=smpl_c2.orient.detach(),
                    body_pose=shared_body_pose.unsqueeze(0).expand(n_frame, -1)
                    if shared_body_pose.dim() == 1 else shared_body_pose,
                    left_hand_pose=smpl_c2.left_hand_pose.detach(),
                    right_hand_pose=smpl_c2.right_hand_pose.detach(),
                    betas=shared_beta.unsqueeze(0).expand(n_frame, -1),
                    transl=smpl_c2.transl.detach(),
                    scale=smpl_c2.scale.expand(n_frame),
                )
                verts_c4_s3l, joints_c4_s3l, _ = forward_smplx_tensor_batch(
                    smplx_model=model,
                    global_orient=smpl_c4.orient.detach(),
                    body_pose=shared_body_pose.unsqueeze(0).expand(n_frame, -1)
                    if shared_body_pose.dim() == 1 else shared_body_pose,
                    left_hand_pose=smpl_c4.left_hand_pose.detach(),
                    right_hand_pose=smpl_c4.right_hand_pose.detach(),
                    betas=shared_beta.unsqueeze(0).expand(n_frame, -1),
                    transl=smpl_c4.transl.detach(),
                    scale=smpl_c4.scale.expand(n_frame),
                )

                # Keypoint losses (cam2 and cam4)
                kp_c2, _ = compute_keypoint_2d_loss_with_proj(
                    joints_all=joints_c2_s3l, dets_tensor=kp_tensor_c2,
                    keypoints_format=key_fmt_c2, project_fn=self._project_cam2,
                    W=erp_w, H=erp_h, conf_thresh=self.conf_thresh,
                )
                kp_c4, _ = compute_keypoint_2d_loss_with_proj(
                    joints_all=joints_c4_s3l, dets_tensor=kp_tensor_c4,
                    keypoints_format=key_fmt_c4, project_fn=self._project_cam2,
                    W=erp_w, H=erp_h, conf_thresh=self.conf_thresh,
                )

                # DenseKP losses
                dkp_c2 = self.densekp_loss.compute(verts_c2_s3l, densekp_list_c2, self._project_cam2, erp_w, erp_h)
                dkp_c4 = self.densekp_loss.compute(verts_c4_s3l, densekp_list_c4, self._project_cam2, erp_w, erp_h)

                # Mask fill losses
                mf_c2 = mask_fill_loss_fn.compute(verts_c2_s3l, samples_c2, self._project_cam2)
                mf_c4 = mask_fill_loss_fn.compute(verts_c4_s3l, samples_c4, self._project_cam2)

                # Shape regularization
                shape_losses = {}
                shape_losses["loss_shape_prior"] = torch.mean(shared_beta ** 2)
                shape_losses["loss_shape_bound"] = torch.mean(
                    torch.relu(torch.abs(shared_beta) - self.current_shape_soft_bound) ** 2
                )
                shape_losses["loss_shape_anchor"] = torch.mean((shared_beta - shape_anchor) ** 2)
                shape_losses["loss_shape_norm"] = torch.relu(torch.norm(shared_beta) - self.shape_norm_bound) ** 2

                # Pose anchor (body_pose stays frozen in stage3)
                pose_anch = torch.mean((shared_body_pose.detach() - pose_anchor) ** 2)

                zero = shared_beta.new_zeros(())
                loss = zero
                raw_losses = {}  # raw (unweighted) for diagnostics
                weighted_losses = {}  # weighted contributions

                def _add(k, v, w):
                    nonlocal loss
                    raw_losses[k] = float(v.detach().cpu().item())
                    loss = loss + v * w
                    weighted_losses[k] = float(v.detach().cpu().item()) * w

                _add("kp_c2", kp_c2.get("body_2d_det", zero), s3_keypoint_weight)
                _add("kp_c4", kp_c4.get("body_2d_det", zero), s3_keypoint_weight)
                _add("dkp_c2", dkp_c2, s3_densekp_weight)
                _add("dkp_c4", dkp_c4, s3_densekp_weight)
                _add("mf_c2", mf_c2, s3_mask_fill_weight * w_cam_c2)
                _add("mf_c4", mf_c4, s3_mask_fill_weight * w_cam_c4)
                _add("shape_prior", shape_losses["loss_shape_prior"], s3_shape_prior_weight)
                _add("shape_bound", shape_losses["loss_shape_bound"], s3_shape_bound_weight)
                _add("shape_anchor", shape_losses["loss_shape_anchor"], s3_shape_anchor_weight)
                _add("shape_norm", shape_losses["loss_shape_norm"], s3_shape_norm_weight)
                _add("pose_anchor", pose_anch, s3_pose_anchor_weight)

                nonlocal s3_latest_loss, s3_latest_detail
                s3_latest_loss = float(loss.detach().cpu().item())
                s3_latest_detail = {"raw": raw_losses, "weighted": weighted_losses}
                loss.backward()
                return loss

            s3_latest_loss = 0.0
            s3_latest_detail = {}
            print("  opt (stage3_mask_fill)...")
            loss_prev = 1e9
            for it in range(s3_max_iter):
                stage3_optimizer.step(stage3_closure)
                shared_beta.data.clamp_(-self.shape_hard_clip, self.shape_hard_clip)
                if it >= 10 and loss_prev - s3_latest_loss < loss_prev * self.stop_ratio:
                    break
                loss_prev = s3_latest_loss
                raw = s3_latest_detail.get("raw", {})
                wgt = s3_latest_detail.get("weighted", {})
                detail_parts = []
                for k in sorted(wgt.keys()):
                    rv = raw.get(k, 0.0)
                    wv = wgt.get(k, 0.0)
                    detail_parts.append(f"{k}={rv:.4f}/{wv:.4f}")
                detail_str = "  ".join(detail_parts)
                print(f"[stage3_mask_fill] iter={it+1:03d}/{s3_max_iter} total={s3_latest_loss:.4f}  |  {detail_str}")

            beta_after_s3 = shared_beta.detach().clone()
            delta_beta = beta_after_s3 - beta_before_s3
            print(f"  beta after S3: {beta_after_s3.cpu().numpy().tolist()}")
            print(f"  delta beta:   {delta_beta.cpu().numpy().tolist()}")
            print(f"  max|delta|={delta_beta.abs().max().item():.6f}  |delta|_2={delta_beta.norm().item():.6f}")

            delta_pose = shared_body_pose.detach() - pose_before_s3
            print(f"  max|delta_pose|={delta_pose.abs().max().item():.6f}  |delta_pose|_2={delta_pose.norm().item():.6f}")

            # Save post-optimization summary
            with torch.no_grad():
                verts_c2_post, _, _ = forward_smplx_tensor_batch(
                    smplx_model=model,
                    global_orient=smpl_c2.orient,
                    body_pose=shared_body_pose.unsqueeze(0).expand(n_frame, -1)
                    if shared_body_pose.dim() == 1 else shared_body_pose,
                    left_hand_pose=smpl_c2.left_hand_pose,
                    right_hand_pose=smpl_c2.right_hand_pose,
                    betas=shared_beta.unsqueeze(0).expand(n_frame, -1),
                    transl=smpl_c2.transl,
                    scale=smpl_c2.scale.expand(n_frame),
                )
                verts_c4_post, _, _ = forward_smplx_tensor_batch(
                    smplx_model=model,
                    global_orient=smpl_c4.orient,
                    body_pose=shared_body_pose.unsqueeze(0).expand(n_frame, -1)
                    if shared_body_pose.dim() == 1 else shared_body_pose,
                    left_hand_pose=smpl_c4.left_hand_pose,
                    right_hand_pose=smpl_c4.right_hand_pose,
                    betas=shared_beta.unsqueeze(0).expand(n_frame, -1),
                    transl=smpl_c4.transl,
                    scale=smpl_c4.scale.expand(n_frame),
                )
            verts_c2_post_np = [verts_c2_post[i].cpu().numpy() for i in range(n_frame)]
            verts_c4_post_np = [verts_c4_post[i].cpu().numpy() for i in range(n_frame)]

            post_samples_c2, post_stats_c2 = prepare_mask_fill_samples_for_sequence(
                frame_indices=frame_indices,
                verts_list=verts_c2_post_np,
                image_root=self.image_root,
                seg_dir=self.seg_dir,
                bbox_dict=bbox_c2,
                faces=model.faces.numpy() if hasattr(model.faces, "numpy") else model.faces,
                erp_w=erp_w,
                erp_h=erp_h,
                cam="cam2",
                rng=rng_s3,
            )
            post_samples_c4, post_stats_c4 = prepare_mask_fill_samples_for_sequence(
                frame_indices=frame_indices,
                verts_list=verts_c4_post_np,
                image_root=self.image_root,
                seg_dir=self.seg_dir,
                bbox_dict=bbox_c4,
                faces=model.faces.numpy() if hasattr(model.faces, "numpy") else model.faces,
                erp_w=erp_w,
                erp_h=erp_h,
                cam="cam4",
                rng=rng_s3,
            )

            summary["cam2"] = {str(frame_indices[i]): s for i, s in enumerate(post_stats_c2)}
            summary["cam4"] = {str(frame_indices[i]): s for i, s in enumerate(post_stats_c4)}

            print("  Stage 3 mask fill refine completed.")

            if output_dir is not None:
                summary_path = Path(output_dir) / "mask_fill_stage3_summary.json"
                summary_path.parent.mkdir(parents=True, exist_ok=True)
                with open(summary_path, "w", encoding="utf-8") as f:
                    json.dump(summary, f, indent=2, ensure_ascii=False)
                print(f"  Stage 3 summary saved to {summary_path}")

        # ==================================================================
        # End Stage 3
        # ==================================================================

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
