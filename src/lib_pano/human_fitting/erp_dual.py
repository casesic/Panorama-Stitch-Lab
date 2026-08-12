"""HMR 初值 + 单主观测源（DenseKP 或 SAM3D dense）的两阶段双视角 ERP SMPL-X 优化器。"""

from __future__ import annotations

import math
from pathlib import Path
from types import SimpleNamespace
from typing import Literal

import torch
from torch import nn

from .base import BaseSMPLXFitter
from .constraints import (
    DenseKPLoss,
    compute_shape_regularization_losses,
    suffix_non_shape_losses,
)
from .loss_config import (
    DualCamLossWeights,
    OptimizationProfile,
    build_base_weights,
    build_optimization_stages,
)
from .projection import project_cam2, project_cam4, project_erp
from .constraints.sam3d_dense import (
    compute_sam3d_dense_sequence_reprojection_loss,
    load_default_sam3d_sample_ids,
)


PrimaryObservationSource = Literal["densekp", "sam3d_dense"]

BETA_ATANH_EPS = 1e-6


class DualCamERPFitter(BaseSMPLXFitter):
    """两阶段双相机 ERP 优化器：HMR 初值 + 单主观测源（DenseKP 或 SAM3D dense），cam2/cam4 共享 beta 与 body_pose。"""

    def __init__(
        self,
        primary_observation_source: PrimaryObservationSource,
        gender: str = "neutral",
        R_cam4: torch.Tensor | None = None,
        densekp_filter_path: str | None = None,
        sam3d_dense_num_points: int = 1024,
        optimization_profile: OptimizationProfile = "legacy",
        beta_abs_limit: float = 2.5,
    ):
        super().__init__(gender=gender)
        if primary_observation_source not in ("densekp", "sam3d_dense"):
            raise ValueError(f"primary_observation_source must be 'densekp' or 'sam3d_dense', got {primary_observation_source!r}")
        if beta_abs_limit <= 0:
            raise ValueError(f"beta_abs_limit must be > 0, got {beta_abs_limit}")
        self.primary_observation_source: PrimaryObservationSource = primary_observation_source
        self.conf_thresh = 0.57
        self.fps = 30.0
        self.max_speed_mps = 3.0
        self.shape_norm_bound = 0.9
        self.shape_hard_clip = 0.55
        self.current_shape_soft_bound = 0.35
        self.lr_lbfgs = 0.3
        self.max_iter_closure = 10
        self.stop_ratio = 1e-3
        self.hand_overall_weight = 2.0
        self.sam3d_dense_num_points = int(sam3d_dense_num_points)
        self.loss_weights = DualCamLossWeights()
        if R_cam4 is None:
            raise ValueError("R_cam4 must be provided for DualCamERPFitter")
        self.R_cam4 = R_cam4.clone().detach().float()

        self.optimization_profile: OptimizationProfile = optimization_profile
        if optimization_profile not in ("legacy", "staged_charbonnier_parity"):
            raise ValueError(f"Unknown optimization_profile: {optimization_profile!r}")
        self.beta_abs_limit = float(beta_abs_limit)
        self.last_fit_diagnostics = None

        self.densekp_loss = None
        self._sam3d_dense_sample_ids = None
        if self.primary_observation_source == "densekp":
            project_root = Path(__file__).resolve().parents[3]
            data_dir = project_root / "tools" / "CameraHMR" / "data" / "train-eval-utils"
            id_mask = None
            if densekp_filter_path is not None:
                from lib_pano.fitting_io.densekp_filter import load_densekp_filter
                id_mask = load_densekp_filter(densekp_filter_path)
            self.densekp_loss = DenseKPLoss(data_dir=data_dir, weight=self.loss_weights.densekp, id_mask=id_mask)
        else:
            if densekp_filter_path is not None:
                raise ValueError("densekp_filter_path is only valid for primary_observation_source='densekp'")
            self._sam3d_dense_sample_ids = load_default_sam3d_sample_ids(num_points=self.sam3d_dense_num_points)

    def _primary_obs_points_2d(self, obs):
        if self.primary_observation_source == "densekp":
            return obs.keypoints_2d
        return obs.points_2d

    def _transl_init_from_primary(self, primary_observations, image_size):
        erp_w, erp_h = image_size
        n_frame = len(primary_observations)
        transl = torch.zeros((n_frame, 3), dtype=torch.float32)
        valid_mask = torch.zeros(n_frame, dtype=torch.bool)
        for i, obs in enumerate(primary_observations):
            if obs is None or not bool(obs.valid.any()):
                continue
            pts = self._primary_obs_points_2d(obs)[obs.valid]
            x1, y1 = float(pts[:, 0].min()), float(pts[:, 1].min())
            x2, y2 = float(pts[:, 0].max()), float(pts[:, 1].max())
            u_c = (x1 + x2) / 2.0; v_c = (y1 + y2) / 2.0
            lon = (u_c / erp_w - 0.5) * (2.0 * math.pi)
            lat = (v_c / erp_h - 0.5) * math.pi
            dir_x = math.sin(lon) * math.cos(lat)
            dir_y = -math.sin(lat)
            dir_z = math.cos(lon) * math.cos(lat)
            bbox_h_px = max(1.0, y2 - y1)
            fov_h = bbox_h_px / erp_h * math.pi
            dist = min(15.0, max(0.5, 1.7 / (2.0 * max(1e-3, math.tan(fov_h / 2.0)))))
            transl[i, 0] = dir_x * dist; transl[i, 1] = dir_y * dist; transl[i, 2] = dir_z * dist
            valid_mask[i] = True
        return transl, valid_mask

    @staticmethod
    def _project_erp(points_3d, W, H): return project_erp(points_3d, W, H)

    @staticmethod
    def _project_cam2(points_3d, W, H): return project_cam2(points_3d, W, H)

    def _project_cam4(self, points_3d, W, H): return project_cam4(points_3d, W, H, self.R_cam4)

    def _init_params(self, smplx_init_c2, smplx_init_c4, primary_observations_c2, primary_observations_c4, image_size):
        n_frame = len(primary_observations_c2)
        smpl_c2 = smplx_init_c2 if smplx_init_c2 is not None else self._build_default_smplx(n_frame)
        smpl_c4 = smplx_init_c4 if smplx_init_c4 is not None else self._build_default_smplx(n_frame)
        transl_init_c2, transl_valid_c2 = self._transl_init_from_primary(primary_observations_c2, image_size)
        transl_init_c4, transl_valid_c4 = self._transl_init_from_primary(primary_observations_c4, image_size)
        for i in range(n_frame):
            if bool(transl_valid_c2[i]):
                smpl_c2.transl.data[i] = transl_init_c2[i].to(smpl_c2.transl.device, smpl_c2.transl.dtype)
            if bool(transl_valid_c4[i]):
                smpl_c4.transl.data[i] = transl_init_c4[i].to(smpl_c4.transl.device, smpl_c4.transl.dtype)
        smpl_c2 = self._prepare_smplx_param_for_optimization(smpl_c2)
        smpl_c4 = self._prepare_smplx_param_for_optimization(smpl_c4)
        smpl_c2.scale = smpl_c2.scale.reshape(-1)[:1].clone().detach()
        smpl_c4.scale = smpl_c4.scale.reshape(-1)[:1].clone().detach()
        beta_c2 = getattr(smpl_c2, "beta", None)
        beta_c4 = getattr(smpl_c4, "beta", None)
        if beta_c2 is None and beta_c4 is None:
            beta_init = torch.zeros(10, dtype=smpl_c2.transl.dtype, device=smpl_c2.transl.device)
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

    def _validate_primary_observations(self, observations):
        from lib_pano.fitting_io.densekp_io import DenseKPResult
        from lib_pano.fitting_io.sam3d_dense_io import SAM3DDenseResult
        expected = DenseKPResult if self.primary_observation_source == "densekp" else SAM3DDenseResult
        for i, obs in enumerate(observations):
            if obs is None: continue
            if not isinstance(obs, expected):
                raise TypeError(f"primary observation at index {i} must be {expected.__name__} or None, got {type(obs).__name__}")

    def _compute_primary_loss(self, verts_seq, observations, erp_w, erp_h):
        if self.primary_observation_source == "densekp":
            return self.densekp_loss.compute(verts_seq, observations, self._project_cam2, erp_w, erp_h)
        projected = self._project_cam2(verts_seq, erp_w, erp_h)
        loss, _ = compute_sam3d_dense_sequence_reprojection_loss(
            projected_points_2d=projected, obs_list=observations,
            sample_ids=self._sam3d_dense_sample_ids, num_points=self.sam3d_dense_num_points, compute_stats=False,
        )
        return loss

    def fit(self, image_size, smplx_init_c2, smplx_init_c4, primary_observations_c2, primary_observations_c4):
        from .smplx_sequence import forward_smplx_tensor_batch

        primary_observations_c2 = list(primary_observations_c2)
        primary_observations_c4 = list(primary_observations_c4)
        n_frame = len(primary_observations_c2)
        if n_frame == 0: raise ValueError("fit() requires at least one frame of primary observations")
        if len(primary_observations_c4) != n_frame:
            raise ValueError(f"primary_observations_c2/c4 length mismatch: {n_frame} vs {len(primary_observations_c4)}")
        self._validate_primary_observations(primary_observations_c2)
        self._validate_primary_observations(primary_observations_c4)
        erp_w, erp_h = image_size
        model = self._get_model()

        smpl_c2, smpl_c4, shared_beta, shared_body_pose = self._init_params(
            smplx_init_c2=smplx_init_c2, smplx_init_c4=smplx_init_c4,
            primary_observations_c2=primary_observations_c2, primary_observations_c4=primary_observations_c4,
            image_size=image_size,
        )

        diagnostics = {
            "optimization_profile": self.optimization_profile,
            "primary_observation_source": self.primary_observation_source,
            "n_frame": n_frame,
            "beta_abs_limit": self.beta_abs_limit if self.optimization_profile == "staged_charbonnier_parity" else None,
            "stages": [],
        }

        if self.optimization_profile == "staged_charbonnier_parity":
            result = self._fit_staged(
                n_frame, erp_w, erp_h, model, smpl_c2, smpl_c4, shared_beta, shared_body_pose,
                primary_observations_c2, primary_observations_c4, diagnostics,
            )
        else:
            result = self._fit_legacy(
                n_frame, erp_w, erp_h, model, smpl_c2, smpl_c4, shared_beta, shared_body_pose,
                primary_observations_c2, primary_observations_c4, diagnostics,
            )

        self.last_fit_diagnostics = diagnostics
        return result

    # ============================================================
    # Legacy path (unchanged behavior)
    # ============================================================

    def _fit_legacy(self, n_frame, erp_w, erp_h, model, smpl_c2, smpl_c4, shared_beta, shared_body_pose,
                    primary_observations_c2, primary_observations_c4, diagnostics):
        from .smplx_sequence import forward_smplx_tensor_batch

        shape_anchor = shared_beta.clone().detach()
        pose_anchor = shared_body_pose.clone().detach()
        base_weights = build_base_weights(n_frame=n_frame, primary_observation_source=self.primary_observation_source,
                                          weights=self.loss_weights, optimization_profile="legacy")
        optimization_stages = build_optimization_stages(stop_ratio=self.stop_ratio, smpl_c2=smpl_c2, smpl_c4=smpl_c4,
                                                        body_pose_param=shared_body_pose,
                                                        legacy_shared_beta=shared_beta, optimization_profile="legacy")
        has_primary_c2 = any(obs is not None for obs in primary_observations_c2)
        has_primary_c4 = any(obs is not None for obs in primary_observations_c4)

        def make_loss_fn(compute_primary_c2, compute_primary_c4):
            verts_c2, _, _ = forward_smplx_tensor_batch(
                smplx_model=model, global_orient=smpl_c2.orient,
                body_pose=shared_body_pose.unsqueeze(0).expand(n_frame, -1) if shared_body_pose.dim() == 1 else shared_body_pose,
                left_hand_pose=smpl_c2.left_hand_pose, right_hand_pose=smpl_c2.right_hand_pose,
                betas=shared_beta.unsqueeze(0).expand(n_frame, -1), transl=smpl_c2.transl, scale=smpl_c2.scale.expand(n_frame),
            )
            verts_c4, _, _ = forward_smplx_tensor_batch(
                smplx_model=model, global_orient=smpl_c4.orient,
                body_pose=shared_body_pose.unsqueeze(0).expand(n_frame, -1) if shared_body_pose.dim() == 1 else shared_body_pose,
                left_hand_pose=smpl_c4.left_hand_pose, right_hand_pose=smpl_c4.right_hand_pose,
                betas=shared_beta.unsqueeze(0).expand(n_frame, -1), transl=smpl_c4.transl, scale=smpl_c4.scale.expand(n_frame),
            )
            proxy_c2 = SimpleNamespace(transl=smpl_c2.transl, body_pose=shared_body_pose,
                                       left_hand_pose=smpl_c2.left_hand_pose, right_hand_pose=smpl_c2.right_hand_pose, beta=shared_beta)
            proxy_c4 = SimpleNamespace(transl=smpl_c4.transl, body_pose=shared_body_pose,
                                       left_hand_pose=smpl_c4.left_hand_pose, right_hand_pose=smpl_c4.right_hand_pose, beta=shared_beta)
            reg_c2 = self._common_regularization_losses(proxy_c2, shape_anchor, n_frame, pose_anchor=pose_anchor)
            reg_c4 = self._common_regularization_losses(proxy_c4, shape_anchor, n_frame, pose_anchor=pose_anchor)
            loss_primary_c2 = self._compute_primary_loss(verts_c2, primary_observations_c2, erp_w, erp_h) if compute_primary_c2 else verts_c2.new_zeros(())
            loss_primary_c4 = self._compute_primary_loss(verts_c4, primary_observations_c4, erp_w, erp_h) if compute_primary_c4 else verts_c4.new_zeros(())
            losses_all = {"loss_primary_obs_c2": loss_primary_c2, "loss_primary_obs_c4": loss_primary_c4}
            losses_all.update(suffix_non_shape_losses(reg_c2, "c2"))
            losses_all.update(suffix_non_shape_losses(reg_c4, "c4"))
            losses_all.update(compute_shape_regularization_losses(
                common_regularization_fn=self._common_regularization_losses, shared_beta=shared_beta,
                shared_body_pose=shared_body_pose, shape_anchor=shape_anchor, ref_param=smpl_c2, n_frame=n_frame,
            ))
            return losses_all

        latest_loss = 0.0; latest_losses_detail = {}
        for stage_name, get_params, max_iter, stage_stop, weight_mult in optimization_stages:
            stage_weights = {k: base_weights[k] * weight_mult.get(k, 1.0) for k in base_weights}
            compute_c2 = stage_weights.get("loss_primary_obs_c2", 0.0) > 0 and has_primary_c2
            compute_c4 = stage_weights.get("loss_primary_obs_c4", 0.0) > 0 and has_primary_c4
            params = get_params()
            param_names = []  # legacy: no per-profile names, diagnostics records shapes only
            optimizer = torch.optim.LBFGS(params, line_search_fn="strong_wolfe", lr=self.lr_lbfgs, max_iter=self.max_iter_closure)
            closure_count = [0]
            def closure():
                nonlocal latest_loss, latest_losses_detail
                closure_count[0] += 1
                for p in params: p.grad = None
                optimizer.zero_grad()
                losses_all = make_loss_fn(compute_primary_c2=compute_c2, compute_primary_c4=compute_c4)
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
                if it > 0 and loss_prev - latest_loss < loss_prev * stage_stop: break
                loss_prev = latest_loss
                detail_str = "  ".join(f"{k}={latest_losses_detail.get(k, 0.0) * stage_weights.get(k, 1.0):.4f}"
                                       for k in stage_weights if stage_weights.get(k, 0.0) > 0)
                print(f"[{stage_name}] iter={it+1:03d}/{max_iter} total={latest_loss:.4f}  |  {detail_str}")
            diagnostics["stages"].append({
                "name": stage_name,
                "optimized_parameter_names": param_names,
                "optimized_parameter_shapes": [list(p.shape) for p in params] if params else [],
                "effective_weights": {k: base_weights[k] * weight_mult.get(k, 1.0) for k in base_weights},
                "closure_count": closure_count[0],
                "final_total_loss": latest_loss,
                "final_weighted_losses": {k: latest_losses_detail.get(k, 0.0) * stage_weights.get(k, 1.0) for k in stage_weights},
            })

        for smpl_param in (smpl_c2, smpl_c4):
            for attr in ["orient", "transl", "left_hand_pose", "right_hand_pose", "beta"]:
                getattr(smpl_param, attr).requires_grad_(False)
        shared_beta.requires_grad_(False); shared_body_pose.requires_grad_(False)
        shared_beta_final = shared_beta.detach().clone()
        smpl_c2.beta = shared_beta_final.clone(); smpl_c4.beta = shared_beta_final.clone()
        smpl_c2.body_pose = shared_body_pose.detach().clone(); smpl_c4.body_pose = shared_body_pose.detach().clone()

        beta_anchor_vals = shape_anchor.detach().cpu().tolist()
        beta_final_vals = shared_beta_final.detach().cpu().tolist()
        diagnostics.update({
            "beta_parameterization": "legacy_direct",
            "beta_anchor": beta_anchor_vals, "beta_anchor_max_abs": max(abs(v) for v in beta_anchor_vals),
            "beta_final": beta_final_vals, "beta_final_max_abs": max(abs(v) for v in beta_final_vals),
            "beta_saturation_ratio": None,
            "beta_delta_norm": sum((a - b) ** 2 for a, b in zip(beta_anchor_vals, beta_final_vals)) ** 0.5,
        })
        return smpl_c2, smpl_c4, shared_beta_final

    # ============================================================
    # Staged charbonnier parity path
    # ============================================================

    def _fit_staged(self, n_frame, erp_w, erp_h, model, smpl_c2, smpl_c4, shared_beta, shared_body_pose,
                    primary_observations_c2, primary_observations_c4, diagnostics):
        from .smplx_sequence import forward_smplx_tensor_batch

        limit = self.beta_abs_limit
        beta_anchor = shared_beta.detach().clone().reshape(-1)
        ba_max = float(beta_anchor.abs().max())
        safe_bound = limit * (1.0 - BETA_ATANH_EPS)
        if ba_max >= safe_bound:
            raise RuntimeError(f"INFEASIBLE_BETA_ANCHOR: max_abs={ba_max:.6f} >= safe_bound={safe_bound:.6f}; increase --beta_abs_limit")
        beta_latent = nn.Parameter(torch.atanh(beta_anchor / limit))

        diagnostics.update({
            "beta_parameterization": "tanh_latent",
            "beta_anchor": beta_anchor.cpu().tolist(), "beta_anchor_max_abs": ba_max,
        })

        pose_anchor = shared_body_pose.clone().detach()
        base_weights = build_base_weights(n_frame=n_frame, primary_observation_source=self.primary_observation_source,
                                          weights=self.loss_weights, optimization_profile="staged_charbonnier_parity")
        optimization_stages = build_optimization_stages(
            stop_ratio=self.stop_ratio, smpl_c2=smpl_c2, smpl_c4=smpl_c4,
            body_pose_param=shared_body_pose, beta_latent_param=beta_latent,
            optimization_profile="staged_charbonnier_parity",
        )

        def _beta_physical():
            return limit * torch.tanh(beta_latent)

        def _beta_physical_expanded():
            return _beta_physical().unsqueeze(0).expand(n_frame, -1)

        has_primary_c2 = any(obs is not None for obs in primary_observations_c2)
        has_primary_c4 = any(obs is not None for obs in primary_observations_c4)

        def make_loss_fn(include_pose_anchor, include_shape_anchor):
            bp_exp = shared_body_pose.unsqueeze(0).expand(n_frame, -1) if shared_body_pose.dim() == 1 else shared_body_pose

            verts_c2, _, _ = forward_smplx_tensor_batch(
                smplx_model=model, global_orient=smpl_c2.orient, body_pose=bp_exp,
                left_hand_pose=smpl_c2.left_hand_pose, right_hand_pose=smpl_c2.right_hand_pose,
                betas=_beta_physical_expanded(), transl=smpl_c2.transl, scale=smpl_c2.scale.expand(n_frame),
            )
            verts_c4, _, _ = forward_smplx_tensor_batch(
                smplx_model=model, global_orient=smpl_c4.orient, body_pose=bp_exp,
                left_hand_pose=smpl_c4.left_hand_pose, right_hand_pose=smpl_c4.right_hand_pose,
                betas=_beta_physical_expanded(), transl=smpl_c4.transl, scale=smpl_c4.scale.expand(n_frame),
            )

            losses = {
                "loss_primary_obs_c2": self._compute_primary_loss(verts_c2, primary_observations_c2, erp_w, erp_h) if has_primary_c2 else verts_c2.new_zeros(()),
                "loss_primary_obs_c4": self._compute_primary_loss(verts_c4, primary_observations_c4, erp_w, erp_h) if has_primary_c4 else verts_c4.new_zeros(()),
            }
            if include_pose_anchor:
                losses["loss_pose_anchor_shared"] = torch.mean((shared_body_pose - pose_anchor) ** 2)
            if include_shape_anchor:
                losses["loss_shape_anchor"] = torch.mean((_beta_physical() - beta_anchor) ** 2)
            return losses

        latest_loss = 0.0; latest_losses_detail = {}
        for stage_name, get_params, max_iter, stage_stop, weight_mult in optimization_stages:
            include_pa = weight_mult.get("loss_pose_anchor_shared", 0.0) > 0
            include_sa = weight_mult.get("loss_shape_anchor", 0.0) > 0
            stage_weights = {k: base_weights[k] * weight_mult.get(k, 1.0) for k in base_weights}
            params = get_params()
            param_names = []
            if stage_name == "stage1_camera_alignment":
                param_names = ["cam2_orient", "cam2_transl", "cam4_orient", "cam4_transl"]
            elif stage_name == "stage2_pose_warmup":
                param_names = ["cam2_orient", "cam2_transl", "cam4_orient", "cam4_transl", "shared_body_pose"]
            elif stage_name == "stage2_pose_shape_refinement":
                param_names = ["cam2_orient", "cam2_transl", "cam4_orient", "cam4_transl", "shared_body_pose", "beta_latent"]
            optimizer = torch.optim.LBFGS(params, line_search_fn="strong_wolfe", lr=self.lr_lbfgs, max_iter=self.max_iter_closure)
            closure_count = [0]
            def closure():
                nonlocal latest_loss, latest_losses_detail
                nonlocal latest_loss, latest_losses_detail
                closure_count[0] += 1
                for p in params: p.grad = None
                optimizer.zero_grad()
                losses_all = make_loss_fn(include_pose_anchor=include_pa, include_shape_anchor=include_sa)
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
                if it > 0 and loss_prev - latest_loss < loss_prev * stage_stop: break
                loss_prev = latest_loss
                detail_str = "  ".join(f"{k}={latest_losses_detail.get(k, 0.0) * stage_weights.get(k, 1.0):.4f}"
                                       for k in stage_weights if stage_weights.get(k, 0.0) > 0)
                print(f"[{stage_name}] iter={it+1:03d}/{max_iter} total={latest_loss:.4f}  |  {detail_str}")
            diagnostics["stages"].append({
                "name": stage_name,
                "optimized_parameter_names": param_names,
                "optimized_parameter_shapes": [list(p.shape) for p in params] if params else [],
                "effective_weights": {k: base_weights[k] * weight_mult.get(k, 1.0) for k in base_weights},
                "closure_count": closure_count[0],
                "final_total_loss": latest_loss,
                "final_weighted_losses": {k: latest_losses_detail.get(k, 0.0) * stage_weights.get(k, 1.0) for k in stage_weights},
            })

        for smpl_param in (smpl_c2, smpl_c4):
            for attr in ["orient", "transl", "left_hand_pose", "right_hand_pose"]:
                getattr(smpl_param, attr).requires_grad_(False)
        shared_body_pose.requires_grad_(False)
        beta_latent.requires_grad_(False)

        beta_final = _beta_physical().detach().clone()  # [10] shared beta
        smpl_c2.beta = beta_final.clone()
        smpl_c4.beta = beta_final.clone()
        smpl_c2.body_pose = shared_body_pose.detach().clone(); smpl_c4.body_pose = shared_body_pose.detach().clone()
        for attr in ["beta"]:
            getattr(smpl_c2, attr).requires_grad_(False)
            getattr(smpl_c4, attr).requires_grad_(False)

        beta_final_vals = beta_final.cpu().tolist()
        diagnostics.update({
            "beta_final": beta_final_vals, "beta_final_max_abs": max(abs(v) for v in beta_final_vals),
            "beta_saturation_ratio": max(abs(v) for v in beta_final_vals) / limit,
            "beta_delta_norm": float(torch.norm(beta_final - beta_anchor).item()),
        })
        return smpl_c2, smpl_c4, beta_final
