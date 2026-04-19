from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional, Sequence

import torch
from lib_dynamic_hvip.fitting.smplx_seq_forward_module import SMPLX_Seq_Forward_Module

from single_fitting.debug_shared import DS_PARAMS

from .projection import DEFAULT_T_CAM4_INIT


@dataclass
class HandJointConfig:
    det_indices: list
    smplx_indices_l: list
    smplx_indices_r: list
    weight: float = 1.0


DEFAULT_FINGERTIP_CONFIG = HandJointConfig(
    det_indices=[4, 8, 12, 16, 20],
    smplx_indices_l=[39, 27, 30, 36, 33],
    smplx_indices_r=[54, 42, 45, 51, 48],
    weight=1.0,
)


def resolve_delta_t(delta_t: Optional[torch.Tensor], ref: torch.Tensor) -> torch.Tensor:
    if delta_t is None:
        return DEFAULT_T_CAM4_INIT.to(ref.device, dtype=ref.dtype)
    return delta_t.to(ref.device, dtype=ref.dtype)


def unpack_smplx_result(result) -> list:
    if isinstance(result, SMPLX_Seq_Forward_Module.SMPLX_Param_Seq):
        return [result.get_param_at_frame(i) for i in range(result.num_batch)]
    if isinstance(result, SMPLX_Seq_Forward_Module.SMPLX_Param):
        return [result]
    raise TypeError(type(result))


class BaseSMPLXFitter:
    def __init__(self, gender: str = "neutral"):
        self.SMPLX = SMPLX_Seq_Forward_Module
        self._fitter = SMPLX_Seq_Forward_Module(gender=gender)
        self.gender = gender
        self.ds = DS_PARAMS
        self.fps = 30.0
        self.max_speed_mps = 4.0
        self.shape_norm_bound = 3.5
        self.shape_hard_clip = 4.5
        self.current_shape_soft_bound = 2.0
        self.lr_lbfgs = 0.8
        self.max_iter_closure = 20
        self.stop_ratio = 1e-4
        self.hand_joint_configs: list = [DEFAULT_FINGERTIP_CONFIG]
        self.hand_overall_weight: float = 1.0

    def _get_model(self):
        return self._fitter._get_model()

    def forward_param_to_numpy(self, smplx_param):
        with torch.no_grad():
            verts_t, joints_t, joints_body_t = self._fitter.forward_param(smplx_param)
        return verts_t.cpu().numpy(), joints_t.cpu().numpy(), joints_body_t.cpu().numpy()

    def get_model_faces(self, batch_size: int = 1):
        model = self._get_model()
        return model.faces.numpy() if hasattr(model.faces, "numpy") else model.faces

    def _stack_keypoints(self, detection_results_list: Sequence) -> torch.Tensor:
        return torch.stack([torch.from_numpy(dr.keypoints).float() for dr in detection_results_list], dim=0)

    def _stack_hand(self, hand_detection_list):
        if hand_detection_list is None or not any(h is not None for h in hand_detection_list):
            return None
        return torch.stack(
            [torch.from_numpy(h).float() if h is not None else torch.zeros(42, 3) for h in hand_detection_list],
            dim=0,
        )

    def _build_default_smplx(self, n_frame: int):
        return self.SMPLX.SMPLX_Param_Seq(num_batch=n_frame, gender=self.gender)

    def _prepare_smplx_param_for_optimization(self, smpl_param):
        # Match the original fitter behavior: optimize detached leaf tensors.
        smpl_param.orient = smpl_param.orient.clone().detach().requires_grad_(True)
        smpl_param.transl = smpl_param.transl.clone().detach().requires_grad_(True)
        smpl_param.body_pose = smpl_param.body_pose.clone().detach().requires_grad_(True)
        smpl_param.left_hand_pose = smpl_param.left_hand_pose.clone().detach().requires_grad_(True)
        smpl_param.right_hand_pose = smpl_param.right_hand_pose.clone().detach().requires_grad_(True)
        smpl_param.beta = smpl_param.beta.clone().detach().requires_grad_(True)
        smpl_param.scale = smpl_param.scale.clone().detach()
        return smpl_param

    def _init_transl_from_bbox(
        self,
        keypoints_tensor: torch.Tensor,
        image_size,
    ) -> Optional[torch.Tensor]:
        return None

    def _hand_joint_loss(self, joints_2d: torch.Tensor, hand_tensor) -> torch.Tensor:
        zero = joints_2d.new_zeros(())
        if hand_tensor is None:
            return zero

        total_loss = zero
        lhand_det = hand_tensor[:, :21, :]
        rhand_det = hand_tensor[:, 21:, :]
        for config in self.hand_joint_configs:
            det_idx = torch.tensor(config.det_indices, dtype=torch.long, device=joints_2d.device)
            smplx_idx_l = torch.tensor(config.smplx_indices_l, dtype=torch.long, device=joints_2d.device)
            smplx_idx_r = torch.tensor(config.smplx_indices_r, dtype=torch.long, device=joints_2d.device)
            proj_l = joints_2d[:, smplx_idx_l, :]
            proj_r = joints_2d[:, smplx_idx_r, :]
            det_l = lhand_det[:, det_idx, :]
            det_r = rhand_det[:, det_idx, :]
            loss_l = (torch.norm(proj_l - det_l[:, :, :2], dim=2) * det_l[:, :, 2]).mean()
            loss_r = (torch.norm(proj_r - det_r[:, :, :2], dim=2) * det_r[:, :, 2]).mean()
            total_loss = total_loss + (loss_l + loss_r) * config.weight
        return total_loss * self.hand_overall_weight

    def _common_regularization_losses(
        self,
        smpl_param,
        shape_anchor,
        n_frame: int,
        pose_anchor: torch.Tensor | None = None,
    ) -> Dict[str, torch.Tensor]:
        losses_all = {}
        if n_frame > 1:
            t_diff = smpl_param.transl[1:] - smpl_param.transl[:-1]
            t_spd = torch.norm(t_diff, dim=1) * self.fps
            losses_all["loss_transl_diff"] = torch.relu(t_spd - self.max_speed_mps).mean()
            p_diff = (smpl_param.body_pose[1:] - smpl_param.body_pose[:-1]).reshape(-1, 21, 3)
            losses_all["loss_pose_diff"] = torch.norm(p_diff, dim=2).mean()
        else:
            z = torch.zeros((), dtype=smpl_param.body_pose.dtype, device=smpl_param.body_pose.device)
            losses_all["loss_transl_diff"] = z
            losses_all["loss_pose_diff"] = z
        losses_all["loss_pose_prior"] = torch.norm(smpl_param.body_pose, dim=1).mean()
        losses_all["loss_hand_prior"] = (
            torch.norm(smpl_param.left_hand_pose, dim=1).mean()
            + torch.norm(smpl_param.right_hand_pose, dim=1).mean()
        )
        losses_all["loss_shape_prior"] = torch.mean(smpl_param.beta**2)
        losses_all["loss_shape_bound"] = torch.mean(
            torch.relu(torch.abs(smpl_param.beta) - self.current_shape_soft_bound) ** 2
        )
        losses_all["loss_shape_anchor"] = torch.mean((smpl_param.beta - shape_anchor) ** 2)
        losses_all["loss_shape_norm"] = torch.relu(torch.norm(smpl_param.beta) - self.shape_norm_bound) ** 2
        if pose_anchor is not None:
            losses_all["loss_pose_anchor"] = torch.mean((smpl_param.body_pose - pose_anchor) ** 2)
        else:
            losses_all["loss_pose_anchor"] = torch.zeros(
                (), dtype=smpl_param.body_pose.dtype, device=smpl_param.body_pose.device
            )
        return losses_all

    def _run_optimization(
        self,
        smpl_param,
        base_weights: Dict[str, float],
        optimization_stages: Sequence,
        make_loss_fn: Callable[[], Dict[str, torch.Tensor]],
        after_step: Optional[Callable[[], None]] = None,
    ):
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
                if after_step is not None:
                    after_step()
                if it > 0 and loss_prev - latest_loss < loss_prev * stage_stop:
                    break
                loss_prev = latest_loss
                detail_str = "  ".join(
                    f"{k}={latest_losses_detail.get(k, 0.0) * stage_weights.get(k, 1.0):.4f}"
                    for k in stage_weights
                    if stage_weights.get(k, 0.0) > 0
                )
                print(f"[{stage_name}] iter={it+1:03d}/{max_iter} total={latest_loss:.4f}  |  {detail_str}")

        for attr in ["orient", "transl", "body_pose", "left_hand_pose", "right_hand_pose", "beta"]:
            getattr(smpl_param, attr).requires_grad_(False)
        return smpl_param
