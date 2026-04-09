from __future__ import annotations

from typing import Callable, Dict, Optional, Sequence

import torch

from .debug_shared import DS_PARAMS, R_CAM4, build_smpl_init_seq

R_CAM4_TORCH = torch.tensor(R_CAM4, dtype=torch.float32)
DEFAULT_T_CAM4_INIT = torch.tensor([0.0, 0.0, -0.17355913120508845], dtype=torch.float32)


def ds_project_torch(
    points_3d: torch.Tensor,
    xi: float = DS_PARAMS["xi"],
    alpha: float = DS_PARAMS["alpha"],
    fx: float = DS_PARAMS["fx"],
    fy: float = DS_PARAMS["fy"],
    cx: float = DS_PARAMS["cx"],
    cy: float = DS_PARAMS["cy"],
) -> torch.Tensor:
    x = points_3d[..., 0]
    y = points_3d[..., 1]
    z = points_3d[..., 2]
    r2 = x**2 + y**2
    d1 = torch.sqrt(r2 + z**2).clamp_min(1e-6)
    d2 = torch.sqrt(r2 + (xi * d1 + z) ** 2).clamp_min(1e-6)
    w1 = (alpha * d2 + (1.0 - alpha) * (xi * d1 + z)).clamp_min(1e-6)
    return torch.stack([fx * x / w1 + cx, fy * y / w1 + cy], dim=-1)


def world_to_cam4_torch(points_world: torch.Tensor, delta_t: torch.Tensor) -> torch.Tensor:
    r = R_CAM4_TORCH.to(points_world.device, dtype=points_world.dtype)
    pts_flat = points_world.reshape(-1, 3)
    delta_t = delta_t.to(points_world.device, dtype=points_world.dtype).reshape(1, 3)
    pts_cam4 = pts_flat @ r.T - delta_t
    return pts_cam4.reshape(points_world.shape)


def project_world_to_cam2_torch(points_world: torch.Tensor) -> torch.Tensor:
    return ds_project_torch(points_world)


def project_world_to_cam4_torch(points_world: torch.Tensor, delta_t: torch.Tensor) -> torch.Tensor:
    return ds_project_torch(world_to_cam4_torch(points_world, delta_t))


def init_smplx_from_smpl54_seq(smpl54, index_seq, smpl_fitter_cls, smplx_fitter_cls):
    sub = build_smpl_init_seq(smpl_fitter_cls, smpl54, index_seq)
    return smplx_fitter_cls.init_from_smpl54_seq(sub)


def resolve_delta_t(delta_t: Optional[torch.Tensor], ref: torch.Tensor) -> torch.Tensor:
    if delta_t is None:
        return DEFAULT_T_CAM4_INIT.to(ref.device, dtype=ref.dtype)
    return delta_t.to(ref.device, dtype=ref.dtype)


def unpack_smplx_result(result) -> list:
    from lib_dynamic_hvip.fitting.smplx_seq_forward_module import SMPLX_Seq_Forward_Module

    if isinstance(result, SMPLX_Seq_Forward_Module.SMPLX_Param_Seq):
        return [result.get_param_at_frame(i) for i in range(result.num_batch)]
    if isinstance(result, SMPLX_Seq_Forward_Module.SMPLX_Param):
        return [result]
    raise TypeError(type(result))


def forward_param_to_numpy(fitter, smplx_param):
    with torch.no_grad():
        verts_t, joints_t, joints_body_t = fitter._fitter.forward_param(smplx_param)
    return verts_t.cpu().numpy(), joints_t.cpu().numpy(), joints_body_t.cpu().numpy()


def get_model_faces(fitter, batch_size: int = 1):
    model = fitter._fitter._get_model(batch_size)
    return model.faces.numpy() if hasattr(model.faces, "numpy") else model.faces


class BaseSMPLXFitter:
    def __init__(self, gender: str = "neutral"):
        from lib_dynamic_hvip.fitting.smplx_seq_forward_module import SMPLX_Seq_Forward_Module

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

    def _get_model(self, batch_size: int):
        return self._fitter._get_model(batch_size)

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

    def _common_regularization_losses(self, smpl_param, shape_anchor, n_frame: int) -> Dict[str, torch.Tensor]:
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
            finger_tip_idx = [4, 8, 12, 16, 20]
            lhand_tip_smplx = [39, 27, 30, 36, 33]
            rhand_tip_smplx = [54, 42, 45, 51, 48]
            dev = joints_all.device
            l_tip = torch.tensor(lhand_tip_smplx, dtype=torch.long, device=dev)
            r_tip = torch.tensor(rhand_tip_smplx, dtype=torch.long, device=dev)
            t_idx = torch.tensor(finger_tip_idx, dtype=torch.long, device=dev)
            lhand_det = hand_keypoints_detected[:, :21, :]
            rhand_det = hand_keypoints_detected[:, 21:, :]
            proj_l = joints_2d_all[:, l_tip, :]
            proj_r = joints_2d_all[:, r_tip, :]
            det_l = lhand_det[:, t_idx, :]
            det_r = rhand_det[:, t_idx, :]
            losses["hand_2d_det"] = (
                (torch.norm(proj_l - det_l[:, :, :2], dim=2) * det_l[:, :, 2]).mean()
                + (torch.norm(proj_r - det_r[:, :, :2], dim=2) * det_r[:, :, 2]).mean()
            ) * hand_weight
        return losses, joints_2d_all

    def fit(self, detection_results_list: list, image_size: int = 2880, smpl_param_init=None, hand_detection_list=None):
        from lib_dynamic_hvip.smpl.smpl_lib import forward_smplx_tensor_batch

        n_frame = len(detection_results_list)
        model = self._get_model(n_frame)
        key_fmt = detection_results_list[0].keypoints_format
        kp_tensor = self._stack_keypoints(detection_results_list)
        hand_tensor = self._stack_hand(hand_detection_list)
        smpl_param = smpl_param_init if smpl_param_init is not None else self._build_default_smplx(n_frame)
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

    def _compute_hand_loss(self, joints_world: torch.Tensor, hand_tensor: Optional[torch.Tensor], camera: str, delta_t: torch.Tensor) -> torch.Tensor:
        if hand_tensor is None:
            return torch.zeros(1, device=joints_world.device)
        finger_tip_idx = [4, 8, 12, 16, 20]
        lhand_tip_smplx = [39, 27, 30, 36, 33]
        rhand_tip_smplx = [54, 42, 45, 51, 48]
        dev = joints_world.device
        l_tip = torch.tensor(lhand_tip_smplx, dtype=torch.long, device=dev)
        r_tip = torch.tensor(rhand_tip_smplx, dtype=torch.long, device=dev)
        t_idx = torch.tensor(finger_tip_idx, dtype=torch.long, device=dev)
        joints_2d = project_world_to_cam2_torch(joints_world) if camera == "cam2" else project_world_to_cam4_torch(joints_world, delta_t)
        lhand_det = hand_tensor[:, :21, :]
        rhand_det = hand_tensor[:, 21:, :]
        proj_l = joints_2d[:, l_tip, :]
        proj_r = joints_2d[:, r_tip, :]
        det_l = lhand_det[:, t_idx, :]
        det_r = rhand_det[:, t_idx, :]
        return (
            (torch.norm(proj_l - det_l[:, :, :2], dim=2) * det_l[:, :, 2]).mean()
            + (torch.norm(proj_r - det_r[:, :, :2], dim=2) * det_r[:, :, 2]).mean()
        )

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
        model = self._get_model(n_frame)
        kp_cam2 = self._stack_keypoints(det_list_cam2)
        kp_cam4 = self._stack_keypoints(det_list_cam4)
        hand_cam2 = self._stack_hand(hand_list_cam2)
        hand_cam4 = self._stack_hand(hand_list_cam4)
        smpl_param = smpl_param_init if smpl_param_init is not None else self._build_default_smplx(n_frame)
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
            ("stage1_global", lambda: [smpl_param.orient, smpl_param.transl, delta_t], 25, self.stop_ratio, {"body_2d_cam2": 1.0, "body_2d_cam4": 1.0, "loss_pose_prior": 0.0, "loss_hand_prior": 0.0, "loss_shape_prior": 0.0, "loss_shape_bound": 0.0, "loss_shape_anchor": 0.0, "loss_shape_norm": 0.0, "loss_transl_diff": 0.0, "loss_pose_diff": 0.0, "loss_delta_t": 1.0}),
            ("stage2_body", lambda: [smpl_param.orient, smpl_param.transl, smpl_param.body_pose, delta_t], 50, self.stop_ratio * 0.7, {"body_2d_cam2": 1.2, "body_2d_cam4": 1.2, "loss_pose_prior": 0.5, "loss_hand_prior": 0.0, "loss_shape_prior": 0.0, "loss_shape_bound": 0.0, "loss_shape_anchor": 0.0, "loss_shape_norm": 0.0, "loss_transl_diff": 1.0, "loss_pose_diff": 0.8, "loss_delta_t": 1.0}),
            ("stage3_hand", lambda: [smpl_param.orient, smpl_param.transl, smpl_param.body_pose, smpl_param.left_hand_pose, smpl_param.right_hand_pose, delta_t], 60, self.stop_ratio * 0.7, {"body_2d_cam2": 1.5, "body_2d_cam4": 1.5, "hand_2d_cam2": 1.0, "hand_2d_cam4": 1.0, "loss_pose_prior": 0.5, "loss_hand_prior": 1.0, "loss_shape_prior": 0.0, "loss_shape_bound": 0.0, "loss_shape_anchor": 0.0, "loss_shape_norm": 0.0, "loss_transl_diff": 1.0, "loss_pose_diff": 0.8, "loss_delta_t": 1.0}),
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
            losses_all = self._common_regularization_losses(smpl_param, shape_anchor, n_frame)
            losses_all["body_2d_cam2"] = self.compute_loss_2d(joints_world, kp_cam2, kp_fmt_cam2, camera="cam2", delta_t=delta_t)["body_2d_det"]
            losses_all["body_2d_cam4"] = self.compute_loss_2d(joints_world, kp_cam4, kp_fmt_cam4, camera="cam4", delta_t=delta_t)["body_2d_det"]
            losses_all["hand_2d_cam2"] = self._compute_hand_loss(joints_world, hand_cam2, "cam2", delta_t)
            losses_all["hand_2d_cam4"] = self._compute_hand_loss(joints_world, hand_cam4, "cam4", delta_t)
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
