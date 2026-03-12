from typing import Any, Dict, List, Optional, Tuple
import sys
import os

import numpy as np
import torch

# 动态导入 omni_project_torch（来自 lib_pano.camera.omni_camera）
def _try_import_omni_project():
    """尝试从多个路径导入 omni_project_torch，找不到时返回 None。"""
    # 优先用已有 sys.path
    try:
        from lib_pano.camera.omni_camera import omni_project_torch
        return omni_project_torch
    except ImportError:
        pass
    # 尝试相对路径（smpl_fitter_sample.py → ../../camera/）
    _here = os.path.dirname(os.path.abspath(__file__))
    _src  = os.path.abspath(os.path.join(_here, '..', '..'))
    if _src not in sys.path:
        sys.path.insert(0, _src)
    try:
        from lib_pano.camera.omni_camera import omni_project_torch
        return omni_project_torch
    except ImportError:
        return None

_omni_project_torch = _try_import_omni_project()

from constants_skeleton import SMPL_ALL_54
try:
    # 优先使用师兄的完整库（含 lib_dynamic_hvip）
    from smpl_lib import (
        forward_smpl_full_tensor,
        forward_smpl_full_tensor_batch,
        get_smpl_full,
    )
except Exception:
    # 回退到本地独立实现（仅依赖 smplx，无需私有库）
    from smpl_runner import (
        forward_smpl_full_tensor,
        forward_smpl_full_tensor_batch,
        get_smpl_full,
    )


class SMPLFitterSample:

    class SMPLParam:
        def __init__(self, dim_shape: int = 10, gender: str = "neutral"):
            self.orient = torch.zeros(3, dtype=torch.float32)
            self.transl = torch.zeros(3, dtype=torch.float32)
            self.pose = torch.zeros(69, dtype=torch.float32)
            self.scale = torch.ones(1, dtype=torch.float32)
            if dim_shape == 10:
                self.beta = torch.zeros(10, dtype=torch.float32)
            elif dim_shape == 300:
                self.beta = torch.zeros(300, dtype=torch.float32)
            else:
                raise ValueError(dim_shape)
            self.gender = gender

    class SMPLParamSeq:
        def __init__(self, num_batch: int, dim_shape: int = 10, gender: str = "neutral"):
            self.num_batch = int(num_batch)
            self.orient = torch.zeros(self.num_batch, 3, dtype=torch.float32)
            self.transl = torch.zeros(self.num_batch, 3, dtype=torch.float32)
            self.pose = torch.zeros(self.num_batch, 69, dtype=torch.float32)
            self.scale = torch.ones(1, dtype=torch.float32)
            if dim_shape == 10:
                self.beta = torch.zeros(10, dtype=torch.float32)
            elif dim_shape == 300:
                self.beta = torch.zeros(300, dtype=torch.float32)
            else:
                raise ValueError(dim_shape)
            self.gender = gender

    def __init__(self, dim_shape: int = 10):
        if dim_shape == 10:
            version_smpl = "v100"
        elif dim_shape == 300:
            version_smpl = "v110"
        else:
            raise ValueError(dim_shape)
        self.dim_shape = dim_shape
        self.smpl_models = {
            "neutral": get_smpl_full(version=version_smpl, gender="neutral"),
            "male": get_smpl_full(version=version_smpl, gender="male"),
            "female": get_smpl_full(version=version_smpl, gender="female"),
        }

    @staticmethod
    def _copy_seq_param_like(src: Any) -> SMPLParamSeq:
        for key in ["num_batch", "orient", "transl", "pose", "beta", "scale", "gender"]:
            if not hasattr(src, key):
                raise AttributeError(f"smpl_param_init missing attribute: {key}")
        dst = SMPLFitterSample.SMPLParamSeq(
            num_batch=int(src.num_batch),
            dim_shape=int(src.beta.shape[0]),
            gender=str(src.gender),
        )
        dst.orient = src.orient.clone().detach().to(torch.float32)
        dst.transl = src.transl.clone().detach().to(torch.float32)
        dst.pose = src.pose.clone().detach().to(torch.float32)
        dst.beta = src.beta.clone().detach().to(torch.float32)
        dst.scale = src.scale.clone().detach().to(torch.float32)
        return dst

    def forward_smpl_param_seq(self, smpl_param_seq: SMPLParamSeq) -> Tuple[torch.Tensor, torch.Tensor]:
        gender = smpl_param_seq.gender
        assert gender in ["neutral", "male", "female"], gender
        bsz = int(smpl_param_seq.num_batch)

        body_pose = smpl_param_seq.pose
        poses = torch.zeros((bsz, 72), device=body_pose.device, dtype=body_pose.dtype)
        poses[:, 3:] = body_pose
        poses[:, :3] = smpl_param_seq.orient

        betas = smpl_param_seq.beta.reshape(1, -1).repeat(bsz, 1)
        scales = smpl_param_seq.scale.reshape(1).repeat(bsz)
        transl = smpl_param_seq.transl

        verts, joints_54 = forward_smpl_full_tensor_batch(
            smpl_model=self.smpl_models[gender],
            poses=poses,
            shapes=betas,
            trans=transl,
            scale=scales,
            ret_joints_54=True,
        )
        return verts, joints_54

    def forward_smpl_param_single(self, smpl_param: SMPLParam) -> Tuple[torch.Tensor, torch.Tensor]:
        gender = smpl_param.gender
        assert gender in ["neutral", "male", "female"], gender

        body_pose = smpl_param.pose
        poses = torch.zeros((72,), device=body_pose.device, dtype=body_pose.dtype)
        poses[3:] = body_pose
        poses[:3] = smpl_param.orient

        verts, joints_54 = forward_smpl_full_tensor(
            smpl_model=self.smpl_models[gender],
            poses=poses,
            shapes=smpl_param.beta,
            trans=smpl_param.transl,
            scale=smpl_param.scale,
            ret_joints_54=True,
        )
        return verts, joints_54

    @staticmethod
    def sum_losses_with_weight(dict_losses: Dict[str, torch.Tensor], weight: Dict[str, float]) -> torch.Tensor:
        loss = torch.zeros((), dtype=torch.float32, device=next(iter(dict_losses.values())).device)
        for key, val in dict_losses.items():
            loss = loss + val * float(weight[key])
        return loss

    @staticmethod
    def compute_loss_keypoints_2d_from_single_view_seq_by_keypoints_weight(
        camera_k: torch.Tensor,
        joints_54_current: torch.Tensor,
        keypoints_detected: torch.Tensor,
        keypoints_format: Dict[str, int],
        keypoints_weight: Optional[Dict[str, float]] = None,
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        if keypoints_weight is None:
            keypoints_weight = {}

        bsz = joints_54_current.shape[0]
        j_det = len(keypoints_format.keys())
        assert camera_k.shape == (3, 3), camera_k.shape
        assert joints_54_current.shape == (bsz, 54, 3), joints_54_current.shape
        assert keypoints_detected.shape == (bsz, j_det, 3), keypoints_detected.shape

        index_54_to_std = []
        index_kp_to_std = []
        weights_list = []
        for name_joint, index_joint in keypoints_format.items():
            if name_joint in SMPL_ALL_54:
                index_54_to_std.append(SMPL_ALL_54[name_joint])
                index_kp_to_std.append(index_joint)
                weights_list.append(float(keypoints_weight.get(name_joint, 1.0)))

        index_54_to_std_t = torch.tensor(index_54_to_std, dtype=torch.long, device=joints_54_current.device)
        index_kp_to_std_t = torch.tensor(index_kp_to_std, dtype=torch.long, device=keypoints_detected.device)
        weights_t = torch.tensor(weights_list, dtype=torch.float32, device=joints_54_current.device)

        joints_2d_54 = joints_54_current @ camera_k.T
        joints_2d_54 = joints_2d_54[:, :, :2] / (joints_2d_54[:, :, 2:3] + 1e-9)
        joints_2d_std = joints_2d_54[:, index_54_to_std_t, :]
        keypoints_detected_std = keypoints_detected[:, index_kp_to_std_t, :]

        loss_per_joint = torch.norm(joints_2d_std - keypoints_detected_std[:, :, :2], dim=2)  # (B,J)
        w_norm = weights_t / (weights_t.sum() + 1e-8)
        w_norm = w_norm.view(1, -1).expand(bsz, -1)
        weighted_loss = loss_per_joint * keypoints_detected_std[:, :, 2] * w_norm
        dict_loss = {"body_2d_det": weighted_loss.sum(dim=1).mean()}
        return dict_loss, joints_2d_54

    @staticmethod
    def compute_loss_keypoints_2d_omni(
        omni_params: Dict[str, float],
        joints_54_current: torch.Tensor,
        keypoints_detected: torch.Tensor,
        keypoints_format: Dict[str, int],
        keypoints_weight: Optional[Dict[str, float]] = None,
    ) -> Tuple[Dict[str, torch.Tensor], None]:
        """
        Omni-radtan 鱼眼相机下的 2D 关键点投影损失。

        Args:
            omni_params: 包含 {xi, fx, fy, cx, cy, k1, k2, p1, p2} 的字典
            joints_54_current: (B, 54, 3) SMPL 关节点，相机坐标系
            keypoints_detected: (B, J, 3) 检测到的 2D 关键点 (x, y, conf)
            keypoints_format: 关节名称 -> 检测索引
            keypoints_weight: 各关节的权重（可选）
        """
        if _omni_project_torch is None:
            raise ImportError(
                "找不到 lib_pano.camera.omni_camera，"
                "请确保 src/ 目录在 sys.path 中。"
            )
        if keypoints_weight is None:
            keypoints_weight = {}

        bsz   = joints_54_current.shape[0]
        j_det = len(keypoints_format)
        assert joints_54_current.shape == (bsz, 54, 3), joints_54_current.shape
        assert keypoints_detected.shape == (bsz, j_det, 3), keypoints_detected.shape

        index_54_to_std, index_kp_to_std, weights_list = [], [], []
        for name_joint, index_joint in keypoints_format.items():
            if name_joint in SMPL_ALL_54:
                index_54_to_std.append(SMPL_ALL_54[name_joint])
                index_kp_to_std.append(index_joint)
                weights_list.append(float(keypoints_weight.get(name_joint, 1.0)))

        dev = joints_54_current.device
        idx54  = torch.tensor(index_54_to_std, dtype=torch.long, device=dev)
        idx_kp = torch.tensor(index_kp_to_std, dtype=torch.long, device=dev)
        weights_t = torch.tensor(weights_list, dtype=torch.float32, device=dev)

        # omni-radtan 投影：(B, J, 3) → (B, J, 2)
        joints_sel = joints_54_current[:, idx54, :]   # (B, J, 3)
        joints_2d = _omni_project_torch(
            joints_sel,
            fx=omni_params['fx'], fy=omni_params['fy'],
            cx=omni_params['cx'], cy=omni_params['cy'],
            k1=omni_params['k1'], k2=omni_params['k2'],
            p1=omni_params['p1'], p2=omni_params['p2'],
            xi=omni_params['xi'],
        )  # (B, J, 2)

        kps_std = keypoints_detected[:, idx_kp, :]  # (B, J, 3)
        loss_per_joint = torch.norm(joints_2d - kps_std[:, :, :2], dim=2)  # (B, J)
        w_norm = weights_t / (weights_t.sum() + 1e-8)
        w_norm = w_norm.view(1, -1).expand(bsz, -1)
        weighted_loss = loss_per_joint * kps_std[:, :, 2] * w_norm
        dict_loss = {"body_2d_det": weighted_loss.sum(dim=1).mean()}
        return dict_loss, None

    def fit_from_multi_frame_seq(
        self,
        detection_results_list: List[Any],
        camera: Optional[np.ndarray] = None,
        smpl_param_init: Optional[Any] = None,
        omni_params: Optional[Dict[str, float]] = None,
    ) -> SMPLParamSeq:
        """
        多帧序列 SMPL 拟合。支持针孔相机和 omni-radtan 鱼眼相机。

        Args:
            detection_results_list: 每帧的检测结果（含 keypoints 和 keypoints_format）
            camera  : 针孔相机内参矩阵 K (3×3)，使用针孔模型时必填
            smpl_param_init: 热启动初始参数（可选）
            omni_params: omni-radtan 相机参数字典（与 camera 二选一）
                         必须包含: {xi, fx, fy, cx, cy, k1, k2, p1, p2}
        """
        assert omni_params is not None or camera is not None, \
            "camera (针孔K矩阵) 和 omni_params 必须至少提供一个"
        assert isinstance(detection_results_list, list) and len(detection_results_list) > 0
        n_frames = len(detection_results_list)

        # 1) Nose positions in camera space
        # n_cam, d_cam = plane_cam
        # assert isinstance(nose_position, np.ndarray), type(nose_position)
        # assert nose_position.ndim == 3 and nose_position.shape[1:] == (1, 3), nose_position.shape
        # assert nose_position.shape[0] == n_frames, (nose_position.shape, n_frames)
        # positions_nose = torch.from_numpy(nose_position[:, 0, :].astype(np.float32))  # (N,3)

        # 2) Keypoints stack
        camera_tensor = torch.from_numpy(np.asarray(camera, dtype=np.float32)) \
            if camera is not None else None
        keypoints_list = []
        key_fmt = None
        for i in range(n_frames):
            dr = detection_results_list[i]
            kp = torch.from_numpy(np.asarray(dr.keypoints, dtype=np.float32))
            keypoints_list.append(kp)
            if key_fmt is None:
                key_fmt = dr.keypoints_format
            else:
                assert key_fmt == dr.keypoints_format
        keypoints_tensor = torch.stack(keypoints_list, dim=0)  # (N,J,3)

        # 3) Init SMPL sequence
        if smpl_param_init is not None:
            smpl_param = self._copy_seq_param_like(smpl_param_init)
            assert smpl_param.num_batch == n_frames
        else:
            smpl_param = SMPLFitterSample.SMPLParamSeq(num_batch=n_frames, dim_shape=self.dim_shape, gender="neutral")
        # smpl_param.transl = smpl_param.transl + positions_nose

        smpl_param.orient = smpl_param.orient.clone().detach().requires_grad_(True)
        smpl_param.transl = smpl_param.transl.clone().detach().requires_grad_(True)
        smpl_param.pose = smpl_param.pose.clone().detach().requires_grad_(True)
        smpl_param.beta = smpl_param.beta.clone().detach().requires_grad_(True)
        smpl_param.scale = smpl_param.scale.clone().detach()

        # 4) Optim hyper-params
        lr_lbfgs = 0.5
        max_iter_per_closure = 60
        max_closure = 60
        loss_percent_stop = 0.025

        # n_cam_tensor = torch.from_numpy(np.asarray(n_cam, dtype=np.float32))
        # d_cam_tensor = torch.tensor(float(d_cam), dtype=torch.float32)

        weight_vr_3d = 2.0 * n_frames
        weight_of_losses = {
            "body_2d_det": 2.0 * n_frames / 20.0,
            "shape": 2e-4,
            "depth_prior": 50.0,   # 防止身体跑到相机后方 (Z<0)
            # "loss_vr": weight_vr_3d,
            # "loss_vr_2d": weight_vr_3d / 10.0,
            # "loss_foot_penetration": n_frames * 0.5,
            # "loss_foot_distance": n_frames * 4.0,
        }

        # keypoints_weight = {
        #     "Nose": 1.0,
        #     "L_Eye": 0.0,
        #     "R_Eye": 0.0,
        #     "L_Ear": 0.0,
        #     "R_Ear": 0.0,
        #     "L_Shoulder": 3.0,
        #     "R_Shoulder": 3.0,
        #     "L_Hip": 2.0,
        #     "R_Hip": 2.0,
        #     "L_Ankle": 2.0,
        #     "R_Ankle": 2.0,
        #     "L_Heel": 1.5,
        #     "R_Heel": 1.5,
        #     "L_BigToe": 1.5,
        #     "R_BigToe": 1.5,
        #     "L_SmallToe": 1.5,
        #     "R_SmallToe": 1.5,
        #     "L_Wrist": 1.5,
        #     "R_Wrist": 1.5,
        #     "L_Elbow": 1.2,
        #     "R_Elbow": 1.2,
        # }

        optimizer = torch.optim.LBFGS(
            [smpl_param.orient, smpl_param.transl, smpl_param.pose, smpl_param.beta],
            line_search_fn="strong_wolfe",
            lr=lr_lbfgs,
            max_iter=max_iter_per_closure,
        )

        def closure():
            optimizer.zero_grad()
            _, joints_54 = self.forward_smpl_param_seq(smpl_param)
            joints_54 = joints_54.reshape((-1, 54, 3))

            losses_all: Dict[str, torch.Tensor] = {}

            if omni_params is not None:
                losses_kp2d, _ = self.compute_loss_keypoints_2d_omni(
                    omni_params=omni_params,
                    keypoints_detected=keypoints_tensor,
                    joints_54_current=joints_54,
                    keypoints_format=key_fmt,
                )
            else:
                losses_kp2d, _ = self.compute_loss_keypoints_2d_from_single_view_seq_by_keypoints_weight(
                    camera_k=camera_tensor,
                    keypoints_detected=keypoints_tensor,
                    joints_54_current=joints_54,
                    keypoints_format=key_fmt,
                )
            losses_all.update(losses_kp2d)

            # 形状正则化：抑制极端体型（beta 偏离 0 才有惩罚）
            losses_all["shape"] = smpl_param.beta.pow(2).mean()

            # 深度先验：惩罚 Z < 1.5m，防止身体跑到相机后方
            # relu(1.5 - z)^2：z 越小惩罚越大；z>=1.5m 时惩罚为 0
            losses_all["depth_prior"] = torch.relu(
                1.5 - smpl_param.transl[:, 2]
            ).pow(2).mean()

            # point_cam = positions_nose
            # point_proj = point_cam @ camera_tensor.T
            # point_2d = point_proj[:, :2] / (point_proj[:, 2:] + 1e-9)
            # nose_2d = joints_2d_54_current[:, 24, :]
            # nose_3d = joints_54[:, 24, :]

            # losses_all["loss_vr_2d"] = torch.norm(point_2d - nose_2d, dim=1).mean()
            # losses_all["loss_vr"] = torch.norm(nose_3d - positions_nose, dim=1).mean()

            # foot_idx = [7, 8, 10, 11]
            # joints_feet_cam = joints_54[:, foot_idx, :]  # (N,4,3)
            # dist_foot = joints_feet_cam @ n_cam_tensor + d_cam_tensor  # (N,4)
            # losses_all["loss_foot_penetration"] = torch.relu(-dist_foot).mean()
            # contact_mask = (dist_foot < 0.05).float()
            # losses_all["loss_foot_distance"] = (dist_foot.abs() * contact_mask).mean()

            loss = self.sum_losses_with_weight(losses_all, weight_of_losses)
            loss.backward()
            return loss

        print("opt multi-frame...")
        loss_prev = 1e9
        for _ in range(max_closure):
            optimizer.step(closure)
            loss_now = float(closure().item())
            if (loss_prev - loss_now) < (loss_prev * loss_percent_stop):
                break
            loss_prev = loss_now

        smpl_param.orient.requires_grad_(False)
        smpl_param.transl.requires_grad_(False)
        smpl_param.pose.requires_grad_(False)
        smpl_param.beta.requires_grad_(False)
        smpl_param.scale.requires_grad_(False)
        return smpl_param
