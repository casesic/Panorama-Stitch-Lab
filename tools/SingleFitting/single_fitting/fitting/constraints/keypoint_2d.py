from __future__ import annotations

from typing import Dict

import torch


def compute_keypoint_2d_loss_with_proj(
    joints_all: torch.Tensor,
    dets_tensor: torch.Tensor,
    keypoints_format: Dict[str, int],
    project_fn,
    W: int,
    H: int,
    conf_thresh: float,
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
        conf_mask = (det[:, :, 2] > conf_thresh).float()
        losses["body_2d_det"] = (loss_body * det[:, :, 2] * conf_mask * w_t.unsqueeze(0)).sum(dim=1).mean()

    return losses, joints_2d_all
