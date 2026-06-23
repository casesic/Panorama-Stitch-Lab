"""形状/姿态正则化损失：拆分 shape 类与非 shape 类损失项并按需加后缀。"""

from __future__ import annotations

from types import SimpleNamespace

import torch


SHAPE_LOSS_KEYS = (
    "loss_shape_prior",
    "loss_shape_bound",
    "loss_shape_anchor",
    "loss_shape_norm",
)

NON_SHAPE_LOSS_KEYS = (
    "loss_transl_diff",
    "loss_pose_diff",
    "loss_pose_prior",
    "loss_hand_prior",
    "loss_pose_anchor",
)


def compute_shape_regularization_losses(
    common_regularization_fn,
    shared_beta: torch.Tensor,
    shared_body_pose: torch.Tensor,
    shape_anchor: torch.Tensor,
    ref_param,
    n_frame: int,
):
    """用共享 beta/body_pose 构造代理参数计算正则化损失，仅返回 shape 类损失项。"""
    proxy = SimpleNamespace(
        transl=ref_param.transl,
        body_pose=shared_body_pose,
        left_hand_pose=ref_param.left_hand_pose,
        right_hand_pose=ref_param.right_hand_pose,
        beta=shared_beta,
    )
    losses = common_regularization_fn(proxy, shape_anchor, n_frame)
    return {key: losses[key] for key in SHAPE_LOSS_KEYS}


def suffix_non_shape_losses(losses_all: dict, suffix: str) -> dict:
    """从损失字典中取出非 shape 类损失项并为键名追加后缀。"""
    return {f"{key}_{suffix}": losses_all[key] for key in NON_SHAPE_LOSS_KEYS}
