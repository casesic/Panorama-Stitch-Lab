"""双相机拟合的损失配置：各损失项权重、按帧数缩放的基础权重表、分阶段优化的阶段定义。"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class DualCamLossWeights:
    """双相机 SMPL-X 拟合各损失项的权重配置。"""
    body_2d: float = 0.1
    pose_prior: float = 0.05
    hand_prior: float = 0.02
    pose_anchor: float = 1.0
    transl_diff: float = 0.0
    pose_diff: float = 0.0
    shape_prior: float = 6.0
    shape_bound: float = 20.0
    shape_anchor: float = 30.0
    shape_norm: float = 10.0
    densekp: float = 1.0
    sam3d_dense: float = 0.5


def build_base_weights(
    n_frame: int,
    weights: DualCamLossWeights,
) -> dict:
    """按帧数缩放各权重，生成 c2/c4 双相机各损失项的基础权重字典。"""
    return {
        "body_2d_det_c2": n_frame * weights.body_2d,
        "body_2d_det_c4": n_frame * weights.body_2d,
        "loss_pose_prior_c2": n_frame * weights.pose_prior,
        "loss_pose_prior_c4": n_frame * weights.pose_prior,
        "loss_hand_prior_c2": n_frame * weights.hand_prior,
        "loss_hand_prior_c4": n_frame * weights.hand_prior,
        "loss_pose_anchor_c2": n_frame * weights.pose_anchor,
        "loss_pose_anchor_c4": n_frame * weights.pose_anchor,
        "loss_transl_diff_c2": n_frame * weights.transl_diff,
        "loss_transl_diff_c4": n_frame * weights.transl_diff,
        "loss_pose_diff_c2": n_frame * weights.pose_diff,
        "loss_pose_diff_c4": n_frame * weights.pose_diff,
        "loss_shape_prior": n_frame * weights.shape_prior,
        "loss_shape_bound": n_frame * weights.shape_bound,
        "loss_shape_anchor": n_frame * weights.shape_anchor,
        "loss_shape_norm": n_frame * weights.shape_norm,
        "loss_densekp_c2": n_frame * weights.densekp,
        "loss_densekp_c4": n_frame * weights.densekp,
        "loss_sam3d_dense_c2": n_frame * weights.sam3d_dense,
        "loss_sam3d_dense_c4": n_frame * weights.sam3d_dense,
    }


def build_optimization_stages(
    stop_ratio: float,
    smpl_c2,
    smpl_c4,
    shared_body_pose,
    shared_beta,
) -> list:
    """定义两阶段 LBFGS 优化：
    Stage 1 仅优化平移/朝向；
    Stage 2 加入 body pose、shape、DenseKP 与 SAM dense surface 辅助约束。
    """
    return [
        (
            "stage1_transl_orient",
            lambda: [smpl_c2.orient, smpl_c2.transl, smpl_c4.orient, smpl_c4.transl],
            40,
            stop_ratio,
            {
                "body_2d_det_c2": 1.0,
                "body_2d_det_c4": 1.0,
                "loss_pose_prior_c2": 0.0,
                "loss_pose_prior_c4": 0.0,
                "loss_hand_prior_c2": 0.0,
                "loss_hand_prior_c4": 0.0,
                "loss_pose_anchor_c2": 0.0,
                "loss_pose_anchor_c4": 0.0,
                "loss_transl_diff_c2": 1.0,
                "loss_transl_diff_c4": 1.0,
                "loss_pose_diff_c2": 0.0,
                "loss_pose_diff_c4": 0.0,
                "loss_shape_prior": 0.0,
                "loss_shape_bound": 0.0,
                "loss_shape_anchor": 0.1,
                "loss_shape_norm": 0.0,
                "loss_densekp_c2": 0.0,
                "loss_densekp_c4": 0.0,
                "loss_sam3d_dense_c2": 0.0,
                "loss_sam3d_dense_c4": 0.0,
            },
        ),
        (
            "stage2_body",
            lambda: [
                smpl_c2.orient,
                smpl_c2.transl,
                smpl_c4.orient,
                smpl_c4.transl,
                shared_body_pose,
                shared_beta,
            ],
            60,
            stop_ratio,
            {
                "body_2d_det_c2": 1.0,
                "body_2d_det_c4": 1.0,
                "loss_pose_prior_c2": 1.0,
                "loss_pose_prior_c4": 1.0,
                "loss_hand_prior_c2": 0.0,
                "loss_hand_prior_c4": 0.0,
                "loss_pose_anchor_c2": 3.0,
                "loss_pose_anchor_c4": 3.0,
                "loss_transl_diff_c2": 1.0,
                "loss_transl_diff_c4": 1.0,
                "loss_pose_diff_c2": 1.0,
                "loss_pose_diff_c4": 1.0,
                "loss_shape_prior": 0.1,
                "loss_shape_bound": 1.0,
                "loss_shape_anchor": 0.1,
                "loss_shape_norm": 1.0,
                "loss_densekp_c2": 1.0,
                "loss_densekp_c4": 1.0,
                "loss_sam3d_dense_c2": 1.0,
                "loss_sam3d_dense_c4": 1.0,
            },
        ),
    ]
