"""双相机拟合的损失配置：各损失项权重、按帧数缩放的基础权重表、分阶段优化的阶段定义。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

OptimizationProfile = Literal["legacy", "staged_charbonnier_parity"]


@dataclass
class DualCamLossWeights:
    """双相机 SMPL-X 拟合各损失项的权重配置。"""
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
    sam3d_dense: float = 1.0


def build_base_weights(
    n_frame: int,
    primary_observation_source: str,
    weights: DualCamLossWeights,
    optimization_profile: OptimizationProfile = "legacy",
) -> dict:
    """按帧数缩放各权重，生成 c2/c4 双相机各损失项的基础权重字典。"""
    if primary_observation_source == "densekp":
        primary_weight = weights.densekp
    elif primary_observation_source == "sam3d_dense":
        primary_weight = weights.sam3d_dense
    else:
        raise ValueError(f"primary_observation_source must be 'densekp' or 'sam3d_dense', got {primary_observation_source!r}")

    if optimization_profile == "staged_charbonnier_parity":
        return {
            "loss_primary_obs_c2": primary_weight,
            "loss_primary_obs_c4": primary_weight,
            "loss_pose_anchor_shared": 1.0,
            "loss_shape_anchor": 10.0,
        }

    # legacy
    return {
        "loss_primary_obs_c2": n_frame * primary_weight,
        "loss_primary_obs_c4": n_frame * primary_weight,
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
    }


def build_optimization_stages(
    stop_ratio: float,
    smpl_c2,
    smpl_c4,
    body_pose_param,
    legacy_shared_beta=None,
    beta_latent_param=None,
    optimization_profile: OptimizationProfile = "legacy",
) -> list:
    """定义 LBFGS 优化阶段。

    legacy: 两阶段 (stage1_transl_orient, stage2_body)

    staged_charbonnier_parity: 三阶段
      stage1_camera_alignment (orient/transl only, primary only)
      stage2_pose_warmup (orient/transl + shared_body_pose, primary + pose_anchor)
      stage2_pose_shape_refinement (all + beta_latent, primary + pose_anchor + shape_anchor)
    """
    if optimization_profile == "staged_charbonnier_parity":
        return [
            (
                "stage1_camera_alignment",
                lambda: [smpl_c2.orient, smpl_c2.transl, smpl_c4.orient, smpl_c4.transl],
                30,
                stop_ratio,
                {
                    "loss_primary_obs_c2": 1.0,
                    "loss_primary_obs_c4": 1.0,
                    "loss_pose_anchor_shared": 0.0,
                    "loss_shape_anchor": 0.0,
                },
            ),
            (
                "stage2_pose_warmup",
                lambda: [smpl_c2.orient, smpl_c2.transl, smpl_c4.orient, smpl_c4.transl, body_pose_param],
                40,
                stop_ratio,
                {
                    "loss_primary_obs_c2": 1.0,
                    "loss_primary_obs_c4": 1.0,
                    "loss_pose_anchor_shared": 1.0,
                    "loss_shape_anchor": 0.0,
                },
            ),
            (
                "stage2_pose_shape_refinement",
                lambda: [
                    smpl_c2.orient, smpl_c2.transl, smpl_c4.orient, smpl_c4.transl,
                    body_pose_param, beta_latent_param,
                ],
                50,
                stop_ratio,
                {
                    "loss_primary_obs_c2": 1.0,
                    "loss_primary_obs_c4": 1.0,
                    "loss_pose_anchor_shared": 1.0,
                    "loss_shape_anchor": 1.0,
                },
            ),
        ]

    # legacy
    return [
        (
            "stage1_transl_orient",
            lambda: [smpl_c2.orient, smpl_c2.transl, smpl_c4.orient, smpl_c4.transl],
            40,
            stop_ratio,
            {
                "loss_primary_obs_c2": 1.0,
                "loss_primary_obs_c4": 1.0,
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
            },
        ),
        (
            "stage2_body",
            lambda: [
                smpl_c2.orient, smpl_c2.transl, smpl_c4.orient, smpl_c4.transl,
                body_pose_param, legacy_shared_beta,
            ],
            60,
            stop_ratio,
            {
                "loss_primary_obs_c2": 1.0,
                "loss_primary_obs_c4": 1.0,
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
            },
        ),
    ]
