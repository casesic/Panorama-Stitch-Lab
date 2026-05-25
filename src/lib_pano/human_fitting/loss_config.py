from __future__ import annotations

from dataclasses import dataclass


@dataclass
class DualCamLossWeights:
    body_2d: float = 1.0
    silhouette: float = 0.3
    pose_prior: float = 0.05
    hand_prior: float = 0.02
    pose_anchor: float = 1.0
    transl_diff: float = 0.0
    pose_diff: float = 0.0
    shape_prior: float = 6.0
    shape_bound: float = 20.0
    shape_anchor: float = 30.0
    shape_norm: float = 15.0
    densekp: float = 0.01


def build_base_weights(
    n_frame: int,
    use_silhouette_loss: bool,
    weights: DualCamLossWeights,
) -> dict:
    sil_weight = weights.silhouette if use_silhouette_loss else 0.0
    return {
        "body_2d_det_c2": n_frame * weights.body_2d,
        "body_2d_det_c4": n_frame * weights.body_2d,
        "loss_sil_c2": n_frame * sil_weight,
        "loss_sil_c4": n_frame * sil_weight,
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
    }


def build_optimization_stages(
    stop_ratio: float,
    smpl_c2,
    smpl_c4,
    shared_body_pose,
    shared_beta,
) -> list:
    return [
        (
            "stage1_transl_orient",
            lambda: [smpl_c2.orient, smpl_c2.transl, smpl_c4.orient, smpl_c4.transl],
            40,
            stop_ratio,
            {
                "body_2d_det_c2": 1.0,
                "body_2d_det_c4": 1.0,
                "loss_sil_c2": 0.0,
                "loss_sil_c4": 0.0,
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
                "loss_sil_c2": 1.0,
                "loss_sil_c4": 1.0,
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
            },
        ),
    ]
