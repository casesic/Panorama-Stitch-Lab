"""SAM 3D Body dense surface reprojection loss（单视角/双视角/序列）。

约定：传入的 projected_points_2d 必须已是 equirect_rotate 全图坐标；本模块只算 loss，
不做投影 / 坐标变换 / 黑区或 mask 过滤 / 随机采样。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from lib_pano.fitting_io.sam3d_dense_io import SAM3DDenseResult
from lib_pano.human3d.mesh_surface_sampling import (
    get_valid_sampled_vertex_ids,
    load_vertex_ids,
)

SMPLX_NUM_VERTS = 10475


@dataclass
class SAM3DDenseLossStats:
    """单视角 dense loss 统计（请求点数 / 有效点数 / 平均与最大 2D L2 像素误差）。"""

    requested_num_points: int
    valid_num_points: int
    mean_l2_px: float
    max_l2_px: float


def get_default_sam3d_sample_asset_path(num_points: int) -> Path:
    """返回固定采样 asset 路径 human3d/assets/smplx_surface_ids_fps_{num_points}.npy（由 __file__ 推导）。"""
    assets_dir = Path(__file__).resolve().parents[2] / "human3d" / "assets"
    return assets_dir / f"smplx_surface_ids_fps_{num_points}.npy"


def load_default_sam3d_sample_ids(num_points: int = 1024) -> np.ndarray:
    """读取固定采样的 SMPL-X surface vertex ids，返回一维 `np.int32`。"""
    asset_path = get_default_sam3d_sample_asset_path(num_points)
    return load_vertex_ids(
        asset_path,
        total_vertices=SMPLX_NUM_VERTS,
        expected_count=num_points,
    )


def select_valid_sam3d_dense_points(
    obs: SAM3DDenseResult,
    sample_ids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """按 obs.valid 从固定采样 ids 筛出有效点，返回 (valid_ids[K] int32, obs_points_2d[K,2] float32)，保持原顺序。"""
    valid_ids = get_valid_sampled_vertex_ids(sample_ids, obs.valid)
    obs_points_2d = obs.points_2d[valid_ids].astype(np.float32)
    return valid_ids, obs_points_2d


def _validate_inputs(
    projected_points_2d: torch.Tensor,
    obs: SAM3DDenseResult,
    num_points: int,
    min_valid_points: int,
) -> None:
    """对单视角 loss 的输入做形状与语义检查，不满足时抛出清晰 ValueError。"""
    if num_points <= 0:
        raise ValueError(f"num_points 必须为正，实际为 {num_points}")
    if min_valid_points < 0:
        raise ValueError(f"min_valid_points 不能为负，实际为 {min_valid_points}")
    if tuple(projected_points_2d.shape) != (SMPLX_NUM_VERTS, 2):
        raise ValueError(
            f"projected_points_2d 形状必须为 ({SMPLX_NUM_VERTS}, 2)，"
            f"实际为 {tuple(projected_points_2d.shape)}"
        )
    if obs.vertex_ids.shape != (SMPLX_NUM_VERTS,):
        raise ValueError(
            f"obs.vertex_ids 形状必须为 ({SMPLX_NUM_VERTS},)，实际为 {obs.vertex_ids.shape}"
        )
    if obs.points_2d.shape != (SMPLX_NUM_VERTS, 2):
        raise ValueError(
            f"obs.points_2d 形状必须为 ({SMPLX_NUM_VERTS}, 2)，实际为 {obs.points_2d.shape}"
        )
    if obs.valid.shape != (SMPLX_NUM_VERTS,):
        raise ValueError(
            f"obs.valid 形状必须为 ({SMPLX_NUM_VERTS},)，实际为 {obs.valid.shape}"
        )
    if not np.array_equal(obs.vertex_ids, np.arange(SMPLX_NUM_VERTS, dtype=np.int32)):
        raise ValueError(
            "obs.vertex_ids 必须与 np.arange(10475) 一一对应，"
            "当前实现假设 observation 顶点语义与 SMPL-X 10475 顶点一致"
        )


def compute_sam3d_dense_reprojection_loss(
    projected_points_2d: torch.Tensor,
    obs: SAM3DDenseResult,
    sample_ids: np.ndarray | None = None,
    num_points: int = 1024,
    min_valid_points: int = 16,
    robust_eps: float = 1.0,
    compute_stats: bool = True,
) -> tuple[torch.Tensor, SAM3DDenseLossStats | None]:
    """单视角 dense 2D reprojection loss（Charbonnier robust）；compute_stats=False 跳过统计避免 GPU→CPU 同步，stats 返回 None。"""
    _validate_inputs(projected_points_2d, obs, num_points, min_valid_points)

    if sample_ids is None:
        sample_ids = load_default_sam3d_sample_ids(num_points)

    requested_num_points = int(np.asarray(sample_ids).reshape(-1).shape[0])
    valid_ids, obs_points_2d = select_valid_sam3d_dense_points(obs, sample_ids)
    num_valid = int(valid_ids.shape[0])

    if num_valid < min_valid_points or num_valid == 0:
        loss = projected_points_2d.new_zeros(())
        if compute_stats:
            stats = SAM3DDenseLossStats(
                requested_num_points=requested_num_points,
                valid_num_points=num_valid,
                mean_l2_px=0.0,
                max_l2_px=0.0,
            )
        else:
            stats = None
        return loss, stats

    device = projected_points_2d.device
    idx = torch.as_tensor(valid_ids.astype(np.int64), device=device)
    pred = projected_points_2d.index_select(0, idx)
    obs_t = torch.as_tensor(obs_points_2d, dtype=pred.dtype, device=device)

    diff = pred - obs_t
    sq = (diff ** 2).sum(dim=1)
    loss_i = torch.sqrt(sq + robust_eps ** 2) - robust_eps
    loss = loss_i.mean()

    if compute_stats:
        l2 = torch.sqrt(sq)
        stats = SAM3DDenseLossStats(
            requested_num_points=requested_num_points,
            valid_num_points=num_valid,
            mean_l2_px=float(l2.mean().detach().cpu()),
            max_l2_px=float(l2.max().detach().cpu()),
        )
    else:
        stats = None
    return loss, stats


def compute_dual_view_sam3d_dense_loss(
    projected_cam2: torch.Tensor,
    obs_cam2: SAM3DDenseResult | None,
    projected_cam4: torch.Tensor,
    obs_cam4: SAM3DDenseResult | None,
    sample_ids: np.ndarray | None = None,
    num_points: int = 1024,
    min_valid_points: int = 16,
    robust_eps: float = 1.0,
    compute_stats: bool = True,
) -> tuple[torch.Tensor, dict[str, SAM3DDenseLossStats | None]]:
    """双视角平均 dense loss：两边存在取 0.5*(c2+c4)，仅一边存在返回该边，均无返回 0；compute_stats 透传，dict 形如 {"cam2":..,"cam4":..}。"""
    loss_cam2 = None
    loss_cam4 = None
    stats_cam2 = None
    stats_cam4 = None

    if obs_cam2 is not None:
        loss_cam2, stats_cam2 = compute_sam3d_dense_reprojection_loss(
            projected_cam2, obs_cam2, sample_ids, num_points, min_valid_points,
            robust_eps, compute_stats,
        )
    if obs_cam4 is not None:
        loss_cam4, stats_cam4 = compute_sam3d_dense_reprojection_loss(
            projected_cam4, obs_cam4, sample_ids, num_points, min_valid_points,
            robust_eps, compute_stats,
        )

    if loss_cam2 is not None and loss_cam4 is not None:
        loss = 0.5 * (loss_cam2 + loss_cam4)
    elif loss_cam2 is not None:
        loss = loss_cam2
    elif loss_cam4 is not None:
        loss = loss_cam4
    else:
        loss = projected_cam2.new_zeros(())

    return loss, {"cam2": stats_cam2, "cam4": stats_cam4}


def compute_sam3d_dense_sequence_reprojection_loss(
    projected_points_2d: torch.Tensor,
    obs_list: list[SAM3DDenseResult | None] | None,
    sample_ids: np.ndarray | None = None,
    num_points: int = 1024,
    min_valid_points: int = 16,
    robust_eps: float = 1.0,
    compute_stats: bool = True,
) -> tuple[torch.Tensor, list[SAM3DDenseLossStats | None] | None]:
    """序列级 dense loss（projected [T,10475,2]）：每帧独立 robust mean，再对有效帧取 mean（避免有效点多的帧支配）；compute_stats=False 跳过统计避免同步。"""
    if projected_points_2d.ndim != 3:
        raise ValueError(
            f"projected_points_2d 必须为 3 维 [T, {SMPLX_NUM_VERTS}, 2]，"
            f"实际 ndim={projected_points_2d.ndim}"
        )
    if tuple(projected_points_2d.shape[1:]) != (SMPLX_NUM_VERTS, 2):
        raise ValueError(
            f"projected_points_2d 后两维必须为 ({SMPLX_NUM_VERTS}, 2)，"
            f"实际为 {tuple(projected_points_2d.shape[1:])}"
        )
    if num_points <= 0:
        raise ValueError(f"num_points 必须为正，实际为 {num_points}")
    if min_valid_points < 0:
        raise ValueError(f"min_valid_points 不能为负，实际为 {min_valid_points}")

    num_frames = int(projected_points_2d.shape[0])
    if obs_list is not None and len(obs_list) != num_frames:
        raise ValueError(
            f"obs_list 长度 {len(obs_list)} 必须等于帧数 T={num_frames}"
        )

    # 整段无 observation：按全 None 处理，最终因无帧参与而返回零 loss
    frames_obs = [None] * num_frames if obs_list is None else obs_list

    # 固定采样：整段只确定一次，所有帧共享同一组 vertex ids
    if sample_ids is None:
        sample_ids = load_default_sam3d_sample_ids(num_points=num_points)

    losses: list[torch.Tensor] = []
    stats_per_frame: list[SAM3DDenseLossStats | None] | None = (
        [] if compute_stats else None
    )

    for t in range(num_frames):
        obs = frames_obs[t]

        if obs is None:
            if compute_stats:
                stats_per_frame.append(None)
            continue

        valid_ids, _ = select_valid_sam3d_dense_points(obs, sample_ids)

        if len(valid_ids) < min_valid_points:
            # 有效点不足：该帧不参与 sequence 平均
            if compute_stats:
                _, frame_stats = compute_sam3d_dense_reprojection_loss(
                    projected_points_2d=projected_points_2d[t],
                    obs=obs,
                    sample_ids=sample_ids,
                    num_points=num_points,
                    min_valid_points=min_valid_points,
                    robust_eps=robust_eps,
                    compute_stats=True,
                )
                stats_per_frame.append(frame_stats)
            continue

        frame_loss, frame_stats = compute_sam3d_dense_reprojection_loss(
            projected_points_2d=projected_points_2d[t],
            obs=obs,
            sample_ids=sample_ids,
            num_points=num_points,
            min_valid_points=min_valid_points,
            robust_eps=robust_eps,
            compute_stats=compute_stats,
        )
        losses.append(frame_loss)
        if compute_stats:
            stats_per_frame.append(frame_stats)

    if len(losses) > 0:
        loss = torch.stack(losses).mean()
    else:
        loss = projected_points_2d.new_zeros(())

    return loss, stats_per_frame


# Example:
# sample_ids = load_default_sam3d_sample_ids(num_points=1024)
# loss_cam2, stats_cam2 = compute_sam3d_dense_reprojection_loss(
#     projected_points_2d=projected_cam2,
#     obs=obs_cam2,
#     sample_ids=sample_ids,
# )
#
# loss_dual, stats_dual = compute_dual_view_sam3d_dense_loss(
#     projected_cam2=projected_cam2,
#     obs_cam2=obs_cam2,
#     projected_cam4=projected_cam4,
#     obs_cam4=obs_cam4,
#     sample_ids=sample_ids,
# )
#
# Sequence example:
# loss_seq, stats_seq = compute_sam3d_dense_sequence_reprojection_loss(
#     projected_points_2d=projected_cam2_seq,  # [T, 10475, 2]
#     obs_list=obs_list_cam2,
#     sample_ids=sample_ids,
#     compute_stats=False,
# )
