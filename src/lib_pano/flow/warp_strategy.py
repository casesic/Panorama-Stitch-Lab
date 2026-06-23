"""变形策略：根据网格 warp 场生成单向/双向 remap，封装 source/target 的变形与遮罩。"""

import cv2
import numpy as np
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
from lib_pano.flow.mesh_warp_field import MeshWarpField


@dataclass
class MeshConfig:
    """网格变形场参数配置。"""
    grid_size: int = 32
    decay_radius: float = 6.0
    max_propagation: float = 8.0
    min_valid_ratio: float = 0.3
    invalid_cell_strategy: str = 'interpolate'
    feather_width: float = 20.0
    min_weight: float = 0.05
    min_weight_ratio: float = 0.02
    use_weighted_sampling: bool = True


@dataclass
class WarpStrategy(Enum):
    """变形策略枚举：单向（仅 source 向 target 变形）或双向（各变形一半）。"""
    MESH_UNIDIRECTIONAL = 'mesh_unidirectional'
    MESH_BIDIRECTIONAL = 'mesh_bidirectional'


@dataclass
class WarpResult:
    """变形结果：变形后的 source/target 图像、对应有效遮罩与全图光流。"""
    warped_source: np.ndarray
    warped_target: np.ndarray
    warped_source_mask: Optional[np.ndarray]
    warped_target_mask: Optional[np.ndarray]
    full_flow: np.ndarray


def _build_mesh_warper(img_shape: tuple, mesh_config: MeshConfig) -> MeshWarpField:
    """根据图像尺寸和配置构建 MeshWarpField 实例。"""
    h, w = img_shape[:2]
    return MeshWarpField(
        full_h=h,
        full_w=w,
        grid_size=mesh_config.grid_size,
        decay_radius=mesh_config.decay_radius,
        max_propagation=mesh_config.max_propagation,
        min_valid_ratio=mesh_config.min_valid_ratio,
        invalid_cell_strategy=mesh_config.invalid_cell_strategy,
        feather_width=mesh_config.feather_width,
        use_weighted_sampling=mesh_config.use_weighted_sampling,
        min_weight=mesh_config.min_weight,
        min_weight_ratio=mesh_config.min_weight_ratio,
    )


def _warp_unidirectional(
    warper,
    img_source: np.ndarray,
    img_target: np.ndarray,
    flow_roi: np.ndarray,
    roi_bbox: tuple,
    valid_mask: Optional[np.ndarray],
    weight_map: Optional[np.ndarray],
    source_valid_mask: Optional[np.ndarray],
    target_valid_mask: Optional[np.ndarray],
) -> WarpResult:
    """单向变形：仅将 source 按全图光流 remap 到 target 坐标系，target 保持不变。"""
    map_x, map_y, full_flow = warper.compute(flow_roi, roi_bbox, valid_mask, weight_map=weight_map)

    warped_source = cv2.remap(
        img_source,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
    )

    if source_valid_mask is not None:
        warped_source_mask = cv2.remap(
            source_valid_mask,
            map_x,
            map_y,
            interpolation=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
        )
    else:
        warped_source_mask = None

    warped_target = img_target.copy()
    warped_target_mask = target_valid_mask.copy() if target_valid_mask is not None else None

    return WarpResult(
        warped_source=warped_source,
        warped_target=warped_target,
        warped_source_mask=warped_source_mask,
        warped_target_mask=warped_target_mask,
        full_flow=full_flow,
    )


def _warp_bidirectional(
    warper,
    img_source: np.ndarray,
    img_target: np.ndarray,
    flow_roi: np.ndarray,
    roi_bbox: tuple,
    valid_mask: Optional[np.ndarray],
    weight_map: Optional[np.ndarray],
    source_valid_mask: Optional[np.ndarray],
    target_valid_mask: Optional[np.ndarray],
) -> WarpResult:
    """双向变形：source 和 target 各沿光流方向变形一半，在中间位置汇合。"""
    if isinstance(warper, MeshWarpField):
        _, _, full_flow = warper.compute(flow_roi, roi_bbox, valid_mask, weight_map=weight_map)
    else:
        _, _, full_flow = warper.compute(flow_roi, roi_bbox, valid_mask)
    flow_mid = full_flow * 0.5

    h, w = img_source.shape[:2]
    grid_y, grid_x = np.mgrid[0:h, 0:w].astype(np.float32)

    map_x_source = grid_x + flow_mid[..., 0]
    map_y_source = grid_y + flow_mid[..., 1]

    map_x_target = grid_x - flow_mid[..., 0]
    map_y_target = grid_y - flow_mid[..., 1]

    warped_source = cv2.remap(
        img_source,
        map_x_source,
        map_y_source,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
    )
    warped_target = cv2.remap(
        img_target,
        map_x_target,
        map_y_target,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
    )

    if source_valid_mask is not None:
        warped_source_mask = cv2.remap(
            source_valid_mask,
            map_x_source,
            map_y_source,
            interpolation=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
        )
    else:
        warped_source_mask = None

    if target_valid_mask is not None:
        warped_target_mask = cv2.remap(
            target_valid_mask,
            map_x_target,
            map_y_target,
            interpolation=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
        )
    else:
        warped_target_mask = None

    return WarpResult(
        warped_source=warped_source,
        warped_target=warped_target,
        warped_source_mask=warped_source_mask,
        warped_target_mask=warped_target_mask,
        full_flow=full_flow,
    )


def apply_warp(
    img_source: np.ndarray,
    img_target: np.ndarray,
    flow_roi: np.ndarray,
    roi_bbox: tuple,
    valid_mask: Optional[np.ndarray],
    strategy: WarpStrategy,
    source_valid_mask: Optional[np.ndarray] = None,
    target_valid_mask: Optional[np.ndarray] = None,
    mesh_config: MeshConfig = MeshConfig(),
    weight_map: Optional[np.ndarray] = None,
) -> WarpResult:
    """变形入口：根据策略选择单向或双向网格变形并返回 WarpResult。"""
    if strategy == WarpStrategy.MESH_UNIDIRECTIONAL:
        warper = _build_mesh_warper(img_source.shape, mesh_config)
        return _warp_unidirectional(
            warper=warper,
            img_source=img_source,
            img_target=img_target,
            flow_roi=flow_roi,
            roi_bbox=roi_bbox,
            valid_mask=valid_mask,
            weight_map=weight_map,
            source_valid_mask=source_valid_mask,
            target_valid_mask=target_valid_mask,
        )

    if strategy == WarpStrategy.MESH_BIDIRECTIONAL:
        warper = _build_mesh_warper(img_source.shape, mesh_config)
        return _warp_bidirectional(
            warper=warper,
            img_source=img_source,
            img_target=img_target,
            flow_roi=flow_roi,
            roi_bbox=roi_bbox,
            valid_mask=valid_mask,
            weight_map=weight_map,
            source_valid_mask=source_valid_mask,
            target_valid_mask=target_valid_mask,
        )

    raise ValueError(f'Unsupported warp strategy: {strategy}')
