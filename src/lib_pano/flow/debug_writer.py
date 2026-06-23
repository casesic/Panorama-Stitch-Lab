"""光流拼接调试输出：将中间 flow/mask/warp 结果按调试等级选择性保存为 PNG 图像。"""

import os
import cv2
import numpy as np
from typing import Optional
from lib_pano.flow.warp_strategy import WarpResult


MINIMAL_DEBUG_FILES = {
    "support_roi_mask.png",
    "flow_weight_map.png",
    "semantic_weight_map.png",
    "combined_weight_map.png",
    "full_flow_vis.png",
}


def _visualize_flow_bgr(flow: np.ndarray) -> np.ndarray:
    """将光流 (dx,dy) 可视化为 HSV 彩色图，色调=方向，亮度=大小。"""
    magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1], angleInDegrees=False)
    hsv = np.zeros((flow.shape[0], flow.shape[1], 3), dtype=np.uint8)
    hsv[..., 0] = ((angle * 180.0 / np.pi) / 2.0).astype(np.uint8)
    hsv[..., 1] = 255
    if float(magnitude.max()) > 0.0:
        hsv[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


class DebugWriter:
    """按调试等级 (minimal/full/none) 选择性保存中间结果图像。"""
    def __init__(self, debug_dir: str, enabled: bool = True, debug_level: str = "minimal"):
        self.debug_dir = debug_dir
        self.enabled = enabled
        self.debug_level = debug_level
        if enabled:
            os.makedirs(debug_dir, exist_ok=True)

    def _should_save(self, filename: str) -> bool:
        """根据调试等级判断该文件是否应该保存。"""
        if not self.enabled:
            return False
        if self.debug_level == "none":
            return False
        if self.debug_level == "full":
            return True
        if self.debug_level == "minimal":
            return filename in MINIMAL_DEBUG_FILES
        raise ValueError(f"Unsupported debug_level: {self.debug_level}")

    def _save(self, filename: str, image: np.ndarray):
        """受控写入：先检查 _should_save，通过则保存到 debug_dir。"""
        if not self._should_save(filename):
            return
        cv2.imwrite(os.path.join(self.debug_dir, filename), image)

    def write_image(self, name: str, image: np.ndarray):
        """保存任意图像（自动补 .png 后缀）。"""
        filename = name if name.lower().endswith(".png") else f"{name}.png"
        self._save(filename, image)

    def write_crops(self, crop_source: np.ndarray, crop_target: np.ndarray):
        """保存 ROI 裁剪后的 source 和 target 图像。"""
        self._save('crop_source.png', crop_source)
        self._save('crop_target.png', crop_target)

    def write_flow(self, flow_roi: np.ndarray, valid_mask: Optional[np.ndarray] = None):
        """保存 ROI 光流可视化图和可选的有效性 mask。"""
        self._save('flow_roi_vis.png', _visualize_flow_bgr(flow_roi))
        if valid_mask is not None:
            self.write_mask('flow_validity_mask.png', valid_mask)

    def write_flow_vis(self, name: str, flow: np.ndarray):
        """以指定名称保存光流可视化图。"""
        filename = name if name.lower().endswith(".png") else f"{name}.png"
        self._save(filename, _visualize_flow_bgr(flow))

    def write_weight_map(self, name: str, weight: np.ndarray, max_value: float | None = None):
        """保存权重图（自动归一化到 0-255）。"""
        filename = name if name.lower().endswith(".png") else f"{name}.png"

        weight = np.nan_to_num(weight.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        weight = np.maximum(weight, 0.0)

        if max_value is None:
            max_value = float(weight.max())
        max_value = max(float(max_value), 1e-6)

        vis = np.clip(weight / max_value * 255.0, 0.0, 255.0).astype(np.uint8)
        self._save(filename, vis)

    def write_mask(self, name: str, mask: np.ndarray):
        """保存 mask（自动处理 bool/float/uint8 各种类型）。"""
        filename = name if name.lower().endswith('.png') else f'{name}.png'
        mask_to_save = mask
        if mask_to_save.dtype == np.bool_:
            mask_to_save = mask_to_save.astype(np.uint8) * 255
        elif np.issubdtype(mask_to_save.dtype, np.floating):
            finite_mask = np.nan_to_num(mask_to_save, nan=0.0, posinf=1.0, neginf=0.0)
            if finite_mask.max() <= 1.0:
                mask_to_save = np.clip(finite_mask * 255.0, 0.0, 255.0).astype(np.uint8)
            else:
                mask_to_save = np.clip(finite_mask, 0.0, 255.0).astype(np.uint8)
        else:
            mask_to_save = mask_to_save.astype(np.uint8)
            if mask_to_save.max() <= 1:
                mask_to_save = mask_to_save * 255
        self._save(filename, mask_to_save)

    def write_alignment_debug(
        self,
        crop_source: np.ndarray,
        crop_target: np.ndarray,
        flow_roi: np.ndarray,
    ):
        """保存 warp 前/后的 overlay 对比图，用于快速检查光流对齐效果。"""
        h, w = flow_roi.shape[:2]
        grid_y, grid_x = np.mgrid[0:h, 0:w].astype(np.float32)
        map_x = grid_x + flow_roi[..., 0].astype(np.float32)
        map_y = grid_y + flow_roi[..., 1].astype(np.float32)
        warped_crop_source = cv2.remap(
            crop_source,
            map_x,
            map_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )

        before = cv2.addWeighted(crop_target, 0.5, crop_source, 0.5, 0.0)
        after = cv2.addWeighted(crop_target, 0.5, warped_crop_source, 0.5, 0.0)

        target_black = crop_target.max(axis=2) < 5
        source_black = crop_source.max(axis=2) < 5
        warped_source_black = warped_crop_source.max(axis=2) < 5

        before[target_black] = crop_source[target_black]
        before[source_black] = crop_target[source_black]
        after[target_black] = warped_crop_source[target_black]
        after[warped_source_black] = crop_target[warped_source_black]

        self._save('alignment_before.png', before)
        self._save('alignment_after.png', after)

    def write_warp_result(self, warp_result: WarpResult):
        """保存双向 warp 结果和全图光流可视化。"""
        self._save('warped_source.png', warp_result.warped_source)
        self._save('warped_target.png', warp_result.warped_target)
        self._save('full_flow_vis.png', _visualize_flow_bgr(warp_result.full_flow))
        if warp_result.warped_source_mask is not None:
            self._save('warped_source_mask.png', (warp_result.warped_source_mask > 0).astype(np.uint8) * 255)
        if warp_result.warped_target_mask is not None:
            self._save('warped_target_mask.png', (warp_result.warped_target_mask > 0).astype(np.uint8) * 255)

    def write_final(self, image: np.ndarray, valid_mask: Optional[np.ndarray] = None):
        """保存最终合成图及其 valid mask。"""
        self._save('final_composed.png', image)
        if valid_mask is not None:
            self._save('final_valid_mask.png', valid_mask)
