import os
import cv2
import numpy as np
from typing import Optional
from lib_pano.flow.warp_strategy import WarpResult


def _visualize_flow_bgr(flow: np.ndarray) -> np.ndarray:
    magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1], angleInDegrees=False)
    hsv = np.zeros((flow.shape[0], flow.shape[1], 3), dtype=np.uint8)
    hsv[..., 0] = ((angle * 180.0 / np.pi) / 2.0).astype(np.uint8)
    hsv[..., 1] = 255
    if float(magnitude.max()) > 0.0:
        hsv[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


class DebugWriter:
    def __init__(self, debug_dir: str, enabled: bool = True):
        self.debug_dir = debug_dir
        self.enabled = enabled
        if enabled:
            os.makedirs(debug_dir, exist_ok=True)

    def _save(self, filename: str, image: np.ndarray):
        if not self.enabled:
            return
        cv2.imwrite(os.path.join(self.debug_dir, filename), image)

    def write_crops(self, crop_source: np.ndarray, crop_target: np.ndarray):
        self._save('crop_source.png', crop_source)
        self._save('crop_target.png', crop_target)

    def write_flow(self, flow_roi: np.ndarray, valid_mask: Optional[np.ndarray] = None):
        self._save('flow_roi_vis.png', _visualize_flow_bgr(flow_roi))
        if valid_mask is not None:
            self._save('flow_validity_mask.png', (valid_mask * 255).astype(np.uint8))

    def write_alignment_debug(
        self,
        crop_source: np.ndarray,
        crop_target: np.ndarray,
        flow_roi: np.ndarray,
    ):
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
        self._save('warped_source.png', warp_result.warped_source)
        self._save('warped_target.png', warp_result.warped_target)
        self._save('full_flow_vis.png', _visualize_flow_bgr(warp_result.full_flow))
        if warp_result.warped_source_mask is not None:
            self._save('warped_source_mask.png', (warp_result.warped_source_mask > 0).astype(np.uint8) * 255)
        if warp_result.warped_target_mask is not None:
            self._save('warped_target_mask.png', (warp_result.warped_target_mask > 0).astype(np.uint8) * 255)

    def write_final(self, image: np.ndarray, valid_mask: Optional[np.ndarray] = None):
        self._save('final_composed.png', image)
        if valid_mask is not None:
            self._save('final_valid_mask.png', valid_mask)
