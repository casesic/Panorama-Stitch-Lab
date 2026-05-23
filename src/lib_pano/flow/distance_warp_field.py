import cv2
import numpy as np
from typing import Optional, Tuple


class DistanceWarpField:
    def __init__(
        self,
        full_h: int,
        full_w: int,
        decay_radius_px: float = 200.0,
        max_propagation_px: float = 400.0,
        seed_erode_kernel: int = 3,
        hole_close_kernel: int = 7,
        extrapolation_blur_sigma: float = 8.0,
        downsample_factor: int = 2,
    ) -> None:
        self.H = full_h
        self.W = full_w
        self.decay_radius_px = decay_radius_px
        self.max_propagation_px = max_propagation_px
        self.seed_erode_kernel = seed_erode_kernel
        self.hole_close_kernel = hole_close_kernel
        self.extrapolation_blur_sigma = extrapolation_blur_sigma
        self.downsample_factor = max(1, int(downsample_factor))

    def compute(
        self,
        flow_roi: np.ndarray,
        roi_bbox: Tuple[int, int, int, int],
        valid_mask: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        seed_flow = np.zeros((self.H, self.W, 2), dtype=np.float32)
        seed_mask = np.zeros((self.H, self.W), dtype=np.uint8)

        x_min, y_min, x_max, y_max = roi_bbox
        roi_h = y_max - y_min
        roi_w = x_max - x_min

        seed_flow[y_min:y_max, x_min:x_max] = flow_roi.astype(np.float32)
        if valid_mask is None:
            roi_valid = np.ones((roi_h, roi_w), dtype=np.uint8) * 255
        else:
            roi_valid = (valid_mask > 0.5).astype(np.uint8) * 255
        seed_mask[y_min:y_max, x_min:x_max] = roi_valid

        closed_mask = seed_mask.copy()
        if self.hole_close_kernel > 1:
            close_kernel = np.ones((self.hole_close_kernel, self.hole_close_kernel), np.uint8)
            closed_mask = cv2.morphologyEx(closed_mask, cv2.MORPH_CLOSE, close_kernel)

        cleaned_mask = closed_mask.copy()
        if self.seed_erode_kernel > 1:
            erode_kernel = np.ones((self.seed_erode_kernel, self.seed_erode_kernel), np.uint8)
            cleaned_mask = cv2.erode(cleaned_mask, erode_kernel, iterations=1)

        d = self.downsample_factor
        low_h = max(1, int(np.ceil(self.H / d)))
        low_w = max(1, int(np.ceil(self.W / d)))

        if d > 1:
            seed_flow_low = cv2.resize(seed_flow, (low_w, low_h), interpolation=cv2.INTER_NEAREST)
            cleaned_mask_low = cv2.resize(cleaned_mask, (low_w, low_h), interpolation=cv2.INTER_NEAREST)
            protected_mask_low = cv2.resize(closed_mask, (low_w, low_h), interpolation=cv2.INTER_NEAREST)
            decay_radius = self.decay_radius_px / d
            max_propagation = self.max_propagation_px / d
            blur_sigma = self.extrapolation_blur_sigma / d
        else:
            seed_flow_low = seed_flow
            cleaned_mask_low = cleaned_mask
            protected_mask_low = closed_mask
            decay_radius = self.decay_radius_px
            max_propagation = self.max_propagation_px
            blur_sigma = self.extrapolation_blur_sigma

        seed_binary = cleaned_mask_low > 0
        protected_binary = protected_mask_low > 0

        if not np.any(seed_binary):
            full_flow = np.zeros((self.H, self.W, 2), dtype=np.float32)
        else:
            inv_mask = (~seed_binary).astype(np.uint8)
            dist, labels = cv2.distanceTransformWithLabels(
                inv_mask,
                cv2.DIST_L2,
                5,
                labelType=cv2.DIST_LABEL_PIXEL,
            )

            seed_coords = np.column_stack(np.where(inv_mask == 0))
            label_lookup = np.zeros((len(seed_coords) + 1, 2), dtype=np.int32)
            label_lookup[1:, 0] = seed_coords[:, 0]
            label_lookup[1:, 1] = seed_coords[:, 1]
            nearest_coords = label_lookup[labels]

            nearest_y = nearest_coords[..., 0]
            nearest_x = nearest_coords[..., 1]
            inherited_flow = seed_flow_low[nearest_y, nearest_x]

            weight = np.clip(1.0 - dist / max(decay_radius, 1e-6), 0.0, 1.0)
            weight[dist > max_propagation] = 0.0
            full_flow_low = inherited_flow * weight[..., None].astype(np.float32)

            if blur_sigma > 1e-6:
                blurred_flow = cv2.GaussianBlur(
                    full_flow_low,
                    (0, 0),
                    sigmaX=blur_sigma,
                    sigmaY=blur_sigma,
                )
                full_flow_low = np.where(protected_binary[..., None], full_flow_low, blurred_flow)

            if d > 1:
                full_flow = cv2.resize(full_flow_low, (self.W, self.H), interpolation=cv2.INTER_LINEAR)
            else:
                full_flow = full_flow_low

        protected_mask_full = closed_mask > 0
        full_flow[protected_mask_full] = seed_flow[protected_mask_full]

        grid_x, grid_y = np.meshgrid(np.arange(self.W), np.arange(self.H))
        map_x = (grid_x + full_flow[..., 0]).astype(np.float32)
        map_y = (grid_y + full_flow[..., 1]).astype(np.float32)
        return map_x, map_y, full_flow
