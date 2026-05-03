import cv2
import numpy as np
from typing import Optional, Tuple


class MeshWarpField:
    def __init__(
        self,
        full_h: int,
        full_w: int,
        grid_size: int = 32,
        decay_radius: float = 6.0,
        max_propagation: Optional[float] = 8.0,
        min_valid_ratio: float = 0.3,
        invalid_cell_strategy: str = 'interpolate',
        feather_width: float = 20.0,
        use_weighted_sampling: bool = True,
        min_weight: float = 0.05,
        min_weight_ratio: float = 0.02,
    ) -> None:
        self.H = full_h
        self.W = full_w
        self.grid_size = grid_size
        self.decay_radius = decay_radius
        self.max_propagation = max_propagation
        self.min_valid_ratio = min_valid_ratio
        self.invalid_cell_strategy = invalid_cell_strategy
        self.feather_width = feather_width
        self.use_weighted_sampling = use_weighted_sampling
        self.min_weight = min_weight
        self.min_weight_ratio = min_weight_ratio

        self.grid_h = int(np.ceil(full_h / grid_size))
        self.grid_w = int(np.ceil(full_w / grid_size))
        self._last_roi_grid_bounds = (0, 0, 0, 0)

    def _sample_grid_flow(
        self,
        flow_roi: np.ndarray,
        roi_bbox: Tuple[int, int, int, int],
        valid_mask: Optional[np.ndarray],
        weight_map: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        x_min, y_min, x_max, y_max = roi_bbox
        grid_flow = np.zeros((self.grid_h, self.grid_w, 2), dtype=np.float32)
        grid_mask = np.zeros((self.grid_h, self.grid_w), dtype=np.bool_)

        g_x_min = x_min // self.grid_size
        g_y_min = y_min // self.grid_size
        g_x_max = min(x_max // self.grid_size + 1, self.grid_w)
        g_y_max = min(y_max // self.grid_size + 1, self.grid_h)
        self._last_roi_grid_bounds = (g_x_min, g_y_min, g_x_max, g_y_max)

        for gy in range(g_y_min, g_y_max):
            for gx in range(g_x_min, g_x_max):
                px_start = gx * self.grid_size
                py_start = gy * self.grid_size
                px_end = min((gx + 1) * self.grid_size, self.W)
                py_end = min((gy + 1) * self.grid_size, self.H)

                intersect_x0 = max(px_start, x_min)
                intersect_y0 = max(py_start, y_min)
                intersect_x1 = min(px_end, x_max)
                intersect_y1 = min(py_end, y_max)

                if intersect_x1 <= intersect_x0 or intersect_y1 <= intersect_y0:
                    continue

                roi_slice = flow_roi[
                    intersect_y0 - y_min: intersect_y1 - y_min,
                    intersect_x0 - x_min: intersect_x1 - x_min,
                ]

                if self.use_weighted_sampling and weight_map is not None:
                    weight_slice = weight_map[
                        intersect_y0 - y_min: intersect_y1 - y_min,
                        intersect_x0 - x_min: intersect_x1 - x_min,
                    ].astype(np.float32)
                    weight_slice = np.clip(np.nan_to_num(weight_slice, nan=0.0), 0.0, 1.0)

                    if valid_mask is not None:
                        mask_slice = valid_mask[
                            intersect_y0 - y_min: intersect_y1 - y_min,
                            intersect_x0 - x_min: intersect_x1 - x_min,
                        ]
                        weight_slice = np.where(mask_slice > 0, weight_slice, 0.0)

                    strong_weight = weight_slice > self.min_weight
                    cell_area = weight_slice.size
                    weight_sum = float(np.sum(weight_slice[strong_weight]))
                    weight_ratio = weight_sum / float(cell_area) if cell_area > 0 else 0.0

                    if np.any(strong_weight) and weight_ratio >= self.min_weight_ratio:
                        weights = weight_slice[strong_weight]
                        weighted_flow = roi_slice[strong_weight] * weights[:, None]
                        grid_flow[gy, gx] = np.sum(weighted_flow, axis=0) / max(weight_sum, 1e-6)
                        grid_mask[gy, gx] = True
                elif valid_mask is not None:
                    mask_slice = valid_mask[
                        intersect_y0 - y_min: intersect_y1 - y_min,
                        intersect_x0 - x_min: intersect_x1 - x_min,
                    ]

                    valid_pixels = roi_slice[mask_slice > 0.5]
                    total_pixels = mask_slice.size
                    valid_ratio = float(len(valid_pixels)) / float(total_pixels) if total_pixels > 0 else 0.0

                    if valid_ratio >= self.min_valid_ratio:
                        grid_flow[gy, gx] = np.mean(valid_pixels, axis=0)
                        grid_mask[gy, gx] = True
                else:
                    grid_flow[gy, gx] = np.mean(roi_slice, axis=(0, 1))
                    grid_mask[gy, gx] = True

        return grid_flow, grid_mask

    def _build_base_flow(self, grid_flow: np.ndarray, grid_mask: np.ndarray) -> np.ndarray:
        base_flow = grid_flow.copy()
        has_data_rows = np.any(grid_mask, axis=1)
        for gy in range(self.grid_h):
            if not has_data_rows[gy]:
                continue

            row_mask = grid_mask[gy, :]
            indices = np.where(row_mask)[0]
            first_idx, last_idx = indices[0], indices[-1]

            base_flow[gy, :first_idx] = base_flow[gy, first_idx]
            base_flow[gy, last_idx + 1:] = base_flow[gy, last_idx]

        valid_row_indices = np.where(has_data_rows)[0]
        if len(valid_row_indices) > 0:
            if valid_row_indices[0] > 0:
                top_row = base_flow[valid_row_indices[0]]
                base_flow[:valid_row_indices[0], :] = top_row

            if valid_row_indices[-1] < self.grid_h - 1:
                bottom_row = base_flow[valid_row_indices[-1]]
                base_flow[valid_row_indices[-1] + 1:, :] = bottom_row

            for k in range(len(valid_row_indices) - 1):
                r1 = valid_row_indices[k]
                r2 = valid_row_indices[k + 1]
                if r2 - r1 > 1:
                    for r in range(r1 + 1, r2):
                        ratio = (r - r1) / (r2 - r1)
                        base_flow[r] = (1 - ratio) * base_flow[r1] + ratio * base_flow[r2]

        return base_flow

    def _build_weight_mask(self, grid_mask: np.ndarray) -> np.ndarray:
        mask_uint8 = grid_mask.astype(np.uint8)
        inv_mask = 1 - mask_uint8

        if np.all(inv_mask == 1):
            dist_map = np.ones_like(inv_mask, dtype=np.float32) * 9999
        elif np.all(inv_mask == 0):
            dist_map = np.zeros_like(inv_mask, dtype=np.float32)
        else:
            dist_map = cv2.distanceTransform(inv_mask, cv2.DIST_L2, 5)

        weight_mask = np.clip(1.0 - dist_map / max(self.decay_radius, 1e-6), 0.0, 1.0)
        if self.max_propagation is not None:
            weight_mask = np.where(dist_map <= self.max_propagation, weight_mask, 0.0)

        if self.invalid_cell_strategy == 'freeze':
            g_x_min, g_y_min, g_x_max, g_y_max = self._last_roi_grid_bounds
            roi_grid_mask = np.zeros_like(grid_mask, dtype=np.bool_)
            roi_grid_mask[g_y_min:g_y_max, g_x_min:g_x_max] = True
            frozen_cells = roi_grid_mask & (~grid_mask)
            weight_mask[frozen_cells] = 0.0

        return weight_mask[..., None].astype(np.float32)

    def compute(
        self,
        flow_roi: np.ndarray,
        roi_bbox: Tuple[int, int, int, int],
        valid_mask: Optional[np.ndarray] = None,
        weight_map: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        grid_flow, grid_mask = self._sample_grid_flow(flow_roi, roi_bbox, valid_mask, weight_map)
        base_flow = self._build_base_flow(grid_flow, grid_mask)
        weight_mask = self._build_weight_mask(grid_mask)
        final_grid_flow = base_flow * weight_mask
        full_flow = cv2.resize(final_grid_flow, (self.W, self.H), interpolation=cv2.INTER_LINEAR)

        # ROI 内有效像素直接用 flow_roi 原始值覆盖，跳过 mesh 平均
        # x_min, y_min, x_max, y_max = roi_bbox
        # if valid_mask is not None:
        #     valid_region = valid_mask > 0.5
        # else:
        #     valid_region = np.ones((y_max - y_min, x_max - x_min), dtype=np.bool_)

        # roi_in_full = full_flow[y_min:y_max, x_min:x_max].copy()
        # if np.any(valid_region):
        #     valid_uint8 = valid_region.astype(np.uint8)
        #     if self.feather_width <= 1e-6:
        #         alpha = valid_region.astype(np.float32)
        #     else:
        #         interior_valid = cv2.erode(valid_uint8, np.ones((3, 3), np.uint8), iterations=1)
        #         dist_to_boundary = cv2.distanceTransform(interior_valid, cv2.DIST_L2, 5)
        #         alpha = np.clip(dist_to_boundary / self.feather_width, 0.0, 1.0)
        #         alpha = np.where(valid_region, alpha, 0.0).astype(np.float32)

        #     alpha = alpha[..., None]
        #     roi_in_full[valid_region] = (
        #         alpha[valid_region] * flow_roi[valid_region]
        #         + (1.0 - alpha[valid_region]) * roi_in_full[valid_region]
        #     )

        # full_flow[y_min:y_max, x_min:x_max] = roi_in_full

        grid_x, grid_y = np.meshgrid(np.arange(self.W), np.arange(self.H))
        map_x = (grid_x + full_flow[..., 0]).astype(np.float32)
        map_y = (grid_y + full_flow[..., 1]).astype(np.float32)
        return map_x, map_y, full_flow
