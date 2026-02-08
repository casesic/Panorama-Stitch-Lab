import cv2
import numpy as np
import torch
import torch.nn.functional as F
from typing import List, Tuple, Optional, Union

class Mesh_Grid_Warper:
    def __init__(self, full_h: int, full_w: int, grid_size: int = 32) -> None:
        """
        Initialize the Mesh Grid Warper.

        Width and height are needed to pre-calculate the grid dimensions for the 
        global warp field, ensuring the mesh covers the entire image canvas.
        The grid size determines the granularity of the warp; a smaller grid allows 
        for more local deformation control but increases computational cost and 
        risk of local distortions.
        """
        self.H = full_h
        self.W = full_w
        self.grid_size = grid_size
        
        self.grid_h = int(np.ceil(full_h / grid_size))
        self.grid_w = int(np.ceil(full_w / grid_size))

    def compute_mesh_flow_from_roi(
        self, 
        flow_roi: np.ndarray, 
        roi_bbox: Tuple[int, int, int, int], 
        validity_mask: Optional[np.ndarray] = None, 
        return_flow: bool = False
    ) -> Union[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """
        Infer a global mesh deformation field from local ROI optical flow using Masked Decay.

        Why Masked Decay:
        Naive warping based strictly on optical flow works well for the overlapping 
        regions but often causes severe distortion or tearing in non-overlapping areas 
        where flow is undefined. We use a 'Masked Decay' strategy to smoothly transition 
        from the active alignment in the overlap region to a rigid identity transform 
        in the far field. This ensures that the stitched image remains natural-looking 
        outside the seam, preserving the structure of the original image where no 
        alignment information is available.
        """
        x_min, y_min, x_max, y_max = roi_bbox
        
        grid_flow = np.zeros((self.grid_h, self.grid_w, 2), dtype=np.float32)
        
        # Track which grids have valid flow data to guide the extrapolation
        grid_mask = np.zeros((self.grid_h, self.grid_w), dtype=np.bool_)

        g_x_min = x_min // self.grid_size
        g_y_min = y_min // self.grid_size
        g_x_max = min(x_max // self.grid_size + 1, self.grid_w)
        g_y_max = min(y_max // self.grid_size + 1, self.grid_h)
        
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
                
                if intersect_x1 > intersect_x0 and intersect_y1 > intersect_y0:
                    roi_slice = flow_roi[
                        intersect_y0 - y_min : intersect_y1 - y_min, 
                        intersect_x0 - x_min : intersect_x1 - x_min
                    ]
                    
                    if validity_mask is not None:
                        mask_slice = validity_mask[
                            intersect_y0 - y_min : intersect_y1 - y_min, 
                            intersect_x0 - x_min : intersect_x1 - x_min
                        ]
                        
                        valid_pixels = roi_slice[mask_slice > 0.5]
                        
                        if len(valid_pixels) > 0:
                            mean_motion = np.mean(valid_pixels, axis=0)
                            grid_flow[gy, gx] = mean_motion
                            grid_mask[gy, gx] = True
                    else:
                        mean_motion = np.mean(roi_slice, axis=(0, 1))
                        grid_flow[gy, gx] = mean_motion
                        grid_mask[gy, gx] = True

        # --- Base Flow Construction ---
        base_flow = grid_flow.copy() 
        has_data_rows = np.any(grid_mask, axis=1)

        for gy in range(self.grid_h):
            if not has_data_rows[gy]:
                continue
            
            row_mask = grid_mask[gy, :]
            indices = np.where(row_mask)[0]
            first_idx, last_idx = indices[0], indices[-1]
            
            base_flow[gy, :first_idx] = base_flow[gy, first_idx]
            base_flow[gy, last_idx+1:] = base_flow[gy, last_idx]

        valid_row_indices = np.where(has_data_rows)[0]
        if len(valid_row_indices) > 0:
            if valid_row_indices[0] > 0:
                top_row = base_flow[valid_row_indices[0]]
                base_flow[:valid_row_indices[0], :] = top_row
            
            if valid_row_indices[-1] < self.grid_h - 1:
                bottom_row = base_flow[valid_row_indices[-1]]
                base_flow[valid_row_indices[-1]+1:, :] = bottom_row
                
            for k in range(len(valid_row_indices) - 1):
                r1 = valid_row_indices[k]
                r2 = valid_row_indices[k+1]
                if r2 - r1 > 1:
                    for r in range(r1 + 1, r2):
                        ratio = (r - r1) / (r2 - r1)
                        base_flow[r] = (1 - ratio) * base_flow[r1] + ratio * base_flow[r2]
        
        # --- Weight Mask Construction ---
        mask_uint8 = grid_mask.astype(np.uint8)
        
        inv_mask = 1 - mask_uint8 
        
        if np.all(inv_mask == 1): 
             dist_map = np.ones_like(inv_mask, dtype=np.float32) * 9999
        elif np.all(inv_mask == 0): 
             dist_map = np.zeros_like(inv_mask, dtype=np.float32)
        else:
            dist_map = cv2.distanceTransform(inv_mask, cv2.DIST_L2, 5) 
        
        decay_radius = 12.0 
        
        weight_mask = np.clip(1.0 - dist_map / decay_radius, 0.0, 1.0)
        
        weight_mask = weight_mask[..., None] 

        # --- Fusion ---
        final_grid_flow = base_flow * weight_mask
            
        final_grid_flow = cv2.GaussianBlur(final_grid_flow, (3, 3), 0)

        full_flow = cv2.resize(final_grid_flow, (self.W, self.H), interpolation=cv2.INTER_LINEAR)
        
        grid_x, grid_y = np.meshgrid(np.arange(self.W), np.arange(self.H))
        
        map_x = (grid_x + full_flow[..., 0]).astype(np.float32)
        map_y = (grid_y + full_flow[..., 1]).astype(np.float32)
        
        if return_flow:
            return map_x, map_y, full_flow
        return map_x, map_y

    def warp_image(self, img: np.ndarray, map_x: np.ndarray, map_y: np.ndarray) -> np.ndarray:
        return cv2.remap(img, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
