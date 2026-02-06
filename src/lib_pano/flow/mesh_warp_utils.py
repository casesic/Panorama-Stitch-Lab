import cv2
import numpy as np
import torch
import torch.nn.functional as F

class MeshGridWarper:
    def __init__(self, full_h, full_w, grid_size=32):
        """
        Args:
            full_h, full_w: 原始全景图的高度和宽度
            grid_size: 网格大小 (pixels)，越小越精细但越容易抖动，建议 32 或 64
        """
        self.H = full_h
        self.W = full_w
        self.grid_size = grid_size
        
        # 计算网格的维度 (Grid Dimensions)
        # 我们向上取整，确保覆盖全图
        self.grid_h = int(np.ceil(full_h / grid_size))
        self.grid_w = int(np.ceil(full_w / grid_size))

    def compute_mesh_flow_from_roi(self, flow_roi, roi_bbox, validity_mask=None, return_flow=False):
        """
        根据局部 ROI 的光流，推断出全局的网格形变场。
        采用 Masked Decay 策略：重叠区精确对齐，非重叠区平滑衰减至静止。
        
        Args:
            flow_roi: (h_roi, w_roi, 2) numpy array, RAFT/GMFlow output.
            roi_bbox: (x_min, y_min, x_max, y_max) tuple, ROI 在全图中的位置.
            validity_mask: (h_roi, w_roi) numpy array, optional. 
                           光流可靠性掩码 (0 or 1.0)。如果提供，会过滤掉不可靠的像素（如遮挡区）。
            return_flow: bool, 是否返回原始的光流场 (H, W, 2)
            
        Returns:
            map_x, map_y: 全图尺寸的 remap 映射表,可以直接传给 cv2.remap
            (optional) full_flow: 全图的光流场 (H, W, 2)
        """
        x_min, y_min, x_max, y_max = roi_bbox
        
        # 1. 初始化全图的低分辨率 Grid Flow (h_grid, w_grid, 2)
        grid_flow = np.zeros((self.grid_h, self.grid_w, 2), dtype=np.float32)
        
        # 记录哪些 Grid 是被 ROI 覆盖并“激活”的（有真实数据且有效的）
        grid_mask = np.zeros((self.grid_h, self.grid_w), dtype=np.bool_)

        # 2. 将 ROI 内的像素级光流“聚合”到网格控制点上
        # 计算 ROI 在 Grid 坐标系下的范围
        g_x_min = x_min // self.grid_size
        g_y_min = y_min // self.grid_size
        g_x_max = min(x_max // self.grid_size + 1, self.grid_w)
        g_y_max = min(y_max // self.grid_size + 1, self.grid_h)
        
        for gy in range(g_y_min, g_y_max):
            for gx in range(g_x_min, g_x_max):
                # 转换回像素坐标范围
                px_start = gx * self.grid_size
                py_start = gy * self.grid_size
                px_end = min((gx + 1) * self.grid_size, self.W)
                py_end = min((gy + 1) * self.grid_size, self.H)
                
                # 计算该 Grid 与 ROI 的交集
                intersect_x0 = max(px_start, x_min)
                intersect_y0 = max(py_start, y_min)
                intersect_x1 = min(px_end, x_max)
                intersect_y1 = min(py_end, y_max)
                
                if intersect_x1 > intersect_x0 and intersect_y1 > intersect_y0:
                    # 提取对应的 roi flow 片段
                    roi_slice = flow_roi[
                        intersect_y0 - y_min : intersect_y1 - y_min, 
                        intersect_x0 - x_min : intersect_x1 - x_min
                    ]
                    
                    if validity_mask is not None:
                        # 提取对应的 validity mask 片段
                        mask_slice = validity_mask[
                            intersect_y0 - y_min : intersect_y1 - y_min, 
                            intersect_x0 - x_min : intersect_x1 - x_min
                        ]
                        
                        # 仅使用有效的像素计算均值
                        # 注意：valid_pixels 可能是空的（如果全是遮挡）
                        valid_pixels = roi_slice[mask_slice > 0.5]
                        
                        if len(valid_pixels) > 0:
                            mean_motion = np.mean(valid_pixels, axis=0)
                            grid_flow[gy, gx] = mean_motion
                            grid_mask[gy, gx] = True
                        else:
                            # 如果该Grid下所有像素都不可靠，该怎么办？
                            # 策略：不标记为grid_mask=True，让它被后续的平滑/外推填补
                            # 或者强制为0
                            pass
                    else:
                        # 旧逻辑：无脑取均值
                        mean_motion = np.mean(roi_slice, axis=(0, 1))
                        grid_flow[gy, gx] = mean_motion
                        grid_mask[gy, gx] = True

        # 3. 外推/传播 (Extrapolation / Propagation) - Masked Decay Strategy
        # 需求是：非Overlap区域尽量不动，但不能撕裂。
        # 本策略分三步：
        # A. 生成一个填满全图的 "Base Flow"，假设整张图都跟随 ROI 移动（解决连续性）。
        # B. 生成一个 "Weight Mask"，在 ROI 处为 1，向远处平滑衰减到 0（解决非Overlap不动）。
        # C. Final Flow = Base Flow * Weight Mask。

        # --- A. 构建 Base Flow (全图填充) ---
        base_flow = grid_flow.copy() # (H, W, 2)
        has_data_rows = np.any(grid_mask, axis=1) # (H,)

        # A.1 水平填充：对每一行有数据的，把左右空缺填满边缘值
        for gy in range(self.grid_h):
            if not has_data_rows[gy]:
                continue
            
            row_mask = grid_mask[gy, :]
            indices = np.where(row_mask)[0]
            first_idx, last_idx = indices[0], indices[-1]
            
            # 左侧填入 first_val
            base_flow[gy, :first_idx] = base_flow[gy, first_idx]
            # 右侧填入 last_val
            base_flow[gy, last_idx+1:] = base_flow[gy, last_idx]

        # A.2 垂直填充：对于完全没数据的行，寻找最近的有数据行复制
        valid_row_indices = np.where(has_data_rows)[0]
        if len(valid_row_indices) > 0:
            # 上方空白行
            if valid_row_indices[0] > 0:
                top_row = base_flow[valid_row_indices[0]]
                base_flow[:valid_row_indices[0], :] = top_row
            
            # 下方空白行
            if valid_row_indices[-1] < self.grid_h - 1:
                bottom_row = base_flow[valid_row_indices[-1]]
                base_flow[valid_row_indices[-1]+1:, :] = bottom_row
                
            # 中间空白行
            for k in range(len(valid_row_indices) - 1):
                r1 = valid_row_indices[k]
                r2 = valid_row_indices[k+1]
                if r2 - r1 > 1:
                    # 线性插值
                    for r in range(r1 + 1, r2):
                        ratio = (r - r1) / (r2 - r1)
                        base_flow[r] = (1 - ratio) * base_flow[r1] + ratio * base_flow[r2]
        
        # --- B. 构建 Weight Mask (距离衰减) ---
        # 我们需要计算网格中每个点到 ROI (grid_mask=True) 的距离
        mask_uint8 = grid_mask.astype(np.uint8)
        
        # distanceTransform 计算的是到 "0" 的距离
        # 所以我们翻转 mask: ROI=0, BG=1. 这样 distanceTransform 返回的就是“到 ROI 的距离”
        inv_mask = 1 - mask_uint8 
        
        # 如果全是一样（比如全0或全1），需要特殊处理防止报错
        if np.all(inv_mask == 1): # 没有 ROI
             dist_map = np.ones_like(inv_mask, dtype=np.float32) * 9999
        elif np.all(inv_mask == 0): # 全是 ROI
             dist_map = np.zeros_like(inv_mask, dtype=np.float32)
        else:
            dist_map = cv2.distanceTransform(inv_mask, cv2.DIST_L2, 5) # 计算最近0像素距离
        
        # 设定衰减半径 (Grid count). 8 grids * 32 px = 256 px
        decay_radius = 12.0 
        
        # Weight = max(0, 1 - d / R)
        # ROI 内部 d=0 -> Weight=1
        # 近处 d<R -> Weight 1->0
        # 远处 d>R -> Weight=0
        weight_mask = np.clip(1.0 - dist_map / decay_radius, 0.0, 1.0)
        
        # 扩展维度以便广播
        weight_mask = weight_mask[..., None] # (H, W, 1)

        # --- C. 融合 ---
        final_grid_flow = base_flow * weight_mask
            
        # 4. 平滑处理
        # 对 grid_flow 进行高斯模糊，消除行与行之间的突变
        final_grid_flow = cv2.GaussianBlur(final_grid_flow, (3, 3), 0)

        # 5. 上采样回像素级 (Upsample)
        full_flow = cv2.resize(final_grid_flow, (self.W, self.H), interpolation=cv2.INTER_LINEAR)
        
        # 6. 生成 Map
        grid_x, grid_y = np.meshgrid(np.arange(self.W), np.arange(self.H))
        
        map_x = (grid_x + full_flow[..., 0]).astype(np.float32)
        map_y = (grid_y + full_flow[..., 1]).astype(np.float32)
        
        if return_flow:
            return map_x, map_y, full_flow
        return map_x, map_y

    def warp_image(self, img, map_x, map_y):
        return cv2.remap(img, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
