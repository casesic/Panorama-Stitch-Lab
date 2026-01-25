import cv2
import numpy as np

def get_roi_overlap_mask(img1: np.ndarray, img2: np.ndarray, threshold: float = 1e-6) -> np.ndarray:
    """
    计算两张图的共同非黑区域（ROI Mask）。
    假设黑色 (0,0,0) 为无效区域。
    """
    # 转灰度
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # 提取非黑区域 Mask (255=valid, 0=invalid)
    mask1 = (gray1 > threshold).astype(np.uint8) * 255
    mask2 = (gray2 > threshold).astype(np.uint8) * 255

    if mask1.shape != mask2.shape:
        # Resize mask2 to match mask1 if shape mismatch
        h1, w1 = mask1.shape
        mask2 = cv2.resize(mask2, (w1, h1), interpolation=cv2.INTER_NEAREST)

    # 共同重叠区
    overlap_mask = cv2.bitwise_and(mask1, mask2)
    return overlap_mask

def extract_overlap_roi(img1: np.ndarray, img2: np.ndarray, overlap_mask: np.ndarray):
    """
    从两张图中根据 Mask 提取 Bounding Box 的重叠内容。
    修改：增加 x 轴过滤逻辑。
    对于每一个 x 坐标 (Column)，如果该列中 Overlap 的有效像素高度 (y 轴投影) 小于图像高度的一半，则认为该列无效。
    这一步是为了去除上下边缘的噪声或误匹配区域，只保留中间连通性好的重叠带。
    """
    h, w = overlap_mask.shape
    
    # 1. 计算每一列的非零像素数量
    col_counts = np.count_nonzero(overlap_mask, axis=0)
    
    # 2. 阈值：高度的一半
    threshold_h = h * 0.5
    
    # 3. 找到满足条件的列索引
    valid_cols = np.where(col_counts >= threshold_h)[0]
    
    if len(valid_cols) == 0:
        # 如果没有满足条件的列，回退到原来的逻辑（只要有像素就算）
        # 或者直接返回 None
        print("[FlowUtils] Warning: No columns satisfy vertical density threshold. Fallback to bounding rect.")
        y_indices, x_indices = np.where(overlap_mask > 0)
        if len(y_indices) == 0:
            return None, None, None
        valid_cols = np.unique(x_indices)
    
    # 确定 X 范围
    x_min = np.min(valid_cols)
    x_max = np.max(valid_cols)
    
    # 4. 在确定的 X 范围内，确定 Y 的范围
    # 通常取该范围内所有有效像素的 Y 边界
    roi_mask_sub = overlap_mask[:, x_min:x_max+1]
    y_indices_sub, _ = np.where(roi_mask_sub > 0)
    
    if len(y_indices_sub) == 0:
         return None, None, None

    y_min = np.min(y_indices_sub)
    y_max = np.max(y_indices_sub)

    crop1 = img1[y_min:y_max+1, x_min:x_max+1].copy()
    crop2 = img2[y_min:y_max+1, x_min:x_max+1].copy()

    sub_mask = overlap_mask[y_min:y_max+1, x_min:x_max+1]
    
    # 非重叠区域设为黑色
    non_overlap_indices = (sub_mask == 0)
    
    crop1[non_overlap_indices] = 0
    crop2[non_overlap_indices] = 0

    roi_bbox = (x_min, y_min, x_max, y_max)
    
    return crop1, crop2, roi_bbox

def blend_priority(img1, img2):
    """
    优先使用img1，img1为黑色/透明时使用img2
    """
    mask = np.all(img1 > 0, axis=2).astype(np.uint8) * 255
    
    # 应用掩码
    result = img1.copy()
    result[mask == 0] = img2[mask == 0]
    
    return result


def compose_final_panorama(target_pano: np.ndarray, source_pano: np.ndarray, 
                         blended_roi: np.ndarray, roi_bbox: tuple, 
                         overlap_mask: np.ndarray, threshold: int = 1) -> np.ndarray:
    """
    Combines the blended ROI overlap back into the full panorama canvas along with 
    non-overlapping parts of the source image.
    
    Args:
        target_pano: The full target panorama image (base canvas).
        source_pano: The full source panorama image.
        blended_roi: The aligned and blended crop of the overlap region.
        roi_bbox: (x_min, y_min, x_max, y_max) of the overlap ROI in the full image.
        overlap_mask: Boolean or binary mask of the overlap region on the full image.
        threshold: Threshold for considering a pixel valid (non-black).

    Returns:
        final_pano: The fully composed panorama.
    """
    # A. 初始化最终画布，先拷贝 Target 的全部内容
    final_pano = target_pano.copy()

    # B. 将非 Overlap 的 Source 区域贴回去
    # 1. Source 的有效区域 (非黑)
    source_gray = cv2.cvtColor(source_pano, cv2.COLOR_BGR2GRAY)
    source_valid_mask = (source_gray > threshold).astype(np.uint8) * 255

    # 2. Source 的非 Overlap 区域 = Valid Mask - Overlap Mask
    source_non_overlap_mask = cv2.bitwise_and(source_valid_mask, cv2.bitwise_not(overlap_mask))

    # 使用 numpy 索引进行复制
    non_overlap_idx = source_non_overlap_mask > 0
    final_pano[non_overlap_idx] = source_pano[non_overlap_idx]

    # C. 将对齐且融合好的 Overlap Crop 贴回去
    x_min, y_min, x_max, y_max = roi_bbox

    # 提取 crop 对应的局部 mask
    roi_mask = overlap_mask[y_min:y_max+1, x_min:x_max+1]
    roi_mask_bool = roi_mask > 0

    # 校验尺寸一致性
    roi_h_actual, roi_w_actual = blended_roi.shape[:2]
    target_slice_h = (y_max + 1) - y_min
    target_slice_w = (x_max + 1) - x_min

    if roi_h_actual != target_slice_h or roi_w_actual != target_slice_w:
        # print("Warning: ROI size mismatch in composition!")
        min_h = min(roi_h_actual, target_slice_h)
        min_w = min(roi_w_actual, target_slice_w)
        blended_roi = blended_roi[:min_h, :min_w]
        roi_mask_bool = roi_mask_bool[:min_h, :min_w]
        # 更新 slice 坐标
        y_max = y_min + min_h - 1
        x_max = x_min + min_w - 1

    # 在全图的对应位置
    full_roi_slice = (slice(y_min, y_max+1), slice(x_min, x_max+1))

    # 获取当前全图该区域的引用
    roi_on_canvas = final_pano[full_roi_slice]

    # 只更新 mask 内的像素
    roi_on_canvas[roi_mask_bool] = blended_roi[roi_mask_bool]

    return final_pano

