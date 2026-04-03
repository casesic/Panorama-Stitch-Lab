import cv2
import numpy as np
from typing import Optional, Tuple


def _ensure_binary_mask(mask: np.ndarray) -> np.ndarray:
    return (mask > 0).astype(np.uint8) * 255


def soft_overlap_from_masks(
    mask_1: np.ndarray,
    mask_2: np.ndarray,
    extend_y: int = 0,
) -> np.ndarray:
    mask_1 = _ensure_binary_mask(mask_1)
    mask_2 = _ensure_binary_mask(mask_2)

    overlap = cv2.bitwise_and(mask_1, mask_2)

    overlap_rows = np.any(overlap > 0, axis=1)  # 每行是否有1
    if not np.any(overlap_rows):
        return np.zeros_like(overlap)  # 没有交集直接返回空

    row_min = np.argmax(overlap_rows)           # 第一个有交集的行
    row_max = len(overlap_rows) - np.argmax(overlap_rows[::-1]) - 1  # 最后一个

    row_min_ext = max(0, row_min - extend_y)
    row_max_ext = min(mask_1.shape[0] - 1, row_max + extend_y)

    result = np.zeros_like(overlap)
    result[row_min_ext:row_max_ext + 1, :] = 1

    return result


def extract_overlap_roi(
    image_1: np.ndarray,
    image_2: np.ndarray,
    overlap_mask: np.ndarray
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[Tuple[int, int, int, int]]]:
    
    y_indices, x_indices = np.where(overlap_mask > 0)
    if len(y_indices) == 0:
        return None, None, None
    
    y_min, y_max = np.min(y_indices), np.max(y_indices)
    x_min, x_max = np.min(x_indices), np.max(x_indices)
    y_max_exclusive = y_max + 1
    x_max_exclusive = x_max + 1

    crop_1 = image_1[y_min:y_max_exclusive, x_min:x_max_exclusive].copy()
    crop_2 = image_2[y_min:y_max_exclusive, x_min:x_max_exclusive].copy()

    roi_bbox = (x_min, y_min, x_max_exclusive, y_max_exclusive)

    return crop_1, crop_2, roi_bbox

def blend_priority(
    image_source: np.ndarray, 
    image_target: np.ndarray, 
    threshold: int = 10, 
    use_target_mask: bool = False, 
    save_cleaned_source_path: Optional[str] = None, 
    erode_kernel_size: int = 0,
    source_valid_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    使用灰度值判断, 避免单通道为0的问题
    :param use_target_mask: 是否利用 image_target 的Mask来过滤 image_source (清除warp产生的artifacts)
    :param save_cleaned_source_path: 如果不为None, 且启用了use_target_mask, 则将清洗后的source保存到此路径
    :param erode_kernel_size: 腐蚀Mask的核大小, 用于去除边缘插值产生的杂色. 默认为3.
    """
    image_to_process = image_source
    if use_target_mask:
        image_to_process = image_source.copy()
        gray_target = cv2.cvtColor(image_target, cv2.COLOR_BGR2GRAY)
        mask_target = (gray_target > 1) # Boolean mask, threshold 1 to catch any non-black
        image_to_process[~mask_target] = 0
        
        if save_cleaned_source_path:
            cv2.imwrite(save_cleaned_source_path, image_to_process)

    if source_valid_mask is not None:
        mask = (source_valid_mask > 0).astype(np.uint8) * 255
    else:
        # 转换为灰度图
        gray = cv2.cvtColor(image_to_process, cv2.COLOR_BGR2GRAY)
        mask = (gray > threshold).astype(np.uint8) * 255
    
    # 增加腐蚀操作, 收缩 Mask, 以去除边缘插值引入的灰黑色伪影
    if erode_kernel_size > 0:
        # 使用几何有效掩码时，仅沿水平方向腐蚀，避免吞掉图像顶部/底部有效区域。
        if source_valid_mask is not None:
            kernel = np.ones((1, erode_kernel_size), np.uint8)
        else:
            kernel = np.ones((erode_kernel_size, erode_kernel_size), np.uint8)
        mask = cv2.erode(mask, kernel, iterations=1)
    
    result = image_to_process.copy()
    # Replace background where mask is 0
    result[mask == 0] = image_target[mask == 0]
    
    return result

# def blend_priority(
#     image_source: np.ndarray, 
#     image_target: np.ndarray, 
#     threshold: int = 10, 
#     use_target_mask: bool = False, 
#     save_cleaned_source_path: Optional[str] = None, 
#     erode_kernel_size: int = 0,
#     feather_radius: int = 40
# ) -> np.ndarray:
#     """
#     使用灰度值判断, 避免单通道为0的问题
#     :param use_target_mask: 是否利用 image_target 的Mask来过滤 image_source (清除warp产生的artifacts)
#     :param save_cleaned_source_path: 如果不为None, 且启用了use_target_mask, 则将清洗后的source保存到此路径
#     :param erode_kernel_size: 腐蚀Mask的核大小, 用于去除边缘插值产生的杂色. 默认为3.
#     :param feather_radius: 羽化半径(像素). >0时在source mask边界做距离变换渐变融合，消除硬接缝. 默认40.
#     """
#     image_to_process = image_source
#     if use_target_mask:
#         image_to_process = image_source.copy()
#         gray_target = cv2.cvtColor(image_target, cv2.COLOR_BGR2GRAY)
#         mask_target = (gray_target > 1) # Boolean mask, threshold 1 to catch any non-black
#         image_to_process[~mask_target] = 0
        
#         if save_cleaned_source_path:
#             cv2.imwrite(save_cleaned_source_path, image_to_process)

#     # 转换为灰度图
#     gray = cv2.cvtColor(image_to_process, cv2.COLOR_BGR2GRAY)
#     mask = (gray > threshold).astype(np.uint8) * 255
    
#     # 凸包边界腐蚀：只腐蚀鱼眼投影的外圆弧边界（凸包外沿），
#     # 不影响内部细节（手臂、手指等）。
#     # 相比普通 erode，它能去除鱼眼 bilinear 插值灰色边缘，同时保留细肢体。
#     if erode_kernel_size > 0:
#         contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#         if contours:
#             all_pts = np.vstack([c.reshape(-1, 2) for c in contours])
#             hull_pts = cv2.convexHull(all_pts)
#             hull_mask = np.zeros_like(mask)
#             cv2.fillPoly(hull_mask, [hull_pts], 255)
#             k = np.ones((erode_kernel_size, erode_kernel_size), np.uint8)
#             eroded_hull = cv2.erode(hull_mask, k, iterations=1)
#             # 仅保留在腐蚀凸包内部的有效像素
#             mask = cv2.bitwise_and(mask, eroded_hull)

#     if feather_radius > 0:
#         # 羽化策略：alpha 基于「当前像素到 target-only 区域的距离」
#         # - source-only 区域（target 无效）：dist 大 → alpha≈1 → 完全用 source
#         # - 重叠区靠近 source-only 侧（如人物头部）：dist 大 → alpha 大 → source 主导，建筑不透出
#         # - 重叠区靠近 target-only 侧：dist 小 → alpha 小 → 平滑过渡到 target
#         gray_target = cv2.cvtColor(image_target, cv2.COLOR_BGR2GRAY)
#         target_valid = (gray_target > threshold).astype(np.uint8) * 255

#         # target-only 区域：target 有效 且 source 无效
#         target_only = cv2.bitwise_and(target_valid, cv2.bitwise_not(mask))  # uint8 0/255

#         # 计算每个像素到最近的 target-only 像素的距离
#         # distanceTransform 输入：255=非 target-only，0=target-only
#         not_target_only = (target_only == 0).astype(np.uint8) * 255
#         dist_to_target_only = cv2.distanceTransform(not_target_only, cv2.DIST_L2, 5)

#         # alpha: dist 大（远离 target-only）→ 接近 1（用 source）；dist 小 → 接近 0（用 target）
#         alpha = np.clip(dist_to_target_only / feather_radius, 0.0, 1.0).astype(np.float32)

#         src_f = image_to_process.astype(np.float32)
#         tgt_f = image_target.astype(np.float32)
#         alpha_3c = alpha[:, :, np.newaxis]
#         blended = (alpha_3c * src_f + (1.0 - alpha_3c) * tgt_f).astype(np.uint8)

#         result = blended.copy()
#         # source 无效处直接用 target（不受 alpha 影响）
#         result[mask == 0] = image_target[mask == 0]
#     else:
#         result = image_to_process.copy()
#         result[mask == 0] = image_target[mask == 0]
    
#     return result


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
        roi_bbox: (x_min, y_min, x_max, y_max) of the overlap ROI in the full image,
            using half-open bounds compatible with image[y_min:y_max, x_min:x_max].
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
    roi_mask = overlap_mask[y_min:y_max, x_min:x_max]
    roi_mask_bool = roi_mask > 0

    # 校验尺寸一致性
    roi_h_actual, roi_w_actual = blended_roi.shape[:2]
    target_slice_h = y_max - y_min
    target_slice_w = x_max - x_min

    if roi_h_actual != target_slice_h or roi_w_actual != target_slice_w:
        # print("Warning: ROI size mismatch in composition!")
        min_h = min(roi_h_actual, target_slice_h)
        min_w = min(roi_w_actual, target_slice_w)
        blended_roi = blended_roi[:min_h, :min_w]
        roi_mask_bool = roi_mask_bool[:min_h, :min_w]
        # 更新 slice 坐标
        y_max = y_min + min_h
        x_max = x_min + min_w

    # 在全图的对应位置
    full_roi_slice = (slice(y_min, y_max), slice(x_min, x_max))

    # 获取当前全图该区域的引用
    roi_on_canvas = final_pano[full_roi_slice]

    # 只更新 mask 内的像素
    roi_on_canvas[roi_mask_bool] = blended_roi[roi_mask_bool]

    return final_pano

