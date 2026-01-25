# 用光流法优化近景对齐
import os
import numpy as np
import cv2 as cv
from lib_pano import local_setting as ls

def feather_mask(mask: np.ndarray, dilate_kernel_size: tuple = (31, 31), 
                 blur_kernel_size: tuple = (51, 51), blur_sigma: float = 15.0) -> np.ndarray:
    """对 mask 进行膨胀和羽化处理，避免硬边界。
    """
    mask_expanded = mask.copy()
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, dilate_kernel_size)
    mask_expanded = cv.dilate(mask_expanded, kernel, iterations=1)
    
    # 高斯模糊羽化边界（关键：让光流平滑过渡）
    mask_feathered = cv.GaussianBlur(mask_expanded, blur_kernel_size, blur_sigma)
    mask_feathered = mask_feathered.astype(np.float32) / 255.0
    mask_feathered = np.expand_dims(mask_feathered, axis=-1)  # H*W*1
    
    return mask_feathered

def visualize_optical_flow(flow: np.ndarray, img_shape: tuple, show_window: bool = True) -> np.ndarray:
    """将光流可视化为 HSV 彩色图。
    """
    window_name = "Optical Flow Visualization"
    hsv = np.zeros(img_shape, dtype=np.uint8)
    mag, ang = cv.cartToPolar(flow[..., 0], flow[..., 1])
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 1] = 255
    hsv[..., 2] = cv.normalize(mag, None, 0, 255, cv.NORM_MINMAX)
    flow_bgr = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)
    
    if show_window:
        cv.imshow(window_name, flow_bgr)
        cv.waitKey(0)
    
    return flow_bgr

def refine_near_region_with_optical_flow(img_ref_path: str,  
                                         img_src_paths: list,
                                         near_mask_paths: list,
                                         use_optical_flow: bool = False,
                                         ):
    """对多个源图像分别应用光流对齐（可选），然后与参考图像融合成全景。
    
    参数:
        use_optical_flow: 是否使用光流对齐（False=直接距离变换融合，True=光流+融合）
    """
    if len(img_src_paths) != len(near_mask_paths):
        raise ValueError("源图像和mask数量不匹配")
    
    # 读取参考图像
    img_ref = cv.imread(img_ref_path)
    
    if use_optical_flow:
        # ===== 方案A: 使用光流对齐 =====
        print("使用光流对齐...")
        gray_ref = cv.cvtColor(img_ref, cv.COLOR_BGR2GRAY)
        h, w = img_ref.shape[:2]
        grid_y, grid_x = np.mgrid[0:h, 0:w].astype(np.float32)
        
        refined_images = []
        near_close_masks = []
        
        for i, (src_path, mask_path) in enumerate(zip(img_src_paths, near_mask_paths)):
            print(f"处理 src_{i+1} 与 ref 的光流对齐...")
            
            img_src = cv.imread(src_path)
            near_close_mask = get_near_close_overlap_mask(mask_path)
            near_close_masks.append(near_close_mask)
            
            gray_src = cv.cvtColor(img_src, cv.COLOR_BGR2GRAY)
            flow = cv.calcOpticalFlowFarneback(   
                prev=gray_ref, next=gray_src, flow=None,
                pyr_scale=0.5, levels=5, winsize=21,
                iterations=5, poly_n=7, poly_sigma=1.5, flags=0
            )
            
            mask_feathered = feather_mask(near_close_mask, 
                                          dilate_kernel_size=(31, 31),
                                          blur_kernel_size=(51, 51),
                                          blur_sigma=15.0)
            
            flow_refined = flow * mask_feathered
            map_x = grid_x + flow_refined[:, :, 0]
            map_y = grid_y + flow_refined[:, :, 1]
            
            img_src_refined = cv.remap(img_src, map_x, map_y, 
                                        interpolation=cv.INTER_LINEAR, 
                                        borderMode=cv.BORDER_REPLICATE)
            refined_images.append(img_src_refined)
    else:
        # ===== 方案B: 不用光流，直接融合（推荐） =====
        print("不使用光流，直接距离变换融合...")
        refined_images = []
        near_close_masks = []
        for src_path in img_src_paths:
            img_src = cv.imread(src_path)
            refined_images.append(img_src)
    
    # 融合所有图像
    print(f"融合 1 + {len(refined_images)} 张图像...")
    stitched = stitch_multiple_images(img_ref, refined_images)
    
    return stitched, refined_images
    
    
def get_near_close_overlap_mask(near_mask_path:str):
    near_mask = cv.imread(near_mask_path, cv.IMREAD_GRAYSCALE)
    # 上面变黑，下面保留
    h, w = near_mask.shape
    overlap_mask = np.zeros_like(near_mask)
    overlap_mask[h//2:, :] = near_mask[h//2:, :]
    
    return overlap_mask

def stitch_multiple_images(img_ref: np.ndarray, img_src_list: list) -> np.ndarray:
    """将参考图与多个源图像融合成全景。

    """
    out_dir = os.path.join(ls.DATA_OUTPUT_DIR, 'preview')
    os.makedirs(out_dir, exist_ok=True)
    
    # 从参考图像开始
    result = img_ref
    
    # 逐个融合源图像
    for i, img_src in enumerate(img_src_list):
        print(f"步骤{i+1}: 融合 src_{i+1}...")
        
        result = stitch_two_images(result, img_src)
       
    
    # 保存最终结果
    output_path = os.path.join(out_dir, 'refined_panorama.png')
    cv.imwrite(output_path, result)
    print(f"全景图已保存: {output_path}")
    
    return result

def stitch_two_images(img1: np.ndarray, img2: np.ndarray, 
                     ) -> np.ndarray:
    """融合两张图像（羽化融合）。
    
    """
    mask1 = (np.any(img1 != 0, axis=2)).astype(np.uint8) * 255
    mask2 = (np.any(img2 != 0, axis=2)).astype(np.uint8) * 255
    mask1_bool = (mask1 > 0)
    mask2_bool = (mask2 > 0)
    overlap = mask1_bool & mask2_bool
    
    
    # 距离变换计算权重
    dr = cv.distanceTransform(mask1, cv.DIST_L2, 5).astype(np.float32)
    ds = cv.distanceTransform(mask2, cv.DIST_L2, 5).astype(np.float32)
    
    eps = 1e-6
    weight_2 = np.zeros_like(dr, dtype=np.float32)
    
    # 非重叠区域
    weight_2[mask2_bool & ~mask1_bool] = 1.0
    weight_2[mask1_bool & ~mask2_bool] = 0.0
    
    # 重叠区域按距离混合
    denom = ds + dr + eps
    weight_2[overlap] = ds[overlap] / denom[overlap]
    
    # 融合
    weight_3 = np.stack([weight_2, weight_2, weight_2], axis=2)
    stitched = (img1.astype(np.float32) * (1.0 - weight_3) + 
                img2.astype(np.float32) * weight_3)
    stitched = np.clip(stitched, 0, 255).astype(np.uint8)
    
    return stitched

