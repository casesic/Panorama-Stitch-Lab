import cv2 as cv
import numpy as np
from typing import List, Tuple
import math
import os
from lib_pano.io.params_read import load_R_and_T, load_opt_R_and_T, load_ground_params
from lib_pano.camera.camera import get_R_by_euler 

def normalize_vectors(vectors: np.ndarray, eps: float = 1e-12):
    '''
    将 H*W*3 的向量组归一化为单位向量。
    '''
    norms = np.linalg.norm(vectors, axis=-1, keepdims=True)  # 计算每个向量的范数
    norms_safe = np.where(norms < eps, eps, norms)  
    return vectors / norms_safe

def sphere_to_equirect_coords(xyz: np.ndarray, img_width: int, img_height: int):
    
    assert xyz.shape[-1] == 3 and xyz.ndim == 3         # xyz的shape为(H,W,3)
    
    x, y, z = xyz[:, :, 0], xyz[:, :, 1], xyz[:, :, 2]
    # 计算纬度 [-π/2, π/2]
    lat = np.arcsin(y)
    # 计算经度 [-π, π]
    lon = np.arctan2(x, z)
    
    u = (lon + np.pi) * img_width / (2 * np.pi)
    v = (lat + np.pi / 2) * img_height / np.pi 
    
    return u, v
    

def generate_equirect_pixels(H: int, W: int):
    """
    返回像素坐标网格 uu, vv, shape 为 (H, W)。
    eg: H = 4, W = 6 
    u = [0,1,2,3,4,5], v = [0,1,2,3]
    uu = [[0,1,2,3,4,5],   
          [0,1,2,3,4,5],
          [0,1,2,3,4,5],
          [0,1,2,3,4,5]]
    vv = [[0,0,0,0,0,0],
         [1,1,1,1,1,1],
         [2,2,2,2,2,2],
         [3,3,3,3,3,3]]
    """
    
    u = np.linspace(0, W - 1, W)
    v = np.linspace(0, H - 1, H)
    uu, vv = np.meshgrid(u, v)
    return uu, vv

def equirect_pixels_to_sphere(uu: np.ndarray, vv: np.ndarray, H: int, W: int, flatten: bool = True):
    """将 equirect 像素坐标 (uu,vv) 映射为球面上的 3D 向量。
    返回 shape 为 (H,W,3)（flatten=False）或 (H*W,3)（flatten=True）。
    """
    lat = (vv / float(H) - 0.5) * math.pi
    lon = (uu / float(W) - 0.5) * 2.0 * math.pi
    
    # lon = lon + 2*math.pi / 3 
    # lon = np.where(lon > np.pi, lon - 2.0 * np.pi, lon)
    
    cos_lat = np.cos(lat)
    X = cos_lat * np.sin(lon)
    Y = np.sin(lat)
    Z = cos_lat * np.cos(lon)

    dirs = np.stack([X, Y, Z], axis=-1)  # H,W,3
    if flatten:
        return dirs.reshape((-1, 3))
    return dirs

def blend_near_region(pano_f: np.ndarray, sampled_stack: np.ndarray, 
                     near_start: int, out_height: int, out_width: int) -> np.ndarray:
    """
    对近景区域进行距离变换混合处理。
    """
    num_cam = sampled_stack.shape[0]
    print(f"处理近景区域 (从行 {near_start} 到 {out_height})...")
    
    # 为每个相机生成有效mask（非黑色像素）
    valid_masks = []
    for cam_idx in range(num_cam):
        # 判断是否为非黑色像素
        mask_cam = np.any(sampled_stack[cam_idx] > 1e-6, axis=2).astype(np.uint8) * 255
        valid_masks.append(mask_cam)
    
    # 为每个相机计算距离变换
    distance_maps = np.zeros((num_cam, out_height, out_width), dtype=np.float32)
    for cam_idx in range(num_cam):
        dist = cv.distanceTransform(valid_masks[cam_idx], cv.DIST_L2, 5)
        distance_maps[cam_idx] = dist
    
    # 遍历近景所有像素
    for row in range(near_start, out_height):
        for col in range(out_width):
            # 获取有效相机（该像素非黑色）
            valid_cams_bool = np.array([valid_masks[i][row, col] > 0 for i in range(num_cam)])
            num_valid = valid_cams_bool.sum()
            
            if num_valid == 0:
                pano_f[row, col] = 0
                continue
            elif num_valid == 1:
                # 只有一个相机，直接用该相机
                cam_idx = np.where(valid_cams_bool)[0][0]
                pano_f[row, col] = sampled_stack[cam_idx, row, col, :]
                continue
            
            # 多个相机：距离变换混合
            distances = distance_maps[valid_cams_bool, row, col]
            total_dist = distances.sum()
            
            if total_dist < 1e-6:
                # 距离太小，简单平均
                colors = sampled_stack[valid_cams_bool, row, col, :]
                blended_color = colors.mean(axis=0)
            else:
                # 按距离加权混合（类似 refine_near_region）
                weights = distances / total_dist
                colors = sampled_stack[valid_cams_bool, row, col, :]
                blended_color = (colors * weights[:, None]).sum(axis=0)
            
            pano_f[row, col] = blended_color
    
    return pano_f

def get_y_regions(out_height: int) -> Tuple[int, int, int, int]:
    y_regions = [
        {
            'y_range': (0, 600),  # 经纬图像素行：0-519
        },
        {
            'y_range': (600, out_height),  # 经纬图像素行：600-959
        }
    ]
    return y_regions

def process_camera_region(R: np.ndarray, image: np.ndarray, 
                          xyz_world_region_flat: np.ndarray,
                          region_h: int, out_width: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    处理单个相机在某个区域的投影和采样。
    
    Returns:
        sampled_f: 采样后的图像 (region_h, W, 3)
        weight: 权重图 (region_h, W)
    """
    img_height, img_width = image.shape[:2]
    
    xyz_camera_flat = R @ xyz_world_region_flat  # (3, region_h*W)
    xyz_camera = xyz_camera_flat.T.reshape((region_h, out_width, 3))  # (region_h, W, 3)
    xyz_camera = normalize_vectors(xyz_camera)
    
    z_cam = xyz_camera[:, :, 2]
    front_mask = z_cam > 0
    # 转换为该相机的经纬图坐标
    u, v = sphere_to_equirect_coords(xyz_camera, img_width, img_height)
    
    valid_mask = (u >= 0) & (u < img_width) & (v >= 0) & (v < img_height)
    valid_mask = valid_mask & front_mask
    map_x = u.astype(np.float32)
    map_y = v.astype(np.float32)
    map_x[~valid_mask] = -1
    map_y[~valid_mask] = -1
    warped_img = cv.remap(image, map_x, map_y, interpolation=cv.INTER_LINEAR, 
                        borderMode=cv.BORDER_CONSTANT, borderValue=0)
    
    sampled_f = warped_img.astype(np.float32) / 255.0
    w = np.maximum(0.0, z_cam)  # 简单用 z 值作为权重
    w *= valid_mask.astype(np.float32)
    
    return sampled_f, w

def prepare_region_rotation_matrices(calib_opt_path: str, region: dict, ground_params_path: str) -> List[np.ndarray]:
    """
    为特定区域准备优化后的旋转矩阵或单应性矩阵。
    """
    R_list_region, T_list_region = load_opt_R_and_T(calib_opt_path)
    
    # 获取区域的y范围
    y_range = region.get('y_range', (0, 960))
    y_min, y_max = y_range
    
    # 判断是否为地面区域（这里假设y >= 600为地面）
    is_ground_region = (y_min >= 600)
    
    if not is_ground_region:
        print(f"  远景区域 y_range={y_range}: 使用纯旋转矩阵")
        return R_list_region
    
    print(f"  地面区域 y_range={y_range}: 使用平面单应性矩阵")
    
    n, d = load_ground_params(ground_params_path)

    n = n.reshape(3, 1)  # 变为列向量 (3, 1)
    
    H_list = []
    for i, (R, T) in enumerate(zip(R_list_region, T_list_region)):
        
        T_col = T.reshape(3, 1)
        H = R + (T_col @ n.T) / d
        H_list.append(H)
        
        camera_id = [2, 4, 6][i]
        print(f"  相机{camera_id}的单应性矩阵已计算")
    
    return R_list_region

def process_all_regions(image_list: List[np.ndarray], xyz_world: np.ndarray,
                       calib_opt_path: str, ground_params_path: str, y_regions: List[dict],
                       out_height: int, out_width: int):
    """
    按Y轴区域分段处理所有相机。
    
    Returns:
        sampled_stack: (num_cam, H, W, 3)
        weight_stack: (num_cam, H, W)
    """
    num_cam = len(image_list)
    sampled_stack = np.zeros((num_cam, out_height, out_width, 3), dtype=np.float64)
    weight_stack = np.zeros((num_cam, out_height, out_width), dtype=np.float64)
    
    # 按Y轴区域分段处理
    for region in y_regions:
        y_start, y_end = region['y_range']
        print(f"处理Y轴区域 [{y_start}, {y_end})...")
        
        R_list_region = prepare_region_rotation_matrices(calib_opt_path, region, ground_params_path)
        
        # 获取该区域的世界坐标（通过像素行索引xyz_world数组）
        # xyz_world[y, x]存储的是该像素对应的球面单位向量（值范围-1到1）
        xyz_world_region = xyz_world[y_start:y_end, :, :]  # (region_h, W, 3)
        region_h = y_end - y_start
        xyz_world_region_flat = xyz_world_region.reshape(-1, 3).T  # (3, region_h*W)
        
        # 对每个相机处理该区域
        for cam_idx, (R, image) in enumerate(zip(R_list_region, image_list)):
            sampled_f, w = process_camera_region(R, image, xyz_world_region_flat, 
                                                region_h, out_width)
            # 存储到对应区域
            sampled_stack[cam_idx, y_start:y_end, :, :] = sampled_f
            weight_stack[cam_idx, y_start:y_end, :] = w
    
    return sampled_stack, weight_stack

def save_debug_images(sampled_stack: np.ndarray, output_path: str):
    """
    保存每个相机的warped结果用于调试。
    """
    num_cam = sampled_stack.shape[0]
    for cam_idx in range(num_cam):
        warped_debug = (sampled_stack[cam_idx] * 255.0).astype(np.uint8)
        output_step_path = os.path.splitext(output_path)[0] + f'_step_{cam_idx}.png'
        cv.imwrite(output_step_path, warped_debug)

def blend_panorama(sampled_stack: np.ndarray, weight_stack: np.ndarray,
                  out_height: int, out_width: int,
                  enable_near_blend: bool = False) -> np.ndarray:
    """
    融合所有相机的采样结果生成全景图。
    
    Args:
        enable_near_blend: 是否启用近景距离混合
    """
    max_weight = weight_stack.max(axis=0)
    valid = max_weight > 1e-8
    
    # 初始化输出（远景用argmax）
    best_idx = np.argmax(weight_stack, axis=0)  # H,W
    arr = sampled_stack.transpose(1, 2, 0, 3)  # H,W,num,3
    rows = np.arange(out_height)[:, None]
    cols = np.arange(out_width)[None, :]
    pano_f = arr[rows, cols, best_idx]  # H,W,3
    
    # 近景区域：使用距离变换混合（可选）
    if enable_near_blend:
        near_ratio = 0.50  # 近景区域占比
        near_start = int(out_height * (1 - near_ratio))
        pano_f = blend_near_region(pano_f, sampled_stack, near_start, out_height, out_width)
    
    pano_out = np.zeros((out_height, out_width, 3), dtype=np.uint8)
    pano_out[valid] = (pano_f[valid] * 255.0).astype(np.uint8)
    return pano_out

def stitch_images(image_list: List[np.ndarray], calib_opt_path: str, ground_params_path: str,
                 output_path: str, output_size: Tuple[int, int]) -> np.ndarray:
    """
    主拼接函数：协调整个全景图拼接流程。
    """
    out_width, out_height = output_size
   
    # 1. 生成经纬图坐标对应的球面向量
    uu, vv = generate_equirect_pixels(out_height, out_width)
    xyz_world = equirect_pixels_to_sphere(uu, vv, out_height, out_width, flatten=False)
    
    
    # 3. 获取分区域的优化旋转矩阵配置
    y_regions = get_y_regions(out_height)
    
    # 4. 处理所有区域的所有相机
    sampled_stack, weight_stack = process_all_regions(
        image_list, xyz_world, calib_opt_path, ground_params_path, y_regions, out_height, out_width)
    
    # 5. 保存调试图像
    save_debug_images(sampled_stack, output_path)
    
    # 6. 融合生成最终全景图
    pano_out = blend_panorama(sampled_stack, weight_stack, out_height, out_width, 
                             enable_near_blend=False)
    
    return pano_out

def stitch_pano(equirect_images_path: List[str],
                calib_opt_path: str,
                ground_params_path: str,
                output_path: str,
                output_size: Tuple[int, int], 
                ) :
    print(f"load {len(equirect_images_path)} equirect images for stitching...")
    image_list = []
    for idx, img_path in enumerate(equirect_images_path):
        img = cv.imread(img_path)
        image_list.append(img)
    panorama = stitch_images(image_list, calib_opt_path, ground_params_path, output_path, output_size)
    cv.imwrite(output_path, panorama)
    print(f"panorama saved to {output_path}")