from  typing import List, Tuple, Dict, Any
import numpy as np
from lib_pano.io.params_read import load_R_and_T
from lib_pano.feature.feature_filter import load_matches
from lib_pano.stitch.pano_stitch import equirect_pixels_to_sphere
import cv2 as cv
from scipy.spatial.transform import Rotation 
from lib_pano.camera.camera import get_R_by_euler, get_camera_std_euler_by_R
from scipy.optimize import minimize
import json
import os
import lib_pano.local_setting as ls

def load_all_matches(npz_paths: List[str]) -> Dict[Tuple[int, int], Tuple[np.ndarray, np.ndarray]]:
    match_data: Dict[Tuple[int, int], Tuple[np.ndarray, np.ndarray]] = {}
    for npz_path in npz_paths:
        kpts0, kpts1, matches, conf, valid_mask = load_matches(npz_path)
        idx = np.where(valid_mask)[0]
        for i in idx:
            cam0_idx = int(npz_path.split('_cam')[1].split('_')[0])
            cam1_idx = int(npz_path.split('_cam')[2].split('_')[0])
            if (cam0_idx, cam1_idx) not in match_data:
                match_data[(cam0_idx, cam1_idx)] = ([], [])
            match_data[(cam0_idx, cam1_idx)][0].append(kpts0[i])
            match_data[(cam0_idx, cam1_idx)][1].append(kpts1[matches[i].astype(int)])
    for key in match_data:
        match_data[key] = (np.array(match_data[key][0], dtype=np.float32), np.array(match_data[key][1], dtype=np.float32))
    return match_data
        

def get_matches_in_y_region(
    npz_paths: List[str],
    y_min: int,
    y_max: int
) -> Dict:
    """
    获取指定Y轴区域的特征点匹配情况
    
    """
    match_keys = [[2,4], [2,6],[4,6]]
    # 加载所有匹配数据
    match_data = load_all_matches(npz_paths)
    match_data_y_range: Dict[Tuple[int, int], Tuple[np.ndarray, np.ndarray]] = {}
    for match_key in match_keys:
        kpts0, kpts1 =  match_data[tuple(match_key)]
        y_coords = kpts0[:, 1]  # Y坐标是第二列
        mask = (y_coords >= y_min) & (y_coords <= y_max)
        kpts0_filtered = kpts0[mask]
        kpts1_filtered = kpts1[mask]
        match_data_y_range[tuple(match_key)] = (kpts0_filtered, kpts1_filtered)
    return match_data_y_range
    

# 先利用检测到的特征点估计仿射矩阵,来对图像进行warp优化
def optimize_2d_image_warp(image_paths: List[str],  npz_paths: List[str], background_image_path: str):
    '''
    利用特征点匹配对图像进行warp优化
    可以更改pts_dst和background_image_path来选择参考图像
    '''
    background_image = cv.imread(background_image_path)
    for image_path, npz_path in zip(image_paths, npz_paths):
        kpts0, kpts1, matches, conf, valid_mask = load_matches(npz_path)
        idx = np.where(valid_mask)[0]
        pts_dst = kpts0[idx]  # dst是cam2
        pts_src = kpts1[matches[idx].astype(int)]
        # 计算仿射变换矩阵
        H, maskH = cv.findHomography(pts_src, pts_dst, cv.RANSAC)
        
        image = cv.imread(image_path)
        warped = cv.warpPerspective(image, H, (background_image.shape[1], background_image.shape[0]))
       
        mask = cv.inRange(warped, (1, 1, 1), (255, 255, 255))  # 筛选非纯黑区域
        blur_size = 21  # 模糊核大小，越大边缘越柔和（奇数）
        mask = cv.GaussianBlur(mask.astype(np.float32)/255, (blur_size, blur_size), 0)
        mask_3ch = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
        background_image = (background_image * (1 - mask_3ch) + warped * mask_3ch).astype(np.uint8)
        
        cv.imwrite(image_path.replace('.png', '_blend.png'), background_image)

def normalize_vectors(vectors: np.ndarray, eps: float = 1e-12):
    '''
    归一化3d点: N*3
    '''
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)  # 计算每个向量的范数
    norms_safe = np.where(norms < eps, eps, norms)
    return vectors / norms_safe

def pts_2d_to_3d_sphere(pts_2d: np.ndarray, H: int, W: int) -> np.ndarray:
    '''
    2d: N,2 -> 3d: N,3
    '''
    u = pts_2d[:, 0]
    v = pts_2d[:, 1]
    xyz = equirect_pixels_to_sphere(u, v, H, W)  # N,3
    return xyz
    
def rotate_sphere_pts(pts_3d: np.ndarray, R: np.ndarray) :
    
    return (R @ pts_3d.T).T  # 矩阵乘法：3×3 @ 3×N → 3×N → 转置为N×3

def reproj_error_sphere(R_euler: np.ndarray,
                        pts_3d_src: np.ndarray,
                        pts_3d_dst: np.ndarray,
                        R_base_euler: np.ndarray):
    
    R = get_R_by_euler(R_euler[0], R_euler[1], R_euler[2])  # wor_to_cam
    R_src_to_dst = R.T    # cam_to_wor
    pts_3d_src_wor = rotate_sphere_pts(pts_3d_src, R_src_to_dst)
    
    R_base = get_R_by_euler(R_base_euler[0], R_base_euler[1], R_base_euler[2])  # wor_to_cam
    pts_3d_src_in_dst = rotate_sphere_pts(pts_3d_src_wor, R_base)  # N,3
    
    cos_dist = np.sum(pts_3d_src_in_dst * pts_3d_dst, axis=1)
    cos_dist = np.clip(cos_dist, -1.0, 1.0)  # 避免数值溢出
    angle_err = np.arccos(cos_dist)  # 弧度误差
    return np.mean(angle_err)
    
    
def prepare_3d_matches(match_data: Dict[Tuple[int, int], Tuple[np.ndarray, np.ndarray]],
                       H: int, W: int) -> Dict[str, np.ndarray]:
    """
    预处理匹配数据：将2D匹配点转为3D球面点
    :param match_data: 原始2D匹配数据
    :param H/W: 经纬图尺寸
    :return: 3D球面点字典，key为匹配对标识
    """
    pts_2_in_24, pts_4_in_24 = match_data[(2,4)]
    pts_2_in_26, pts_6_in_26 = match_data[(2,6)]
    pts_4_in_46, pts_6_in_46 = match_data[(4,6)]
    
    return {
        "pts_3d_2_in_24": pts_2d_to_3d_sphere(pts_2_in_24, H, W),
        "pts_3d_4_in_24": pts_2d_to_3d_sphere(pts_4_in_24, H, W),
        "pts_3d_2_in_26": pts_2d_to_3d_sphere(pts_2_in_26, H, W),
        "pts_3d_6_in_26": pts_2d_to_3d_sphere(pts_6_in_26, H, W),
        "pts_3d_4_in_46": pts_2d_to_3d_sphere(pts_4_in_46, H, W),
        "pts_3d_6_in_46": pts_2d_to_3d_sphere(pts_6_in_46, H, W),
    }

def init_optimization_params(R_list: List[np.ndarray]) -> Tuple[np.ndarray, List[tuple]]:
    """
    初始化优化参数：欧拉角初始值 + 优化边界
    :param R_list: 初始旋转矩阵列表 [R2, R4, R6]
    :return: 优化初始值x0 + 欧拉角边界bounds
    """
    R2, R4_init, R6_init = R_list[0], R_list[1], R_list[2]
    
    R4_euler_init = get_camera_std_euler_by_R(R4_init)
    R6_euler_init = get_camera_std_euler_by_R(R6_init)
    x0 = np.concatenate([R4_euler_init, R6_euler_init])
    
    angle_bound = np.deg2rad(5)
    bounds = [
        (R4_euler_init[0]-angle_bound, R4_euler_init[0]+angle_bound),  # cam4 roll
        (R4_euler_init[1]-angle_bound, R4_euler_init[1]+angle_bound),  # cam4 pitch
        (R4_euler_init[2]-angle_bound, R4_euler_init[2]+angle_bound),  # cam4 yaw
        (R6_euler_init[0]-angle_bound, R6_euler_init[0]+angle_bound),  # cam6 roll
        (R6_euler_init[1]-angle_bound, R6_euler_init[1]+angle_bound),  # cam6 pitch
        (R6_euler_init[2]-angle_bound, R6_euler_init[2]+angle_bound),  # cam6 yaw
    ]
    return x0, bounds

def build_cost_function(pts_3d_dict: Dict[str, np.ndarray], R2_euler: np.ndarray):
    """
    构建全局优化的代价函数（闭包形式，复用3D点和基准矩阵）
    :param pts_3d_dict: 3D球面点字典
    :param R2: cam2的基准旋转矩阵
    :return: 代价函数
    """
    def total_cost(x: np.ndarray) -> float:
       
        R4_euler = x[:3]
        R6_euler = x[3:]
        
        err4_2 = reproj_error_sphere(
            R4_euler, 
            pts_3d_dict["pts_3d_4_in_24"], 
            pts_3d_dict["pts_3d_2_in_24"], 
            R2_euler
        )
        
        err6_2 = reproj_error_sphere(
            R6_euler, 
            pts_3d_dict["pts_3d_6_in_26"], 
            pts_3d_dict["pts_3d_2_in_26"], 
            R2_euler
        )
        
        err4_6 = reproj_error_sphere(
            R4_euler, 
            pts_3d_dict["pts_3d_4_in_46"], 
            pts_3d_dict["pts_3d_6_in_46"], 
            R6_euler
        )
        
        return err4_2 * 1.0 + err6_2 * 1.0 + err4_6 * 1.0
    
    return total_cost

def run_optimization(x0: np.ndarray, bounds: List[tuple], cost_func) :
    """
    执行优化
    :param x0: 初始值
    :param bounds: 优化边界
    :param cost_func: 代价函数
    :return: 优化结果对象
    """
    res = minimize(
        cost_func,
        x0,
        method='L-BFGS-B',
        bounds=bounds,
        options={
            'maxiter': 2000,
            'gtol': 1e-6,
            'disp': True
        }
    )
    return res

def save_optimized_params_to_json(R_list: List[np.ndarray], T_list: List[np.ndarray], 
                                    output_path: str):

    new_data = {
        'name': '三个相机（优化后）',
        'sequence': '2&4&6',
        'model': 'Omni-radtan',
            'R2': R_list[0].tolist(),
            'R4': R_list[1].tolist(),
            'R6': R_list[2].tolist(),
            'T2': T_list[0].tolist(),
            'T4': T_list[1].tolist(),
            'T6': T_list[2].tolist(),
    }
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(new_data, f, indent=4, ensure_ascii=False)
    
    print(f'\n优化后的标定参数已保存到: {output_path}')

def parse_optimization_result(res: Any, R_list: List[np.ndarray], pts_3d_dict: Dict[str, np.ndarray], R2_euler: np.ndarray) :
    """
    解析优化结果：恢复欧拉角和旋转矩阵
    :param res: 优化器返回结果
    :param R_list: 初始旋转矩阵列表 [R2, R4, R6]
    :return: 解析后的优化结果字典
    """
    R2 = R_list[0]
    
    R4_euler_init = get_camera_std_euler_by_R(R_list[1])
    R6_euler_init = get_camera_std_euler_by_R(R_list[2])
    
    R4_euler_opt = res.x[:3]
    R6_euler_opt = res.x[3:]
    
    R4_opt = get_R_by_euler(R4_euler_opt[0], R4_euler_opt[1], R4_euler_opt[2])
    R6_opt = get_R_by_euler(R6_euler_opt[0], R6_euler_opt[1], R6_euler_opt[2])
    
    cost_func = build_cost_function(pts_3d_dict, R2_euler)  
    initial_error = cost_func(np.concatenate([R4_euler_init, R6_euler_init]))
    
    print("===== 旋转矩阵优化结果 =====")
    print(f"cam4 初始欧拉角（弧度）: {np.round(R4_euler_init, 6)}")
    print(f"cam4 优化后欧拉角（弧度）: {np.round(R4_euler_opt, 6)}")
    print(f"cam6 初始欧拉角（弧度）: {np.round(R6_euler_init, 6)}")
    print(f"cam6 优化后欧拉角（弧度）: {np.round(R6_euler_opt, 6)}")
    print(f"初始总误差: {initial_error:.6f} 弧度")
    print(f"优化后总误差: {res.fun:.6f} 弧度")
    print(f"优化是否收敛: {res.success}")
    
    return {
        'R2': R2,
        'R4_opt': R4_opt,
        'R6_opt': R6_opt,
        'R4_euler_opt': R4_euler_opt,
        'R6_euler_opt': R6_euler_opt,
        'initial_error': initial_error,
        'optimized_error': res.fun,
        'converged': res.success
    }

# 尝试用得到的特征点来优化R
def global_optimize_R(calib_path: str, npz_paths: List[str]):
    match_data = load_all_matches(npz_paths)
    R_list, T_list = load_R_and_T(calib_path)  # wor_to_cam
    R2_eluer = get_camera_std_euler_by_R(R_list[0])
    H = 960
    W = 1920
    pts_3d_dict = prepare_3d_matches(match_data, H, W)
    x0, bounds = init_optimization_params(R_list)
    
    cost_func = build_cost_function(pts_3d_dict, R2_eluer)
    res = run_optimization(x0, bounds, cost_func)
    
    optimize_result = parse_optimization_result(res, R_list, pts_3d_dict, R2_eluer)
    
    # 构建优化后的R和T列表
    R_optimized = [optimize_result['R2'], optimize_result['R4_opt'], optimize_result['R6_opt']]

    save_optimized_params_to_json(R_optimized, T_list, ls.CALIB_OPT_FILE)
    
    return optimize_result
        
def local_optimize_R(calib_path: str, npz_paths: List[str], y_scope: Tuple[int, int]):
    
    match_data = get_matches_in_y_region(npz_paths, y_scope[0], y_scope[1])
    R_list, T_list = load_R_and_T(calib_path)  # wor_to_cam
    R2_eluer = get_camera_std_euler_by_R(R_list[0])
    H = 960
    W = 1920
    pts_3d_dict = prepare_3d_matches(match_data, H, W)
    x0, bounds = init_optimization_params(R_list)
    cost_func = build_cost_function(pts_3d_dict, R2_eluer)
    res = run_optimization(x0, bounds, cost_func)
    optimize_result = parse_optimization_result(res, R_list, pts_3d_dict, R2_eluer)
    return optimize_result

def multi_local_optimize_R(calib_path: str, npz_paths: List[str], y_scopes: List[Tuple[int, int]]):
    # [500, 680], [680, 800], [800, 960]
    optimize_results = []
    for y_scope in y_scopes:
        print(f"===== 优化Y轴范围: {y_scope} =====")
        optimize_result = local_optimize_R(calib_path, npz_paths, y_scope)
        optimize_results.append((y_scope, optimize_result))
    return optimize_results
