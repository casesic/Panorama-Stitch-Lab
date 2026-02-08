from lib_pano.calibration.optimize_R import load_all_matches, prepare_3d_matches, reproj_error_sphere
import numpy as np
from lib_pano.io.params_read import get_inv_RT, load_opt_R_and_T
from lib_pano.calibration.optimize_R import get_camera_std_euler_by_R, get_R_by_euler, rotate_sphere_pts, run_optimization
from typing import Dict, Tuple
import json

# 我需要知道d的高度

def init_optimization_params() -> Tuple[np.ndarray, list]:
    
    # 地面参数：使用球坐标 (theta, phi, d) 代替 (a, b, d)
    # 初始法向量 [0, 1, 0] 对应 theta=π/2, phi=π/2
    theta_init = np.pi / 2  # 极角：从z轴正方向的夹角
    phi_init = np.pi / 2    # 方位角：在xy平面上从x轴正方向的夹角
    d_init = 1.2
    ground_params = np.array([theta_init, phi_init, d_init])
    x0 = ground_params
    
    theta_bound = np.deg2rad(10)  # 允许法向量倾斜10度
    phi_bound = np.deg2rad(10)    # 允许方位角变化10度
    d_bound = 1
    
    bounds = [
        (theta_init - theta_bound, theta_init + theta_bound),  # theta: 极角
        (phi_init - phi_bound, phi_init + phi_bound),          # phi: 方位角
        (d_init - d_bound, d_init + d_bound),  # d: 平面到原点的距离
    ]
    
    return x0, bounds

def save_ground_params_to_json(n: np.ndarray, d: float, output_path: str) -> None:
    """
    保存地面参数到JSON文件
    """
    ground_params = {
        'ground_normal': n.tolist(),
        'ground_distance': float(d)
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(ground_params, f, indent=4, ensure_ascii=False)
    
    print(f'\n地面参数已保存到: {output_path}')

def calculate_ground_error_1(ground_params: np.ndarray,
                          R_euler: np.ndarray, 
                          R_base_euler: np.ndarray,
                          T: np.ndarray,
                          T_base: np.ndarray,
                          ground_pts_3d_src: np.ndarray, 
                          ground_pts_3d_dst: np.ndarray) -> float:
   
    ''''
    对任意一对地面匹配点:
    ax+by+cz = d
    1. 基准相机视线向量 \vec{m}_w → 地面交点 P_w = s_w · \vec{m}_w
       其中 s_w = (d - \vec{n}·t_w) / (\vec{n}·\vec{m}_w)
       (t_w = 0,基准相机光心为原点）→ s_w = d / (\vec{n}·\vec{m}_w)
    
    2. 待优化相机视线向量 \vec{m}_i → 转换到基准坐标系 \vec{m}_i^w = R·\vec{m}_i
       → 地面交点 P_i^w = s_i · \vec{m}_i^w + t
       其中 s_i = (d - \vec{n}·t) / (\vec{n}·\vec{m}_i^w)
    
    3. 残差 e_g = ||P_w - P_i^w||₂² （d是变量，参与残差的求导和优化）
    '''
    theta, phi, d = ground_params
    # 球坐标转笛卡尔坐标，保证单位法向量
    a = np.sin(theta) * np.cos(phi)
    b = np.sin(theta) * np.sin(phi)
    c = np.cos(theta)
    n = np.array([a, b, c]).reshape(3, 1)  # 地面法向量（自动归一化）

    R = get_R_by_euler(R_euler[0], R_euler[1], R_euler[2])
    R_cam_to_wor, T_cam_to_wor = get_inv_RT(R, T)
    R_base = get_R_by_euler(R_base_euler[0], R_base_euler[1], R_base_euler[2])
    R_base_cam_to_wor, T_base_cam_to_wor = get_inv_RT(R_base, T_base)
    pts_src_to_wor = rotate_sphere_pts(ground_pts_3d_src, R_cam_to_wor) 
    # print(f"pts_src_to_wor: {pts_src_to_wor[0:10]}")
    dot_n_pts_src = np.dot(pts_src_to_wor, n).reshape(-1)  # (N,)
    dot_n_T = np.dot(n.T, T_cam_to_wor)[0]  # 标量
    s_src = (d - dot_n_T) / np.maximum(np.abs(dot_n_pts_src), 1e-6)  # (N,)
    # print(f"s_src: {s_src[0:10]}")
    P_src_world = s_src[:, None] * pts_src_to_wor + T_cam_to_wor[None, :]  # (N,3)
    
    pts_dst_world = rotate_sphere_pts(ground_pts_3d_dst, R_base_cam_to_wor) 
    # print(f"pts_dst_world: {pts_dst_world[0:10]}")
    dot_n_pts_dst = np.dot(pts_dst_world, n).reshape(-1)  # (N,)
    dot_n_T_base = np.dot(n.T, T_base_cam_to_wor)[0]  # 标量
    s_dst = (d - dot_n_T_base) / np.maximum(np.abs(dot_n_pts_dst), 1e-6)  # (N,)
    # print(f"s_dst: {s_dst[0:10]}")
    P_dst_world = s_dst[:, None] * pts_dst_world + T_base_cam_to_wor[None, :]  # (N,3)
    
    ground_error = np.mean(np.sum((P_src_world - P_dst_world) ** 2, axis=1))
    return ground_error


def calculate_ground_error(ground_params: np.ndarray,
                             R_euler: np.ndarray, 
                             R_base_euler: np.ndarray,
                             T: np.ndarray,
                             T_base: np.ndarray,
                             ground_pts_3d_src: np.ndarray, 
                             ground_pts_3d_base: np.ndarray) -> float:
    """
    使用单应性矩阵计算地面误差：H = R_21 + (t_21 @ n^T) / d
    
    基于球面单应性变换：
    1. 计算相对位姿：
       R_21 = R_src @ R_base^T  (从base坐标系到src坐标系的旋转)
       t_21 = T_src - R_21 @ T_base  (相对平移)
    
    2. 构建单应性矩阵：
       H = R_21 + (t_21 @ n^T) / d
       其中 n 是地面法向量(世界坐标系)，d 是平面到原点距离
    
    3. 变换球面方向向量：
       m_src' = H @ m_base (变换后归一化到单位球面)
    
    4. 计算球面误差：
       一个新想法，用角度计算error会更合理一些
       error = ||m_src' - m_base||²  (单位球面上的欧氏距离)
    
    """
    theta, phi, d = ground_params
    # 球坐标转笛卡尔坐标
    a = np.sin(theta) * np.cos(phi)
    b = np.sin(theta) * np.sin(phi)
    c = np.cos(theta)
    n = np.array([a, b, c]).reshape(3, 1)  # 地面法向量 (3, 1)
    # 获取旋转矩阵 (world to camera)
    R_src = get_R_by_euler(R_euler[0], R_euler[1], R_euler[2])
    R_base = get_R_by_euler(R_base_euler[0], R_base_euler[1], R_base_euler[2])
    
    n_base = R_base @ n 
    d_base = d + n.T@ R_base.T @ T_base.reshape(3, 1)  # 调整d到基准相机坐标系
    # 计算相对位姿 R_21 = R_src @ R_base^T
    R_21 = R_src @ R_base.T
    
    # 计算相对平移 t_21 = T_src - R_21 @ T_base
    T_base_col = T_base.reshape(3, 1)
    T_src_col = T.reshape(3, 1)
    t_21 = T_src_col - R_21 @ T_base_col  # (3, 1)
    
    # 构建单应性矩阵 H = R_21 + (t_21 @ n^T) / d
    H = R_21 + (t_21 @ n_base.T) / d_base  # (3, 3)
    
    m_src_pred = (H @ ground_pts_3d_base.T).T  # (N, 3)
    
    # 归一化到球面
    norms = np.linalg.norm(m_src_pred, axis=1, keepdims=True)
    m_src_pred_normalized = m_src_pred / np.maximum(norms, 1e-8)
    
    # 计算角度误差（球面距离）
    cos_dist = np.sum(m_src_pred_normalized * ground_pts_3d_src, axis=1)
    cos_dist = np.clip(cos_dist, -1.0, 1.0)
    angle_errors = np.arccos(cos_dist)  # 弧度
    error = np.mean(angle_errors ** 2)  # 使用角度误差的平方
    
    return error


def build_cost_function(ground_pts_3d_dict: Dict[str, np.ndarray], R_list: list, T_list: list):
    def total_cost(x: np.ndarray) -> float:
       
        ground_params = x[0:3]  # theta, phi, d
        
        T2 = T_list[0]
        T4 = T_list[1]
        T6 = T_list[2]
        R2_euler = get_camera_std_euler_by_R(R_list[0])
        R4_euler = get_camera_std_euler_by_R(R_list[1])
        R6_euler = get_camera_std_euler_by_R(R_list[2])
        # 地面误差
        err4_2_ground = calculate_ground_error(
            ground_params,
            R4_euler, 
            R2_euler,
            T4,
            T2,
            ground_pts_3d_dict["pts_3d_4_in_24"],
            ground_pts_3d_dict["pts_3d_2_in_24"]
        )
        err6_2_ground = calculate_ground_error(
            ground_params,
            R6_euler, 
            R2_euler,
            T6,
            T2,
            ground_pts_3d_dict["pts_3d_6_in_26"],
            ground_pts_3d_dict["pts_3d_2_in_26"]
        )
        
        err6_4_ground = calculate_ground_error(
            ground_params,
            R6_euler, 
            R4_euler,
            T6,
            T4,
            ground_pts_3d_dict["pts_3d_6_in_46"],
            ground_pts_3d_dict["pts_3d_4_in_46"]
        )
        ground_loss = err4_2_ground + err6_2_ground + err6_4_ground
        return ground_loss
    
    return total_cost

def parser_optimize_results(result, R_list: list, T_list: list, ground_pts_3d_dict: Dict[str, np.ndarray]):
    
    
    # 解析优化后的参数
    ground_params_opt = result.x[0:3]  # theta, phi, d
    
    # 计算初始误差
    ground_params_init = np.array([np.pi/2, np.pi/2, 1.2])  # 初始地面参数 (theta, phi, d)
    x_init = ground_params_init
    cost_func = build_cost_function(ground_pts_3d_dict, R_list, T_list)
    initial_error = cost_func(x_init)
    
    # 将球坐标转换为笛卡尔坐标的法向量
    theta_opt, phi_opt, d_opt = ground_params_opt
    a_opt = np.sin(theta_opt) * np.cos(phi_opt)
    b_opt = np.sin(theta_opt) * np.sin(phi_opt)
    c_opt = np.cos(theta_opt)
    n_opt = np.array([a_opt, b_opt, c_opt])
    
    # 初始法向量
    theta_init, phi_init, d_init = ground_params_init
    a_init = np.sin(theta_init) * np.cos(phi_init)
    b_init = np.sin(theta_init) * np.sin(phi_init)
    c_init = np.cos(theta_init)
    n_init = np.array([a_init, b_init, c_init])
    
    # 输出结果
    print("\n===== 地面参数优化结果 =====")
    
    print(f"\n初始地面法向量: [{n_init[0]:.6f}, {n_init[1]:.6f}, {n_init[2]:.6f}] (模长: {np.linalg.norm(n_init):.6f}), d初始值: {d_init:.6f}")
    print(f"优化地面法向量: [{n_opt[0]:.6f}, {n_opt[1]:.6f}, {n_opt[2]:.6f}] (模长: {np.linalg.norm(n_opt):.6f}), d优化值: {d_opt:.6f}")
    
    # 计算法向量与垂直方向的夹角
    vertical = np.array([0, 1, 0])
    tilt_angle = np.rad2deg(np.arccos(np.clip(np.dot(n_opt, vertical), -1, 1)))
    print(f"地面倾斜角（与水平面的夹角）: {90 - tilt_angle:.2f}°")
    
    print(f"\n初始总误差: {initial_error:.6f}")
    print(f"优化后总误差: {result.fun:.6f}")
    print(f"误差减少: {initial_error - result.fun:.6f} ({(1 - result.fun/initial_error)*100:.2f}%)")
    print(f"优化是否收敛: {result.success}")
    print(f"优化消息: {result.message}")
    
    return n_opt, d_opt


def optimize_ground(ground_npz_paths: str, calib_opt_path: str, save_path: str = None):

    ground_match_data = load_all_matches(ground_npz_paths)
    R_list, T_list = load_opt_R_and_T(calib_opt_path)    # opt world to camera
    H = 960
    W = 1920
    ground_pts_3d_dict = prepare_3d_matches(ground_match_data, H, W)
    
    # 计算T_list每个元素的模长
    for i, T in enumerate(T_list):
        norm_T = np.linalg.norm(T)
        print(f"相机{i}的平移向量模长: {norm_T:.6f}")
    
    x0, bounds = init_optimization_params()
    cost_func = build_cost_function(ground_pts_3d_dict, R_list, T_list)
    # loss = cost_func(x0)  # 计算初始误差
    # print(f"初始总误差: {loss:.6f}")
    result = run_optimization(x0, bounds, cost_func)
    
    n_opt, d_opt = parser_optimize_results(result, R_list, T_list, ground_pts_3d_dict)
    
    # 保存地面参数
    if save_path:
        save_ground_params_to_json(n_opt, d_opt, save_path)
    
    return n_opt, d_opt