# 用来读取参数

import json
from scipy.spatial.transform import Rotation 
import numpy as np
import math

def load_single_K_and_distortion(path_calib: str, camera_name: str):
    '''
    加载单个相机的内参K和畸变参数
    '''
    with open(path_calib, 'r') as f:
        data = json.load(f)
    camera_configs = data["camera_configs"]
    results = camera_configs["results"]
    if camera_name not in results:
        raise ValueError(f"相机名称 {camera_name} 不存在于标定文件中")
    camera_data = results[camera_name]
    model = camera_configs.get("model", "unknown")
    distortion = np.array(camera_data["distortion"], dtype=np.float32)
    K_params = camera_data["K"]
    xi = K_params[0]  # Omni模型特有参数
    fx, fy = K_params[1], K_params[2]
    cx, cy = K_params[3], K_params[4]
    
    K_matrix = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]
    ], dtype=np.float32)
    return K_matrix, distortion, xi, model
    

def load_R_and_T(path_calib: str):
    '''
    这里拿到的大概是wor_to_cam的R和T
    假设q (T_1_0) 是 0->1 的四元数
    从标定文件加载外参：四元数转旋转矩阵(R)和平移向量(T)
    '''
    q_name_list = ["q (T_1_0)","q (T_2_1)"]
    t_name_list = ["t (T_1_0)","t (T_2_1)"]
    # 我想获得q(T_2_0) t(T_2_0) 和 q(T_1_2) t(T_1_2) ,然后有这个来获得R和T
    R_wor_to_cam_list = []  # 三个相机_>世界的旋转矩阵
    T_wor_to_cam_list = []
    R_wor_to_cam_list.append(np.eye(3, dtype=np.float32))
    T_wor_to_cam_list.append(np.zeros(3, dtype=np.float32))
    with open(path_calib, 'r') as f:
        data = json.load(f)
    results = data["camera_configs"]["results"]
    
    for q_name, t_name in zip(q_name_list, t_name_list):
        q = results[q_name] 
        rot = Rotation.from_quat(q)
        R = rot.as_matrix().astype(np.float32)
        t = np.array(results[t_name], dtype=np.float32)
        
        R_prev = R_wor_to_cam_list[-1]
        T_prev = T_wor_to_cam_list[-1]
        # 主要看q_1_0 是1_>0 还是0_>1
        # 0_>1
        # R_world = R@R_prev
        # T_world = R@T_prev + t
        # 1_>0 
        R_world = R_prev @ R  
        T_world = R_prev @ t + T_prev  
        
        R_wor_to_cam_list.append(R_world)
        T_wor_to_cam_list.append(T_world)
    return R_wor_to_cam_list, T_wor_to_cam_list


def Get_inv_RT(R: np.ndarray, T: np.ndarray):
    # 组成4*4矩阵
    M = np.eye(4, dtype=np.float32)
    M[0:3, 0:3] = R
    M[0:3, 3] = T
    M_inv = np.linalg.inv(M)
    R_inv = M_inv[0:3, 0:3]
    T_inv = M_inv[0:3, 3]
    return R_inv, T_inv

def load_R_and_T_2(path_calib: str):
    '''
    return 
    R_cam_to_wor_list
    T_cam_to_wor_list
    假设q (T_1_0) 是 0->1 的四元数
    '''
    q_name_list = ["q (T_1_0)","q (T_2_1)"]
    t_name_list = ["t (T_1_0)","t (T_2_1)"]
    # 我想获得q(T_2_0) t(T_2_0) 和 q(T_1_2) t(T_1_2) ,然后有这个来获得R和T
    R_cam_to_wor_list = []  # 三个相机_>世界的旋转矩阵
    T_cam_to_wor_list = []
    R_cam_to_wor_list.append(np.eye(3, dtype=np.float32))
    T_cam_to_wor_list.append(np.zeros(3, dtype=np.float32))
    with open(path_calib, 'r') as f:
        data = json.load(f)
    results = data["camera_configs"]["results"]
    for q_name, t_name in zip(q_name_list, t_name_list):
        q = results[q_name] # w,x,y,z
        rot = Rotation.from_quat(q)
        R = rot.as_matrix().astype(np.float32)  # 0->1
        t = np.array(results[t_name], dtype=np.float32)
        R_1, T_1 = Get_inv_RT(R, t)  # 1->0
        R_prev = R_cam_to_wor_list[-1]
        T_prev = T_cam_to_wor_list[-1]
        R_world = R_prev @ R_1  
        T_world = R_prev @ T_1 + T_prev  
        R_cam_to_wor_list.append(R_world)
        T_cam_to_wor_list.append(T_world)
        
    
    return R_cam_to_wor_list, T_cam_to_wor_list
    

def load_opt_R_and_T(path_opt_calib: str):
    '''
    从优化后的标定文件加载外参R和T , R: WOR_TO_CAM
    '''
    with open(path_opt_calib, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    R_list = []
    T_list = []
    
    for cam_id in ['2', '4', '6']:
        R_key = f'R{cam_id}'
        T_key = f'T{cam_id}'
        
        R = np.array(data[R_key], dtype=np.float32)
        T = np.array(data[T_key], dtype=np.float32)
        
        R_list.append(R)
        T_list.append(T)
    
    return R_list, T_list
    

def load_all_K_and_distortion(path_calib: str):
    # Try to detect if we should use 135 cams (camera1, 3, 5) or 200 cams (camera2, 4, 6)
    # But for now hardcode or try-catch
    camera_name_list = ["camera2", "camera4", "camera6"]
    # Quick hack: check if these exist
    with open(path_calib, 'r') as f:
        data = json.load(f)
    if "camera1" in data["camera_configs"]["results"]:
         camera_name_list = ["camera1", "camera3", "camera5"]

    K_list = []
    distortion_list = []
    xi_list = []
    model_list = []
    for camera_name in camera_name_list:
        K, distortion, xi, model = load_single_K_and_distortion(path_calib, camera_name)
        K_list.append(K)
        distortion_list.append(distortion)  
        xi_list.append(xi)
        model_list.append(model)
    return camera_name_list, K_list, distortion_list, xi_list, model_list

def create_rotation_matrix_y_axis(angle_rad: float) -> np.ndarray:
    """
    创建绕Y轴旋转的3x3旋转矩阵,将鱼眼的赤道对准图像中心
    """
    rot = np.eye(3, dtype=np.float32)
    rot[0, 0] = math.cos(angle_rad)
    rot[0, 2] = math.sin(angle_rad)
    rot[2, 0] = -math.sin(angle_rad)
    rot[2, 2] = math.cos(angle_rad)
    return rot

def load_ground_params(ground_params_path: str):
    """
    从JSON文件加载地面参数
    Returns:
        n: 地面法向量 [a, b, c] (numpy array)
        d: 地面到原点的距离 (float)
    """
    with open(ground_params_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    n = np.array(data["ground_normal"], dtype=np.float32)
    d = float(data["ground_distance"])
    
    return n, d

def show_all_params(path_calib: str):
    camera_name_list, K_list, distortion_list, xi_list = load_all_K_and_distortion(path_calib)
    R_wor_to_cam_list, T_wor_to_cam_list = load_R_and_T(path_calib)
    R_cam_to_wor_list_2, T_cam_to_wor_list_2 = load_R_and_T_2(path_calib)
    # 计算并打印复合四元数和平移
    for camera_name, K, distortion in zip(camera_name_list, K_list, distortion_list):
        print(f"相机名称: {camera_name}")
        print(f"内参K: {K}")
        print(f"xi: {xi_list[camera_name_list.index(camera_name)]}")
        print(f"畸变参数: {distortion}")
        R = R_wor_to_cam_list[camera_name_list.index(camera_name)]
        print(f"外参R wor_to_cam:\n {R}\n")
        R_2 = R_cam_to_wor_list_2[camera_name_list.index(camera_name)]
        print(f"R cam_to_wor 2: \n{R_2}\n")
        T = T_wor_to_cam_list[camera_name_list.index(camera_name)]
        print(f"平移向量T wor_to_cam: {T}")
        print("---------------")
    return camera_name_list, K_list, distortion_list, xi_list, R_wor_to_cam_list, T_wor_to_cam_list
    
    