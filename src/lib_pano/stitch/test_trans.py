'''
lib_pano.stitch.test_trans 的 Docstring
记录世界坐标系下的点(X_W,Y_W,Z_W)在投射到鱼眼图像的每一步坐标转换关系，便于我理解和debug
1. (X,Y,Z)直接投射
2. (X/d, Y/d, Z/d) 通过外参投射看看差别
'''
import numpy as np
from lib_pano.io.params_read import load_R_and_T_2, load_all_K_and_distortion, load_R_and_T


    
def generate_3d_points_in_world():
    # 这是按列的,一列是一个点的坐标
    points_world = np.array([
        [0, 1, 89, 9], # x
        [0, 1, -37, -2], # y
        [-0.5, 1000, 24, 4]  # z
    ])
    return points_world

def world_to_camera(points_world, R, T):
    points_camera = R @ points_world +T
    return points_camera

def project_to_unit_sphere(points_world):
    norms = np.linalg.norm(points_world, axis=0)  # axis=0 按列算范数
    points_unit_sphere = points_world / norms
    return points_unit_sphere

def project_to_normalized_image(points_unit_sphere, xi):
    X, Y, Z = points_unit_sphere
    s = Z + xi
    mx = X / s
    my = Y / s
    points_normalized = np.vstack((mx, my))
    return points_normalized
    
def camera_to_normalized_image(points_camera, xi):
    d = np.linalg.norm(points_camera, axis=0) # axis=0 按列算范数
    x, y, z = points_camera
    s = 1/(xi * d + z)
    mx = s * x
    my = s * y
    mz = s * z    # (mx,my,mz)组成一个什么样的形状？
    points_normalized = np.vstack((mx, my))
    return points_normalized

def apply_distortion(points_normalized, k1, k2, p1, p2):
    mx, my = points_normalized
    r2 = mx**2 + my**2
    dr = 1+ k1 * r2 + k2 * r2**2
    dx = 2*p1*mx*my + p2*(r2 + 2*mx**2) 
    dy = p1 * (r2 + 2*my**2) + 2*p2*mx*my
    mx_distorted = mx * dr + dx
    my_distorted = my * dr + dy
    points_distorted = np.vstack((mx_distorted, my_distorted))
    return points_distorted

def apply_intrinsics(points_distorted, fx, fy, cx, cy):
    mx_distorted, my_distorted = points_distorted
    u = fx * mx_distorted + cx
    v = fy * my_distorted + cy
    points_image = np.vstack((u, v))
    return points_image 

def load_data(calib_path: str):
    R_wor_to_cam_list, T_wor_to_cam_list = load_R_and_T(calib_path)
    _, K_list, distortion_list, xi_list = load_all_K_and_distortion(calib_path)
    return R_wor_to_cam_list, T_wor_to_cam_list, K_list, distortion_list, xi_list

def unapply_intrinsics(points_image, fx, fy, cx, cy):
    u, v = points_image
    mx_distorted = (u - cx) / fx
    my_distorted = (v - cy) / fy
    points_distorted = np.vstack((mx_distorted, my_distorted))
    return points_distorted

def unapply_distortion(points_distorted, k1, k2, p1, p2):
    mx_distorted, my_distorted = points_distorted
    mx = mx_distorted.copy()
    my = my_distorted.copy()
    for _ in range(5):  # 迭代5次
        r2 = mx**2 + my**2
        dr = 1 + k1 * r2 + k2 * r2**2
        dx = 2*p1*mx*my + p2*(r2 + 2*mx**2)
        dy = p1 * (r2 + 2*my**2) + 2*p2*mx*my
        mx = (mx_distorted - dx) / dr
        my = (my_distorted - dy) / dr
    points_normalized = np.vstack((mx, my))
    return points_normalized


def sphere_to_equirectangular(points_unit_sphere, H: int, W: int):
    X, Y, Z = points_unit_sphere 
    phi = np.arcsin(Y)  # [-pi/2, pi/2]
    theta = np.arctan2(Z, X)  # [-pi, pi]
    u = (theta + np.pi) / (2.0 * np.pi)  # [0,1]
    v = (phi + (np.pi / 2.0)) / np.pi  # [0,1]
    u_pixels = u * W
    v_pixels = v * H
    return u_pixels, v_pixels
    
    
    
def test_trans(calib_path :str):
    R_wor_to_cam_list, T_wor_to_cam_list, K_list, distortion_list, xi_list = load_data(calib_path)
    points_world = generate_3d_points_in_world()
    
    points_unit_sphere = project_to_unit_sphere(points_world)
    # 1.常规从世界坐标系映射
    # 2.从世界坐标系单位球面映射
    for i in range(3):  # 三个相机
        print(f"--- 相机 {i} ---")
        R_wor_to_cam = R_wor_to_cam_list[i]
        T_wor_to_cam = T_wor_to_cam_list[i].reshape(3,1)  # 转成列向量
        R = R_wor_to_cam
        T = T_wor_to_cam
        K = K_list[i]
        distortion = distortion_list[i]
        xi = xi_list[i]
        fx, fy = K[0,0], K[1,1]
        cx, cy = K[0,2], K[1,2]
        k1, k2, p1, p2 = distortion
        
        points_camera = world_to_camera(points_world, R, T)
        points_unit_camera = project_to_unit_sphere(points_camera)
        u_pixels, v_pixels = sphere_to_equirectangular(points_unit_camera, 960, 1920)
        # 实际上对比的是0，1，2的uw_pixels, vw_pixels,事实上存在误差 KEYPOINT 也许它给的R我还是理解错了
        print(f"直接通过世界坐标经过wor_to_cam R和T投影到经纬图坐标 u,v:\n{u_pixels}\n{v_pixels}")
        
        T0 = np.array([[0],[0],[0]]).reshape(3,1)
        points_unitW_camera = world_to_camera(points_unit_sphere, R, T0)
        uw_pixels, vw_pixels = sphere_to_equirectangular(points_unitW_camera, 960, 1920)
        print(f"通过世界单位单位球坐标经过wor_to_cam R投影到经纬图坐标 u,v:\n{uw_pixels}\n{vw_pixels}")
        
        points_unitc_sphere = world_to_camera(points_unit_camera, R.T, T0)  # 这里实际是cam_to_wor
        uw_pixels, vw_pixels = sphere_to_equirectangular(points_unitc_sphere, 960, 1920 )
        print(f"从相机坐标系投影回世界单位球面去展开成经纬图坐标 u,v:\n{uw_pixels}\n{vw_pixels}")
        
        points_normalized = camera_to_normalized_image(points_camera, xi)
        points_unit_normalized = project_to_normalized_image(points_unitW_camera, xi)
        
        points_distorted = apply_distortion(points_normalized, k1, k2, p1, p2)
        points_unit_distorted = apply_distortion(points_unit_normalized, k1, k2, p1, p2)
        
        points_image = apply_intrinsics(points_distorted, fx, fy, cx, cy)
        points_unit_image = apply_intrinsics(points_unit_distorted, fx, fy, cx, cy)
        # print(f"Points world:\n{points_world}")
        # print(f"Points image from world coords:\n{points_image}")
        # print(f"Points image from world unit sphere:\n{points_unit_image}")
    
    
    
    
    

