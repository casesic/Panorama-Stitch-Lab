import cv2 as cv
import numpy as np
from typing import List
from lib_pano.calibration.optimize_R import load_all_matches, pts_2d_to_3d_sphere, rotate_sphere_pts
from lib_pano.io.params_read import load_opt_R_and_T, get_inv_RT, load_ground_params
from lib_pano.stitch.pano_stitch import sphere_to_equirect_coords

CAMERA_COLORS = {2: (0, 0, 255), 4: (255, 0, 0), 6: (0, 255, 0)}  # 红蓝绿
CAMERA_INDICES = {2: 0, 4: 1, 6: 2}

def _transform_to_world(pts_3d, R, T, is_ground, use_homography, ground_params_path):
    """统一的坐标变换函数"""
    if not use_homography or not is_ground.any():
        R_inv, _ = get_inv_RT(R, T)
        return rotate_sphere_pts(pts_3d, R_inv)
    
    pts_world = np.zeros_like(pts_3d)
    if (~is_ground).any():
        R_inv, _ = get_inv_RT(R, T)
        pts_world[~is_ground] = rotate_sphere_pts(pts_3d[~is_ground], R_inv)
    GROUND_N, GROUND_D = load_ground_params(ground_params_path) 
    n = GROUND_N.reshape(3, 1)  # 确保是列向量
    H_inv = np.linalg.inv(R + (T.reshape(3, 1) @ n.T) / GROUND_D)
    # H_inv = np.linalg.inv(R)  # debug
    pts_world[is_ground] = rotate_sphere_pts(pts_3d[is_ground], H_inv)
    pts_world[is_ground] /= np.linalg.norm(pts_world[is_ground], axis=1, keepdims=True)
    return pts_world

def _project_and_draw(vis_img, pts_world, color, is_ground, out_width, out_height, draw_circle=False):
    """投影并绘制点"""
    u, v = sphere_to_equirect_coords(pts_world.reshape(1, -1, 3), out_width, out_height)
    u, v = u.flatten().astype(int), v.flatten().astype(int)
    
    for i, (x, y) in enumerate(zip(u, v)):
        if 0 <= x < out_width and 0 <= y < out_height:
            cv.circle(vis_img, (x, y), 3, color, -1)
            if draw_circle and is_ground[i]:
                cv.circle(vis_img, (x, y), 5, color, 1)

def visualize_feature_points_on_panorama(panorama_path: str, npz_paths: List[str],
                                         calib_opt_path: str, output_path: str,
                                         use_ground_homography: bool = True,
                                         ground_y_threshold: int = 600, ground_params_path: str = None):
    """在全景图上可视化各相机特征点"""
    pano = cv.imread(panorama_path)
    out_height, out_width = pano.shape[:2]
    vis_img = pano.copy()
    
    match_data = load_all_matches(npz_paths)
    R_list, T_list = load_opt_R_and_T(calib_opt_path)
    
    print("开始可视化特征点...")
    for (cam_src, cam_dst), (kpts_src, kpts_dst) in match_data.items():
        print(f"  匹配对: cam{cam_src} -> cam{cam_dst}")
        
        is_ground = kpts_src[:, 1] >= ground_y_threshold
        
        for cam_id, kpts in [(cam_src, kpts_src), (cam_dst, kpts_dst)]:
            idx = CAMERA_INDICES[cam_id]
            pts_3d = pts_2d_to_3d_sphere(kpts, out_height, out_width)
            pts_world = _transform_to_world(pts_3d, R_list[idx], T_list[idx], 
                                           is_ground, use_ground_homography, ground_params_path)
            _project_and_draw(vis_img, pts_world, CAMERA_COLORS[cam_id], 
                            is_ground, out_width, out_height, use_ground_homography)
    
    # 添加图例
    legend = np.ones((80, out_width, 3), dtype=np.uint8) * 255
    for i, (cam_id, color) in enumerate(CAMERA_COLORS.items()):
        x = 50 + i * 150
        cv.circle(legend, (x, 30), 5, color, -1)
        cv.putText(legend, f"Camera {cam_id}", (x + 15, 35),
                  cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    
    final_img = np.vstack([legend, vis_img])
    cv.imwrite(output_path, final_img)
    print(f"\n已保存到: {output_path}")
    return final_img

def visualize_feature_matching_errors(panorama_path: str, npz_paths: List[str],
                                      calib_opt_path: str, output_path: str,
                                      use_ground_homography: bool = True,
                                      ground_y_threshold: int = 600, ground_params_path: str = None):
    """用连线可视化匹配误差（绿色=小，红色=大）"""
    pano = cv.imread(panorama_path)
    out_height, out_width = pano.shape[:2]
    vis_img = pano.copy()
    
    match_data = load_all_matches(npz_paths)
    R_list, T_list = load_opt_R_and_T(calib_opt_path)
    all_errors = []
    
    print("计算匹配误差...")
    for (cam_src, cam_dst), (kpts_src, kpts_dst) in match_data.items():
        print(f"  匹配对: cam{cam_src} -> cam{cam_dst}")
        
        pts_src_3d = pts_2d_to_3d_sphere(kpts_src, out_height, out_width)
        pts_dst_3d = pts_2d_to_3d_sphere(kpts_dst, out_height, out_width)
        is_ground = kpts_src[:, 1] >= ground_y_threshold
        
        pts_src_world = _transform_to_world(pts_src_3d, R_list[CAMERA_INDICES[cam_src]], 
                                           T_list[CAMERA_INDICES[cam_src]], is_ground, use_ground_homography, ground_params_path)
        pts_dst_world = _transform_to_world(pts_dst_3d, R_list[CAMERA_INDICES[cam_dst]], 
                                           T_list[CAMERA_INDICES[cam_dst]], is_ground, use_ground_homography, ground_params_path)
        
        # 计算角度误差
        angle_errors = np.arccos(np.clip(np.sum(pts_src_world * pts_dst_world, axis=1), -1, 1))
        all_errors.extend(angle_errors)
        
        # 投影并绘制
        u_src, v_src = sphere_to_equirect_coords(pts_src_world.reshape(1, -1, 3), out_width, out_height)
        u_dst, v_dst = sphere_to_equirect_coords(pts_dst_world.reshape(1, -1, 3), out_width, out_height)
        
        for i, (x1, y1, x2, y2) in enumerate(zip(u_src.flat, v_src.flat, u_dst.flat, v_dst.flat)):
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            if 0 <= x1 < out_width and 0 <= y1 < out_height and 0 <= x2 < out_width and 0 <= y2 < out_height:
                # 误差映射颜色：绿->黄->红
                norm = min(np.rad2deg(angle_errors[i]) / 5.0, 1.0)
                color = (int(255 * norm), 255, 0) if norm < 0.5 else (0, int(255 * (2 - 2 * norm)), 255)
                cv.line(vis_img, (x1, y1), (x2, y2), color, 1)
    
    # 统计信息和图例
    legend = np.ones((120, out_width, 3), dtype=np.uint8) * 255
    for i in range(400):
        norm = i / 400
        color = (int(255 * norm), 255, 0) if norm < 0.5 else (0, int(255 * (2 - 2 * norm)), 255)
        cv.line(legend, (50 + i, 30), (50 + i, 60), color, 1)
    
    cv.putText(legend, "0°", (40, 80), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    cv.putText(legend, "5°", (440, 80), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    
    if all_errors:
        mean, max_e, median = np.rad2deg([np.mean(all_errors), np.max(all_errors), np.median(all_errors)])
        cv.putText(legend, f"Mean:{mean:.2f}° Max:{max_e:.2f}° Median:{median:.2f}° N:{len(all_errors)}", 
                  (50, 105), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        print(f"\n平均:{mean:.2f}° 最大:{max_e:.2f}° 中位数:{median:.2f}°")
    
    final_img = np.vstack([legend, vis_img])
    cv.imwrite(output_path, final_img)
    print(f"已保存到: {output_path}")
    return final_img

def visualize_optical_flow_color(flow: np.ndarray, save_path: str = None) -> np.ndarray:
    """
    可视化稠密光流为 HSV 颜色图。
    颜色代表方向，亮度/饱和度代表大小。
    
    Args:
        flow: (H, W, 2) 光流场
        save_path: 可选，保存路径
    Returns:
        bgr: 可视化结果图像
    """
    h, w = flow.shape[:2]
    hsv = np.zeros((h, w, 3), dtype=np.uint8)
    hsv[..., 1] = 255

    mag, ang = cv.cartToPolar(flow[..., 0], flow[..., 1])
    
    # 色调 (Hue) 代表角度 (0-179)
    # angel in radians -> degrees / 2
    hsv[..., 0] = ang * 180 / np.pi / 2
    
    # 值 (Value) 代表大小
    # 归一化到 0-255 用于显示
    if np.max(mag) > 0:
        hsv[..., 2] = cv.normalize(mag, None, 0, 255, cv.NORM_MINMAX)
    else:
        hsv[..., 2] = 0
    
    bgr = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)
    
    if save_path:
        cv.imwrite(save_path, bgr)
        print(f"光流颜色图已保存到: {save_path}")
        
    return bgr

def visualize_optical_flow_arrows(image: np.ndarray, flow: np.ndarray, 
                                step: int = 16, scale: float = 1.0, 
                                color: tuple = (0, 255, 0), save_path: str = None) -> np.ndarray:
    """
    在图像上绘制光流向量箭头 (稀疏采样)。
    
    Args:
        image: 背景图像
        flow: 光流场 (H, W, 2)
        step: 采样步长 (每隔 step 像素画一个箭头)
        scale: 箭头的长度缩放比例
        color: 箭头颜色
        save_path: 可选，保存路径
    Returns:
        vis_img: 可视化结果图像
    """
    h, w = image.shape[:2]
    vis_img = image.copy()
    
    for y in range(0, h, step):
        for x in range(0, w, step):
            # 防止索引越界
            if y >= flow.shape[0] or x >= flow.shape[1]:
                continue
                
            dx, dy = flow[y, x]
            
            # 只有当光流显著时才绘制
            if abs(dx) > 0.5 or abs(dy) > 0.5:
                end_x = int(x + dx * scale)
                end_y = int(y + dy * scale)
                
                cv.arrowedLine(vis_img, (x, y), (end_x, end_y), color, 1, tipLength=0.3)
                
    if save_path:
        cv.imwrite(save_path, vis_img)
        print(f"光流箭头图已保存到: {save_path}")
        
    return vis_img
