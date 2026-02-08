"""One-step panorama stitching for three fisheye cameras."""
import os
import time
from typing import List, Tuple

import numpy as np
import cv2
from lib_pano import local_setting as ls
from lib_pano.io.params_read import load_all_K_and_distortion, load_R_and_T, load_opt_R_and_T
from lib_pano.camera.camera import get_camera_std_euler_by_R, get_R_by_euler


def forward_unified_to_normalized(X: np.ndarray, Y: np.ndarray, Z: np.ndarray, xi: float):
    """统一球面模型（unified / omni）前向投影：
    """
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    Z = np.asarray(Z, dtype=np.float64)
    
    s =  Z + xi

    mx = X / s
    my = Y / s
    return mx, my


def forward_kb4_to_pixel(X: np.ndarray, Y: np.ndarray, Z: np.ndarray, 
                         k1: float, k2: float, k3: float, k4: float,
                         fx: float, fy: float, cx: float, cy: float):
    """Kannala-Brandt (Equidistant) Model Forward Projection
    World/Cam coords -> Pixel coords
    """
    r = np.sqrt(X*X + Y*Y + Z*Z)
    theta = np.arccos(Z / (r + 1e-8))
    phi = np.arctan2(Y, X)
    
    theta_2 = theta * theta
    theta_4 = theta_2 * theta_2
    theta_6 = theta_4 * theta_2
    theta_8 = theta_4 * theta_4
    
    theta_d = theta * (1.0 + k1 * theta_2 + k2 * theta_4 + k3 * theta_6 + k4 * theta_8)
    
    x_dist = theta_d * np.cos(phi)
    y_dist = theta_d * np.sin(phi)
    
    u = fx * x_dist + cx
    v = fy * y_dist + cy
    
    return u, v


def forward_ds_to_pixel(X: np.ndarray, Y: np.ndarray, Z: np.ndarray, 
                        xi: float, alpha: float,
                        fx: float, fy: float, cx: float, cy: float):
    """Double Sphere Model Forward Projection"""
    # d1 = sqrt(x^2 + y^2 + z^2)
    # z' = z + xi * d1
    # d2 = sqrt(x^2 + y^2 + z'^2)
    # denom = alpha * d2 + (1 - alpha) * z'
    
    # Normalized:
    # u = x / denom
    # v = y / denom
    
    d1 = np.sqrt(X*X + Y*Y + Z*Z)
    z_prime = Z + xi * d1
    
    d2 = np.sqrt(X*X + Y*Y + z_prime*z_prime)
    denom = alpha * d2 + (1.0 - alpha) * z_prime
    
    # Valid check?
    
    u_n = X / (denom + 1e-8)
    v_n = Y / (denom + 1e-8)
    
    u = fx * u_n + cx
    v = fy * v_n + cy
    
    return u, v



def apply_radtan_distortion(mx: np.ndarray, my: np.ndarray, k1: float, k2: float, p1: float, p2: float):
    """对无畸变归一化坐标应用 Radtan 畸变模型，返回畸变后的归一化坐标 (xd, yd)。"""
    x = mx
    y = my
    r2 = x * x + y * y
    radial = 1.0 + k1 * r2 + k2 * (r2 ** 2)
    x_dist = x * radial + 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x)
    y_dist = y * radial + p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y
    return x_dist, y_dist


def normalize_dirs_cam(dirs_cam: np.ndarray, eps: float = 1e-12):
    """将 3xN 的列向量组归一化为单位向量。

    - dirs_cam: shape (3, N)
    - 返回归一化后的数组，形状仍为 (3, N)。若某列范数小于 eps，则用 eps 替代以避免除0。
    """
    norms = np.linalg.norm(dirs_cam, axis=0) # axis=0 按列算范数
    norms_safe = np.where(norms < eps, eps, norms)
    return dirs_cam / norms_safe


def generate_equirectangular_dirs(H: int, W: int) -> np.ndarray:
    """为 equirectangular 全景生成每像素的单位方向向量（世界坐标系）。
    返回 shape=(H,W,3) 的数组，单位长度。
    映射约定：u  [0,W) -> theta  [-pi, pi), v  [0,H) -> phi  [pi/2, -pi/2]
    theta: 经度，phi: 纬度
    """
    us = (np.arange(W, dtype=np.float64) ) / W  # [0,1]
    vs = (np.arange(H, dtype=np.float64) ) / H  # [0,1]
    # 0对应-pi,W对应Pi 我想让pi对应W/2 us=0.5, theta = pi us=0, theta =0
    # theta = us * (2.0 * np.pi) - np.pi  # 修改这一部分，可以决定从哪里开出来 Z 轴为蓝色  X 轴为红色、Y 轴为绿色
    # 想让-2/3*pi处对应(0,0,1)点 即us=1/6  x=0,y=0,z=1 th=0
    # w_offset = -1.0 / 6.0 
    w_offset = 0.2
    theta = (us * 2.0 -1 + w_offset) * np.pi  
    theta = np.where(theta > np.pi, theta - 2.0 * np.pi, theta)
    phi = (vs - 0.5) * np.pi

    th, ph = np.meshgrid(theta, phi)
    # x = cosph * cos(th) 图像映射到最右边(不使用外参) x=cosph * sin(th):图像映射到中间
    sinph = np.sin(ph)
    cosph = np.cos(ph)
    z = cosph * np.cos(th)  
    y = sinph    # ph=pi/2 y=1
    x = cosph * np.sin(th)  

    dirs = np.stack([x, y, z], axis=-1)  # H,W,3
    return dirs

def remap_and_accumulate(
    pano_dirs: np.ndarray,
    images: List[np.ndarray],
    Ks: List[np.ndarray],
    dists: List[np.ndarray],
    xis: List[float],
    Rs: List[np.ndarray],
    Ts: List[np.ndarray],
    models: List[str],  # Added models list
    output_size: Tuple[int, int] = (1024, 512),
):
    """对一组相机进行反向映射并融合：
    - pano_dirs: (H,W,3) 世界坐标下的单位方向向量
    - images: 每个相机的 H_i x W_i x 3 BGR 图像 (uint8)
    - Ks,dists,xis,Rs,Ts: 校准参数

    返回融合后的 BGR uint8 全景图
    """
    H, W = output_size[1], output_size[0]
    # H = pano_dirs.shape[0] # Overwriting param H? Careful.
    # W = pano_dirs.shape[1] # Prefer using output_size or shape of dirs if matched.
    H_grid, W_grid = pano_dirs.shape[:2]


    dirs_flat = pano_dirs.reshape(-1, 3).T  # 3,N 列向量形式，便于右乘外参
    # 为接缝合并准备每相机的采样与权重栈（硬边界：选择权重最高的相机）
    num_cam = len(images)
    
    # Ensure num_cam matches provided params
    if len(models) < num_cam:
        # Fallback if models not provided correctly
        models = ["unknown"] * num_cam

    sampled_stack = np.zeros((num_cam, H_grid, W_grid, 3), dtype=np.float64)
    weight_stack = np.zeros((num_cam, H_grid, W_grid), dtype=np.float64)
    # 修改一下 让R[1] = R[2] R[2] =R[1]  PROBLEM
    # Rs[1], Rs[2] = Rs[2], Rs[1]  
    for idx, img in enumerate(images):
        K = Ks[idx]
        dist = dists[idx]
        xi = xis[idx]
        R = Rs[idx]
        t = Ts[idx]
        model = models[idx]

        ih, iw = img.shape[:2]

        fx = float(K[0, 0]); fy = float(K[1, 1]); cx = float(K[0, 2]); cy = float(K[1, 2])
        
        # Decide projection method
        use_ds = False
        use_kb = False
        
        if 'ds' in model or 'double' in model:
            use_ds = True
        elif 'equi' in model or 'kb' in model:
            use_kb = True
        elif model == 'ds-none': # Specific case from json
            use_ds = True
            
        if use_ds:
             # DS params: usually distortion=[xi, alpha, 0, 0] in that json
             # Double check param interpretation
             xi_ds = float(dist[0])
             alpha = float(dist[1])
             # xi from K is usually for Omni, ignore if DS uses dist[0]
        elif use_kb:
             k1 = float(dist[0])
             k2 = float(dist[1])
             p1 = float(dist[2]) # k3
             p2 = float(dist[3]) # k4
        else:
             # Default Unified/Omni
             k1 = float(dist[0]) if dist.size > 0 else 0.0
             k2 = float(dist[1]) if dist.size > 1 else 0.0
             p1 = float(dist[2]) if dist.size > 2 else 0.0
             p2 = float(dist[3]) if dist.size > 3 else 0.0


        # 将世界点/方向转换到相机坐标系（包含外参平移）：
        # 真实关系为 Pc = R^T * (Pw - t)。使用行向量表示,做转置了以后,则为 Pc_row = (Pw_row - t_row) @ R
        # R_inv, T_inv = Get_inv_RT(R, t) # wor_to_cam
        R_wor_to_cam , T_wor_to_cam = R, t
        # euler_X, euler_Y, euler_Z = get_camera_std_euler_by_R(R_wor_to_cam)
        # print(f"euler_X:{euler_X}, euler_y:{euler_Y}, euler_Z:{euler_Z}\n")
        dirs_cam = R_wor_to_cam @ dirs_flat   # 3,N 列向量
        # dirs_cam = dirs_flat
        dirs_cam = normalize_dirs_cam(dirs_cam)
        # print(dirs_cam.shape)
        Xc = dirs_cam[0, :]
        Yc = dirs_cam[1, :]
        Zc = dirs_cam[2, :]

        
        # Valid mask based on Zc
        if use_ds or use_kb:
            # Allow slightly back
            # But usually we filter by image circle boundary (FOV) later?
            # Let's allow all for now and filter by map_x/map_y range
            pass
        else:
            # Unified old logic
            facing = Zc > 1e-6
            if not np.any(facing):
                continue

        if use_ds:
            map_x_flat, map_y_flat = forward_ds_to_pixel(
                Xc, Yc, Zc, xi_ds, alpha, fx, fy, cx, cy
            )
        elif use_kb:
            map_x_flat, map_y_flat = forward_kb4_to_pixel(
                Xc, Yc, Zc, k1, k2, p1, p2, fx, fy, cx, cy
            )
        else:
            # Unified
            mx, my = forward_unified_to_normalized(Xc, Yc, Zc, xi)
            xd, yd = apply_radtan_distortion(mx, my, k1, k2, p1, p2)
            map_x_flat = xd * fx + cx
            map_y_flat = yd * fy + cy

        map_x = map_x_flat.reshape(H_grid, W_grid).astype(np.float32)
        map_y = map_y_flat.reshape(H_grid, W_grid).astype(np.float32)

        # mask: 在图像范围内并且朝向相机
        # Check image bounds
        in_bounds = (map_x >= 0) & (map_x <= (iw - 1)) & (map_y >= 0) & (map_y <= (ih - 1))
        
        # Zc Check logic again
        if use_ds or use_kb:
            z_valid = Zc.reshape(H_grid, W_grid) > 0
            valid_mask = in_bounds & z_valid
        else:
            z_valid = Zc.reshape(H_grid, W_grid) > 1e-6
            valid_mask = in_bounds & z_valid

        map_x[~valid_mask] = -1  # 设置无效点，remap 会跳过
        map_y[~valid_mask] = -1
        # 使用 OpenCV remap（双线性插值）拉取颜色，注意输入图像为 BGR
        sampled = cv2.remap(img, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        
        # Save intermediate sampled images for debugging (requested for 135 mode)
        debug_path = os.path.join(ls.DATA_OUTPUT_DIR, 'preview', f"debug_sampled_cam_{idx}.png")
        cv2.imwrite(debug_path, sampled)
        print(f"Saved debug image: {debug_path}")
        
        sampled_f = sampled.astype(np.float64) / 255.0

        # 权重：使用 camera-facing 的 Zc（绝对值/余弦）作为权重（更正面贡献更大）
        w = np.maximum(0.0, Zc.reshape(H_grid, W_grid))
        w *= valid_mask.astype(np.float64)

        # 存入栈，用于后续按最大权重选取（接缝合并，硬边界）
        sampled_stack[idx, :, :, :] = sampled_f
        weight_stack[idx, :, :] = w    
        
    
    # 基于权重栈按像素选择权重最大的相机（接缝合并）
    eps = 1e-8
    max_weight = weight_stack.max(axis=0)
    valid = max_weight > eps

    # 对每个像素选取权重最大的相机索引
    best_idx = np.argmax(weight_stack, axis=0)  # H,W

    # 从 sampled_stack 中按 best_idx 取值
    # sampled_stack: num,H,W,3 -> 转为 H,W,num,3 以便索引
    arr = sampled_stack.transpose(1, 2, 0, 3)  # H,W,num,3
    rows = np.arange(H_grid)[:, None]
    cols = np.arange(W_grid)[None, :]
    pano_f = arr[rows, cols, best_idx]  # H,W,3

    pano_out = np.zeros((H_grid, W_grid, 3), dtype=np.uint8)
    pano_out[valid] = (pano_f[valid] * 255.0).astype(np.uint8)

    return pano_out

def load_images_for_names(images_path: List[str]):
    images = []
    for p in images_path:
        img = cv2.imread(p, cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(p)
        images.append(img)
    return images   
            


def one_step_stitch(images_path: List[str],
                    calib_path: str,
                    calib_opt_path: str,
                    output_path: str,
                    pano_size: Tuple[int, int] = (1920, 960)):

    names, K_list, dist_list, xi_list, model_list = load_all_K_and_distortion(calib_path)
    
    # Simple heuristic to switch extrinsic loader
    if "camera1" in names or "camera3" in names:
        # Assuming 135 series use the standard kalibr 'chain' format in the main calib file
        print(f"[Info] Detected 135 series ({names}), loading extrinsics from {calib_path} using chain format.")
        R_list, t_list = load_R_and_T(calib_path)
    else:
        # Assuming 246 series use the 'opt' format
        print(f"[Info] Detected 246 series ({names}), loading extrinsics from {calib_opt_path}.")
        R_list, t_list = load_opt_R_and_T(calib_opt_path)  # Debug
    
    
    # R_list = trans_R(R_list)
    # 加载图片
    images = load_images_for_names(images_path)

    # 生成全景方向
    W, H = pano_size
    pano_dirs = generate_equirectangular_dirs(H, W)

    print("Start remap-and-fuse...")
    pano = remap_and_accumulate(pano_dirs, images, K_list, dist_list, xi_list, R_list, t_list, model_list, output_size=pano_size)
    print(f"Panorama generated, saving to: {output_path}")
    cv2.imwrite(output_path, pano)


