import math
import numpy as np
import cv2
from lib_pano.io.params_read import load_single_K_and_distortion


# 替换版：增加几个调试开关
def direction_to_pixel(mei_params, dirs, xi_sign=+1, swap_fx_fy=False):
    xi = mei_params['xi']
    fx = mei_params['fx']
    fy = mei_params['fy']
    cx = mei_params['cx']
    cy = mei_params['cy']
    k1 = mei_params['k1']
    k2 = mei_params['k2']
    p1 = mei_params['p1']
    p2 = mei_params['p2']

    if swap_fx_fy:
        fx, fy = fy, fx

    X = dirs[:, 0]
    Y = dirs[:, 1]
    Z = dirs[:, 2]

    # 尝试使用 Z + xi 或 Z - xi（xi_sign 取 +1 或 -1）
    denom = Z + xi_sign * xi
    eps = 1e-8
    # 标记无效点，防止极端放大
    invalid_mask = denom <= eps
    denom_safe = denom.copy()
    denom_safe[invalid_mask] = np.nan

    x = X / denom_safe
    y = Y / denom_safe

    r2 = x * x + y * y
    radial = 1.0 + k1 * r2 + k2 * (r2 ** 2)
    x_dist = x * radial + 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x)
    y_dist = y * radial + p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y

    u = fx * x_dist + cx
    v = fy * y_dist + cy

    pixels = np.stack([u, v], axis=1)
    return pixels

def create_equirectangular_map(mei_params, H_out, W_out, flip_v=False, xi_sign=+1, swap_fx_fy=False):
    u = np.linspace(0, W_out - 1, W_out)
    v = np.linspace(0, H_out - 1, H_out)
    uu, vv = np.meshgrid(u, v)

    # 若 flip_v=True 输入图像是正的, flip_v=False 输入图像是倒的
    if flip_v:
        lat = (vv / float(H_out) - 0.5) * math.pi
    else:
        lat = (0.5 - vv / float(H_out)) * math.pi

    lon = (uu / float(W_out) - 0.5) * 2.0 * math.pi

    cos_lat = np.cos(lat)
    X = cos_lat * np.sin(lon)
    Y = np.sin(lat)
    Z = cos_lat * np.cos(lon)

    dirs = np.stack([X, Y, Z], axis=-1).reshape((-1, 3))
    pixels = direction_to_pixel(mei_params, dirs, xi_sign=xi_sign, swap_fx_fy=swap_fx_fy)
    map_x = pixels[:, 0].reshape((H_out, W_out)).astype(np.float32)
    map_y = pixels[:, 1].reshape((H_out, W_out)).astype(np.float32)
    # 将 Z < 0（向后/不可见）的位置置为无效（remap 使用 -1 作为不可采样信号）
    forward_mask = (dirs[:, 2] > 0).reshape((H_out, W_out))
    map_x[~forward_mask] = -1.0
    map_y[~forward_mask] = -1.0
    return map_x, map_y


def omnidir_to_equirect(input_img, mei_params, out_w=2048, out_h=1024, border_mode=cv2.BORDER_CONSTANT, flip_v=False):
    H_out = out_h
    W_out = out_w
    map_x, map_y = create_equirectangular_map(mei_params, H_out, W_out, flip_v=flip_v)

    # 对超出原图范围的映射点，remap 会用 borderMode 填充
    equi = cv2.remap(input_img, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=border_mode)
    return equi


def double_sphere_to_equirectangular(input_img, mei_params, out_w=1920, out_h=960):
    """
    基于 Kalibr Double Sphere (DS) 模型的鱼眼图转经纬图
    Params keys used: 'xi_ds', 'alpha', 'fx', 'fy', 'cx', 'cy'
    """
    xi = mei_params.get('xi_ds', 0.0)
    alpha = mei_params.get('alpha', 0.0)
    fx = mei_params['fx']
    fy = mei_params['fy']
    cx = mei_params['cx']
    cy = mei_params['cy']
    
    h, w = out_h, out_w
    u = np.linspace(0, w - 1, w)
    v = np.linspace(0, h - 1, h)
    uu, vv = np.meshgrid(u, v)

    # 1. Equirect -> Camera Vector
    # v=0(Top) -> Lat=pi/2
    lat = (0.5 - vv / float(h)) * np.pi
    lon = (uu / float(w) - 0.5) * 2.0 * np.pi
    
    # Camera frame: Z-Forward, X-Right, Y-Down
    # World frame: Y-Up, Z-Forward (center of pano), X-Right
    # So Y_c = -Y_w
    
    cos_lat = np.cos(lat)
    sin_lat = np.sin(lat)
    cos_lon = np.cos(lon)
    sin_lon = np.sin(lon)
    
    # Using standard conversion
    x = cos_lat * sin_lon
    y = -sin_lat
    z = cos_lat * cos_lon
    
    # 2. DS Projection (3D -> 2D)
    # d1 = sqrt(x^2 + y^2 + z^2). Since unit vector, d1=1.
    d1 = 1.0
    
    # z' = z + xi * d1
    z_prime = z + xi
    
    # d2 = sqrt(x^2 + y^2 + z'^2)
    x2_y2 = x*x + y*y
    d2 = np.sqrt(x2_y2 + z_prime*z_prime)
    
    # denom = alpha * d2 + (1 - alpha) * z'
    denom = alpha * d2 + (1.0 - alpha) * z_prime
    
    # Avoid division by zero
    valid_mask = np.abs(denom) > 1e-6
    
    # Normalized coords
    # u_n = x / denom
    # v_n = y / denom
    # Using 'where' to handle safe division
    safe_denom = denom.copy()
    safe_denom[~valid_mask] = np.nan
    
    u_norm = x / safe_denom
    v_norm = y / safe_denom
    
    # Pixel coords
    map_x = fx * u_norm + cx
    map_y = fy * v_norm + cy
    
    map_x = map_x.astype(np.float32)
    map_y = map_y.astype(np.float32)
    
    # Mask out invalid points (e.g. behind camera)
    # Since z is the forward axis, z < 0 implies point is behind camera plane (FOV > 180)
    # User requested to ban z < 0
    map_x[z <= 0] = -1.0
    map_y[z <= 0] = -1.0
    
    output = cv2.remap(input_img, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    return output


def fisheye_to_equirectangular(
    fisheye_img, 
    mei_params,
    out_w=1920,
    out_h=960
):
    """
    基于 Kannala-Brandt (KB4) 模型的鱼眼图转经纬图 (Equidistant)
    """
    k1 = mei_params['k1']
    k2 = mei_params['k2']
    k3 = mei_params.get('p1', 0.0) # p1 slot used for k3
    k4 = mei_params.get('p2', 0.0) # p2 slot used for k4
    
    fx, fy = mei_params['fx'], mei_params['fy']
    cx, cy = mei_params['cx'], mei_params['cy']
    
    h, w = out_h, out_w
    
    u = np.linspace(0,  w - 1, w)
    v = np.linspace(0, h - 1, h)
    uu, vv = np.meshgrid(u, v)
    
    lat = (0.5 - vv / float(h)) * np.pi
    lon = (uu / float(w) - 0.5) * 2.0 * np.pi

    cos_lat = np.cos(lat)
    sin_lat = np.sin(lat)
    cos_lon = np.cos(lon)
    sin_lon = np.sin(lon)

    x_c = cos_lat * sin_lon 
    y_c = -sin_lat               
    z_c = cos_lat * cos_lon      
    
    theta = np.arccos(z_c) 
    phi = np.arctan2(y_c, x_c)

    theta_2 = theta ** 2
    theta_4 = theta_2 ** 2
    theta_6 = theta_2 * theta_4
    theta_8 = theta_4 ** 2
    
    theta_d = theta * (1.0 + k1 * theta_2 + k2 * theta_4 + k3 * theta_6 + k4 * theta_8)
    
    u_fish = fx * theta_d * np.cos(phi) + cx
    v_fish = fy * theta_d * np.sin(phi) + cy
    
    # Mask z < 0 to avoid wrapping artifacts or projecting back-sphere
    u_fish[z_c <= 0] = -1.0
    v_fish[z_c <= 0] = -1.0

    equirectangular = cv2.remap(
        fisheye_img, 
        u_fish.astype(np.float32), 
        v_fish.astype(np.float32),
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0,0,0)
    )
    
    return equirectangular


def test_main(calib_path: str, camera_name:str, image_path: str, out_path: str, flip_v: bool):
    # Updated to receive model name
    K, distortion, xi, model = load_single_K_and_distortion(calib_path, camera_name)
    
    # Universal params extraction
    fx, fy = K[0,0], K[1,1]
    cx, cy = K[0,2], K[1,2]

    use_ds = False
    use_kb = False
    
    # Check model string
    if 'ds' in model or 'double' in model:
        use_ds = True
    elif 'equi' in model or 'kb' in model:
        use_kb = True
    
    # For 135 calibration explicitly mentioned in task:
    if model == 'ds-none':
        use_ds = True

    mei_params = {
        'fx': fx, 'fy': fy, 'cx': cx, 'cy': cy
    }

    if use_ds:
        print(f"[Info] Detected Double Sphere model: {model}")
        # In this specific custom json, DS params seem to be in 'distortion' array
        mei_params['xi_ds'] = distortion[0]
        mei_params['alpha'] = distortion[1]
        
    elif use_kb:
        print(f"[Info] Detected Equidistant/KB model: {model}")
        mei_params['k1'] = distortion[0]
        mei_params['k2'] = distortion[1]
        mei_params['p1'] = distortion[2]
        mei_params['p2'] = distortion[3]
        
    else:
        # Default / Fallback / Omni
        # If camera is one of the 135s but model was not detected as DS, maybe default to DS if params look like it?
        # But stick to safe logic
        print(f"[Info] Using default Omni/MEI model: {model}")
        mei_params['xi'] = xi
        mei_params['k1'] = distortion[0]
        mei_params['k2'] = distortion[1]
        mei_params['p1'] = distortion[2]
        mei_params['p2'] = distortion[3]

    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError('无法打开输入图像: %s' % image_path)
    
    if use_ds:
        equi = double_sphere_to_equirectangular(img, mei_params, out_w=1920, out_h=960)
    elif use_kb: 
        equi = fisheye_to_equirectangular(img, mei_params, out_w=1920, out_h=960)
    else:
        # Legacy check if we want to force something for specific cameras
        if camera_name in ['camera1', 'camera3', 'camera5'] and 'ds' not in model:
             # Previously forced KB4 here, but if model is not DS, maybe it IS KB4? 
             # Let's trust the model string first.
             equi = fisheye_to_equirectangular(img, mei_params, out_w=1920, out_h=960)
        else:
             equi = omnidir_to_equirect(img, mei_params, out_w=1920, out_h=960, flip_v=flip_v)

    cv2.imwrite(out_path, equi)
    print('写出：', out_path)
