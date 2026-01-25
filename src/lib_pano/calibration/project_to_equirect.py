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


def test_main(calib_path: str, camera_name:str, image_path: str, out_path: str, flip_v: bool):
    K, distortion, xi = load_single_K_and_distortion(calib_path, camera_name)
    k1, k2 , p1, p2 = distortion[0], distortion[1], distortion[2], distortion[3]
    fx, fy = K[0,0], K[1,1]
    cx, cy = K[0,2], K[1,2]
    mei_params = {
        'k1': k1,
        'k2': k2,
        'p1': p1,
        'p2': p2,
        'xi': xi,
        'fx': fx,
        'fy': fy,
        'cx': cx,
        'cy': cy,
    }
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError('无法打开输入图像: %s' % image_path)

    equi = omnidir_to_equirect(img, mei_params, out_w=1920, out_h=960, flip_v=flip_v)

    cv2.imwrite(out_path, equi)
    print('写出：', out_path)
    
    
# 可以作为一个优化思路吧，把overlap投到赤道上