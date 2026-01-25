import numpy as np
from typing import Tuple
import tqdm
import cv2 as cv
import json
from lib_pano.io.params_read import load_single_K_and_distortion

def undistor(path_src: str, path_dst: str, path_intrinsic: str, method: str = 'r1'):
    '''
    r1: 使用三次方模型去畸变
    r2: 使用二次方模型去畸变
    '''
    src_image = cv.imread(path_src)
    a, b, c, center = load_intrinsic(path_intrinsic)
    H, W = src_image.shape[:2]
    if method == 'r1':
        transform_func = transform_point_1
    else:
        transform_func = transform_point
    map1, map2 = create_map(H, W, transform_func, a, b, c, center)
    dst_image = cv.remap(src_image, map1, map2, interpolation=cv.INTER_LINEAR)
    cv.imwrite(path_dst, dst_image)

def undistor_test(image_distorted: np.ndarray, K: np.ndarray, D: np.ndarray):
    H, W = image_distorted.shape[:2]
    a, b, c = D[0], D[1], 0.0
    fx, fy = K[0,0], K[1,1]
    cx_norm = K[0,2]/fx
    cy_norm = K[1,2]/fy
    center = (cx_norm, cy_norm)
    map1, map2 = create_map(H, W, transform_point, a, b, c, center, fx=fx, fy=fy)
    image_undistorted = cv.remap(image_distorted, map1, map2, interpolation=cv.INTER_LINEAR)
    return image_undistorted
    
def create_map(
    H: int,
    W: int, 
    transform_func: callable, 
    a:float, 
    b:float, 
    c:float, 
    center: Tuple[float, float],
    fx: float = None,
    fy: float = None,
    ):
    map1 = np.zeros((H, W), dtype=np.float32)
    map2 = np.zeros((H, W), dtype=np.float32)
    W_s = W if fx is None else fx
    H_s = H if fy is None else fy
    # size = (width, height)
    for h in tqdm.tqdm(range(H)):
        for w in range(W):
            x_norm = w/W_s
            y_norm = h/H_s
            x_distor_norm, y_distor_norm = transform_func(x_norm, y_norm, a, b, c, center)
            map1[h, w] = x_distor_norm*W_s
            map2[h, w] = y_distor_norm*H_s
    return map1, map2

def load_intrinsic(path_intrinsic: str):
    with open(path_intrinsic, 'r', encoding='utf-8') as f:
        data = json.load(f)
    a = data["distortion"]["coeffs_map"]["a"]
    b = data["distortion"]["coeffs_map"]["b"]
    c = data["distortion"]["coeffs_map"]["c"]
    center = data["distortion"]["center"]["x"], data["distortion"]["center"]["y"]
    return a, b, c, center

def transform_point(
    x_undistor_norm:int, 
    y_undistor_norm:int,
    a:float,
    b:float,
    c:float,
    center: Tuple[float, float],
    ):
    
    cx_norm, cy_norm = center
    x = x_undistor_norm - cx_norm
    y = y_undistor_norm - cy_norm

    r2 = x * x + y * y
    r4 = r2 * r2
    cdist = 1 + a * r2 + b * r4 + c * r2 * r4
        
    x_distor = x * cdist
    y_distor = y * cdist
    
    x_distor_norm = x_distor + cx_norm
    y_distor_norm = y_distor + cy_norm
    return x_distor_norm, y_distor_norm

def transform_point_1(
    x_undistor_norm:int, 
    y_undistor_norm:int,
    a:float,
    b:float,
    c:float,
    center: Tuple[float, float],
    ):
    cx_norm, cy_norm = center
    x = x_undistor_norm - cx_norm
    y = y_undistor_norm - cy_norm
    r = np.sqrt(x * x + y * y)
    d = 1-a-b-c
    cdist = a*r*r*r + b*r*r + c*r + d   # 删除+1看看
    x_distor = x * cdist
    y_distor = y * cdist
    x_distor_norm = x_distor + cx_norm
    y_distor_norm = y_distor + cy_norm
    return x_distor_norm, y_distor_norm
    
def rotate_image_counterclockwise_90(path_src: str):
    src_image = cv.imread(path_src)
    dst_image = cv.rotate(src_image, cv.ROTATE_90_COUNTERCLOCKWISE)
    cv.imwrite(path_src, dst_image)

def build_undistort_map(w, h, path_intrinsic: str):
    with open(path_intrinsic, 'r', encoding='utf-8') as f:
        data = json.load(f)
    a = data["distortion"]["coeffs_map"]["a"]
    b = data["distortion"]["coeffs_map"]["b"]
    c = data["distortion"]["coeffs_map"]["c"]
    cx, cy = data["distortion"]["center"]["x"], data["distortion"]["center"]["y"]
    d = 1 - a - b - c
    j, i = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    x = (i / w) - cx
    y = (j / h) - cy
    r_dest = np.sqrt(x * x + y * y)
    r_src = (a * r_dest + b * r_dest**2 + c * r_dest**3 + d) * r_dest
    # r_src/r_dest OR 1
    scale = np.divide(r_src, r_dest, out=np.ones_like(r_dest), where=r_dest > 1e-8)
    x_src = x * scale + cx
    y_src = y * scale + cy
    map_x = (x_src * w).astype(np.float32)
    map_y = (y_src * h).astype(np.float32)
    return map_x, map_y

def undistort_r1_quick(path_in, path_out, path_intrinsic):
    
    img = cv.imread(path_in)
    map_x, map_y = build_undistort_map(img.shape[1], img.shape[0], path_intrinsic)
    undist = cv.remap(img, map_x, map_y, interpolation=cv.INTER_LINEAR)
    cv.imwrite(path_out, undist)

def undistort_r2_quick(path_calib: str, camera_name: str, path_in: str, path_out: str):
    K, distortion, xi = load_single_K_and_distortion(path_calib, camera_name)
    D = np.array([distortion[0], distortion[1], 0, 0, 0])
    img = cv.imread(path_in)
    # undist = cv.fisheye.undistortImage(img, K, D)
    undist = cv.undistort(img, K, D)
    cv.imwrite(path_out, undist)
    print(f"已保存去畸变图像到 {path_out}")
    
    
if __name__ == "__main__":
    undistor(
        path_src=r"data\origin_1\origin_1.snap.1.png",
        path_dst=r"data\origin_1\origin_1.snap.1.undist2.png", 
        path_intrinsic=r"src\calibration\intrinsic.json"
    )
    # undist2是用transform_point函数生成的, undist3是用transform_point_1函数生成的
