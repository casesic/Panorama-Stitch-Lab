import numpy as np
from typing import Tuple
import tqdm
import cv2 as cv
import json

def undistor(path_src: str, path_dst: str, path_intrinsic: str):
    src_image = cv.imread(path_src)
    a, b, c, center = load_intrinsic(path_intrinsic)
    H, W = src_image.shape[:2]
    map1, map2 = create_map(H, W, transform_point, a, b, c, center)
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
    cdist = a*r*r*r + b*r*r + c*r + 1
    x_distor = x * cdist
    y_distor = y * cdist
    x_distor_norm = x_distor + cx_norm
    y_distor_norm = y_distor + cy_norm
    return x_distor_norm, y_distor_norm
    

if __name__ == "__main__":
    undistor(
        path_src=r"data\origin_1\origin_1.snap.1.png",
        path_dst=r"data\origin_1\origin_1.snap.1.undist2.png", 
        path_intrinsic=r"src\calibration\intrinsic.json"
    )
    # undist2是用transform_point函数生成的, undist3是用transform_point_1函数生成的
