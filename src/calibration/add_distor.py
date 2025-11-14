
from typing import List, Tuple, Dict, Optional, Union, Any

import os
import os.path as osp

from icecream import ic
import numpy as np
from tqdm import tqdm
import pickle
import cv2 as cv
from undistor import undistor_test

def apply_distortion(x_distor, y_distor, k1, k2):
    '''
    given (x_distor, y_distor) in normalized image coord and distortion coeffs k1, k2,
    compute (x_nonedistor, y_nonedistor) in normalized image coord
    we know:
    x_distor = x_nonedistor * (1 + k1*r^2 + k2*r^4)
    y_distor = y_nonedistor * (1 + k1*r^2 + k2*r^4)
    where r^2 = x_nonedistor^2 + y_nonedistor^2
    but it is hard to directly solve for (x_nonedistor, y_nonedistor),
    because it is a 5th degree polynomial.

    therefore we use an iterative method to approximate the solution.
    '''
    x = x_distor
    y = y_distor
    
    for _ in range(10):
        r2 = x * x + y * y
        r4 = r2 * r2
        cdist = 1 + k1 * r2 + k2 * r4
        
        x = x_distor / cdist
        y = y_distor / cdist

    # ic(x, y)
    # ic(x_distor, y_distor)
    # exit()

    x_nonedistor = x
    y_nonedistor = y

    #check
    r2_check = x_nonedistor * x_nonedistor + y_nonedistor * y_nonedistor
    r4_check = r2_check * r2_check
    cdist_check = 1 + k1 * r2_check + k2 * r4_check
    x_distor_check = x_nonedistor * cdist_check
    y_distor_check = y_nonedistor * cdist_check
    # print('Distortion check difference:', x_distor_check - x_distor, y_distor_check - y_distor)
    # exit()
    assert np.allclose(x_distor_check, x_distor, atol=1e-5), (x_distor_check, x_distor)
    assert np.allclose(y_distor_check, y_distor, atol=1e-5), (y_distor_check, y_distor)
        
    return x_nonedistor, y_nonedistor

def make_mapping_nonedistorted_image_to_distorted(
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    k1: float,
    k2: float,
    image_size: Tuple[int, int]
):
    
    map_1 = np.zeros((image_size[1], image_size[0]), dtype=np.float32) #for dim0
    map_2 = np.zeros((image_size[1], image_size[0]), dtype=np.float32) #for dim1

    for h in tqdm(range(image_size[1])):
        for w in range(image_size[0]):
            x_normed = (w - cx) / fx
            y_normed = (h - cy) / fy

            x_nonedistor, y_nonedistor = apply_distortion(x_normed, y_normed, k1, k2)

            w_nonedistor = x_nonedistor * fx + cx
            h_nonedistor = y_nonedistor * fy + cy

            map_1[h, w] = h_nonedistor
            map_2[h, w] = w_nonedistor
        
    return map_1, map_2



if __name__ == '__main__':
    ic.configureOutput(includeContext=False)
    np.set_printoptions(precision=4, suppress=True, linewidth=160)

    fx = 1262.4640877468553
    fy = 1262.4640877468553
    cx = 1920/2
    cy = 1080/2
    k1 = -0.33728837169534864
    k2 = 0.09986194299630018

    K = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]
    ])
    D = np.array([k1, k2, 0, 0, 0])

    path_image_input = osp.join('data', '1.png')
    path_map1_map2_pkl = osp.join('data', 'mapping.pkl')
    path_image_distorted_output = osp.join('data', '1_distorted.png')
    path_image_undistorted_output = osp.join('data', '1_undistorted.png')

    map1, map2 = make_mapping_nonedistorted_image_to_distorted(
        fx=fx,
        fy=fy,
        cx=cx,
        cy=cy,
        k1=k1,
        k2=k2,
        image_size=(1920, 1080)
    )
    mapping = {'map1': map1, 'map2': map2}
    with open(path_map1_map2_pkl, 'wb') as f:
        pickle.dump(mapping, f)

    with open(path_map1_map2_pkl, 'rb') as f:
        mapping_loaded = pickle.load(f)        
    map1 = mapping_loaded['map1']
    map2 = mapping_loaded['map2']
    # 1.实现一个和cv.undistort等价的函数，替换, 使用map1,map2去畸变 remap
    # 2.输出map1, map2, 遍历所有像素, 输入映射函数  opencv要查资料对照
    image_rendered = cv.imread(path_image_input)
    image_distorted = cv.remap(image_rendered, map2, map1, interpolation=cv.INTER_LINEAR)
    # image_undistorted = cv.undistort(image_distorted, K, D)
    image_undistorted = undistor_test(image_distorted, K, D)
    cv.imwrite(path_image_distorted_output, image_distorted)
    cv.imwrite(path_image_undistorted_output, image_undistorted)
