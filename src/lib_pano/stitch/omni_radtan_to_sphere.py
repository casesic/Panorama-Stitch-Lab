import argparse
import time
from typing import Optional, Dict, Any

import numpy as np
import cv2
import os
import sys
import open3d as o3d


from lib_pano.io.params_read import load_single_K_and_distortion, load_all_K_and_distortion



def inverse_omni_unit_sphere(mx: np.ndarray, my: np.ndarray, xi: float, eps: float = 1e-12):
    
    r2 = mx * mx + my * my

    # term1 = 1 - xi^2 * r2
    term1 = 1.0 - (xi * xi) * r2

    # sqrt argument: r2 + term1^2
    sqrt_arg = r2 + term1 * term1
    sqrt_arg = np.maximum(sqrt_arg, eps)

    denom = xi * np.sqrt(sqrt_arg) + 1.0
    # protect against denom near zero
    denom = np.where(np.abs(denom) < eps, eps, denom)

    Z = term1 / denom

    # compute s = xi * sqrt(1 - Z^2 + (xi*Z)^2) + Z
    inner = 1.0 - Z * Z + (xi * Z) * (xi * Z)
    inner = np.maximum(inner, 0.0)
    s = xi * np.sqrt(inner) + Z

    X = mx * s
    Y = my * s

    # normalize
    norm = np.sqrt(X * X + Y * Y + Z * Z)
    norm = np.where(norm <= eps, np.nan, norm)
    X = X / norm
    Y = Y / norm
    Z = Z / norm

    return X, Y, Z

def test():
    # 在直线y=2x+3上的点
    points_2d = np.array([[0, 3], [1, 5], [2, 7], [3 ,9]], dtype=np.float64)
    X,Y,Z = inverse_omni_unit_sphere(points_2d[:,0], points_2d[:,1], xi=1.7)
    theta = np.arccos(Y) * 180 / np.pi  
    phi = np.arctan2(Z, X)
    u = (phi + np.pi) / (2.0 * np.pi) *10
    v = theta / np.pi *10
    points = np.stack([u,v], axis=-1)
    print(points)
   
   

def undistort_pixel_grid(width: int, height: int, fu: float, fv: float, cu: float, cv: float,
                                                 k1: float, k2: float, p1: float, p2: float, step: int = 1):
    # generate sampling grid (x = u, y = v) with given step
    xs = np.arange(0, width, step, dtype=np.int32)
    ys = np.arange(0, height, step, dtype=np.int32)
    grid_x, grid_y = np.meshgrid(xs, ys)  # shape Hs x Ws

    pts = np.stack([grid_x.ravel().astype(np.float32), grid_y.ravel().astype(np.float32)], axis=-1)

    K_cv = np.array([[fu, 0.0, cu], [0.0, fv, cv], [0.0, 0.0, 1.0]], dtype=np.float64)
    dist = np.array([k1, k2, p1, p2, 0.0], dtype=np.float64)

    # cv2.undistortPoints returns normalized undistorted coordinates when P is None
    pts_reshaped = pts.reshape(-1, 1, 2)
    und_norm = cv2.undistortPoints(pts_reshaped, K_cv, dist, P=None)
    und_norm = und_norm.reshape(-1, 2)
    mx = und_norm[:, 0]
    my = und_norm[:, 1]

    return mx, my, grid_x, grid_y


def visualize_omni_radtan(image_path: str, xi: float, fu: float, fv: float, cu: float, cv: float,
                          k1: float, k2: float, p1: float, p2: float,
                          width: int, height: int, downsample: int = 4,
                          visualize: bool = True, save_pcd: Optional[str] = None,
                          max_points: int = int(2e6)) -> Dict[str, Any]:

    t0 = time.time()

    img_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    ih, iw = img_rgb.shape[:2]
    if (iw != width) or (ih != height):
        print(f"Warning: provided width/height ({width},{height}) != image ({iw},{ih}). Using image size.")
        width, height = iw, ih

    # adjust downsample automatically if too many points
    est_points = (width // downsample) * (height // downsample)
    if est_points > max_points:
        factor = int(np.ceil(np.sqrt(est_points / max_points)))
        downsample = max(downsample, factor)
        print(f"Auto-increase downsample to {downsample} to limit points (~{max_points})")

    # get undistorted normalized coordinates directly (mx,my)
    mx, my, grid_x, grid_y = undistort_pixel_grid(width, height, fu, fv, cu, cv,
                                                   k1, k2, p1, p2, step=downsample)

    # inverse Omini to get unit-sphere directions
    X, Y, Z = inverse_omni_unit_sphere(mx, my, xi)

    valid_mask = np.isfinite(X) & np.isfinite(Y) & np.isfinite(Z)
    valid_mask &= (~np.isnan(X)) & (~np.isnan(Y)) & (~np.isnan(Z))
    # also ensure Z is finite and not extremely small
    valid_mask &= (np.abs(Z) > 1e-9)

    pts = np.stack([X, Y, Z], axis=1)
    pts_valid = pts[valid_mask]

    grid_x_flat = grid_x.ravel()
    grid_y_flat = grid_y.ravel()
    colors = img_rgb[grid_y_flat, grid_x_flat, :].astype(np.float32) / 255.0
    colors_valid = colors[valid_mask]

    stats = {
        'image_path': image_path,
        'width': width,
        'height': height,
        'xi': xi,
        'fu': fu,
        'fv': fv,
        'cu': cu,
        'cv': cv,
        'k1': k1,
        'k2': k2,
        'p1': p1,
        'p2': p2,
        'downsample': downsample,
        'total_sampled': int(mx.size),
    }

    stats['valid_points'] = int(np.sum(valid_mask))
    stats['invalid_points'] = int(np.sum(~valid_mask))
    stats['mapping_time_sec'] = time.time() - t0

    print(f"Total sampled pixels: {stats['total_sampled']}")
    print(f"Valid points: {stats['valid_points']}, Invalid: {stats['invalid_points']}")
    print(f"Mapping time: {stats['mapping_time_sec']:.3f}s")


    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts_valid.astype(np.float64))
    pcd.colors = o3d.utility.Vector3dVector(colors_valid.astype(np.float64))

    if save_pcd is not None:
        try:
            o3d.io.write_point_cloud(save_pcd, pcd)
            print(f"Saved point cloud to {save_pcd}")
            stats['pcd_saved'] = True
            stats['pcd_path'] = save_pcd
        except Exception as e:
            print(f"Failed to save point cloud: {e}")
            stats['pcd_saved'] = False

    if visualize:
        coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0, 0, 0])
        o3d.visualization.draw_geometries([pcd, coord], window_name='Omni-Radtan Sphere', width=1024, height=768)

    return stats

def omni_radtan_to_sphere_main(image_path: str, calib_path: str, camera_name: str):
   
    K_matrix, distortion, xi_from_file = load_single_K_and_distortion(calib_path, camera_name)
    # override parameters from file
    fu = float(K_matrix[0, 0])
    fv = float(K_matrix[1, 1])
    cu = float(K_matrix[0, 2])
    cv = float(K_matrix[1, 2])
    xi = float(xi_from_file)
    # distortion array: take first 4 as k1,k2,p1,p2 if available
    
    k1, k2, p1, p2 = [float(distortion[i]) for i in range(4)]
    
    # width/height can be passed or use image size; pass None and function will detect
    width = cu * 2
    height = cv * 2

    stats = visualize_omni_radtan(
    image_path, xi, fu, fv, cu, cv,
    k1, k2, p1, p2, width, height,
    )
    print('Done. Stats:')
    for k, v in stats.items():
        print(f"  {k}: {v}")