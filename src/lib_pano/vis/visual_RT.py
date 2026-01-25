import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from lib_pano.io.params_read import load_all_K_and_distortion, load_R_and_T, load_R_and_T_2
import cv2 as cv
import os

def _compute_frustum_corners_cam(K, depth):
    
    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]
    w = cx * 2.0
    h = cy * 2.0

    corners_pix = np.array([[0.0, 0.0], [w, 0.0], [w, h], [0.0, h]])
    corners_cam = []
    for (u, v) in corners_pix:
        x = (u - cx) / fx
        y = (v - cy) / fy
        z = 1.0
        dir = np.array([x, y, z])
        dir = dir / np.linalg.norm(dir)
        p = dir * depth
        corners_cam.append(p)

    return np.asarray(corners_cam)


def visualize_cameras(K_list, R_list, T_list , names=None, colors=None, frustum_depth=1.5, save_path=None, world_align=None):
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    default_colors = ['tab:red', 'tab:green', 'tab:blue', 'tab:orange', 'tab:purple', 'tab:cyan']

    all_points = []
    for i, (K, R, T) in enumerate(zip(K_list, R_list, T_list)):
        R = np.asarray(R).reshape(3, 3)
        T = np.asarray(T).reshape(3)
        C = T.copy()  # camera center in world coords
        # optionally align world coordinates to a different plotting frame
        if world_align is not None:
            world_align = np.asarray(world_align).reshape(3, 3)
            R_plot = world_align @ R
            T_plot = world_align @ T
            C_plot = world_align @ T_plot.copy()
        else:
            R_plot = R
            T_plot = T
            C_plot = C

        col = colors[i] if (colors and i < len(colors)) else default_colors[i % len(default_colors)]
        name = names[i] if (names and i < len(names)) else f'cam_{i}'

        # draw camera center and label
        ax.scatter([C_plot[0]], [C_plot[1]], [C_plot[2]], color=col, s=40)
        ax.text(C_plot[0], C_plot[1], C_plot[2], name, color=col)

        # draw camera axes (camera->world)
        x_axis = R_plot @ np.array([1.0, 0.0, 0.0])
        y_axis = R_plot @ np.array([0.0, 1.0, 0.0])
        z_axis = R_plot @ np.array([0.0, 0.0, 1.0])
        print(f"Camera {i} axes (world coords): x={x_axis}, y={y_axis}, z={z_axis}")

        ax.quiver(C[0], C[1], C[2], x_axis[0], x_axis[1], x_axis[2], length=frustum_depth * 0.25, color='r')
        ax.quiver(C[0], C[1], C[2], y_axis[0], y_axis[1], y_axis[2], length=frustum_depth * 0.25, color='g')
        ax.quiver(C[0], C[1], C[2], z_axis[0], z_axis[1], z_axis[2], length=frustum_depth * 0.25, color='b')

        # frustum: compute corners in camera frame and transform to world
        corners_cam = _compute_frustum_corners_cam(K, frustum_depth)
        corners_world = (R @ corners_cam.T).T + T  # (4,3)
        if world_align is not None:
            corners_world = (world_align @ corners_world.T).T

        # draw frustum edges (from center to corners)
        for corner in corners_world:
            ax.plot([C_plot[0], corner[0]], [C_plot[1], corner[1]], [C_plot[2], corner[2]], color=col)

        # connect the rectangle of the far plane
        idx = [0, 1, 2, 3, 0]
        for a, b in zip(idx[:-1], idx[1:]):
            pA = corners_world[a]
            pB = corners_world[b]
            ax.plot([pA[0], pB[0]], [pA[1], pB[1]], [pA[2], pB[2]], color=col, linestyle='--')

        all_points.append(C)
        all_points.extend(corners_world.tolist())

    # set labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Camera poses and view frustums')

    # set equal aspect by computing bounds
    all_pts = np.asarray(all_points)
    if all_pts.size > 0:
        x_min, y_min, z_min = np.min(all_pts, axis=0)
        x_max, y_max, z_max = np.max(all_pts, axis=0)
        x_mean = 0.5 * (x_min + x_max)
        y_mean = 0.5 * (y_min + y_max)
        z_mean = 0.5 * (z_min + z_max)
        max_range = max(x_max - x_min, y_max - y_min, z_max - z_min)
        if max_range == 0:
            max_range = 1.0
        half = max_range * 0.6
        ax.set_xlim(x_mean - half, x_mean + half)
        ax.set_ylim(y_mean - half, y_mean + half)
        ax.set_zlim(z_mean - half, z_mean + half)

    if save_path:
        plt.savefig(save_path, dpi=200)
        print(f"Saved visualization to {save_path}")
    else:
        plt.show()


def divide_by_y_scopes(image_path: str, y_scopes: list, output_path):
    # 利用y_scopes对图像进行画线
    image = cv.imread(image_path)
    h, w = image.shape[0], image.shape[1]
    for y_scope in y_scopes:
        y_start, y_end = y_scope
        y_start = int(y_start / 960.0 * h)
        y_end = int(y_end / 960.0 * h)
        cv.line(image, (0, y_start), (w, y_start), (0, 0, 255), 2)
        cv.line(image, (0, y_end), (w, y_end), (0, 0, 255), 2)
    cv.imwrite(output_path, image)

def circle_center(image_path: str, output_path: str, color: tuple, center: tuple = None):    
    image = cv.imread(image_path)
    h, w = image.shape[0], image.shape[1]
    if center is None:
        center_x = w // 2
        center_y = h // 2
        center = (center_x, center_y)
    cv.circle(image, center, radius=5, color=color, thickness=5)
    cv.imwrite(output_path, image)

def circle_center_images(image_paths: list, output_paths: list,  calib_path: str = None):
    palette = [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)]
    if calib_path is not None:
        camera_names, K_list, distortion_list, xi_list = load_all_K_and_distortion(calib_path)
    for idx, (img_path, out_path) in enumerate(zip(image_paths, output_paths)):
        if calib_path is not None:
            K = K_list[idx]
            cx, cy = K[0, 2], K[1, 2]
            center = (int(cx), int(cy))
           
        color = palette[idx % len(palette)]
        color_bgr = (int(color[2] * 255), int(color[1] * 255), int(color[0] * 255))
        circle_center(img_path, out_path, color_bgr, center=center)
    