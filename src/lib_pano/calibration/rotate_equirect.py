import cv2
import numpy as np

from lib_pano.calibration.project_to_equirect import _build_equirect_dirs, _euler_deg_to_matrix


def _dirs_to_equirect_map(
    dirs: np.ndarray,
    width: int,
    height: int,
):
    x = dirs[..., 0]
    y = np.clip(dirs[..., 1], -1.0, 1.0)
    z = dirs[..., 2]

    lon = np.arctan2(z, x)
    lat = np.arcsin(y)

    map_x = ((lon / (2.0 * np.pi)) + 0.5) * width
    map_x = np.mod(map_x, width).astype(np.float32)

    
    map_y = ((lat / np.pi) + 0.5) * height
    map_y = np.clip(map_y, 0.0, height - 1.0).astype(np.float32)


    return map_x, map_y


def rotate_equirectangular_image(
    input_img: np.ndarray,
    yaw_deg: float = 0.0,
    pitch_deg: float = 0.0,
    roll_deg: float = 0.0,
    rotation_matrix: np.ndarray | None = None,
    interpolation: int = cv2.INTER_LINEAR,
    border_value=0,
):
    if input_img is None:
        raise ValueError("input_img is None")

    height, width = input_img.shape[:2]
    if rotation_matrix is None:
        rotation_matrix = _euler_deg_to_matrix(
            yaw_deg=yaw_deg,
            pitch_deg=pitch_deg,
            roll_deg=roll_deg,
        )

    dirs = np.stack(
        _build_equirect_dirs(
            out_h=height,
            out_w=width,
            rotation_matrix=rotation_matrix,
        ),
        axis=-1,
    )
    map_x, map_y = _dirs_to_equirect_map(
        dirs=dirs,
        width=width,
        height=height,
    )

    return cv2.remap(
        input_img,
        map_x,
        map_y,
        interpolation=interpolation,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=border_value,
    )


def equirect_points_to_dirs(
    points_xy: np.ndarray,
    width: int,
    height: int,
):
    points_xy = np.asarray(points_xy, dtype=np.float32)
    u = points_xy[:, 0]
    v = points_xy[:, 1]

    lon = (u / float(width) - 0.5) * (2.0 * np.pi)
    
    lat = (v / float(height) - 0.5) * np.pi
    
    cos_lat = np.cos(lat)
    x = cos_lat * np.cos(lon)
    y = np.sin(lat)
    z = cos_lat * np.sin(lon)
    dirs = np.stack([x, y, z], axis=1)
    norms = np.linalg.norm(dirs, axis=1, keepdims=True)
    return dirs / np.maximum(norms, 1e-8)


def _mask_boundary(mask: np.ndarray):
    mask_u8 = (mask > 0).astype(np.uint8) * 255
    kernel = np.ones((3, 3), dtype=np.uint8)
    eroded = cv2.erode(mask_u8, kernel, iterations=1)
    boundary = cv2.subtract(mask_u8, eroded)
    return boundary


def _sample_points(mask: np.ndarray, stride: int = 4):
    ys, xs = np.where(mask > 0)
    if stride > 1 and ys.size > 0:
        ys = ys[::stride]
        xs = xs[::stride]
    return np.stack([xs, ys], axis=1).astype(np.float32) if ys.size > 0 else np.zeros((0, 2), dtype=np.float32)


def _fit_plane_normal_through_origin(dirs: np.ndarray):
    if dirs.shape[0] < 3:
        raise ValueError("Need at least 3 points to fit a seam plane")
    _, _, vt = np.linalg.svd(dirs, full_matrices=False)
    normal = vt[-1]
    normal = normal / max(np.linalg.norm(normal), 1e-8)
    return normal.astype(np.float32)


def _rotation_matrix_from_vectors(src: np.ndarray, dst: np.ndarray):
    src = np.asarray(src, dtype=np.float32)
    dst = np.asarray(dst, dtype=np.float32)
    src = src / max(np.linalg.norm(src), 1e-8)
    dst = dst / max(np.linalg.norm(dst), 1e-8)

    v = np.cross(src, dst)
    c = float(np.clip(np.dot(src, dst), -1.0, 1.0))
    s = float(np.linalg.norm(v))

    if s < 1e-8:
        if c > 0:
            return np.eye(3, dtype=np.float32)
        aux = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        if abs(src[0]) > 0.9:
            aux = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        axis = np.cross(src, aux)
        axis = axis / max(np.linalg.norm(axis), 1e-8)
        k = np.array(
            [
                [0.0, -axis[2], axis[1]],
                [axis[2], 0.0, -axis[0]],
                [-axis[1], axis[0], 0.0],
            ],
            dtype=np.float32,
        )
        return np.eye(3, dtype=np.float32) + 2.0 * (k @ k)

    k = np.array(
        [
            [0.0, -v[2], v[1]],
            [v[2], 0.0, -v[0]],
            [-v[1], v[0], 0.0],
        ],
        dtype=np.float32,
    )
    return np.eye(3, dtype=np.float32) + k + (k @ k) * ((1.0 - c) / (s * s))


def estimate_seam_rotation_from_masks(
    mask_a: np.ndarray,
    mask_b: np.ndarray,
    boundary_band_px: int = 25,
    sample_stride: int = 4,
):
    if mask_a is None or mask_b is None:
        raise ValueError("mask_a and mask_b must not be None")
    if mask_a.shape != mask_b.shape:
        raise ValueError("mask_a and mask_b must have the same shape")

    h, w = mask_a.shape[:2]
    mask_a_u8 = (mask_a > 0).astype(np.uint8) * 255
    mask_b_u8 = (mask_b > 0).astype(np.uint8) * 255

    boundary_a = _mask_boundary(mask_a_u8)
    boundary_b = _mask_boundary(mask_b_u8)

    kernel_size = max(3, boundary_band_px * 2 + 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    near_a = cv2.bitwise_and(boundary_a, cv2.dilate(boundary_b, kernel))
    near_b = cv2.bitwise_and(boundary_b, cv2.dilate(boundary_a, kernel))
    seam_points_xy = np.concatenate(
        [
            _sample_points(near_a, stride=sample_stride),
            _sample_points(near_b, stride=sample_stride),
        ],
        axis=0,
    )

    if seam_points_xy.shape[0] < 16:
        overlap = cv2.bitwise_and(mask_a_u8, mask_b_u8)
        overlap_boundary = _mask_boundary(overlap)
        seam_points_xy = _sample_points(overlap_boundary, stride=max(1, sample_stride // 2))

    if seam_points_xy.shape[0] < 16:
        raise ValueError("Failed to extract enough seam points from the masks")

    seam_dirs = equirect_points_to_dirs(
        seam_points_xy,
        width=w,
        height=h,
    )
    seam_normal = _fit_plane_normal_through_origin(seam_dirs)

    target_up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    target_down = np.array([0.0, -1.0, 0.0], dtype=np.float32)
    if abs(np.dot(seam_normal, target_up)) <= abs(np.dot(seam_normal, target_down)):
        target_normal = target_up
    else:
        target_normal = target_down

    rotation_matrix = _rotation_matrix_from_vectors(seam_normal, target_normal)

    return {
        "rotation_matrix": rotation_matrix,
        "seam_normal": seam_normal,
        "target_normal": target_normal,
        "seam_points_xy": seam_points_xy,
        "boundary_a": boundary_a,
        "boundary_b": boundary_b,
        "near_a": near_a,
        "near_b": near_b,
    }
