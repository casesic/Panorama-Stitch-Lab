import cv2
import numpy as np

from lib_pano.io.params_read import load_single_K_and_distortion
from lib_pano.io.params_read import load_single_K_and_distortion_ds


def _euler_deg_to_matrix(
    yaw_deg: float = 0.0,
    pitch_deg: float = 0.0,
    roll_deg: float = 0.0,
):
    yaw = np.deg2rad(yaw_deg)
    pitch = np.deg2rad(pitch_deg)
    roll = np.deg2rad(roll_deg)

    cy, sy = np.cos(yaw), np.sin(yaw)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cr, sr = np.cos(roll), np.sin(roll)

    rot_y = np.array(
        [
            [cy, 0.0, sy],
            [0.0, 1.0, 0.0],
            [-sy, 0.0, cy],
        ],
        dtype=np.float32,
    )
    rot_x = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, cp, -sp],
            [0.0, sp, cp],
        ],
        dtype=np.float32,
    )
    rot_z = np.array(
        [
            [cr, -sr, 0.0],
            [sr, cr, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    return (rot_z @ rot_x @ rot_y).astype(np.float32)


def _build_equirect_dirs(
    out_h: int,
    out_w: int,
    rotation_matrix: np.ndarray | None = None,
):
    u = np.arange(out_w, dtype=np.float32)
    v = np.arange(out_h, dtype=np.float32)
    uu, vv = np.meshgrid(u, v)

    lat = (vv / float(out_h) - 0.5) * np.pi
    lon = (uu / float(out_w) - 0.5) * (2.0 * np.pi)

    cos_lat = np.cos(lat)
    x = cos_lat * np.sin(lon)  
    y = np.sin(lat)
    z = cos_lat * np.cos(lon)  

    if rotation_matrix is not None:
        dirs = np.stack([x, y, z], axis=-1)
        rotated = dirs @ np.asarray(rotation_matrix, dtype=np.float32).T
        x = rotated[..., 0]
        y = rotated[..., 1]
        z = rotated[..., 2]

    return x, y, z


def omni_radtan_to_equirectangular(
    input_img,
    K,
    distortion,
    xi,
    out_w=1920,
    out_h=960,
    rotation_matrix: np.ndarray | None = None,
):
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    k1, k2, p1, p2 = distortion[0], distortion[1], distortion[2], distortion[3]

    x, y, z = _build_equirect_dirs(
        out_h=out_h,
        out_w=out_w,
        rotation_matrix=rotation_matrix,
    )

    denom = z + xi
    mx = x / denom
    my = y / denom

    r2 = mx * mx + my * my
    radial = 1.0 + k1 * r2 + k2 * r2 * r2
    x_dist = mx * radial + 2.0 * p1 * mx * my + p2 * (r2 + 2.0 * mx * mx)
    y_dist = my * radial + p1 * (r2 + 2.0 * my * my) + 2.0 * p2 * mx * my

    map_x = (fx * x_dist + cx).astype(np.float32)
    map_y = (fy * y_dist + cy).astype(np.float32)
    map_x[z <= 0] = -1.0
    map_y[z <= 0] = -1.0

    return cv2.remap(
        input_img,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
    )


def double_sphere_to_equirectangular(
    input_img,
    K,
    xi,
    alpha,
    out_w=1920,
    out_h=960,
    rotation_matrix: np.ndarray | None = None,
):
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    x, y, z = _build_equirect_dirs(
        out_h=out_h,
        out_w=out_w,
        rotation_matrix=rotation_matrix,
    )

    r2 = x * x + y * y
    d1 = np.sqrt(r2 + z * z)
    z_prime = z + xi * d1
    d2 = np.sqrt(r2 + z_prime * z_prime)
    denom = alpha * d2 + (1.0 - alpha) * z_prime

    mx = x / denom
    my = y / denom

    map_x = (fx * mx + cx).astype(np.float32)
    map_y = (fy * my + cy).astype(np.float32)
    map_x[z <= -xi] = -1.0
    map_y[z <= -xi] = -1.0

    return cv2.remap(
        input_img,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
    )


def test_main(
    calib_path: str,
    camera_name: str,
    image_path: str,
    out_path: str,
):
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError("Failed to load image: %s" % image_path)

    if calib_path.lower().endswith((".yaml", ".yml")):
        K, _, xi, alpha, _ = load_single_K_and_distortion_ds(calib_path, camera_name)
        equi = double_sphere_to_equirectangular(
            input_img=img,
            K=K,
            xi=xi,
            alpha=alpha,
            out_w=1920,
            out_h=960,
        )
    else:
        K, distortion, xi, _ = load_single_K_and_distortion(calib_path, camera_name)
        equi = omni_radtan_to_equirectangular(
            input_img=img,
            K=K,
            distortion=distortion,
            xi=xi,
            out_w=1920,
            out_h=960,
        )

    cv2.imwrite(out_path, equi)
    print(f"Saved equirect to {out_path} ")
