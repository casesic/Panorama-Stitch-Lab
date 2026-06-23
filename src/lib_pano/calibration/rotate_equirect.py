"""对等距柱状图施加 3D 旋转，通过光线重投影实现全景图的视角旋转。"""

import cv2
import numpy as np

def _build_equirect_dirs(
    out_h: int,
    out_w: int,
    rotation_matrix: np.ndarray | None = None,
):
    """Generate ray direction vectors (x,y,z) for each pixel of an equirectangular image."""
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

def _dirs_to_equirect_map(
    dirs: np.ndarray,
    width: int,
    height: int,
):
    """Convert ray direction vectors to equirectangular pixel coordinates (map_x, map_y)."""
    x = dirs[..., 0]
    y = np.clip(dirs[..., 1], -1.0, 1.0)
    z = dirs[..., 2]

    lon = np.arctan2(x, z)
    lat = np.arcsin(y)

    map_x = ((lon / (2.0 * np.pi)) + 0.5) * width
    map_y = ((lat / np.pi) + 0.5) * height

    return map_x, map_y


def rotate_equirectangular_image(
    input_img: np.ndarray,
    rotation_matrix: np.ndarray | None = None,
    interpolation: int = cv2.INTER_LINEAR,
    border_value=0,
):
    """Apply a 3D rotation to an equirectangular image via ray reprojection."""
    if input_img is None:
        raise ValueError("input_img is None")

    height, width = input_img.shape[:2]

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
    # ERP rotation must wrap horizontally because longitude is periodic.
    map_x = np.mod(map_x, width).astype(np.float32)
    map_y = np.clip(map_y, 0.0, height - 1.0).astype(np.float32)

    return cv2.remap(
        input_img,
        map_x,
        map_y,
        interpolation=interpolation,
        # Keeps linear sampling from falling back to a black border at x=width.
        borderMode=cv2.BORDER_WRAP,
        borderValue=border_value,
    )
