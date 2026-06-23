"""鱼眼图像到等距柱状全景的整图转换：LUT 构建、remap、valid mask 生成。"""

from __future__ import annotations

import cv2
import numpy as np
from tqdm import tqdm

from lib_pano.camera.ds_camera_params import DS_PARAMS, DS_PARAMS_CAM2, DS_PARAMS_CAM4, project


ERP_WIDTH = 5760
ERP_HEIGHT = 2880
OUTPUT_WIDTH = 5760
OUTPUT_HEIGHT = 2880
ROTATION_MATRIX = np.array(
    [1, 0, 0, 0, 0, 1, 0, -1, 0],
    dtype=np.float64,
).reshape(3, 3)
R_EXTRINSIC_CAM2 = np.eye(3, dtype=np.float64)
R_EXTRINSIC_CAM4 = np.array( 
    [-1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 1.0],
    dtype=np.float64,
).reshape(3, 3)


def select_camera_params(camera_group: str) -> tuple[dict[str, float], dict[str, float]]:
    """Return DS parameter sets for cam2 and cam4."""
    if camera_group == "new":
        return DS_PARAMS_CAM2, DS_PARAMS_CAM4
    return DS_PARAMS, DS_PARAMS


def build_equirect_lut(
    ds_params: dict[str, float],
    width: int = ERP_WIDTH,
    height: int = ERP_HEIGHT,
    chunk_rows: int = 128,
    R_extrinsic: np.ndarray | None = None,
    desc: str = "Building LUT",
) -> tuple[np.ndarray, np.ndarray]:
    """Build an equirect-to-fisheye remap LUT once for a given camera."""
    map_x = np.full((height, width), -1.0, dtype=np.float32)
    map_y = np.full((height, width), -1.0, dtype=np.float32)

    u_coords = np.arange(width, dtype=np.float64)
    lon = (u_coords / float(width)) * (2.0 * np.pi) - np.pi
    sin_lon = np.sin(lon)
    cos_lon = np.cos(lon)

    for row_start in tqdm(
        range(0, height, chunk_rows),
        desc=desc,
        unit="chunk",
        leave=False,
    ):
        row_end = min(row_start + chunk_rows, height)
        v_coords = np.arange(row_start, row_end, dtype=np.float64)
        lat = (v_coords / float(height)) * np.pi - (np.pi / 2.0)

        cos_lat = np.cos(lat)[:, None]
        sin_lat = np.sin(lat)[:, None]

        ray = np.stack(
            [
                cos_lat * sin_lon[None, :],
                np.broadcast_to(sin_lat, (row_end - row_start, width)),
                cos_lat * cos_lon[None, :],
            ],
            axis=-1,
        )
        if R_extrinsic is not None:
            ray_rot = ray.reshape(-1, 3) @ R_extrinsic.T @ ROTATION_MATRIX.T
        else:
            ray_rot = ray.reshape(-1, 3) @ ROTATION_MATRIX.T
        fisheye_pixels, valid_mask = project(ray_rot, ds_params)

        pixels_chunk = fisheye_pixels.reshape(row_end - row_start, width, 2)
        valid_chunk = valid_mask.reshape(row_end - row_start, width)
        map_x_chunk = map_x[row_start:row_end]
        map_y_chunk = map_y[row_start:row_end]
        map_x_chunk[valid_chunk] = pixels_chunk[..., 0][valid_chunk]
        map_y_chunk[valid_chunk] = pixels_chunk[..., 1][valid_chunk]

    return map_x, map_y


def convert_fisheye_to_equirect(
    frame: np.ndarray,
    map_x: np.ndarray,
    map_y: np.ndarray,
) -> np.ndarray:
    """Remap one fisheye frame to equirect and resize for output."""
    equirect = cv2.remap(
        frame,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    return cv2.resize(equirect, (OUTPUT_WIDTH, OUTPUT_HEIGHT), interpolation=cv2.INTER_AREA)


def encode_jpeg(frame: np.ndarray, quality: int = 95) -> bytes | None:
    """Encode one frame to JPEG bytes."""
    ok, encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        return None
    return encoded.tobytes()


def build_fisheye_lens_mask(
    fisheye_image: np.ndarray,
    threshold: int = 5,
    close_kernel: int = 51,
    erode_px: int = 8,
) -> np.ndarray:
    """Detect the filled fisheye lens imaging area from a source frame."""
    gray = cv2.cvtColor(fisheye_image, cv2.COLOR_BGR2GRAY)
    rough = (gray > threshold).astype(np.uint8) * 255

    close_op_kernel = None
    if close_kernel > 1:
        kernel_size = close_kernel
        if kernel_size % 2 == 0:
            kernel_size += 1
        close_op_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (kernel_size, kernel_size),
        )
        rough = cv2.morphologyEx(rough, cv2.MORPH_CLOSE, close_op_kernel)

    contours, _ = cv2.findContours(rough, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise RuntimeError("Failed to detect fisheye lens contour.")

    largest_contour = max(contours, key=cv2.contourArea)
    lens_mask = np.zeros_like(gray, dtype=np.uint8)
    cv2.drawContours(lens_mask, [largest_contour], -1, 255, thickness=cv2.FILLED)

    if close_op_kernel is not None:
        lens_mask = cv2.morphologyEx(lens_mask, cv2.MORPH_CLOSE, close_op_kernel)

    if erode_px > 0:
        erode_size = 2 * erode_px + 1
        erode_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (erode_size, erode_size),
        )
        lens_mask = cv2.erode(lens_mask, erode_kernel, iterations=1)

    return lens_mask


def build_erp_valid_mask_from_lut_and_lens(
    lut: tuple[np.ndarray, np.ndarray],
    fisheye_lens_mask: np.ndarray,
    output_size: tuple[int, int],
) -> np.ndarray:
    """Build an ERP valid mask from LUT validity and the fisheye lens area."""
    map_x, map_y = lut
    lut_valid = ((map_x >= 0) & (map_y >= 0)).astype(np.uint8) * 255
    lens_valid = cv2.remap(
        fisheye_lens_mask,
        map_x,
        map_y,
        interpolation=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )

    valid = cv2.bitwise_and(lut_valid, lens_valid)

    out_w, out_h = output_size
    if valid.shape[1] != out_w or valid.shape[0] != out_h:
        valid = cv2.resize(valid, (out_w, out_h), interpolation=cv2.INTER_NEAREST)
    return valid


def build_valid_mask_from_lut(
    lut: tuple[np.ndarray, np.ndarray],
    output_size: tuple[int, int],
) -> np.ndarray:
    """Build a binary ERP valid mask from LUT coverage (pixels where the LUT maps to a fisheye source)."""
    map_x, map_y = lut
    valid = ((map_x >= 0) & (map_y >= 0)).astype(np.uint8) * 255
    out_w, out_h = output_size
    if valid.shape[1] != out_w or valid.shape[0] != out_h:
        valid = cv2.resize(valid, (out_w, out_h), interpolation=cv2.INTER_NEAREST)
    return valid


def make_horizontal_flow_valid_mask(
    valid_mask: np.ndarray,
    camera_name: str,
    trim_edge_margin_px: int = 40,
    trim_edge_percentile: float = 5.0,
) -> np.ndarray:
    """Trim seam-side horizontal ERP mask edges for optical-flow inputs."""
    valid = valid_mask > 0
    h, w = valid.shape[:2]
    if not np.any(valid):
        return valid_mask

    grid_y = np.arange(h)[:, None]

    if camera_name == "cam2":
        bottom_values = []
        for x in range(w):
            ys = np.where(valid[:, x])[0]
            if len(ys) > 0:
                bottom_values.append(int(ys.max()))

        if len(bottom_values) == 0:
            return valid_mask

        safe_bottom = int(np.percentile(bottom_values, trim_edge_percentile))
        safe_bottom = max(0, safe_bottom - trim_edge_margin_px)
        flow_valid = valid & (grid_y <= safe_bottom)

    elif camera_name == "cam4":
        top_values = []
        for x in range(w):
            ys = np.where(valid[:, x])[0]
            if len(ys) > 0:
                top_values.append(int(ys.min()))

        if len(top_values) == 0:
            return valid_mask

        safe_top = int(np.percentile(top_values, 100.0 - trim_edge_percentile))
        safe_top = min(h - 1, safe_top + trim_edge_margin_px)
        flow_valid = valid & (grid_y >= safe_top)

    else:
        raise ValueError(f"Unsupported camera_name: {camera_name}")

    return flow_valid.astype(np.uint8) * 255


def apply_valid_mask_to_image(
    image: np.ndarray,
    valid_mask: np.ndarray,
    background_value: int = 0,
) -> np.ndarray:
    """Set pixels outside valid_mask to background_value."""
    masked = image.copy()
    masked[valid_mask <= 0] = background_value
    return masked
