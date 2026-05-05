from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from lib_pano.flow.semantic_flow_io import save_flow_visualization_paths


def visualize_projected_vertices(img, pts_2d, color=(0, 255, 0), radius=2):
    canvas = img.copy()
    if pts_2d is None or len(pts_2d) == 0:
        return canvas

    h, w = canvas.shape[:2]
    pts_px = np.rint(np.asarray(pts_2d, dtype=np.float64)).astype(np.int32)
    for u, v in pts_px:
        if 0 <= u < w and 0 <= v < h:
            cv2.circle(canvas, (u, v), radius, color, -1)
    return canvas


def visualize_flow_color(flow_roi, confidence_roi):
    flow_mag = np.linalg.norm(flow_roi, axis=2)
    flow_angle = np.arctan2(flow_roi[..., 1], flow_roi[..., 0])

    hsv = np.zeros((flow_roi.shape[0], flow_roi.shape[1], 3), dtype=np.uint8)
    hsv[..., 0] = ((flow_angle + np.pi) / (2 * np.pi) * 179).astype(np.uint8)
    hsv[..., 1] = 255
    max_mag = max(float(flow_mag.max()), 1e-6)
    hsv[..., 2] = (np.clip(flow_mag / max_mag, 0.0, 1.0) * 255).astype(np.uint8)
    flow_color = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    flow_color[confidence_roi == 0] = 0
    return flow_color


def visualize_flow_arrows(flow_roi, confidence_roi):
    flow_arrows = np.zeros((flow_roi.shape[0], flow_roi.shape[1], 3), dtype=np.uint8)
    step = 20
    for r in range(0, flow_roi.shape[0], step):
        for c in range(0, flow_roi.shape[1], step):
            conf = int(confidence_roi[r, c])
            if conf == 0:
                continue
            dx, dy = flow_roi[r, c]
            color = (0, 255, 0) if conf == 2 else (255, 0, 0)
            cv2.arrowedLine(
                flow_arrows,
                (c, r),
                (int(round(c + dx)), int(round(r + dy))),
                color,
                1,
                tipLength=0.3,
            )
    return flow_arrows


def visualize_flow_overlay(image, flow_roi, confidence_roi):
    flow_color = visualize_flow_color(flow_roi, confidence_roi)
    if image.shape[:2] != flow_color.shape[:2]:
        return flow_color
    return cv2.addWeighted(image, 0.5, flow_color, 0.5, 0.0)


def save_flow_debug_images(
    frame_output_dir: Path,
    img_cam2,
    img_cam4,
    pts_2d_cam2,
    pts_2d_cam4,
    flow_results: dict[str, tuple[np.ndarray, np.ndarray] | None],
) -> None:
    paths = save_flow_visualization_paths(frame_output_dir)

    mesh_cam2 = visualize_projected_vertices(img_cam2, pts_2d_cam2)
    mesh_cam4 = visualize_projected_vertices(img_cam4, pts_2d_cam4)
    cv2.imwrite(str(paths["mesh_cam2"]), mesh_cam2, [cv2.IMWRITE_JPEG_QUALITY, 95])
    cv2.imwrite(str(paths["mesh_cam4"]), mesh_cam4, [cv2.IMWRITE_JPEG_QUALITY, 95])

    c2_to_c4 = flow_results.get("c2_to_c4")
    if c2_to_c4 is not None:
        flow_c2c4, conf_c2c4 = c2_to_c4
        cv2.imwrite(str(paths["flow_color_c2_to_c4"]), visualize_flow_color(flow_c2c4, conf_c2c4))
        cv2.imwrite(str(paths["flow_arrows_c2_to_c4"]), visualize_flow_arrows(flow_c2c4, conf_c2c4))

    c4_to_c2 = flow_results.get("c4_to_c2")
    if c4_to_c2 is not None:
        flow_c4c2, conf_c4c2 = c4_to_c2
        cv2.imwrite(str(paths["flow_color_c4_to_c2"]), visualize_flow_color(flow_c4c2, conf_c4c2))
        cv2.imwrite(str(paths["flow_arrows_c4_to_c2"]), visualize_flow_arrows(flow_c4c2, conf_c4c2))
