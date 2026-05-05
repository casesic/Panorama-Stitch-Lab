from __future__ import annotations

import cv2
import numpy as np


CAM2_COLOR = (0, 255, 0)
CAM4_COLOR = (0, 255, 255)
LINE_COLOR = (220, 220, 220)


def make_scaled_shape(width: int, height: int, vis_scale: float) -> tuple[int, int]:
    return max(1, int(round(width * vis_scale))), max(1, int(round(height * vis_scale)))


def points_in_bounds(pts: np.ndarray, width: int, height: int) -> np.ndarray:
    finite = np.isfinite(pts).all(axis=1)
    return (
        finite
        & (pts[:, 0] >= 0)
        & (pts[:, 0] < width)
        & (pts[:, 1] >= 0)
        & (pts[:, 1] < height)
    )


def sample_line_indices(num_pairs: int, max_lines: int, sample_mode: str, seed: int) -> np.ndarray:
    if max_lines <= 0 or num_pairs == 0:
        return np.empty((0,), dtype=np.int64)
    if num_pairs <= max_lines:
        return np.arange(num_pairs, dtype=np.int64)
    if sample_mode == "uniform":
        return np.linspace(0, num_pairs - 1, max_lines, dtype=np.int64)

    rng = np.random.default_rng(seed)
    sampled = rng.choice(num_pairs, size=max_lines, replace=False)
    sampled.sort()
    return sampled.astype(np.int64, copy=False)


def draw_corr_points(
    image: np.ndarray,
    points: np.ndarray,
    *,
    color: tuple[int, int, int],
    radius: int,
) -> np.ndarray:
    canvas = image.copy()
    h, w = canvas.shape[:2]
    draw_pts = np.rint(points).astype(np.int32)
    for x, y in draw_pts:
        if 0 <= x < w and 0 <= y < h:
            cv2.circle(canvas, (int(x), int(y)), radius, color, -1, lineType=cv2.LINE_AA)
    return canvas


def draw_corr_lines(
    image: np.ndarray,
    points_a: np.ndarray,
    points_b: np.ndarray,
    *,
    color: tuple[int, int, int],
    thickness: int,
) -> np.ndarray:
    canvas = image.copy()
    h, w = canvas.shape[:2]
    for p_a, p_b in zip(points_a, points_b):
        p_a_draw = (int(round(p_a[0])), int(round(p_a[1])))
        p_b_draw = (int(round(p_b[0])), int(round(p_b[1])))
        if (
            0 <= p_a_draw[0] < w
            and 0 <= p_a_draw[1] < h
            and 0 <= p_b_draw[0] < w
            and 0 <= p_b_draw[1] < h
        ):
            cv2.line(canvas, p_a_draw, p_b_draw, color, thickness, lineType=cv2.LINE_AA)
    return canvas


def draw_points_canvas(
    pts: np.ndarray,
    width: int,
    height: int,
    vis_scale: float,
    point_radius: int,
    color: tuple[int, int, int],
) -> np.ndarray:
    scaled_w, scaled_h = make_scaled_shape(width, height, vis_scale)
    canvas = np.zeros((scaled_h, scaled_w, 3), dtype=np.uint8)
    valid_pts = pts[points_in_bounds(pts, width, height)]
    radius = max(1, int(round(point_radius * vis_scale)))
    return draw_corr_points(canvas, valid_pts * vis_scale, color=color, radius=radius)


def draw_points_on_bg(
    pts: np.ndarray,
    bg: np.ndarray,
    width: int,
    height: int,
    vis_scale: float,
    point_radius: int,
    color: tuple[int, int, int],
) -> np.ndarray:
    scaled_w, scaled_h = make_scaled_shape(width, height, vis_scale)
    canvas = cv2.resize(bg, (scaled_w, scaled_h), interpolation=cv2.INTER_AREA) if vis_scale != 1.0 else bg.copy()
    valid_pts = pts[points_in_bounds(pts, width, height)]
    radius = max(1, int(round(point_radius * vis_scale)))
    return draw_corr_points(canvas, valid_pts * vis_scale, color=color, radius=radius)


def draw_corr_pair_overlay(
    pairs: np.ndarray,
    width: int,
    height: int,
    vis_scale: float,
    point_radius: int,
    line_thickness: int,
    max_lines: int,
    sample_mode: str,
    seed: int,
) -> np.ndarray:
    scaled_w, scaled_h = make_scaled_shape(width, height, vis_scale)
    canvas = np.zeros((scaled_h, scaled_w, 3), dtype=np.uint8)

    cam2_pts = pairs[:, 0:2]
    cam4_pts = pairs[:, 2:4]
    valid_cam2 = points_in_bounds(cam2_pts, width, height)
    valid_cam4 = points_in_bounds(cam4_pts, width, height)

    radius = max(1, int(round(point_radius * vis_scale)))
    thickness = max(1, int(round(line_thickness * vis_scale)))

    canvas = draw_corr_points(canvas, cam2_pts[valid_cam2] * vis_scale, color=CAM2_COLOR, radius=radius)
    canvas = draw_corr_points(canvas, cam4_pts[valid_cam4] * vis_scale, color=CAM4_COLOR, radius=radius)

    for row_idx in sample_line_indices(len(pairs), max_lines, sample_mode, seed):
        if not (valid_cam2[row_idx] and valid_cam4[row_idx]):
            continue

        p2 = cam2_pts[row_idx]
        p4 = cam4_pts[row_idx]
        p2_draw = (int(round(p2[0] * vis_scale)), int(round(p2[1] * vis_scale)))
        p4_draw = (int(round(p4[0] * vis_scale)), int(round(p4[1] * vis_scale)))
        cv2.line(canvas, p2_draw, p4_draw, LINE_COLOR, thickness, lineType=cv2.LINE_AA)
        cv2.circle(canvas, p2_draw, radius, CAM2_COLOR, -1, lineType=cv2.LINE_AA)
        cv2.circle(canvas, p4_draw, radius, CAM4_COLOR, -1, lineType=cv2.LINE_AA)

    return canvas


def draw_corr_side_by_side(
    pairs: np.ndarray,
    width: int,
    height: int,
    vis_scale: float,
    point_radius: int,
    line_thickness: int,
    max_lines: int,
    sample_mode: str,
    seed: int,
) -> np.ndarray:
    scaled_w, scaled_h = make_scaled_shape(width * 2, height, vis_scale)
    canvas = np.zeros((scaled_h, scaled_w, 3), dtype=np.uint8)

    cam2_pts = pairs[:, 0:2]
    cam4_pts = pairs[:, 2:4]
    valid_cam2 = points_in_bounds(cam2_pts, width, height)
    valid_cam4 = points_in_bounds(cam4_pts, width, height)

    radius = max(1, int(round(point_radius * vis_scale)))
    thickness = max(1, int(round(line_thickness * vis_scale)))

    canvas = draw_corr_points(canvas, cam2_pts[valid_cam2] * vis_scale, color=CAM2_COLOR, radius=radius)
    cam4_offset = np.array([width, 0.0])
    canvas = draw_corr_points(canvas, (cam4_pts[valid_cam4] + cam4_offset) * vis_scale, color=CAM4_COLOR, radius=radius)

    for row_idx in sample_line_indices(len(pairs), max_lines, sample_mode, seed):
        if not (valid_cam2[row_idx] and valid_cam4[row_idx]):
            continue

        p2 = cam2_pts[row_idx]
        p4 = cam4_pts[row_idx] + cam4_offset
        p2_draw = (int(round(p2[0] * vis_scale)), int(round(p2[1] * vis_scale)))
        p4_draw = (int(round(p4[0] * vis_scale)), int(round(p4[1] * vis_scale)))
        cv2.line(canvas, p2_draw, p4_draw, LINE_COLOR, thickness, lineType=cv2.LINE_AA)
        cv2.circle(canvas, p2_draw, radius, CAM2_COLOR, -1, lineType=cv2.LINE_AA)
        cv2.circle(canvas, p4_draw, radius, CAM4_COLOR, -1, lineType=cv2.LINE_AA)

    return canvas


def maybe_color_by_vertex_id(vertex_ids: np.ndarray, default_color: tuple[int, int, int]) -> list[tuple[int, int, int]]:
    return [default_color for _ in range(len(vertex_ids))]
