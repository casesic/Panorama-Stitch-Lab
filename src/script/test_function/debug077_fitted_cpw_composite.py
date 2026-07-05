from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

import config_script  # noqa: F401 — ensure src/ on sys.path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import cv2
import numpy as np

from lib_pano.human_vis.smplx_forward_vis import (
    SMPLXForwarder,
    _load_pickle,
    unpack_smplx_result,
)
from lib_pano.human_vis.dual_erp_vis import (
    project_cam2,
    project_points_erp,
)
from lib_pano.composite.cpw_warp import (
    directional_erode_valid_mask,
    generate_cpw_backward_map,
    target_priority_composite,
    warp_image_with_backward_map,
)


PROJECT_ROOT = Path(config_script.PROJECT_ROOT)
ERP_W = 5760
ERP_H = 2880
NUM_DENSEKP = 138

CPW_DIRECTIONS = ("cam4_to_cam2", "cam2_to_cam4")

CPW_MIN_POINTS = 8
CPW_PAD = 180
CPW_KNN = 12
CPW_POWER = 2.0

COMPOSITE_BLACK_THRESH = 3
DIR_ERODE_PX_CAM2 = 15
DIR_ERODE_PX_CAM4 = 15
DEBUG_PANEL_WIDTH = 420


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fitted DenseKP138 双方 ERP valid-region 筛点对（CPW 链路验证第一版）。"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="数据集名称，例如 20260311_riding 或 20260320_riding_dataset1。",
    )
    parser.add_argument(
        "--segment",
        type=str,
        default="513_518",
        help="Fitting segment name，例如 513_518；或 all（扫描 RESULT_DIR/cam2 下所有 seg_*_seq.pkl）。",
    )
    parser.add_argument(
        "--directions",
        type=str,
        choices=["both", *CPW_DIRECTIONS],
        default="both",
        help="CPW warp 方向：both / cam4_to_cam2 / cam2_to_cam4（默认 both）。",
    )
    parser.add_argument(
        "--fitting_dir",
        type=str,
        default=None,
        help="Fitting 结果根目录（含 cam2/ cam4/ 的 seg_*_seq.pkl）。默认 data_process/Output/<dataset>/fitting/dual_cam_sam3d_dense。",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="输出根目录，下设 data/ 与各方向子目录。默认 data_process/Vis/debug077_fitted_densekp138_valid_cpw/<dataset>。",
    )
    return parser.parse_args()


def get_direction_pairs(directions: str) -> list[tuple[str, str, str]]:
    """把 --directions 解析为 (src_cam, tgt_cam, direction_name) 列表。"""
    table = {
        "cam4_to_cam2": ("cam4", "cam2", "cam4_to_cam2"),
        "cam2_to_cam4": ("cam2", "cam4", "cam2_to_cam4"),
    }
    if directions == "both":
        return [table["cam4_to_cam2"], table["cam2_to_cam4"]]
    return [table[directions]]


def parse_segment_range(segment: str) -> tuple[int, int]:
    parts = segment.split("_")
    if len(parts) != 2:
        raise ValueError(f"segment must match start_end, got: {segment}")
    start_frame, end_frame = int(parts[0]), int(parts[1])
    if start_frame > end_frame:
        raise ValueError(f"segment start must be <= end: {segment}")
    return start_frame, end_frame


def find_image(image_dir: Path, cam_id: str, frame_idx: int) -> Path | None:
    for suffix in (".jpg", ".png"):
        p = image_dir / cam_id / f"{frame_idx:05d}{suffix}"
        if p.exists():
            return p
    return None


def compute_valid_mask(
    bg_path: Path | None,
    black_thresh: int = 5,
    erode_px: int = 10,
) -> np.ndarray | None:
    """与 debug052 一致：max(RGB)<=thresh 的黑像素中，仅剔除与图像边界连通的部分，
    内部深色区域保留为 valid，再用 (2*erode_px+1) 方核向内 erosion。读图失败返回 None。"""
    if bg_path is None:
        return None
    img = cv2.imread(str(bg_path), cv2.IMREAD_COLOR)
    if img is None:
        return None

    black = img.max(axis=2) <= black_thresh  # 黑色候选

    # 只把与边界连通的黑区视为 invalid（内部深色衣物/头发/阴影保留）
    num_labels, labels = cv2.connectedComponents(
        black.astype(np.uint8), connectivity=8
    )
    border_labels = set()
    border_labels.update(labels[0, :].tolist())
    border_labels.update(labels[-1, :].tolist())
    border_labels.update(labels[:, 0].tolist())
    border_labels.update(labels[:, -1].tolist())
    border_labels.discard(0)  # label 0 = 非黑背景

    invalid = np.isin(labels, list(border_labels)) if border_labels else np.zeros_like(black)
    valid = ~invalid

    ksize = 2 * erode_px + 1
    kernel = np.ones((ksize, ksize), dtype=np.uint8)
    valid_eroded = cv2.erode(valid.astype(np.uint8), kernel) > 0
    return valid_eroded


def check_points_in_mask(
    points_2d: np.ndarray,
    valid_mask: np.ndarray | None,
) -> np.ndarray:
    """valid_mask 为 None 全 False；坐标非有限或越界 False；落在 valid 内 True。"""
    n = points_2d.shape[0]
    result = np.zeros(n, dtype=bool)
    if valid_mask is None:
        return result

    h, w = valid_mask.shape[:2]
    finite = np.isfinite(points_2d).all(axis=1)

    xs = np.zeros(n, dtype=np.int64)
    ys = np.zeros(n, dtype=np.int64)
    xs[finite] = np.round(points_2d[finite, 0]).astype(np.int64)
    ys[finite] = np.round(points_2d[finite, 1]).astype(np.int64)

    in_bounds = finite & (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)
    idx = np.where(in_bounds)[0]
    result[idx] = valid_mask[ys[idx], xs[idx]].astype(bool)
    return result


def resize_to_width(image: np.ndarray, width: int) -> np.ndarray:
    h, w = image.shape[:2]
    if w == width:
        return image
    scale = width / w
    new_h = max(1, int(round(h * scale)))
    return cv2.resize(image, (width, new_h), interpolation=cv2.INTER_AREA)


def draw_projected_points(
    image: np.ndarray,
    points_2d: np.ndarray,
    valid: np.ndarray,
) -> np.ndarray:
    """valid=True 绿色半径 7，valid=False 红色半径 3，左上角标注双方有效数。"""
    out = image.copy()
    h, w = out.shape[:2]
    for i in range(points_2d.shape[0]):
        x, y = points_2d[i, 0], points_2d[i, 1]
        if not (np.isfinite(x) and np.isfinite(y)):
            continue
        xi, yi = int(round(x)), int(round(y))
        if not (0 <= xi < w and 0 <= yi < h):
            continue
        if valid[i]:
            cv2.circle(out, (xi, yi), 7, (0, 255, 0), -1, lineType=cv2.LINE_AA)
        else:
            cv2.circle(out, (xi, yi), 3, (0, 0, 255), -1, lineType=cv2.LINE_AA)

    n_valid = int(valid.sum())
    cv2.putText(
        out,
        f"fitted DenseKP138 | valid both: {n_valid}/{NUM_DENSEKP}",
        (30, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        2.0,
        (255, 255, 255),
        3,
        cv2.LINE_AA,
    )
    return out


def draw_pair_lines(
    image_cam2: np.ndarray,
    image_cam4: np.ndarray,
    points_cam2: np.ndarray,
    points_cam4: np.ndarray,
    valid: np.ndarray,
) -> np.ndarray:
    """左 cam2 右 cam4 拼接，只画 valid 连线（cam2 绿 / cam4 黄 / 线浅灰），缩放到宽 1800。"""
    h2, w2 = image_cam2.shape[:2]
    h4, w4 = image_cam4.shape[:2]
    h = max(h2, h4)
    canvas = np.zeros((h, w2 + w4, 3), dtype=np.uint8)
    canvas[:h2, :w2] = image_cam2
    canvas[:h4, w2:w2 + w4] = image_cam4
    off_x = w2

    for i in range(points_cam2.shape[0]):
        if not valid[i]:
            continue
        x2, y2 = points_cam2[i, 0], points_cam2[i, 1]
        x4, y4 = points_cam4[i, 0], points_cam4[i, 1]
        if not (np.isfinite(x2) and np.isfinite(y2) and np.isfinite(x4) and np.isfinite(y4)):
            continue
        p2 = (int(round(x2)), int(round(y2)))
        p4 = (int(round(x4)) + off_x, int(round(y4)))
        cv2.line(canvas, p2, p4, (200, 200, 200), 1, lineType=cv2.LINE_AA)
        cv2.circle(canvas, p2, 7, (0, 255, 0), -1, lineType=cv2.LINE_AA)
        cv2.circle(canvas, p4, 7, (0, 255, 255), -1, lineType=cv2.LINE_AA)

    canvas = resize_to_width(canvas, 1800)
    n_valid = int(valid.sum())
    cv2.putText(
        canvas,
        f"fitted DenseKP138 valid pairs: {n_valid}/{NUM_DENSEKP}",
        (10, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return canvas


def get_crop_bbox_from_points(
    points_2d: np.ndarray,
    image_w: int,
    image_h: int,
    pad: int,
) -> tuple[int, int, int, int]:
    """由点云外接矩形 + pad 得到 clamp 到图像内的 (x1, y1, x2, y2)。"""
    x1 = int(np.floor(points_2d[:, 0].min())) - pad
    y1 = int(np.floor(points_2d[:, 1].min())) - pad
    x2 = int(np.ceil(points_2d[:, 0].max())) + pad
    y2 = int(np.ceil(points_2d[:, 1].max())) + pad
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(image_w, x2)
    y2 = min(image_h, y2)
    return (x1, y1, x2, y2)


def _draw_pts_on_crop(
    crop: np.ndarray,
    pts: np.ndarray,
    origin: tuple[int, int],
    color: tuple[int, int, int],
    radius: int = 4,
) -> np.ndarray:
    out = crop.copy()
    ox, oy = origin
    h, w = out.shape[:2]
    for i in range(pts.shape[0]):
        x, y = int(round(pts[i, 0] - ox)), int(round(pts[i, 1] - oy))
        if 0 <= x < w and 0 <= y < h:
            cv2.circle(out, (x, y), radius, color, -1, lineType=cv2.LINE_AA)
    return out


def build_cpw_debug_panel(
    src_crop: np.ndarray,
    tgt_crop: np.ndarray,
    warped_crop: np.ndarray,
    composite_crop: np.ndarray,
    src_pts: np.ndarray,
    tgt_pts: np.ndarray,
    src_crop_bbox: tuple[int, int, int, int],
    tgt_crop_bbox: tuple[int, int, int, int],
    src_cam: str,
    tgt_cam: str,
) -> np.ndarray:
    """2x3 debug panel：行1 src+pts | tgt+pts | warped，行2 overlay | composite | local pair lines。"""
    src_origin = (src_crop_bbox[0], src_crop_bbox[1])
    tgt_origin = (tgt_crop_bbox[0], tgt_crop_bbox[1])

    src_kp = _draw_pts_on_crop(src_crop, src_pts, src_origin, (0, 255, 0))      # source 绿
    tgt_kp = _draw_pts_on_crop(tgt_crop, tgt_pts, tgt_origin, (0, 255, 255))    # target 黄
    overlay = cv2.addWeighted(tgt_crop, 0.5, warped_crop, 0.5, 0)

    # local pair lines：在 target crop 局部坐标中画（source/target 都减 target 原点）
    pair_canvas = np.zeros_like(tgt_crop)
    ox, oy = tgt_origin
    n = min(len(src_pts), len(tgt_pts))
    for i in range(n):
        s = (int(round(src_pts[i, 0] - ox)), int(round(src_pts[i, 1] - oy)))
        t = (int(round(tgt_pts[i, 0] - ox)), int(round(tgt_pts[i, 1] - oy)))
        cv2.line(pair_canvas, s, t, (160, 160, 160), 1, lineType=cv2.LINE_AA)
        cv2.circle(pair_canvas, s, 4, (0, 255, 0), -1, lineType=cv2.LINE_AA)
        cv2.circle(pair_canvas, t, 4, (0, 255, 255), -1, lineType=cv2.LINE_AA)

    panels = [
        resize_to_width(p, DEBUG_PANEL_WIDTH)
        for p in (src_kp, tgt_kp, warped_crop, overlay, composite_crop, pair_canvas)
    ]
    labels = [
        f"src_{src_cam}+pts", f"tgt_{tgt_cam}+pts", f"warped_{src_cam}",
        "overlay", "target_priority", "local_pair_lines",
    ]
    for p, lbl in zip(panels, labels):
        cv2.putText(p, lbl, (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    def pad_to_height(p: np.ndarray, target_h: int) -> np.ndarray:
        if p.shape[0] == target_h:
            return p
        out = np.zeros((target_h, p.shape[1], 3), dtype=np.uint8)
        out[: p.shape[0]] = p
        return out

    row1_h = max(p.shape[0] for p in panels[:3])
    row2_h = max(p.shape[0] for p in panels[3:])
    row1 = np.hstack([pad_to_height(p, row1_h) for p in panels[:3]])
    row2 = np.hstack([pad_to_height(p, row2_h) for p in panels[3:]])

    max_w = max(row1.shape[1], row2.shape[1])
    top = np.zeros((row1.shape[0], max_w, 3), dtype=np.uint8); top[:, : row1.shape[1]] = row1
    bot = np.zeros((row2.shape[0], max_w, 3), dtype=np.uint8); bot[:, : row2.shape[1]] = row2
    panel = np.vstack([top, bot])

    return panel


def run_cpw_direction(
    frame_idxs: list[int],
    data_dir: Path,
    out_base: Path,
    image_dir: Path,
    src_cam: str,
    tgt_cam: str,
    direction_name: str,
) -> None:
    """读取 data_dir/corr_npz/{frame}.npz，按 src_cam→tgt_cam 做 CPW backward warp + target-priority 合成。"""
    corr_npz_dir = data_dir / "corr_npz"
    out_dir = out_base / direction_name
    warp_dir = out_dir / "warp"
    composite_dir = out_dir / "composite_target_priority"
    overlay_dir = out_dir / "overlay"
    panel_dir = out_dir / "debug_panel"
    for d in (warp_dir, composite_dir, overlay_dir, panel_dir):
        d.mkdir(parents=True, exist_ok=True)

    summary_path = out_dir / "cpw_summary.csv"
    fieldnames = [
        "frame", "status", "used_points", "crop_bbox",
        "disp_mean", "disp_median", "disp_std",
        "warp_valid_pixels", "fill_pixels", "output_composite",
    ]

    print()
    print(f"[CPW] direction={direction_name}  source={src_cam}  target={tgt_cam}")

    with summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for frame_idx in frame_idxs:
            fk = f"{frame_idx:05d}"
            row = {k: "" for k in fieldnames}
            row["frame"] = frame_idx

            npz_path = corr_npz_dir / f"{fk}.npz"
            if not npz_path.exists():
                row["status"] = "skip_no_npz"
                writer.writerow(row)
                print(f"[{fk}][{direction_name}] skip: no npz")
                continue

            data = np.load(npz_path)
            points = {"cam2": data["points_cam2"], "cam4": data["points_cam4"]}
            valid = data["valid"].astype(bool)

            src_pts = points[src_cam][valid]   # source
            tgt_pts = points[tgt_cam][valid]   # target
            n_used = int(valid.sum())
            row["used_points"] = n_used

            if n_used < CPW_MIN_POINTS:
                row["status"] = "skip_few_points"
                writer.writerow(row)
                print(f"[{fk}][{direction_name}] skip: used={n_used} < {CPW_MIN_POINTS}")
                continue

            disp = src_pts - tgt_pts
            disp_norm = np.linalg.norm(disp, axis=1)
            row["disp_mean"] = f"{float(disp_norm.mean()):.3f}"
            row["disp_median"] = f"{float(np.median(disp_norm)):.3f}"
            row["disp_std"] = f"{float(disp_norm.std()):.3f}"

            src_img_path = find_image(image_dir, src_cam, frame_idx)
            tgt_img_path = find_image(image_dir, tgt_cam, frame_idx)
            src_img = cv2.imread(str(src_img_path)) if src_img_path is not None else None
            tgt_img = cv2.imread(str(tgt_img_path)) if tgt_img_path is not None else None
            if src_img is None or tgt_img is None:
                row["status"] = "skip_missing_image"
                writer.writerow(row)
                print(f"[{fk}][{direction_name}] skip: missing image")
                continue

            src_valid_full = compute_valid_mask(src_img_path)
            tgt_valid_full = compute_valid_mask(tgt_img_path)
            if src_valid_full is None or tgt_valid_full is None:
                row["status"] = "skip_no_valid_mask"
                writer.writerow(row)
                print(f"[{fk}][{direction_name}] skip: no valid mask")
                continue

            target_crop_bbox = get_crop_bbox_from_points(tgt_pts, ERP_W, ERP_H, CPW_PAD)
            source_crop_bbox = get_crop_bbox_from_points(src_pts, ERP_W, ERP_H, CPW_PAD)
            row["crop_bbox"] = f"({target_crop_bbox[0]},{target_crop_bbox[1]},{target_crop_bbox[2]},{target_crop_bbox[3]})"

            x1, y1, x2, y2 = target_crop_bbox
            crop_h, crop_w = y2 - y1, x2 - x1
            if crop_h <= 0 or crop_w <= 0:
                row["status"] = "skip_empty_crop"
                writer.writerow(row)
                print(f"[{fk}][{direction_name}] skip: empty crop")
                continue

            # 适配库接口：grid 仅覆盖 target crop（image_shape=crop），target 点转 crop-local，
            # source 点保持全局 → 输出 backward map 直接是 source 全图采样坐标。
            dst_local = tgt_pts - np.array([x1, y1], dtype=tgt_pts.dtype)
            backward_map, _ = generate_cpw_backward_map(
                src_pts=src_pts,
                dst_pts=dst_local,
                image_shape=(crop_h, crop_w),
                k=CPW_KNN,
                power=CPW_POWER,
            )
            if backward_map.size == 0:
                row["status"] = "skip_empty_map"
                writer.writerow(row)
                print(f"[{fk}][{direction_name}] skip: empty cpw map")
                continue

            map_x = backward_map[..., 0]
            map_y = backward_map[..., 1]
            inside = (map_x >= 0) & (map_x < ERP_W) & (map_y >= 0) & (map_y < ERP_H)

            warped_crop = warp_image_with_backward_map(
                src_img, backward_map, interpolation=cv2.INTER_LINEAR,
            )
            warped_src_valid = warp_image_with_backward_map(
                (src_valid_full.astype(np.uint8) * 255), backward_map,
                interpolation=cv2.INTER_NEAREST,
            ) > 0
            warp_valid = inside & warped_src_valid

            tgt_crop = tgt_img[y1:y2, x1:x2]
            tgt_valid_crop = tgt_valid_full[y1:y2, x1:x2]

            # cam2 在底部 → "down" 侵蚀，cam4 在顶部 → "up" 侵蚀
            tgt_erode_dir = "down" if tgt_cam == "cam2" else "up"
            src_erode_dir = "down" if src_cam == "cam2" else "up"
            tgt_erode_px = DIR_ERODE_PX_CAM2 if tgt_cam == "cam2" else DIR_ERODE_PX_CAM4
            src_erode_px = DIR_ERODE_PX_CAM2 if src_cam == "cam2" else DIR_ERODE_PX_CAM4
            target_safe = directional_erode_valid_mask(tgt_valid_crop, tgt_erode_dir, tgt_erode_px)
            warp_safe = directional_erode_valid_mask(warp_valid, src_erode_dir, src_erode_px)

            composite_crop = target_priority_composite(
                tgt_crop, warped_crop, target_safe, warp_safe,
            )
            fill_mask = (~target_safe) & warp_safe

            warped_full = np.zeros_like(tgt_img)
            warped_full[y1:y2, x1:x2] = warped_crop
            composite_full = tgt_img.copy()
            composite_full[y1:y2, x1:x2] = composite_crop
            overlay_crop = cv2.addWeighted(tgt_crop, 0.5, warped_crop, 0.5, 0)

            sx1, sy1, sx2, sy2 = source_crop_bbox
            src_crop = src_img[sy1:sy2, sx1:sx2]
            panel = build_cpw_debug_panel(
                src_crop, tgt_crop, warped_crop, composite_crop,
                src_pts, tgt_pts, source_crop_bbox, target_crop_bbox,
                src_cam, tgt_cam,
            )

            cv2.imwrite(str(warp_dir / f"{fk}.jpg"), warped_full, [cv2.IMWRITE_JPEG_QUALITY, 95])
            composite_path = composite_dir / f"{fk}.jpg"
            cv2.imwrite(str(composite_path), composite_full, [cv2.IMWRITE_JPEG_QUALITY, 95])
            cv2.imwrite(str(overlay_dir / f"{fk}.jpg"), overlay_crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
            cv2.imwrite(str(panel_dir / f"{fk}.jpg"), panel, [cv2.IMWRITE_JPEG_QUALITY, 95])

            warp_valid_pixels = int(warp_valid.sum())
            fill_pixels = int(fill_mask.sum())
            row["status"] = "ok"
            row["warp_valid_pixels"] = warp_valid_pixels
            row["fill_pixels"] = fill_pixels
            row["output_composite"] = str(composite_path)
            writer.writerow(row)
            print(
                f"[{fk}][{direction_name}] used={n_used} crop={row['crop_bbox']} "
                f"disp_mean={row['disp_mean']} fill_pixels={fill_pixels}"
            )

    print(f"[CPW] summary: {summary_path}")


def generate_points_for_segment(
    segment: str,
    R: np.ndarray,
    image_dir: Path,
    forwarder: SMPLXForwarder,
    result_dir: Path,
    data_dir: Path,
    summary_rows: list[dict],
) -> list[int]:
    """前向+投影+双方 valid 筛选，写 data_dir 下 npz/可视化，追加 summary_rows，返回处理过的帧号列表。"""
    start_frame, end_frame = parse_segment_range(segment)
    frame_indices = list(range(start_frame, end_frame + 1))

    pkl_cam2 = result_dir / "cam2" / f"seg_{segment}_seq.pkl"
    pkl_cam4 = result_dir / "cam4" / f"seg_{segment}_seq.pkl"
    if not pkl_cam2.exists() or not pkl_cam4.exists():
        print(f"[SKIP] segment {segment}: missing pkl (cam2={pkl_cam2.exists()}, cam4={pkl_cam4.exists()})")
        return []
    params_cam2 = unpack_smplx_result(_load_pickle(pkl_cam2))
    params_cam4 = unpack_smplx_result(_load_pickle(pkl_cam4))

    total_frames = min(len(frame_indices), len(params_cam2), len(params_cam4))

    points_cam2_dir = data_dir / "points_cam2"
    points_cam4_dir = data_dir / "points_cam4"
    pairs_dir = data_dir / "pairs"
    corr_npz_dir = data_dir / "corr_npz"

    print(f"Segment: {segment}  frames {start_frame}-{end_frame}  usable={total_frames}")

    for idx in range(total_frames):
        frame_idx = frame_indices[idx]
        fk = f"{frame_idx:05d}"

        verts_cam2, _, _ = forwarder.forward_param_to_numpy(params_cam2[idx])
        verts_cam4, _, _ = forwarder.forward_param_to_numpy(params_cam4[idx])

        densekp_3d_cam2 = R @ verts_cam2
        densekp_3d_cam4 = R @ verts_cam4

        # 两边都用 project_cam2
        points_cam2 = project_points_erp(densekp_3d_cam2, ERP_W, ERP_H, project_cam2)
        points_cam4 = project_points_erp(densekp_3d_cam4, ERP_W, ERP_H, project_cam2)

        img_cam2_path = find_image(image_dir, "cam2", frame_idx)
        img_cam4_path = find_image(image_dir, "cam4", frame_idx)
        mask_cam2 = compute_valid_mask(img_cam2_path)
        mask_cam4 = compute_valid_mask(img_cam4_path)

        valid_cam2 = check_points_in_mask(points_cam2, mask_cam2)
        valid_cam4 = check_points_in_mask(points_cam4, mask_cam4)

        finite = (
            np.isfinite(points_cam2).all(axis=1)
            & np.isfinite(points_cam4).all(axis=1)
        )
        valid = valid_cam2 & valid_cam4 & finite

        corr_npz_path = corr_npz_dir / f"{fk}.npz"
        np.savez_compressed(
            corr_npz_path,
            frame_idx=np.array(frame_idx, dtype=np.int32),
            point_ids=np.arange(NUM_DENSEKP, dtype=np.int32),
            points_cam2=points_cam2.astype(np.float32),
            points_cam4=points_cam4.astype(np.float32),
            valid_cam2=valid_cam2.astype(bool),
            valid_cam4=valid_cam4.astype(bool),
            valid=valid.astype(bool),
        )

        img_cam2 = cv2.imread(str(img_cam2_path)) if img_cam2_path is not None else None
        img_cam4 = cv2.imread(str(img_cam4_path)) if img_cam4_path is not None else None

        if img_cam2 is not None:
            vis2 = draw_projected_points(img_cam2, points_cam2, valid)
            cv2.imwrite(str(points_cam2_dir / f"{fk}.jpg"), vis2, [cv2.IMWRITE_JPEG_QUALITY, 95])
        if img_cam4 is not None:
            vis4 = draw_projected_points(img_cam4, points_cam4, valid)
            cv2.imwrite(str(points_cam4_dir / f"{fk}.jpg"), vis4, [cv2.IMWRITE_JPEG_QUALITY, 95])
        if img_cam2 is not None and img_cam4 is not None:
            pair_vis = draw_pair_lines(img_cam2, img_cam4, points_cam2, points_cam4, valid)
            cv2.imwrite(str(pairs_dir / f"{fk}.jpg"), pair_vis, [cv2.IMWRITE_JPEG_QUALITY, 95])

        n_c2 = int(valid_cam2.sum())
        n_c4 = int(valid_cam4.sum())
        n_both = int(valid.sum())
        summary_rows.append(
            {
                "frame": frame_idx,
                "total_points": NUM_DENSEKP,
                "valid_cam2": n_c2,
                "valid_cam4": n_c4,
                "valid_both": n_both,
                "output_npz": str(corr_npz_path),
            }
        )
        print(f"[{fk}] valid cam2={n_c2}/{NUM_DENSEKP} cam4={n_c4}/{NUM_DENSEKP} both={n_both}/{NUM_DENSEKP}")

    return frame_indices[:total_frames]


def discover_segments(result_dir: Path, cam: str = "cam2") -> list[str]:
    """扫描 result_dir/{cam}/seg_*_seq.pkl，返回按起止帧排序的 segment 名列表，如 ['513_518', '704_711']。"""
    scan_dir = result_dir / cam
    entries: list[tuple[int, int, str]] = []
    for p in scan_dir.glob("seg_*_seq.pkl"):
        name = p.name[len("seg_"):-len("_seq.pkl")]
        try:
            start, end = parse_segment_range(name)
        except ValueError:
            continue
        entries.append((start, end, name))
    entries.sort(key=lambda x: (x[0], x[1]))
    return [n for _, _, n in entries]


def main() -> None:
    args = parse_args()

    dataset = args.dataset
    data_root = PROJECT_ROOT / "data_process" / "Output" / dataset
    image_dir = data_root / "image" / "equirect_rotate"
    result_dir = (
        Path(args.fitting_dir) if args.fitting_dir is not None
        else data_root / "fitting" / "dual_cam_sam3d_dense"
    )
    output_root = (
        Path(args.output) if args.output is not None
        else PROJECT_ROOT / "data_process" / "Vis" / "debug077_fitted_densekp138_valid_cpw" / dataset
    )

    regressor_path = (
        PROJECT_ROOT / "tools/CameraHMR/data/train-eval-utils/densekp_regressor_R.npy"
    )
    R = np.load(regressor_path).astype(np.float32)
    assert R.shape == (138, 10475), f"unexpected R shape: {R.shape}"

    forwarder = SMPLXForwarder(gender="neutral")

    if args.segment.lower() == "all":
        segments = discover_segments(result_dir, cam="cam2")
        if not segments:
            raise FileNotFoundError(f"No seg_*_seq.pkl found under {result_dir / 'cam2'}")
        print(f"Discovered {len(segments)} segments: {segments}")
    else:
        segments = [args.segment]

    data_dir = output_root / "data"
    for d in (
        data_dir / "points_cam2",
        data_dir / "points_cam4",
        data_dir / "pairs",
        data_dir / "corr_npz",
    ):
        d.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict] = []
    all_frames: list[int] = []
    for i, segment in enumerate(segments):
        print()
        print("=" * 60)
        print(f"[{i + 1}/{len(segments)}] segment {segment}")
        print("=" * 60)
        all_frames.extend(
            generate_points_for_segment(segment, R, image_dir, forwarder, result_dir, data_dir, summary_rows)
        )

    summary_path = data_dir / "summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["frame", "total_points", "valid_cam2", "valid_cam4", "valid_both", "output_npz"],
        )
        writer.writeheader()
        writer.writerows(summary_rows)
    print()
    print(f"[DONE] points: {len(all_frames)} frames → {data_dir}")
    print(f"summary: {summary_path}")

    for src_cam, tgt_cam, direction_name in get_direction_pairs(args.directions):
        run_cpw_direction(all_frames, data_dir, output_root, image_dir, src_cam, tgt_cam, direction_name)


if __name__ == "__main__":
    main()


r"""

# 默认路径（fitting/output 从 --dataset 自动推导）：双向 CPW
python src/script/test_function/debug077_fitted_densekp138_valid_cpw_composite.py \
  --dataset 20260311_riding \
  --segment 513_518 \
  --directions both

# 所有 segment
python src/script/test_function/debug077_fitted_densekp138_valid_cpw_composite.py \
  --dataset 20260311_riding \
  --segment all \
  --directions both

# 自定义 fitting_dir 与 output（覆盖 dataset 默认值）
python src/script/test_function/debug077_fitted_densekp138_valid_cpw_composite.py \
  --dataset 20260320_riding_dataset2 \
  --segment all \
  --directions both \
  --fitting_dir data_process/Output/20260320_riding_dataset2/fitting/dual_cam_sam3d_dense \
  --output data_process/Output/20260320_riding_dataset2/tps_result/sam3dbody_cpw
"""
