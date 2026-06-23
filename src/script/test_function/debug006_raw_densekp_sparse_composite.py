"""raw DenseKP138（检测值）跨相机配对：按 debug077 的 eroded valid-region 规则筛点，
用 CPW backward warp + cam target-priority 合成，仅输出 debug_panel（两个方向）。

约定：点为检测原始 DenseKP（已在 equirect_rotate 全图坐标）；筛选 = 检测 valid AND
两侧各自落在 eroded border-connected valid mask 内 AND finite；合成有效域复用同一 valid mask。
"""

from __future__ import annotations

import argparse
from pathlib import Path

import config_script  # noqa: F401 — ensure src/ on sys.path

import cv2
import numpy as np

from lib_pano.fitting_io.densekp_io import load_densekp_result
from lib_pano.pipeline.bbox_io import load_bbox_dict
from lib_pano.composite.cpw_warp import (
    directional_erode_valid_mask,
    generate_cpw_backward_map,
    target_priority_composite,
    warp_image_with_backward_map,
)

CPW_MIN_POINTS = 8


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="raw DenseKP138 跨相机配对 + CPW 合成（debug077 筛点规则，仅 debug_panel）。"
    )
    parser.add_argument("--image_root", type=str, required=True)
    parser.add_argument("--densekp_dir", type=str, required=True)
    parser.add_argument("--bbox_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument(
        "--segment",
        type=str,
        nargs="+",
        default=None,
        help="一个或多个段，例如 513_518 704_711。",
    )
    parser.add_argument(
        "--frames",
        type=int,
        nargs="+",
        default=None,
        help="可选：显式帧列表（向后兼容）。",
    )
    parser.add_argument("--src_cam", type=str, default=None)
    parser.add_argument("--tgt_cam", type=str, default=None)
    parser.add_argument(
        "--directions",
        choices=["both", "cam4_to_cam2", "cam2_to_cam4"],
        default="both",
    )
    parser.add_argument("--image_width", type=int, required=True)
    parser.add_argument("--image_height", type=int, required=True)
    parser.add_argument("--pad", type=int, default=200)
    parser.add_argument("--panel_size", type=int, default=420)
    parser.add_argument("--cpw_knn", type=int, default=12)
    parser.add_argument("--cpw_power", type=float, default=2.0)
    parser.add_argument("--dir_erode_px_cam2", type=int, default=15)
    parser.add_argument("--dir_erode_px_cam4", type=int, default=15)
    return parser.parse_args()


def parse_segment_range(segment: str) -> tuple[int, int]:
    parts = segment.split("_")
    if len(parts) != 2:
        raise ValueError(f"segment must match start_end, got: {segment}")
    start_frame, end_frame = int(parts[0]), int(parts[1])
    if start_frame > end_frame:
        raise ValueError(f"segment start must be <= end: {segment}")
    return start_frame, end_frame


def get_frame_groups(args) -> list[tuple[str, list[int]]]:
    if args.segment is not None:
        groups = []
        for segment in args.segment:
            start_frame, end_frame = parse_segment_range(segment)
            groups.append((f"seg_{segment}", list(range(start_frame, end_frame + 1))))
        return groups
    if args.frames is not None:
        return [("frames_manual", args.frames)]
    raise ValueError("Need --segment or --frames.")


def get_direction_pairs(args) -> list[tuple[str, str, str]]:
    if args.src_cam is not None and args.tgt_cam is not None:
        return [(args.src_cam, args.tgt_cam, f"{args.src_cam}_to_{args.tgt_cam}")]
    if args.directions == "both":
        return [("cam4", "cam2", "cam4_to_cam2"), ("cam2", "cam4", "cam2_to_cam4")]
    if args.directions == "cam4_to_cam2":
        return [("cam4", "cam2", "cam4_to_cam2")]
    if args.directions == "cam2_to_cam4":
        return [("cam2", "cam4", "cam2_to_cam4")]
    raise ValueError(f"Unsupported directions: {args.directions}")


def resize_to_width(img: np.ndarray, target_w: int) -> np.ndarray:
    h, w = img.shape[:2]
    if w == target_w:
        return img
    scale = target_w / w
    new_h = max(1, int(round(h * scale)))
    return cv2.resize(img, (target_w, new_h), interpolation=cv2.INTER_AREA)


def draw_keypoints(
    img: np.ndarray,
    kps: np.ndarray,
    offset: tuple[int, int],
    color: tuple[int, int, int],
    radius: int = 4,
) -> np.ndarray:
    out = img.copy()
    ox, oy = offset
    h, w = out.shape[:2]
    for kp in kps:
        x, y = int(round(kp[0] - ox)), int(round(kp[1] - oy))
        if 0 <= x < w and 0 <= y < h:
            cv2.circle(out, (x, y), radius, color, -1, lineType=cv2.LINE_AA)
    return out


def compute_valid_mask(
    img: np.ndarray | None, black_thresh: int = 5, erode_px: int = 10
) -> np.ndarray | None:
    """与 debug052/debug077 一致：max(RGB)<=thresh 的黑像素中，仅剔除与图像边界连通的部分，
    内部深色区域保留为 valid，再用 (2*erode_px+1) 方核向内 erosion。img 为 None 返回 None。"""
    if img is None:
        return None
    black = img.max(axis=2) <= black_thresh
    num_labels, labels = cv2.connectedComponents(black.astype(np.uint8), connectivity=8)
    border_labels = set()
    border_labels.update(labels[0, :].tolist())
    border_labels.update(labels[-1, :].tolist())
    border_labels.update(labels[:, 0].tolist())
    border_labels.update(labels[:, -1].tolist())
    border_labels.discard(0)
    invalid = np.isin(labels, list(border_labels)) if border_labels else np.zeros_like(black)
    valid = ~invalid
    ksize = 2 * erode_px + 1
    kernel = np.ones((ksize, ksize), dtype=np.uint8)
    return cv2.erode(valid.astype(np.uint8), kernel) > 0


def check_points_in_mask(points_2d: np.ndarray, valid_mask: np.ndarray | None) -> np.ndarray:
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


def get_crop_bbox_from_points(
    points_2d: np.ndarray, image_w: int, image_h: int, pad: int
) -> tuple[int, int, int, int]:
    x1 = max(0, int(np.floor(points_2d[:, 0].min())) - pad)
    y1 = max(0, int(np.floor(points_2d[:, 1].min())) - pad)
    x2 = min(image_w, int(np.ceil(points_2d[:, 0].max())) + pad)
    y2 = min(image_h, int(np.ceil(points_2d[:, 1].max())) + pad)
    return (x1, y1, x2, y2)


def build_cpw_debug_panel(
    src_crop,
    tgt_crop,
    warped_crop,
    composite_crop,
    src_pts,
    tgt_pts,
    src_crop_bbox,
    tgt_crop_bbox,
    src_cam,
    tgt_cam,
    panel_w,
) -> np.ndarray:
    """2x3：行1 src+pts | tgt+pts | warped，行2 overlay | target_priority | local_pair_lines。"""
    src_origin = (src_crop_bbox[0], src_crop_bbox[1])
    tgt_origin = (tgt_crop_bbox[0], tgt_crop_bbox[1])
    src_kp = draw_keypoints(src_crop, src_pts, src_origin, (0, 255, 0))  # source 绿
    tgt_kp = draw_keypoints(tgt_crop, tgt_pts, tgt_origin, (0, 255, 255))  # target 黄
    overlay = cv2.addWeighted(tgt_crop, 0.5, warped_crop, 0.5, 0)

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
        resize_to_width(p, panel_w)
        for p in (src_kp, tgt_kp, warped_crop, overlay, composite_crop, pair_canvas)
    ]
    labels = [
        f"src_{src_cam}+pts",
        f"tgt_{tgt_cam}+pts",
        f"warped_{src_cam}",
        "overlay",
        "target_priority",
        "local_pair_lines",
    ]
    for p, lbl in zip(panels, labels):
        cv2.putText(p, lbl, (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    def pad_to_height(p, h):
        if p.shape[0] == h:
            return p
        out = np.zeros((h, p.shape[1], 3), dtype=np.uint8)
        out[: p.shape[0]] = p
        return out

    row1_h = max(p.shape[0] for p in panels[:3])
    row2_h = max(p.shape[0] for p in panels[3:])
    row1 = np.hstack([pad_to_height(p, row1_h) for p in panels[:3]])
    row2 = np.hstack([pad_to_height(p, row2_h) for p in panels[3:]])
    max_w = max(row1.shape[1], row2.shape[1])
    top = np.zeros((row1.shape[0], max_w, 3), dtype=np.uint8)
    top[:, : row1.shape[1]] = row1
    bot = np.zeros((row2.shape[0], max_w, 3), dtype=np.uint8)
    bot[:, : row2.shape[1]] = row2
    return np.vstack([top, bot])


def process_frame(
    frame_idx,
    image_root,
    densekp_dir,
    bbox_dict_src,
    bbox_dict_tgt,
    panel_dir,
    erp_h,
    erp_w,
    src_cam,
    tgt_cam,
    pad,
    panel_w,
    cpw_knn,
    cpw_power,
    dir_erode_px_cam2,
    dir_erode_px_cam4,
) -> bool:
    fk = f"{frame_idx:05d}"
    src_bbox = bbox_dict_src.get(str(frame_idx))
    tgt_bbox = bbox_dict_tgt.get(str(frame_idx))
    if src_bbox is None or tgt_bbox is None:
        print(f"[{fk}][{src_cam}->{tgt_cam}] skip: missing bbox")
        return False

    src_dkp = load_densekp_result(frame_idx, densekp_dir, src_cam, erp_w, erp_h, crop_bbox=src_bbox)
    tgt_dkp = load_densekp_result(frame_idx, densekp_dir, tgt_cam, erp_w, erp_h, crop_bbox=tgt_bbox)
    if src_dkp is None or tgt_dkp is None:
        print(f"[{fk}][{src_cam}->{tgt_cam}] skip: missing densekp npz")
        return False

    src_img = cv2.imread(str(image_root / src_cam / f"{fk}.jpg"))
    tgt_img = cv2.imread(str(image_root / tgt_cam / f"{fk}.jpg"))
    if src_img is None or tgt_img is None:
        print(f"[{fk}][{src_cam}->{tgt_cam}] skip: missing image")
        return False

    src_mask = compute_valid_mask(src_img)
    tgt_mask = compute_valid_mask(tgt_img)
    if src_mask is None or tgt_mask is None:
        print(f"[{fk}][{src_cam}->{tgt_cam}] skip: no valid mask")
        return False

    src_pts_all = src_dkp.keypoints_2d
    tgt_pts_all = tgt_dkp.keypoints_2d
    finite = np.isfinite(src_pts_all).all(axis=1) & np.isfinite(tgt_pts_all).all(axis=1)
    in_src = check_points_in_mask(src_pts_all, src_mask)
    in_tgt = check_points_in_mask(tgt_pts_all, tgt_mask)
    valid = src_dkp.valid & tgt_dkp.valid & in_src & in_tgt & finite

    n_used = int(valid.sum())
    if n_used < CPW_MIN_POINTS:
        print(f"[{fk}][{src_cam}->{tgt_cam}] skip: used={n_used} < {CPW_MIN_POINTS}")
        return False

    src_pts = src_pts_all[valid].astype(np.float32)
    tgt_pts = tgt_pts_all[valid].astype(np.float32)

    target_crop_bbox = get_crop_bbox_from_points(tgt_pts, erp_w, erp_h, pad)
    source_crop_bbox = get_crop_bbox_from_points(src_pts, erp_w, erp_h, pad)
    x1, y1, x2, y2 = target_crop_bbox
    crop_h, crop_w = y2 - y1, x2 - x1
    if crop_h <= 0 or crop_w <= 0:
        print(f"[{fk}][{src_cam}->{tgt_cam}] skip: empty crop")
        return False

    # grid 仅覆盖 target crop，target 点转 crop-local、source 点保持全局 → map 为 source 全图采样坐标
    dst_local = tgt_pts - np.array([x1, y1], dtype=tgt_pts.dtype)
    backward_map, _ = generate_cpw_backward_map(
        src_pts, dst_local, (crop_h, crop_w), k=cpw_knn, power=cpw_power
    )
    if backward_map.size == 0:
        print(f"[{fk}][{src_cam}->{tgt_cam}] skip: empty cpw map")
        return False

    map_x = backward_map[..., 0]
    map_y = backward_map[..., 1]
    inside = (map_x >= 0) & (map_x < erp_w) & (map_y >= 0) & (map_y < erp_h)
    warped_crop = warp_image_with_backward_map(src_img, backward_map, interpolation=cv2.INTER_LINEAR)
    warped_src_valid = (
        warp_image_with_backward_map(
            (src_mask.astype(np.uint8) * 255), backward_map, interpolation=cv2.INTER_NEAREST
        )
        > 0
    )
    warp_valid = inside & warped_src_valid

    tgt_crop = tgt_img[y1:y2, x1:x2]
    tgt_valid_crop = tgt_mask[y1:y2, x1:x2]
    tgt_dir = "down" if tgt_cam == "cam2" else "up"
    src_dir = "down" if src_cam == "cam2" else "up"
    tgt_erode = dir_erode_px_cam2 if tgt_cam == "cam2" else dir_erode_px_cam4
    src_erode = dir_erode_px_cam2 if src_cam == "cam2" else dir_erode_px_cam4
    target_safe = directional_erode_valid_mask(tgt_valid_crop, tgt_dir, tgt_erode)
    warp_safe = directional_erode_valid_mask(warp_valid, src_dir, src_erode)
    composite_crop = target_priority_composite(tgt_crop, warped_crop, target_safe, warp_safe)

    sx1, sy1, sx2, sy2 = source_crop_bbox
    src_crop = src_img[sy1:sy2, sx1:sx2]
    panel = build_cpw_debug_panel(
        src_crop,
        tgt_crop,
        warped_crop,
        composite_crop,
        src_pts,
        tgt_pts,
        source_crop_bbox,
        target_crop_bbox,
        src_cam,
        tgt_cam,
        panel_w,
    )
    cv2.imwrite(str(panel_dir / f"{fk}.jpg"), panel, [cv2.IMWRITE_JPEG_QUALITY, 95])

    print(f"[{fk}][{src_cam}->{tgt_cam}] used={n_used} crop=({x1},{y1},{x2},{y2}) saved")
    return True


def main() -> None:
    args = parse_args()
    image_root = Path(args.image_root)
    densekp_dir = Path(args.densekp_dir)
    bbox_dir = Path(args.bbox_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    frame_groups = get_frame_groups(args)
    direction_pairs = get_direction_pairs(args)

    grand_ok = 0
    grand_total = 0
    for group_name, frame_indices in frame_groups:
        print()
        print("*" * 60)
        print(
            f"Frame group: {group_name}  frames {frame_indices[0]}-{frame_indices[-1]} "
            f"(n={len(frame_indices)})"
        )
        print("*" * 60)

        for src_cam, tgt_cam, direction_name in direction_pairs:
            panel_dir = output_dir / direction_name / "debug_panel"
            panel_dir.mkdir(parents=True, exist_ok=True)
            bbox_dict_src = load_bbox_dict(bbox_dir / f"{src_cam}_bboxes.json")
            bbox_dict_tgt = load_bbox_dict(bbox_dir / f"{tgt_cam}_bboxes.json")

            print(f"\n[DIR] {direction_name}  source={src_cam} target={tgt_cam}  -> {panel_dir}")
            n_ok = 0
            for frame_idx in frame_indices:
                if process_frame(
                    frame_idx,
                    image_root,
                    densekp_dir,
                    bbox_dict_src,
                    bbox_dict_tgt,
                    panel_dir,
                    args.image_height,
                    args.image_width,
                    src_cam,
                    tgt_cam,
                    args.pad,
                    args.panel_size,
                    args.cpw_knn,
                    args.cpw_power,
                    args.dir_erode_px_cam2,
                    args.dir_erode_px_cam4,
                ):
                    n_ok += 1
            print(f"[DONE] {direction_name}: {n_ok}/{len(frame_indices)}")
            grand_ok += n_ok
            grand_total += len(frame_indices)

    print(f"\n{'=' * 60}")
    print(f"All groups, all directions done. {grand_ok}/{grand_total} frames OK.")


if __name__ == "__main__":
    main()


"""
python src/script/test_function/debug006_raw_densekp_sparse_composite.py \
  --image_root data_process/Output/20260311_riding/image/equirect_rotate \
  --densekp_dir data_process/Output/20260311_riding/camerahmr/densekp_npz \
  --bbox_dir data_process/Output/20260311_riding/image/equirect_crop_rotate \
  --output_dir data_process/Vis/debug006_raw_densekp_cpw \
  --segment 1856_1928\
  --directions both \
  --image_width 5760 \
  --image_height 2880 \
  --cpw_knn 12 --cpw_power 2.0 \
  --pad 200 --panel_size 420 \
  --dir_erode_px_cam2 20 --dir_erode_px_cam4 20
"""