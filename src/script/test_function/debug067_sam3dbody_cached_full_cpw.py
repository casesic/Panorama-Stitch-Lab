#!/usr/bin/env python3
"""
CPW composite on equirect_rotate full-image pairs from cached SAM 3D Body npz.

Reads pre-computed SAM 3D Body reconstruction npz (debug066 export), pairs
cam2/cam4 vertices by vertex id, restores coordinates to equirect_rotate,
and runs CPW composite. No model loading or inference.

Pipeline per frame:
  1. Read cam2/cam4 cached npz (vertices_2d, valid_z, inside)
  2. Pair vertices by vertex id (crop-upright coords)
  3. Restore coordinates to equirect_rotate full-image coords
  4. CPW composite + debug panel

Output (debug_panel only, both directions under output_dir):
  output_dir/
    cam4_to_cam2/debug_panel/{frame}.jpg
    cam2_to_cam4/debug_panel/{frame}.jpg
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

SRC_ROOT = Path(__file__).resolve().parents[2]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from lib_pano.human3d.sam3dbody_io import load_sam3dbody_recon
from lib_pano.human3d.crop_upright_coord import upright_points_to_equirect_rotate
from lib_pano.correspondence.sam3dbody_pairs import (
    pair_sam3dbody_vertices,
    restore_sam3dbody_pairs_to_full,
    filter_pairs_inside_image,
)
from lib_pano.composite.cpw_warp import (
    generate_cpw_backward_map,
    warp_image_with_backward_map,
    directional_erode_valid_mask,
    target_priority_composite,
)

ERP_W = 5760
ERP_H = 2880


# ---------------------------------------------------------------------------
# drawing helpers
# ---------------------------------------------------------------------------
def resize_to_width(img: np.ndarray, target_w: int) -> np.ndarray:
    h, w = img.shape[:2]
    if w == target_w:
        return img
    scale = target_w / w
    new_h = int(round(h * scale))
    return cv2.resize(img, (target_w, new_h), interpolation=cv2.INTER_LINEAR)


def draw_points(
    img: np.ndarray, pts: np.ndarray, offset: tuple, color: tuple,
    radius: int = 3, max_pts: int = 1500,
) -> np.ndarray:
    out = img.copy()
    total = len(pts)
    if total > max_pts:
        idx = np.linspace(0, total - 1, max_pts).astype(int)
        pts = pts[idx]
    ox, oy = offset
    for i in range(len(pts)):
        x = int(round(pts[i, 0] - ox))
        y = int(round(pts[i, 1] - oy))
        if 0 <= x < out.shape[1] and 0 <= y < out.shape[0]:
            cv2.circle(out, (x, y), radius, color, -1, lineType=cv2.LINE_AA)
    return out


def build_debug_panel(
    src_crop, tgt_crop, warped_crop, composite_tp,
    src_pts, tgt_pts, src_offset, tgt_offset,
    panel_w, max_draw=500,
):
    src_kp = draw_points(src_crop, src_pts, src_offset, (0, 255, 0), radius=4, max_pts=max_draw)
    tgt_kp = draw_points(tgt_crop, tgt_pts, tgt_offset, (0, 255, 255), radius=4, max_pts=max_draw)
    overlay = cv2.addWeighted(tgt_crop, 0.5, warped_crop, 0.5, 0)

    pair_canvas = np.zeros_like(tgt_crop)
    total = min(len(src_pts), len(tgt_pts))
    if total > max_draw:
        idx = np.linspace(0, total - 1, max_draw).astype(int)
    else:
        idx = np.arange(total)
    for i in idx:
        s = (int(round(src_pts[i, 0] - src_offset[0])), int(round(src_pts[i, 1] - src_offset[1])))
        t = (int(round(tgt_pts[i, 0] - tgt_offset[0])), int(round(tgt_pts[i, 1] - tgt_offset[1])))
        cv2.line(pair_canvas, s, t, (180, 180, 180), 1, lineType=cv2.LINE_AA)
        cv2.circle(pair_canvas, s, 3, (0, 255, 0), -1)
        cv2.circle(pair_canvas, t, 3, (0, 255, 255), -1)

    panels = [resize_to_width(p, panel_w) for p in (
        src_kp, tgt_kp, warped_crop, overlay, composite_tp, pair_canvas,
    )]
    row1_h = max(p.shape[0] for p in panels[:3])
    row2_h = max(p.shape[0] for p in panels[3:])

    def pad_to_h(p, h):
        if p.shape[0] == h:
            return p
        out = np.zeros((h, p.shape[1], 3), dtype=np.uint8)
        out[:p.shape[0]] = p
        return out

    row1 = [pad_to_h(p, row1_h) for p in panels[:3]]
    row2 = [pad_to_h(p, row2_h) for p in panels[3:]]
    labels = ["src+kps", "tgt+kps", "warp", "overlay", "target_priority", "pair_lines"]
    for r, lbls in zip([row1, row2], [labels[:3], labels[3:]]):
        for j in range(len(r)):
            cv2.putText(r[j], lbls[j], (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    top = np.hstack(row1)
    bot = np.hstack(row2)
    mw = max(top.shape[1], bot.shape[1])
    tp = np.zeros((top.shape[0], mw, 3), dtype=np.uint8); tp[:, :top.shape[1]] = top
    bp = np.zeros((bot.shape[0], mw, 3), dtype=np.uint8); bp[:, :bot.shape[1]] = bot
    return np.vstack([tp, bp])


# ---------------------------------------------------------------------------
# direction handling + per-direction run (debug_panel only)
# ---------------------------------------------------------------------------
def get_direction_pairs(direction: str) -> list[tuple[str, str, str]]:
    """把 --direction 解析为 (src_cam, tgt_cam, direction_name) 列表。"""
    table = {
        "cam4_to_cam2": ("cam4", "cam2", "cam4_to_cam2"),
        "cam2_to_cam4": ("cam2", "cam4", "cam2_to_cam4"),
    }
    if direction == "both":
        return [table["cam4_to_cam2"], table["cam2_to_cam4"]]
    if direction not in table:
        raise ValueError(f"unknown --direction: {direction}")
    return [table[direction]]


def run_direction(
    direction_name: str,
    src_cam: str,
    tgt_cam: str,
    frame_indices: list[int],
    recon_dir: Path,
    full_dir: Path,
    bbox_by_cam: dict,
    out_base: Path,
    args: argparse.Namespace,
    rng: np.random.RandomState,
) -> tuple[int, int]:
    """对单个方向逐帧做 CPW + target-priority 合成，只输出 out_base/<direction>/debug_panel/{frame}.jpg。"""
    panel_dir = out_base / direction_name / "debug_panel"
    panel_dir.mkdir(parents=True, exist_ok=True)
    bbox_src = bbox_by_cam[src_cam]
    bbox_tgt = bbox_by_cam[tgt_cam]

    print(f"\n[DIR] {direction_name}  source={src_cam}  target={tgt_cam}")
    n_ok = 0
    n_skip = 0

    for frame_idx in frame_indices:
        frame_key = f"{frame_idx:05d}"

        npz_src = recon_dir / src_cam / f"{frame_key}.npz"
        npz_tgt = recon_dir / tgt_cam / f"{frame_key}.npz"
        if not npz_src.exists() or not npz_tgt.exists():
            print(f"[{frame_key}][{direction_name}] SKIP: missing recon npz")
            n_skip += 1
            continue

        recon_src = load_sam3dbody_recon(npz_src)
        recon_tgt = load_sam3dbody_recon(npz_tgt)

        src_up, dst_up, vertex_ids = pair_sam3dbody_vertices(recon_src, recon_tgt)
        if len(vertex_ids) == 0:
            print(f"[{frame_key}][{direction_name}] SKIP: no shared valid points")
            n_skip += 1
            continue

        src_full, dst_full = restore_sam3dbody_pairs_to_full(
            src_up, dst_up, bbox_src[str(frame_idx)], bbox_tgt[str(frame_idx)], frame_idx,
        )

        src_img_path = full_dir / src_cam / f"{frame_key}.jpg"
        tgt_img_path = full_dir / tgt_cam / f"{frame_key}.jpg"
        if not src_img_path.exists() or not tgt_img_path.exists():
            print(f"[{frame_key}][{direction_name}] SKIP: missing full image")
            n_skip += 1
            continue

        src_img = cv2.imread(str(src_img_path))
        tgt_img = cv2.imread(str(tgt_img_path))
        src_h, src_w = src_img.shape[:2]
        tgt_h, tgt_w = tgt_img.shape[:2]

        src_pts, dst_pts, _ = filter_pairs_inside_image(
            src_full, dst_full, vertex_ids, src_img.shape, tgt_img.shape,
        )
        if len(src_pts) == 0:
            print(f"[{frame_key}][{direction_name}] SKIP: no valid full-coord points")
            n_skip += 1
            continue

        if args.max_points > 0 and len(src_pts) > args.max_points:
            sampled = rng.choice(len(src_pts), args.max_points, replace=False)
            src_pts = src_pts[sampled]
            dst_pts = dst_pts[sampled]
        n_used = len(src_pts)

        # target crop bbox（由 dst/target 点决定）
        x1 = max(0, int(np.floor(dst_pts[:, 0].min())) - args.pad)
        y1 = max(0, int(np.floor(dst_pts[:, 1].min())) - args.pad)
        x2 = min(tgt_w, int(np.ceil(dst_pts[:, 0].max())) + args.pad)
        y2 = min(tgt_h, int(np.ceil(dst_pts[:, 1].max())) + args.pad)
        crop_h, crop_w = y2 - y1, x2 - x1
        if crop_h <= 0 or crop_w <= 0:
            print(f"[{frame_key}][{direction_name}] SKIP: empty crop")
            n_skip += 1
            continue

        # source crop bbox（仅供 debug panel 显示）
        sx1 = max(0, int(np.floor(src_pts[:, 0].min())) - args.pad)
        sy1 = max(0, int(np.floor(src_pts[:, 1].min())) - args.pad)
        sx2 = min(src_w, int(np.ceil(src_pts[:, 0].max())) + args.pad)
        sy2 = min(src_h, int(np.ceil(src_pts[:, 1].max())) + args.pad)

        # grid 仅覆盖 target crop（image_shape=crop），target 点转 crop-local，
        # source 点保持全局 → 输出 backward map 直接是 source 全图采样坐标。
        dst_local = dst_pts - np.array([x1, y1], dtype=dst_pts.dtype)
        backward_map, _ = generate_cpw_backward_map(
            src_pts, dst_local, (crop_h, crop_w), k=args.cpw_knn, power=args.cpw_power,
        )
        warped_crop = warp_image_with_backward_map(src_img, backward_map)
        tgt_crop = tgt_img[y1:y2, x1:x2]
        src_crop = src_img[sy1:sy2, sx1:sx2]

        content_valid = np.max(warped_crop, axis=2) > args.black_thresh
        inside = (
            (backward_map[..., 0] >= 0) & (backward_map[..., 0] < src_w) &
            (backward_map[..., 1] >= 0) & (backward_map[..., 1] < src_h)
        )
        warp_valid = inside & content_valid
        target_valid = np.max(tgt_crop, axis=2) > args.black_thresh

        # cam2 在底部 → "down"，cam4 在顶部 → "up"
        tgt_dir = "down" if tgt_cam == "cam2" else "up"
        src_dir = "down" if src_cam == "cam2" else "up"
        tgt_valid_eroded = directional_erode_valid_mask(target_valid, tgt_dir, args.erode_px)
        warp_valid_eroded = directional_erode_valid_mask(warp_valid, src_dir, args.erode_px)
        composite_tp = target_priority_composite(tgt_crop, warped_crop, tgt_valid_eroded, warp_valid_eroded)

        panel = build_debug_panel(
            src_crop, tgt_crop, warped_crop, composite_tp,
            src_pts, dst_pts, (sx1, sy1), (x1, y1), args.panel_size,
            max_draw=min(args.max_points, 500),
        )
        cv2.imwrite(str(panel_dir / f"{frame_key}.jpg"), panel, [cv2.IMWRITE_JPEG_QUALITY, 95])
        print(f"[{frame_key}][{direction_name}] used={n_used} crop=({x1},{y1},{x2},{y2}) saved")
        n_ok += 1

    return n_ok, n_skip


# ---------------------------------------------------------------------------
# parse segment(s)
# ---------------------------------------------------------------------------
def discover_all_frames(recon_dir: Path, cam: str = "cam2") -> list[int]:
    frames = set()
    for p in (recon_dir / cam).glob("*.npz"):
        try:
            frames.add(int(p.stem))
        except ValueError:
            continue
    return sorted(frames)


def parse_segment_arg(segment_str: str) -> list[int]:
    parts = segment_str.split("_")
    if len(parts) != 2:
        raise ValueError(f"segment must be 'start_end' or 'all', got: {segment_str}")
    start, end = int(parts[0]), int(parts[1])
    if start > end:
        raise ValueError(f"segment start must be <= end: {segment_str}")
    return list(range(start, end + 1))


def resolve_frame_indices(segment_args: list[str], recon_dir: Path) -> list[int]:
    all_frames: list[int] = []
    for seg in segment_args:
        if seg.lower() == "all":
            all_frames.extend(discover_all_frames(recon_dir, "cam2"))
        else:
            all_frames.extend(parse_segment_arg(seg))
    if not all_frames:
        raise ValueError("no frames to process (check --segment and recon_dir)")
    return sorted(set(all_frames))


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="CPW composite from cached SAM 3D Body npz (no model loading)."
    )
    parser.add_argument("--dataset_root", type=str, required=True)
    parser.add_argument("--recon_dir", type=str, required=True)
    parser.add_argument("--segment", type=str, nargs="+", default=["all"])
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--direction", type=str, default="cam4_to_cam2")
    parser.add_argument("--max_points", type=int, default=400)
    parser.add_argument("--pad", type=int, default=200)
    parser.add_argument("--cpw_knn", type=int, default=12)
    parser.add_argument("--cpw_power", type=float, default=2.0)
    parser.add_argument("--black_thresh", type=int, default=3)
    parser.add_argument("--erode_px", type=int, default=20)
    parser.add_argument("--panel_size", type=int, default=420)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root)
    recon_dir = Path(args.recon_dir)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    rng = np.random.RandomState(args.seed)

    frame_indices = resolve_frame_indices(args.segment, recon_dir)
    print(f"[INFO] resolved {len(frame_indices)} frames: {frame_indices[0]:05d} ... {frame_indices[-1]:05d}")

    bbox_dir = dataset_root / "image" / "equirect_crop_rotate"
    bbox_by_cam = {
        "cam2": json.loads((bbox_dir / "cam2_bboxes.json").read_text(encoding="utf-8")),
        "cam4": json.loads((bbox_dir / "cam4_bboxes.json").read_text(encoding="utf-8")),
    }
    full_dir = dataset_root / "image" / "equirect_rotate"

    for src_cam, tgt_cam, direction_name in get_direction_pairs(args.direction):
        n_ok, n_skip = run_direction(
            direction_name, src_cam, tgt_cam, frame_indices,
            recon_dir, full_dir, bbox_by_cam, out, args, rng,
        )
        print(f"[DONE] {direction_name}: {n_ok} ok, {n_skip} skipped (total {len(frame_indices)} frames)")
        print(f"panels: {out / direction_name / 'debug_panel'}")


if __name__ == "__main__":
    main()


r'''
# Example usage:
python src/script/test_function/debug067_sam3dbody_cached_full_cpw.py \
  --dataset_root data_process/Output/20260311_riding \
  --recon_dir data_process/Output/20260311_riding/sam3dbody/recon \
  --segment 704_711 \
  --output_dir data_process/Vis/debug067_sam3dbody_cached_full_cpw \
  --direction both \
  --max_points 400 \
  --pad 200 \
  --cpw_knn 12 \
  --cpw_power 2.0 \
  --black_thresh 3 \
  --erode_px 20 \
  --panel_size 420 \
  --seed 0

# All cached frames:
python src/script/test_function/debug067_sam3dbody_cached_full_cpw.py \
  --dataset_root data_process/Output/20260311_riding \
  --recon_dir data_process/Output/20260311_riding/sam3dbody/recon \
  --direction cam2_to_cam4 \
  --segment all \
  --output_dir data_process/Vis/debug067_sam3dbody_cached_full_cpw \
  --max_points 400 --pad 200 --cpw_knn 12 --cpw_power 2.0 \
  --black_thresh 3 --erode_px 20 --panel_size 420 --seed 0
'''
