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

Output:
  output_dir/
    warp/, overlay/, composite_target_priority/, debug_panel/, points/, pairs_npz/
    summary.csv
"""

from __future__ import annotations

import argparse
import csv
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
# output dirs
# ---------------------------------------------------------------------------
def ensure_output_dirs(base: Path) -> dict:
    subs = ["warp", "overlay", "composite_target_priority", "debug_panel", "points", "pairs_npz"]
    dirs = {}
    for s in subs:
        d = base / s
        d.mkdir(parents=True, exist_ok=True)
        dirs[s] = d
    return dirs


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
    parser.add_argument("--max_points", type=int, default=1500)
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
    out_dirs = ensure_output_dirs(out)

    rng = np.random.RandomState(args.seed)

    # resolve frames from cached npz
    frame_indices = resolve_frame_indices(args.segment, recon_dir)
    print(f"[INFO] resolved {len(frame_indices)} frames: {frame_indices[0]:05d} ... {frame_indices[-1]:05d}")

    # load bboxes
    bbox_dir = dataset_root / "image" / "equirect_crop_rotate"
    bbox_cam2 = json.loads((bbox_dir / "cam2_bboxes.json").read_text(encoding="utf-8"))
    bbox_cam4 = json.loads((bbox_dir / "cam4_bboxes.json").read_text(encoding="utf-8"))

    # image dirs
    full_dir = dataset_root / "image" / "equirect_rotate"

    # summary csv
    csv_path = out / "summary.csv"
    csv_file = open(str(csv_path), "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow([
        "frame", "status", "reason",
        "raw_points", "valid_points", "used_points",
        "crop_x1", "crop_y1", "crop_x2", "crop_y2",
        "disp_mean", "disp_median", "disp_std",
    ])

    n_ok = 0
    n_skip = 0

    for frame_idx in frame_indices:
        frame_key = f"{frame_idx:05d}"
        print(f"\n[{frame_key}] processing ...")

        # --------------------------------------------------------------
        # 1. read cached npz via library
        # --------------------------------------------------------------
        npz2_path = recon_dir / "cam2" / f"{frame_key}.npz"
        npz4_path = recon_dir / "cam4" / f"{frame_key}.npz"

        if not npz2_path.exists():
            print(f"[{frame_key}] SKIP: missing {npz2_path}")
            csv_writer.writerow([frame_idx, "skip", "missing cam2 npz", *([""] * 8)])
            n_skip += 1
            continue
        if not npz4_path.exists():
            print(f"[{frame_key}] SKIP: missing {npz4_path}")
            csv_writer.writerow([frame_idx, "skip", "missing cam4 npz", *([""] * 8)])
            n_skip += 1
            continue

        recon_cam2 = load_sam3dbody_recon(npz2_path)
        recon_cam4 = load_sam3dbody_recon(npz4_path)

        # --------------------------------------------------------------
        # 2. pair vertices by vertex id
        # --------------------------------------------------------------
        src_pts_upright, dst_pts_upright, vertex_ids = pair_sam3dbody_vertices(recon_cam4, recon_cam2)
        n_raw = len(vertex_ids)
        print(f"[{frame_key}] raw points: {n_raw}")

        if n_raw == 0:
            print(f"[{frame_key}] SKIP: no valid shared vertex points")
            csv_writer.writerow([frame_idx, "skip", "no valid shared points", 0, *([""] * 7)])
            n_skip += 1
            continue

        # --------------------------------------------------------------
        # 3. restore to equirect_rotate (via library)
        # --------------------------------------------------------------
        bb2 = bbox_cam2[str(frame_idx)]
        bb4 = bbox_cam4[str(frame_idx)]

        src_pts_full, dst_pts_full = restore_sam3dbody_pairs_to_full(
            src_pts_upright, dst_pts_upright, bb4, bb2, frame_idx,
        )

        # --------------------------------------------------------------
        # 4. load full images
        # --------------------------------------------------------------
        src_img_path = full_dir / "cam4" / f"{frame_key}.jpg"
        tgt_img_path = full_dir / "cam2" / f"{frame_key}.jpg"

        if not src_img_path.exists():
            print(f"[{frame_key}] SKIP: missing {src_img_path}")
            csv_writer.writerow([frame_idx, "skip", "missing cam4 full image", *([""] * 8)])
            n_skip += 1
            continue
        if not tgt_img_path.exists():
            print(f"[{frame_key}] SKIP: missing {tgt_img_path}")
            csv_writer.writerow([frame_idx, "skip", "missing cam2 full image", *([""] * 8)])
            n_skip += 1
            continue

        src_img = cv2.imread(str(src_img_path))
        tgt_img = cv2.imread(str(tgt_img_path))
        src_h, src_w = src_img.shape[:2]
        tgt_h, tgt_w = tgt_img.shape[:2]

        # --------------------------------------------------------------
        # 5. filter full-coord points (via library)
        # --------------------------------------------------------------
        src_pts, dst_pts, used_vertex_ids = filter_pairs_inside_image(
            src_pts_full, dst_pts_full, vertex_ids, src_img.shape, tgt_img.shape,
        )
        n_valid_full = len(src_pts)
        print(f"[{frame_key}] valid points (full coord): {n_valid_full}")

        if n_valid_full == 0:
            print(f"[{frame_key}] SKIP: no valid full-coord points")
            csv_writer.writerow([frame_idx, "skip", "no valid full-coord points", *([""] * 7)])
            n_skip += 1
            continue

        if args.max_points > 0 and n_valid_full > args.max_points:
            sampled = rng.choice(len(src_pts), args.max_points, replace=False)
            src_pts = src_pts[sampled]
            dst_pts = dst_pts[sampled]
            used_vertex_ids = used_vertex_ids[sampled]
        n_used = len(src_pts)
        print(f"[{frame_key}] used points: {n_used}")

        # --------------------------------------------------------------
        # 6. crop bbox
        # --------------------------------------------------------------
        x1 = int(np.floor(dst_pts[:, 0].min())) - args.pad
        y1 = int(np.floor(dst_pts[:, 1].min())) - args.pad
        x2 = int(np.ceil(dst_pts[:, 0].max())) + args.pad
        y2 = int(np.ceil(dst_pts[:, 1].max())) + args.pad
        x1 = max(0, x1); y1 = max(0, y1)
        x2 = min(tgt_w, x2); y2 = min(tgt_h, y2)
        crop_bbox = (x1, y1, x2, y2)
        print(f"[{frame_key}] crop_bbox: {crop_bbox}")

        sx1 = int(np.floor(src_pts[:, 0].min())) - args.pad
        sy1 = int(np.floor(src_pts[:, 1].min())) - args.pad
        sx2 = int(np.ceil(src_pts[:, 0].max())) + args.pad
        sy2 = int(np.ceil(src_pts[:, 1].max())) + args.pad
        sx1 = max(0, sx1); sy1 = max(0, sy1)
        sx2 = min(src_w, sx2); sy2 = min(src_h, sy2)

        # --------------------------------------------------------------
        # 7. displacement stats
        # --------------------------------------------------------------
        disp = src_pts - dst_pts
        disp_norm = np.linalg.norm(disp, axis=1)
        d_mean, d_median, d_std = disp_norm.mean(), np.median(disp_norm), disp_norm.std()
        print(f"[{frame_key}] disp: mean={d_mean:.2f} median={d_median:.2f} std={d_std:.2f}")

        # --------------------------------------------------------------
        # 8. CPW backward map (via library)
        # --------------------------------------------------------------
        crop_h = y2 - y1
        crop_w = x2 - x1
        if crop_h <= 0 or crop_w <= 0:
            print(f"[{frame_key}] SKIP: empty crop")
            csv_writer.writerow([frame_idx, "skip", "empty crop", *([""] * 7)])
            n_skip += 1
            continue

        backward_map, stats = generate_cpw_backward_map(
            src_pts, dst_pts, (crop_h, crop_w),
            k=args.cpw_knn, power=args.cpw_power,
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

        # target-priority with directional erosion (via library)
        tgt_valid_eroded = directional_erode_valid_mask(target_valid, "down", args.erode_px)
        warp_valid_eroded = directional_erode_valid_mask(warp_valid, "up", args.erode_px)
        composite_tp = target_priority_composite(tgt_crop, warped_crop, tgt_valid_eroded, warp_valid_eroded)

        # full-size warp
        warped_full = np.zeros_like(tgt_img)
        warped_full[y1:y2, x1:x2] = warped_crop

        # --------------------------------------------------------------
        # 9. save outputs
        # --------------------------------------------------------------
        cv2.imwrite(str(out_dirs["warp"] / f"{frame_key}.jpg"), warped_full, [cv2.IMWRITE_JPEG_QUALITY, 95])
        cv2.imwrite(str(out_dirs["composite_target_priority"] / f"{frame_key}.jpg"), composite_tp, [cv2.IMWRITE_JPEG_QUALITY, 95])
        ov = cv2.addWeighted(tgt_crop, 0.5, warped_crop, 0.5, 0)
        cv2.imwrite(str(out_dirs["overlay"] / f"{frame_key}.jpg"), ov, [cv2.IMWRITE_JPEG_QUALITY, 95])

        # points
        src_pts_img = draw_points(src_img, src_pts, (0, 0), (0, 255, 0), radius=5, max_pts=args.max_points)
        tgt_pts_img = draw_points(tgt_img, dst_pts, (0, 0), (0, 255, 255), radius=5, max_pts=args.max_points)
        cv2.imwrite(str(out_dirs["points"] / f"{frame_key}_cam4_restored_points.jpg"), src_pts_img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        cv2.imwrite(str(out_dirs["points"] / f"{frame_key}_cam2_restored_points.jpg"), tgt_pts_img, [cv2.IMWRITE_JPEG_QUALITY, 95])

        # side-by-side pair lines
        pair_full_h = max(src_h, tgt_h)
        pair_canvas = np.zeros((pair_full_h, src_w + tgt_w, 3), dtype=np.uint8)
        pair_canvas[:src_h, :src_w] = src_img
        pair_canvas[:tgt_h, src_w:] = tgt_img
        n_pts = len(src_pts)
        if n_pts > args.max_points:
            plt_idx = np.linspace(0, n_pts - 1, args.max_points).astype(int)
            s_vis = src_pts[plt_idx]
            d_vis = dst_pts[plt_idx]
        else:
            s_vis = src_pts
            d_vis = dst_pts
        for i in range(len(s_vis)):
            sx_ = int(round(s_vis[i, 0]))
            sy_ = int(round(s_vis[i, 1]))
            dx_ = int(round(d_vis[i, 0])) + src_w
            dy_ = int(round(d_vis[i, 1]))
            cv2.line(pair_canvas, (sx_, sy_), (dx_, dy_), (180, 180, 180), 1, lineType=cv2.LINE_AA)
        for i in range(len(s_vis)):
            sx_ = int(round(s_vis[i, 0]))
            sy_ = int(round(s_vis[i, 1]))
            dx_ = int(round(d_vis[i, 0])) + src_w
            dy_ = int(round(d_vis[i, 1]))
            cv2.circle(pair_canvas, (sx_, sy_), 4, (0, 255, 0), -1)
            cv2.circle(pair_canvas, (dx_, dy_), 4, (0, 255, 255), -1)
        cv2.imwrite(str(out_dirs["points"] / f"{frame_key}_pair_lines.jpg"), pair_canvas, [cv2.IMWRITE_JPEG_QUALITY, 95])

        # debug panel
        panel = build_debug_panel(
            src_crop, tgt_crop, warped_crop, composite_tp,
            src_pts, dst_pts, (sx1, sy1), (x1, y1), args.panel_size,
            max_draw=min(args.max_points, 500),
        )
        cv2.imwrite(str(out_dirs["debug_panel"] / f"{frame_key}.jpg"), panel, [cv2.IMWRITE_JPEG_QUALITY, 95])

        # pairs npz
        np.savez_compressed(
            str(out_dirs["pairs_npz"] / f"{frame_key}_cam4_to_cam2_pairs_equirect_rotate.npz"),
            src_cam="cam4",
            dst_cam="cam2",
            src_pts=src_pts,
            dst_pts=dst_pts,
            vertex_ids=used_vertex_ids,
            frame_idx=frame_idx,
            crop_bbox=np.array([x1, y1, x2, y2], dtype=np.int32),
            image_path_src=str(src_img_path.resolve()),
            image_path_tgt=str(tgt_img_path.resolve()),
        )

        csv_writer.writerow([
            frame_idx, "ok", "",
            n_raw, n_valid_full, n_used,
            x1, y1, x2, y2,
            f"{d_mean:.2f}", f"{d_median:.2f}", f"{d_std:.2f}",
        ])

        print(f"[{frame_key}] saved debug_panel/{frame_key}.jpg")
        n_ok += 1

    csv_file.close()
    print(f"\n[DONE] {n_ok} ok, {n_skip} skipped (total {len(frame_indices)} frames)")
    print(f"summary: {csv_path}")


if __name__ == "__main__":
    main()


r'''
# Example usage:
python src/script/test_function/debug067_sam3dbody_cached_full_cpw.py \
  --dataset_root data_process/Output/20260311_riding \
  --recon_dir data_process/Output/20260311_riding/sam3dbody_recon \
  --segment 513_518 \
  --output_dir data_process/Vis/debug067_sam3dbody_cached_full_cpw/513_518 \
  --direction cam4_to_cam2 \
  --max_points 1500 \
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
  --recon_dir data_process/Output/20260311_riding/sam3dbody_recon \
  --segment all \
  --output_dir data_process/Vis/debug067_sam3dbody_cached_full_cpw/all \
  --max_points 1500 --pad 200 --cpw_knn 12 --cpw_power 2.0 \
  --black_thresh 3 --erode_px 20 --panel_size 420 --seed 0
'''
