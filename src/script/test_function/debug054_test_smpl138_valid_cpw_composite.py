#!/usr/bin/env python3
"""
Test CPW (control point weighted warp) composite using valid DenseKP 138 points.

Reads the delivery package at Circular_Video/01_recv/20260611_smpl_corr,
uses only valid=1 point pairs from erp_corr_json/, and performs a local
inverse-distance-weighted backward warp from source to target camera.

Output: warp, filled composite, overlay, and a 2x3 debug panel per frame.
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

PROJECT_ROOT = SRC_ROOT.parent  # src/.. = project root
PKG_ROOT_DEFAULT = PROJECT_ROOT / "Circular_Video" / "01_recv" / "20260611_smpl_corr"
ERP_W = 5760
ERP_H = 2880


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="CPW composite from valid DenseKP 138 point pairs.")
    p.add_argument("--package_root", type=str, default=str(PKG_ROOT_DEFAULT))
    p.add_argument("--datasets", type=str, nargs="*", default=None)
    p.add_argument("--frames", type=int, nargs="*", default=None)
    p.add_argument("--directions", choices=("both", "cam2_to_cam4", "cam4_to_cam2"), default="both")
    p.add_argument("--output_dir", type=str, default=None)
    p.add_argument("--min_points", type=int, default=8)
    p.add_argument("--pad", type=int, default=180)
    p.add_argument("--cpw_knn", type=int, default=12)
    p.add_argument("--cpw_power", type=float, default=2.0)
    p.add_argument("--max_points", type=int, default=0)
    p.add_argument("--black_thresh", type=int, default=3)
    p.add_argument("--panel_size", type=int, default=420)
    p.add_argument("--save_flow", action="store_true")
    p.add_argument("--dir_erode_px_cam2", type=int, default=15)
    p.add_argument("--dir_erode_px_cam4", type=int, default=15)
    return p.parse_args()


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def find_image(img_dir: Path, cam_id: str, frame_idx: int) -> Path | None:
    for suffix in (".jpg", ".png"):
        p = img_dir / cam_id / f"{frame_idx:05d}{suffix}"
        if p.exists():
            return p
    return None


def load_json(path: Path) -> dict | None:
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"  [WARN] {path}: {e}")
        return None


# ---------------------------------------------------------------------------
# CPW backward map
# ---------------------------------------------------------------------------
def generate_cpw_backward_map(
    target_pts: np.ndarray,   # [N, 2]
    source_pts: np.ndarray,   # [N, 2]
    crop_bbox: tuple[int, int, int, int],
    image_w: int,
    image_h: int,
    knn: int = 12,
    power: float = 2.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """CPW inverse-distance-weighted backward sampling map.

    For each pixel q in the crop region, compute:
        q_source = q + sum(w_i * disp_i) / sum(w_i)
    where disp_i = source_pts[i] - target_pts[i], and
          w_i = 1 / (||q - target_pts[i]|| ** power + eps).

    Returns (map_x, map_y, inside_mask) — all of shape (crop_h, crop_w).
    """
    x1, y1, x2, y2 = crop_bbox
    x1 = max(0, x1); y1 = max(0, y1)
    x2 = min(image_w, x2); y2 = min(image_h, y2)
    crop_h = y2 - y1
    crop_w = x2 - x1

    if crop_h <= 0 or crop_w <= 0:
        return (
            np.zeros((0, 0), dtype=np.float32),
            np.zeros((0, 0), dtype=np.float32),
            np.zeros((0, 0), dtype=bool),
        )

    disp = source_pts - target_pts  # [N, 2]
    N = len(target_pts)
    k = min(knn, N)
    eps = 1e-8

    yy, xx = np.mgrid[y1:y2, x1:x2].astype(np.float32)
    q_grid = np.stack([xx, yy], axis=-1)          # [H, W, 2]
    q_flat = q_grid.reshape(-1, 2)                 # [H*W, 2]

    # Squared distances: [H*W, N]
    diff = q_flat[:, None, :] - target_pts[None, :, :]   # [H*W, N, 2]
    dist2 = np.sum(diff * diff, axis=-1)                   # [H*W, N]

    # For each pixel, find k nearest control points
    if k < N:
        knn_idx = np.argpartition(dist2, k - 1, axis=1)[:, :k]  # [H*W, k]
        knn_dist2 = np.take_along_axis(dist2, knn_idx, axis=1)
        knn_disp = disp[knn_idx]  # [H*W, k, 2]
    else:
        knn_idx = np.arange(N)[None, :]
        knn_dist2 = dist2
        knn_disp = disp[None, :, :]

    # Exact hit: dist < 1e-4 → use exact displacement
    exact_mask = knn_dist2 < 1e-8  # [H*W, k]
    # Weighted interpolation
    weights = 1.0 / (knn_dist2 ** (power * 0.5) + eps)  # dist = sqrt(dist2)
    weights = weights / (weights.sum(axis=1, keepdims=True) + eps)

    # If any exact match, use it directly
    exact_any = exact_mask.any(axis=1)  # [H*W]
    # Default: weighted sum
    weighted_disp = np.sum(weights[..., None] * knn_disp, axis=1)  # [H*W, 2]

    # For exact-match pixels, pick the nearest exact match
    if exact_any.any():
        first_exact = np.argmax(exact_mask, axis=1)  # [H*W]
        exact_disp = knn_disp[np.arange(len(exact_disp)), first_exact]
        weighted_disp[exact_any] = exact_disp[exact_any]

    q_src = q_flat + weighted_disp
    map_x = q_src[:, 0].reshape(crop_h, crop_w).astype(np.float32)
    map_y = q_src[:, 1].reshape(crop_h, crop_w).astype(np.float32)

    inside = (
        (map_x >= 0) & (map_x < image_w) &
        (map_y >= 0) & (map_y < image_h)
    )

    return map_x, map_y, inside


# ---------------------------------------------------------------------------
# debug panel
# ---------------------------------------------------------------------------
def resize_to_width(img: np.ndarray, target_w: int) -> np.ndarray:
    h, w = img.shape[:2]
    if w == target_w:
        return img
    scale = target_w / w
    new_h = int(round(h * scale))
    return cv2.resize(img, (target_w, new_h), interpolation=cv2.INTER_LINEAR)


def draw_kps(img: np.ndarray, pts: np.ndarray, offset: tuple, color: tuple, radius: int = 4) -> np.ndarray:
    out = img.copy()
    ox, oy = offset
    for i in range(len(pts)):
        x, y = int(round(pts[i, 0] - ox)), int(round(pts[i, 1] - oy))
        if 0 <= x < out.shape[1] and 0 <= y < out.shape[0]:
            cv2.circle(out, (x, y), radius, color, -1, lineType=cv2.LINE_AA)
    return out


def build_debug_panel(
    src_crop: np.ndarray,
    tgt_crop: np.ndarray,
    warped_crop: np.ndarray,
    composite_target_priority: np.ndarray,
    src_pts: np.ndarray,
    tgt_pts: np.ndarray,
    crop_offset: tuple[int, int],
    panel_w: int,
) -> np.ndarray:
    """2x3 panel: src+kps | tgt+kps | warp | overlay | target_priority | valid_lines_black"""
    src_kp = draw_kps(src_crop, src_pts, crop_offset, (0, 255, 0))
    tgt_kp = draw_kps(tgt_crop, tgt_pts, crop_offset, (0, 255, 255))
    overlay = cv2.addWeighted(tgt_crop, 0.5, warped_crop, 0.5, 0)

    pair_canvas = np.zeros_like(tgt_crop)
    n = min(len(src_pts), len(tgt_pts))
    for i in range(n):
        s = (int(round(src_pts[i, 0] - crop_offset[0])), int(round(src_pts[i, 1] - crop_offset[1])))
        t = (int(round(tgt_pts[i, 0] - crop_offset[0])), int(round(tgt_pts[i, 1] - crop_offset[1])))
        cv2.line(pair_canvas, s, t, (180, 180, 180), 1, lineType=cv2.LINE_AA)
        cv2.circle(pair_canvas, s, 4, (0, 255, 0), -1)
        cv2.circle(pair_canvas, t, 4, (0, 255, 255), -1)

    panels = [resize_to_width(p, panel_w) for p in (
        src_kp, tgt_kp, warped_crop, overlay, composite_target_priority, pair_canvas,
    )]
    row1_h = max(p.shape[0] for p in panels[:3])
    row2_h = max(p.shape[0] for p in panels[3:])

    def pad(p, h):
        if p.shape[0] == h:
            return p
        out = np.zeros((h, p.shape[1], 3), dtype=np.uint8)
        out[:p.shape[0]] = p
        return out

    row1 = [pad(p, row1_h) for p in panels[:3]]
    row2 = [pad(p, row2_h) for p in panels[3:]]
    labels = ["src+kps", "tgt+kps", "warp", "overlay", "target_priority", "valid_lines_black"]
    for i, (r, lbl) in enumerate(((row1, labels[:3]), (row2, labels[3:]))):
        for j in range(len(r)):
            cv2.putText(r[j], lbl[j], (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    top = np.hstack(row1)
    bot = np.hstack(row2)
    mw = max(top.shape[1], bot.shape[1])
    tp = np.zeros((top.shape[0], mw, 3), dtype=np.uint8); tp[:, :top.shape[1]] = top
    bp = np.zeros((bot.shape[0], mw, 3), dtype=np.uint8); bp[:, :bot.shape[1]] = bot
    return np.vstack([tp, bp])


# ---------------------------------------------------------------------------
# directional erosion
# ---------------------------------------------------------------------------
def directional_erode_valid_mask(valid_mask: np.ndarray, direction: str, ksize: int) -> np.ndarray:
    """Erode valid mask from one boundary only.
    direction="up": top boundary eroded inward (cam4).
    direction="down": bottom boundary eroded inward (cam2).
    """
    if ksize <= 1:
        return valid_mask
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, ksize))
    if direction == "down":
        anchor = (0, 0)
    elif direction == "up":
        anchor = (0, ksize - 1)
    else:
        raise ValueError(f"Unknown direction: {direction}")
    eroded = cv2.erode(valid_mask.astype(np.uint8), kernel, anchor=anchor)
    return eroded > 0


# ---------------------------------------------------------------------------
# per-frame processing
# ---------------------------------------------------------------------------
def process_frame(
    frame_idx: int,
    dataset_dir: Path,
    output_dir: Path,
    direction: str,
    args: argparse.Namespace,
) -> bool:
    json_dir = dataset_dir / "smpl_corr" / "erp_corr_json"
    img_dir = dataset_dir / "image" / "equirect_rotate"

    jp = json_dir / f"{frame_idx:05d}.json"
    if not jp.exists():
        print(f"  [SKIP] missing JSON: {jp}")
        return False

    data = load_json(jp)
    if data is None:
        return False

    pairs = np.array(data["pairs"], dtype=np.float32)
    valid = np.array(data.get("valid", [1] * len(pairs)), dtype=bool)
    values = np.array(data.get("values", [0] * len(pairs)), dtype=np.float32)

    if direction == "cam2_to_cam4":
        src_cam, tgt_cam = "cam2", "cam4"
        src_pts = pairs[:, 0:2]
        tgt_pts = pairs[:, 2:4]
    else:  # cam4_to_cam2
        src_cam, tgt_cam = "cam4", "cam2"
        src_pts = pairs[:, 2:4]
        tgt_pts = pairs[:, 0:2]

    # Filter to valid + finite + in-bounds
    mask = valid.copy()
    mask &= np.isfinite(src_pts).all(axis=1) & np.isfinite(tgt_pts).all(axis=1)
    mask &= (src_pts[:, 0] >= 0) & (src_pts[:, 0] < ERP_W) & (src_pts[:, 1] >= 0) & (src_pts[:, 1] < ERP_H)
    mask &= (tgt_pts[:, 0] >= 0) & (tgt_pts[:, 0] < ERP_W) & (tgt_pts[:, 1] >= 0) & (tgt_pts[:, 1] < ERP_H)

    # Optional: top-k by values
    if args.max_points > 0 and mask.sum() > args.max_points:
        valid_idx = np.where(mask)[0]
        topk = valid_idx[np.argsort(values[mask])[-args.max_points:]]
        mask[:] = False
        mask[topk] = True

    n_used = int(mask.sum())
    if n_used < args.min_points:
        print(f"  [{frame_idx:05d}][{direction}] skip: only {n_used} valid points (min={args.min_points})")
        return False

    src_kps = src_pts[mask]
    tgt_kps = tgt_pts[mask]

    # Crop bbox from union of valid points
    all_pts = np.concatenate([src_kps, tgt_kps], axis=0)
    x1 = int(np.floor(all_pts[:, 0].min())) - args.pad
    y1 = int(np.floor(all_pts[:, 1].min())) - args.pad
    x2 = int(np.ceil(all_pts[:, 0].max())) + args.pad
    y2 = int(np.ceil(all_pts[:, 1].max())) + args.pad
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(ERP_W, x2), min(ERP_H, y2)
    crop_bbox = (x1, y1, x2, y2)

    # Load images
    src_img_path = find_image(img_dir, src_cam, frame_idx)
    tgt_img_path = find_image(img_dir, tgt_cam, frame_idx)
    if src_img_path is None:
        print(f"  [{frame_idx:05d}][{direction}] missing {src_cam} image")
        return False
    if tgt_img_path is None:
        print(f"  [{frame_idx:05d}][{direction}] missing {tgt_cam} image")
        return False

    src_img = cv2.imread(str(src_img_path), cv2.IMREAD_COLOR)
    tgt_img = cv2.imread(str(tgt_img_path), cv2.IMREAD_COLOR)

    # CPW backward map
    map_x, map_y, inside = generate_cpw_backward_map(
        tgt_kps, src_kps, crop_bbox, ERP_W, ERP_H,
        knn=args.cpw_knn, power=args.cpw_power,
    )
    if map_x.size == 0:
        return False

    warped_crop = cv2.remap(
        src_img, map_x, map_y, cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT, borderValue=0,
    )
    tgt_crop = tgt_img[y1:y2, x1:x2]
    src_crop = src_img[y1:y2, x1:x2]

    content_valid = np.max(warped_crop, axis=2) > args.black_thresh
    warp_valid = inside & content_valid

    # Directional erosion to reduce black seam artifacts
    warp_dir = "down" if src_cam == "cam2" else "up"
    tgt_dir = "down" if tgt_cam == "cam2" else "up"
    warp_erode_px = args.dir_erode_px_cam2 if src_cam == "cam2" else args.dir_erode_px_cam4
    tgt_erode_px = args.dir_erode_px_cam2 if tgt_cam == "cam2" else args.dir_erode_px_cam4
    warp_valid_safe = directional_erode_valid_mask(warp_valid, warp_dir, warp_erode_px)
    target_safe = directional_erode_valid_mask(
        np.max(tgt_crop, axis=2) > args.black_thresh, tgt_dir, tgt_erode_px,
    )

    # target-priority: keep target pixels that survived erosion;
    # only fill gaps with warped source
    fill_mask = (~target_safe) & warp_valid_safe
    composite_target_priority = tgt_crop.copy()
    composite_target_priority[fill_mask] = warped_crop[fill_mask]

    # Full ERP warped (black bg)
    warped_full = np.zeros_like(tgt_img)
    warped_full[y1:y2, x1:x2] = warped_crop

    # Output dirs
    out = output_dir / direction
    for d in ("warp", "composite_target_priority", "overlay", "debug_panel"):
        (out / d).mkdir(parents=True, exist_ok=True)

    fk = f"{frame_idx:05d}"
    cv2.imwrite(str(out / "warp" / f"{fk}.jpg"), warped_full, [cv2.IMWRITE_JPEG_QUALITY, 95])
    cv2.imwrite(str(out / "composite_target_priority" / f"{fk}.jpg"), composite_target_priority, [cv2.IMWRITE_JPEG_QUALITY, 95])
    ov = cv2.addWeighted(tgt_crop, 0.5, warped_crop, 0.5, 0)
    cv2.imwrite(str(out / "overlay" / f"{fk}.jpg"), ov, [cv2.IMWRITE_JPEG_QUALITY, 95])

    panel = build_debug_panel(
        src_crop, tgt_crop, warped_crop, composite_target_priority,
        src_kps, tgt_kps, (x1, y1), args.panel_size,
    )
    cv2.imwrite(str(out / "debug_panel" / f"{fk}.jpg"), panel, [cv2.IMWRITE_JPEG_QUALITY, 95])

    if args.save_flow:
        flow = np.zeros((ERP_H, ERP_W, 2), dtype=np.float32)
        flow[y1:y2, x1:x2, 0] = map_x - np.arange(x1, x2, dtype=np.float32)[None, :]
        flow[y1:y2, x1:x2, 1] = map_y - np.arange(y1, y2, dtype=np.float32)[:, None]
        np.save(str(out / f"cpw_flow_{fk}.npy"), flow)

    print(f"  [{frame_idx:05d}][{direction}] valid_used={n_used} crop=({x1},{y1},{x2},{y2}) saved")
    return True


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main() -> None:
    args = parse_args()
    pkg = Path(args.package_root)
    data_dir = pkg / "data"
    if not data_dir.is_dir():
        print(f"[ERROR] data/ not found under {pkg}")
        return

    datasets = args.datasets
    if datasets is None:
        datasets = sorted(d.name for d in data_dir.iterdir() if d.is_dir())

    directions = []
    if args.directions in ("both", "cam2_to_cam4"):
        directions.append("cam2_to_cam4")
    if args.directions in ("both", "cam4_to_cam2"):
        directions.append("cam4_to_cam2")

    output_root = Path(args.output_dir) if args.output_dir else (PROJECT_ROOT / "data_process" / "Vis" / "debug054_cpw_valid_composite")
    output_root.mkdir(parents=True, exist_ok=True)

    print(f"Package root: {pkg}")
    print(f"Datasets: {datasets}")
    print(f"Directions: {directions}")
    print()

    for ds in datasets:
        ds_dir = data_dir / ds
        if not ds_dir.is_dir():
            print(f"[SKIP] {ds_dir} not found")
            continue

        json_dir = ds_dir / "smpl_corr" / "erp_corr_json"
        if not json_dir.is_dir():
            print(f"[SKIP] {ds}: no erp_corr_json/")
            continue

        frame_indices = args.frames
        if frame_indices is None:
            frame_indices = sorted(int(p.stem) for p in json_dir.glob("*.json"))

        out_dir = output_root / ds / "erp"
        print(f"[DATASET] {ds}  frames={len(frame_indices)}")
        stats: dict[str, int] = {}

        for direction in directions:
            n_ok = 0
            for fi in frame_indices:
                ok = process_frame(fi, ds_dir, out_dir, direction, args)
                if ok:
                    n_ok += 1
            stats[direction] = n_ok
            print(f"  {direction}: ok {n_ok}/{len(frame_indices)}")

    print("\n[DONE]")


if __name__ == "__main__":
    main()


r'''
# Both directions, all frames — output to data_process/Vis/debug054_cpw_valid_composite
python src/script/test_function/debug054_test_smpl138_valid_cpw_composite.py \
  --package_root Circular_Video/01_send/20260611_smpl_corr \
  --datasets 20260311_riding \
  --directions both

# Specific frames, single direction
python src/script/test_function/debug054_test_smpl138_valid_cpw_composite.py \
  --package_root Circular_Video/01_recv/20260611_smpl_corr \
  --datasets 20260311_riding \
  --frames 513 704 1217 \
  --directions cam2_to_cam4

# Override output directory
python src/script/test_function/debug054_test_smpl138_valid_cpw_composite.py \
  --package_root Circular_Video/01_recv/20260611_smpl_corr \
  --datasets 20260311_riding \
  --output_dir data_process/Vis/my_cpw_test
'''

