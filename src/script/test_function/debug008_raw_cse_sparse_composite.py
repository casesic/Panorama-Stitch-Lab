from __future__ import annotations

import argparse
from pathlib import Path

import config_script  # noqa: F401 — ensure src/ on sys.path

import cv2
import numpy as np

from lib_pano.fitting.cse_io import load_cse_result
from lib_pano.pipeline.bbox_io import load_bbox_dict


# ---------------------------------------------------------------------------
# argparse
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Raw DensePose CSE sparse correspondence for local human compositing."
    )
    parser.add_argument("--image_root", type=str, required=True)
    parser.add_argument("--cse_npz_dir", type=str, required=True)
    parser.add_argument("--bbox_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--frames", type=int, nargs="+", required=True)
    parser.add_argument("--src_cam", type=str, required=True)
    parser.add_argument("--tgt_cam", type=str, required=True)
    parser.add_argument("--image_width", type=int, required=True)
    parser.add_argument("--image_height", type=int, required=True)
    parser.add_argument(
        "--max_points",
        type=int,
        default=300,
        help="Max number of sparse correspondences (randomly sampled if exceeded).",
    )
    parser.add_argument(
        "--min_pixels_per_vertex",
        type=int,
        default=2,
        help="Minimum CSE pixels for a vertex id to be used.",
    )
    parser.add_argument(
        "--tps_lambda",
        type=float,
        default=0.0,
        help="TPS regularization parameter (0 = exact interpolation).",
    )
    parser.add_argument(
        "--pad",
        type=int,
        default=80,
        help="Padding around bbox for debug panel crops.",
    )
    parser.add_argument(
        "--panel_size",
        type=int,
        default=400,
        help="Target width for each debug panel sub-image.",
    )
    parser.add_argument(
        "--black_thresh",
        type=int,
        default=3,
        help="Pixels in warped_src with all channels <= this value are treated as invalid.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for subsampling correspondences.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# TPS warp (crop region only)
# ---------------------------------------------------------------------------
def fit_tps(
    src_pts: np.ndarray,  # [N, 2]
    dst_pts: np.ndarray,  # [N, 2]
    regularization: float = 0.0,
):
    """Fit TPS mapping src_pts -> dst_pts. Returns the transformer object."""
    tps = cv2.createThinPlateSplineShapeTransformer(regularizationParameter=regularization)
    matches = [cv2.DMatch(i, i, 0) for i in range(len(src_pts))]
    tps.estimateTransformation(
        src_pts.reshape(1, -1, 2).astype(np.float32),
        dst_pts.reshape(1, -1, 2).astype(np.float32),
        matches,
    )
    return tps


def warp_crop_region(
    src_img: np.ndarray,
    tps,
    crop_region: tuple[int, int, int, int],  # (x1, y1, x2, y2) in full ERP coords
    erp_h: int,
    erp_w: int,
    interpolation: int = cv2.INTER_LINEAR,
) -> np.ndarray:
    """Apply TPS warp to a crop region only. Returns warped crop image."""
    x1, y1, x2, y2 = crop_region
    crop_w = x2 - x1
    crop_h = y2 - y1

    y_grid, x_grid = np.mgrid[y1:y2, x1:x2].astype(np.float32)
    grid = np.stack([x_grid, y_grid], axis=-1)  # [crop_h, crop_w, 2]
    grid_flat = grid.reshape(1, -1, 2)
    _, warped_flat = tps.applyTransformation(grid_flat)
    warped_coords = warped_flat.reshape(crop_h, crop_w, 2)

    map_x = warped_coords[..., 0].astype(np.float32)
    map_y = warped_coords[..., 1].astype(np.float32)

    return cv2.remap(
        src_img, map_x, map_y,
        interpolation,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )


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


def draw_keypoints(
    img: np.ndarray,
    kps: np.ndarray,
    offset: tuple[int, int],
    color: tuple[int, int, int],
    radius: int = 3,
) -> np.ndarray:
    out = img.copy()
    ox, oy = offset
    for kp in kps:
        x, y = int(round(kp[0] - ox)), int(round(kp[1] - oy))
        if 0 <= x < out.shape[1] and 0 <= y < out.shape[0]:
            cv2.circle(out, (x, y), radius, color, -1)
    return out


def make_debug_panel(
    tgt_crop: np.ndarray,
    src_crop: np.ndarray,
    tgt_kps: np.ndarray,
    src_kps: np.ndarray,
    warped_crop: np.ndarray,
    filled_warp: np.ndarray,
    filled_target: np.ndarray,
    crop_offset: tuple[int, int],
    panel_w: int,
) -> np.ndarray:
    """Assemble a 2x3 debug panel from pre-cropped images."""
    tgt_kp = draw_keypoints(tgt_crop, tgt_kps, crop_offset, (0, 255, 0), radius=4)
    src_kp = draw_keypoints(src_crop, src_kps, crop_offset, (0, 255, 0), radius=4)

    overlay = cv2.addWeighted(tgt_crop, 0.5, warped_crop, 0.5, 0)

    panels = [
        resize_to_width(tgt_kp, panel_w),
        resize_to_width(src_kp, panel_w),
        resize_to_width(warped_crop, panel_w),
        resize_to_width(overlay, panel_w),
        resize_to_width(filled_warp, panel_w),
        resize_to_width(filled_target, panel_w),
    ]

    row1_h = max(p.shape[0] for p in panels[:3])
    row2_h = max(p.shape[0] for p in panels[3:])

    def pad_to_height(img: np.ndarray, target_h: int) -> np.ndarray:
        h, w = img.shape[:2]
        if h == target_h:
            return img
        out = np.zeros((target_h, w, 3), dtype=np.uint8)
        out[:h, :w] = img
        return out

    row1 = [pad_to_height(p, row1_h) for p in panels[:3]]
    row2 = [pad_to_height(p, row2_h) for p in panels[3:]]

    font = cv2.FONT_HERSHEY_SIMPLEX
    labels = ["target+kps", "source+kps", "warped_src", "overlay", "filled_warp", "filled_target"]

    def add_label(img: np.ndarray, text: str) -> np.ndarray:
        out = img.copy()
        cv2.putText(out, text, (5, 20), font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        return out

    row1_labeled = [add_label(p, l) for p, l in zip(row1, labels[:3])]
    row2_labeled = [add_label(p, l) for p, l in zip(row2, labels[3:])]

    top_row = np.hstack(row1_labeled)
    bot_row = np.hstack(row2_labeled)

    max_w = max(top_row.shape[1], bot_row.shape[1])
    top_padded = np.zeros((top_row.shape[0], max_w, 3), dtype=np.uint8)
    top_padded[:, :top_row.shape[1]] = top_row
    bot_padded = np.zeros((bot_row.shape[0], max_w, 3), dtype=np.uint8)
    bot_padded[:, :bot_row.shape[1]] = bot_row

    return np.vstack([top_padded, bot_padded])


# ---------------------------------------------------------------------------
# CSE correspondence building
# ---------------------------------------------------------------------------
def build_cse_correspondences(
    src_result: CSEResult,
    tgt_result: CSEResult,
    min_pixels_per_vertex: int,
    max_points: int,
    rng: np.random.RandomState,
) -> tuple[np.ndarray, np.ndarray]:
    """Build sparse correspondences from CSE results via vertex-id grouping + median."""
    # Group valid points by vertex_id for src and tgt
    src_valid = src_result.valid
    tgt_valid = tgt_result.valid

    src_vids = src_result.vertex_ids[src_valid]
    tgt_vids = tgt_result.vertex_ids[tgt_valid]
    src_xy = src_result.xy_full[src_valid]
    tgt_xy = tgt_result.xy_full[tgt_valid]

    # Find unique vertex ids and their counts in each
    src_vids_unique, src_counts = np.unique(src_vids, return_counts=True)
    tgt_vids_unique, tgt_counts = np.unique(tgt_vids, return_counts=True)

    src_count_map = dict(zip(src_vids_unique, src_counts))
    tgt_count_map = dict(zip(tgt_vids_unique, tgt_counts))

    # Common vertex ids
    common_vids = np.intersect1d(src_vids_unique, tgt_vids_unique)

    src_pts_list = []
    tgt_pts_list = []
    for vid in common_vids:
        if src_count_map[vid] < min_pixels_per_vertex:
            continue
        if tgt_count_map[vid] < min_pixels_per_vertex:
            continue

        src_mask = src_vids == vid
        tgt_mask = tgt_vids == vid

        src_median = np.median(src_xy[src_mask], axis=0)
        tgt_median = np.median(tgt_xy[tgt_mask], axis=0)

        src_pts_list.append(src_median)
        tgt_pts_list.append(tgt_median)

    if not src_pts_list:
        return np.empty((0, 2)), np.empty((0, 2))

    src_pts = np.array(src_pts_list, dtype=np.float32)
    tgt_pts = np.array(tgt_pts_list, dtype=np.float32)

    # Subsample if too many points
    if len(src_pts) > max_points:
        indices = rng.choice(len(src_pts), size=max_points, replace=False)
        src_pts = src_pts[indices]
        tgt_pts = tgt_pts[indices]

    return src_pts, tgt_pts


# ---------------------------------------------------------------------------
# per-frame processing
# ---------------------------------------------------------------------------
def process_frame(
    frame_idx: int,
    image_root: Path,
    cse_npz_dir: Path,
    bbox_dict_src: dict,
    bbox_dict_tgt: dict,
    output_dir: Path,
    erp_h: int,
    erp_w: int,
    src_cam: str,
    tgt_cam: str,
    tps_lambda: float,
    pad: int,
    panel_w: int,
    black_thresh: int,
    min_pixels_per_vertex: int,
    max_points: int,
    rng: np.random.RandomState,
) -> bool:
    bbox_key = str(frame_idx)
    src_bbox = bbox_dict_src.get(bbox_key)
    tgt_bbox = bbox_dict_tgt.get(bbox_key)

    if src_bbox is None or tgt_bbox is None:
        print(f"[{frame_idx:05d}] missing bbox: src={src_bbox is None}, tgt={tgt_bbox is None}")
        return False

    # Load CSE results (already in full ERP coordinates)
    src_cse = load_cse_result(frame_idx, cse_npz_dir, src_cam, erp_w, erp_h, crop_bbox=src_bbox)
    tgt_cse = load_cse_result(frame_idx, cse_npz_dir, tgt_cam, erp_w, erp_h, crop_bbox=tgt_bbox)

    if src_cse is None:
        print(f"[{frame_idx:05d}] missing src CSE npz")
        return False
    if tgt_cse is None:
        print(f"[{frame_idx:05d}] missing tgt CSE npz")
        return False

    n_src_valid = int(src_cse.valid.sum())
    n_tgt_valid = int(tgt_cse.valid.sum())
    print(f"[{frame_idx:05d}] src CSE valid pixels: {n_src_valid}, tgt: {n_tgt_valid}")

    # Build correspondences via vertex-id grouping
    src_kps, tgt_kps = build_cse_correspondences(
        src_cse, tgt_cse, min_pixels_per_vertex, max_points, rng
    )
    n_corr = len(src_kps)
    print(f"[{frame_idx:05d}] CSE correspondences: {n_corr}")

    if n_corr < 4:
        print(f"[{frame_idx:05d}] too few correspondences ({n_corr}), skipping")
        return False

    # Load full images
    src_img = cv2.imread(str(image_root / src_cam / f"{frame_idx:05d}.jpg"))
    tgt_img = cv2.imread(str(image_root / tgt_cam / f"{frame_idx:05d}.jpg"))

    if src_img is None:
        print(f"[{frame_idx:05d}] failed to read src image")
        return False
    if tgt_img is None:
        print(f"[{frame_idx:05d}] failed to read tgt image")
        return False

    assert src_img.shape == (erp_h, erp_w, 3), f"src image shape mismatch: {src_img.shape}"
    assert tgt_img.shape == (erp_h, erp_w, 3), f"tgt image shape mismatch: {tgt_img.shape}"

    # Crop region from target bbox + padding
    x1c = max(0, tgt_bbox[0] - pad)
    y1c = max(0, tgt_bbox[1] - pad)
    x2c = min(erp_w, tgt_bbox[2] + pad)
    y2c = min(erp_h, tgt_bbox[3] + pad)
    crop_region = (x1c, y1c, x2c, y2c)

    # Fit TPS: tgt_kps -> src_kps  (map tgt coords to src sample coords)
    tps = fit_tps(tgt_kps, src_kps, regularization=tps_lambda)

    # Warp only the crop region
    warped_crop = warp_crop_region(src_img, tps, crop_region, erp_h, erp_w)

    # Determine valid pixels: non-black regions in warped_src
    valid_mask = np.any(warped_crop > black_thresh, axis=2)

    # Crop source and target images to same region for debug panel
    tgt_crop = tgt_img[y1c:y2c, x1c:x2c].copy()
    src_crop = src_img[y1c:y2c, x1c:x2c].copy()

    # filled_warp: valid pixels from warped_src, invalid from target
    filled = tgt_crop.copy()
    filled[valid_mask] = warped_crop[valid_mask]

    # filled_target: valid pixels from target, invalid from warped_src
    valid_target_mask = np.any(tgt_crop > black_thresh, axis=2)
    filled_tgt = warped_crop.copy()
    filled_tgt[valid_target_mask] = tgt_crop[valid_target_mask]

    # Build debug panel
    panel = make_debug_panel(
        tgt_crop=tgt_crop,
        src_crop=src_crop,
        tgt_kps=tgt_kps,
        src_kps=src_kps,
        warped_crop=warped_crop,
        filled_warp=filled,
        filled_target=filled_tgt,
        crop_offset=(x1c, y1c),
        panel_w=panel_w,
    )

    out_path = output_dir / f"{frame_idx:05d}_debug.jpg"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), panel)
    print(f"[{frame_idx:05d}] saved {out_path}")

    return True


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main() -> None:
    args = parse_args()

    image_root = Path(args.image_root)
    cse_npz_dir = Path(args.cse_npz_dir)
    bbox_dir = Path(args.bbox_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    bbox_dict_src = load_bbox_dict(bbox_dir / f"{args.src_cam}_bboxes.json")
    bbox_dict_tgt = load_bbox_dict(bbox_dir / f"{args.tgt_cam}_bboxes.json")

    rng = np.random.RandomState(args.seed)

    print(f"Source camera: {args.src_cam}, Target camera: {args.tgt_cam}")
    print(f"Frames: {args.frames}")
    print(f"Image size: {args.image_width}x{args.image_height}")
    print(f"Max points: {args.max_points}, Min pixels/vertex: {args.min_pixels_per_vertex}")
    print(f"TPS regularization: {args.tps_lambda}")
    print(f"Random seed: {args.seed}")
    print()

    n_ok = 0
    for frame_idx in args.frames:
        ok = process_frame(
            frame_idx=frame_idx,
            image_root=image_root,
            cse_npz_dir=cse_npz_dir,
            bbox_dict_src=bbox_dict_src,
            bbox_dict_tgt=bbox_dict_tgt,
            output_dir=output_dir,
            erp_h=args.image_height,
            erp_w=args.image_width,
            src_cam=args.src_cam,
            tgt_cam=args.tgt_cam,
            tps_lambda=args.tps_lambda,
            pad=args.pad,
            panel_w=args.panel_size,
            black_thresh=args.black_thresh,
            min_pixels_per_vertex=args.min_pixels_per_vertex,
            max_points=args.max_points,
            rng=rng,
        )
        if ok:
            n_ok += 1

    print(f"\nDone. {n_ok}/{len(args.frames)} frames processed successfully.")


if __name__ == "__main__":
    main()


r'''
python src/script/test_function/debug008_raw_cse_sparse_composite.py \
  --image_root data_process/Output/20260311_riding/image/equirect_rotate \
  --cse_npz_dir data_process/Output/20260311_riding/densepose/cse_npz \
  --bbox_dir data_process/Output/20260311_riding/image/equirect_crop_rotate \
  --output_dir data_process/Output/20260311_riding/densepose/raw_cse_sparse_composite \
  --frames 513 514 515 516 517 518 \
  --src_cam cam4 \
  --tgt_cam cam2 \
  --image_width 5760 \
  --image_height 2880 \
  --max_points 300 \
  --min_pixels_per_vertex 2
'''
