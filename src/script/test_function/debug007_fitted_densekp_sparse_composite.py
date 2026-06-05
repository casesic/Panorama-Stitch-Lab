from __future__ import annotations

import argparse
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
    project_cam4,
    project_points_erp,
)
from lib_pano.fitting.densekp_filter import load_densekp_filter
from lib_pano.pipeline.bbox_io import load_bbox_dict


# ---------------------------------------------------------------------------
# argparse
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fitted SMPL-X DenseKP sparse correspondence for local human compositing."
    )
    parser.add_argument("--image_root", type=str, required=True)
    parser.add_argument("--result_dir", type=str, required=True)
    parser.add_argument("--segment", type=str, required=True)
    parser.add_argument("--bbox_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--src_cam", type=str, required=True)
    parser.add_argument("--tgt_cam", type=str, required=True)
    parser.add_argument("--image_width", type=int, required=True)
    parser.add_argument("--image_height", type=int, required=True)
    parser.add_argument(
        "--tps_lambda",
        type=float,
        default=0.0,
        help="TPS regularization parameter (0 = exact interpolation).",
    )
    parser.add_argument(
        "--pad",
        type=int,
        default=200,
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
        "--densekp_filter",
        type=str,
        default=None,
        help="Optional JSON path for DenseKP ID filter (keep_ids / drop_ids).",
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
# DenseKP regressor
# ---------------------------------------------------------------------------
def load_densekp_regressor() -> np.ndarray:
    """Load DenseKP regressor R [138, 10475] mapping SMPL-X verts to 138 surface points.

    Pre-computed from:
      R = downsample_mat [138, 6890] @ smplx2smpl[\"matrix\"] [6890, 10475]
    """
    project_root = Path(__file__).resolve().parents[3]
    npy_path = project_root / "tools/CameraHMR/data/train-eval-utils/densekp_regressor_R.npy"
    if npy_path.exists():
        return np.load(npy_path).astype(np.float32)

    # Fallback: compute from source pickle files (requires functional torch/numpy interop)
    import pickle
    import ctypes
    downsample_path = project_root / "tools/CameraHMR/data/train-eval-utils/downsample_mat.pkl"
    smplx2smpl_path = project_root / "tools/CameraHMR/data/train-eval-utils/smplx2smpl.pkl"

    with open(downsample_path, "rb") as f:
        ds = pickle.load(f)
    ds_dense = ds.to_dense() if ds.is_sparse else ds
    ds_np = _tensor_to_numpy(ds_dense)

    with open(smplx2smpl_path, "rb") as f:
        s2s = pickle.load(f)
    m = s2s["matrix"]
    if hasattr(m, "is_sparse") and m.is_sparse:
        m = m.to_dense()
    m_np = _tensor_to_numpy(m) if hasattr(m, "data_ptr") else np.asarray(m, dtype=np.float64)

    R = ds_np @ m_np
    np.save(npy_path, R.astype(np.float32))
    return R.astype(np.float32)


def _tensor_to_numpy(t) -> np.ndarray:
    """Convert torch tensor to numpy via ctypes to avoid .numpy() (which may fail
    with numpy 2.x + older torch builds)."""
    import ctypes
    import torch
    if not t.is_cpu:
        t = t.cpu()
    t = t.contiguous()
    nbytes = t.numel() * t.element_size()
    ptr = t.data_ptr()
    buf = (ctypes.c_char * nbytes).from_address(ptr)
    dtype_map = {
        torch.float32: np.float32,
        torch.float64: np.float64,
        torch.int64: np.int64,
        torch.int32: np.int32,
    }
    np_dtype = dtype_map.get(t.dtype, np.float32)
    return np.frombuffer(bytes(buf), dtype=np_dtype).reshape(tuple(t.shape)).copy()


# ---------------------------------------------------------------------------
# segment helpers
# ---------------------------------------------------------------------------
def parse_segment_range(segment: str) -> tuple[int, int]:
    parts = segment.split("_")
    if len(parts) != 2:
        raise ValueError(f"segment must match start_end, got: {segment}")
    start_frame, end_frame = int(parts[0]), int(parts[1])
    if start_frame > end_frame:
        raise ValueError(f"segment start must be <= end: {segment}")
    return start_frame, end_frame


# ---------------------------------------------------------------------------
# per-frame processing
# ---------------------------------------------------------------------------
def process_frame(
    frame_idx: int,
    image_root: Path,
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
    src_kps_2d: np.ndarray,  # [138, 2] full ERP coords for src cam
    tgt_kps_2d: np.ndarray,  # [138, 2] full ERP coords for tgt cam
) -> bool:
    bbox_key = str(frame_idx)
    src_bbox = bbox_dict_src.get(bbox_key)
    tgt_bbox = bbox_dict_tgt.get(bbox_key)

    if src_bbox is None or tgt_bbox is None:
        print(f"[{frame_idx:05d}] missing bbox: src={src_bbox is None}, tgt={tgt_bbox is None}")
        return False

    # Filter valid keypoints: finite and within image bounds
    src_finite = np.isfinite(src_kps_2d).all(axis=1)
    tgt_finite = np.isfinite(tgt_kps_2d).all(axis=1)
    src_in = (src_kps_2d[:, 0] >= 0) & (src_kps_2d[:, 0] < erp_w) & (src_kps_2d[:, 1] >= 0) & (src_kps_2d[:, 1] < erp_h)
    tgt_in = (tgt_kps_2d[:, 0] >= 0) & (tgt_kps_2d[:, 0] < erp_w) & (tgt_kps_2d[:, 1] >= 0) & (tgt_kps_2d[:, 1] < erp_h)
    both_valid = src_finite & tgt_finite & src_in & tgt_in
    n_valid = int(both_valid.sum())
    print(f"[{frame_idx:05d}] valid correspondences: {n_valid} / {len(both_valid)}")

    if n_valid < 4:
        print(f"[{frame_idx:05d}] too few valid points ({n_valid}), skipping")
        return False

    src_kps = src_kps_2d[both_valid]
    tgt_kps = tgt_kps_2d[both_valid]

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
    result_dir = Path(args.result_dir)
    bbox_dir = Path(args.bbox_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    erp_w = args.image_width
    erp_h = args.image_height

    # Load segment and frame range
    start_frame, end_frame = parse_segment_range(args.segment)
    frame_indices = list(range(start_frame, end_frame + 1))

    # Load bbox dicts
    bbox_dict_src = load_bbox_dict(bbox_dir / f"{args.src_cam}_bboxes.json")
    bbox_dict_tgt = load_bbox_dict(bbox_dir / f"{args.tgt_cam}_bboxes.json")

    # Load SMPL-X fitting results
    result_pkl_src = result_dir / args.src_cam / f"seg_{args.segment}_seq.pkl"
    result_pkl_tgt = result_dir / args.tgt_cam / f"seg_{args.segment}_seq.pkl"
    result_src = _load_pickle(result_pkl_src)
    result_tgt = _load_pickle(result_pkl_tgt)
    params_src = unpack_smplx_result(result_src)
    params_tgt = unpack_smplx_result(result_tgt)

    total_frames = min(len(frame_indices), len(params_src), len(params_tgt))
    print(f"Segment: {args.segment}, frames: {start_frame}-{end_frame}, usable: {total_frames}")
    print(f"Source camera: {args.src_cam}, Target camera: {args.tgt_cam}")
    print(f"Image size: {erp_w}x{erp_h}")
    print(f"TPS regularization: {args.tps_lambda}")
    print()

    # Load DenseKP regressor and SMPLX forwarder (once)
    print("Loading DenseKP regressor ...")
    R = load_densekp_regressor()  # [138, 10475] numpy
    print(f"  R shape: {R.shape}")

    print("Loading SMPLX forwarder ...")
    forwarder = SMPLXForwarder(gender="neutral")

    # Select projection functions for src/tgt cameras
    project_fn_src = project_cam4 if args.src_cam == "cam4" else project_cam2
    project_fn_tgt = project_cam4 if args.tgt_cam == "cam4" else project_cam2

    # Load DenseKP ID filter if provided
    id_mask = None
    if args.densekp_filter is not None:
        id_mask = load_densekp_filter(args.densekp_filter)
        print(f"DenseKP filter loaded: {id_mask.sum()} / {len(id_mask)} points kept")

    n_ok = 0
    for idx in range(total_frames):
        frame_idx = frame_indices[idx]

        # Forward SMPLX params to get vertices for src and tgt cameras
        verts_src, _, _ = forwarder.forward_param_to_numpy(params_src[idx])
        verts_tgt, _, _ = forwarder.forward_param_to_numpy(params_tgt[idx])

        # DenseKP 3D points [138, 3] by applying regressor to vertices [10475, 3]
        densekp_3d_src = R @ verts_src  # [138, 10475] @ [10475, 3] = [138, 3]
        densekp_3d_tgt = R @ verts_tgt

        # Project to full ERP 2D coordinates
        src_kps_2d = project_points_erp(densekp_3d_src, erp_w, erp_h, project_fn_src)
        tgt_kps_2d = project_points_erp(densekp_3d_tgt, erp_w, erp_h, project_fn_tgt)

        # Apply DenseKP ID filter
        if id_mask is not None:
            src_kps_2d = src_kps_2d[id_mask]
            tgt_kps_2d = tgt_kps_2d[id_mask]

        ok = process_frame(
            frame_idx=frame_idx,
            image_root=image_root,
            bbox_dict_src=bbox_dict_src,
            bbox_dict_tgt=bbox_dict_tgt,
            output_dir=output_dir,
            erp_h=erp_h,
            erp_w=erp_w,
            src_cam=args.src_cam,
            tgt_cam=args.tgt_cam,
            tps_lambda=args.tps_lambda,
            pad=args.pad,
            panel_w=args.panel_size,
            black_thresh=args.black_thresh,
            src_kps_2d=src_kps_2d,
            tgt_kps_2d=tgt_kps_2d,
        )
        if ok:
            n_ok += 1

    print(f"\nDone. {n_ok}/{total_frames} frames processed successfully.")


if __name__ == "__main__":
    main()


r'''
python src/script/test_function/debug007_fitted_densekp_sparse_composite.py \
  --image_root data_process/Output/20260311_riding/image/equirect_rotate \
  --result_dir data_process/Output/20260311_riding/fitting/dual_cam_dense_filter \
  --segment 704_711 \
  --bbox_dir data_process/Output/20260311_riding/image/equirect_crop_rotate \
  --output_dir data_process/Output/20260311_riding/tps_result/kp_densekp_fitted_filter \
  --src_cam cam4 \
  --tgt_cam cam2 \
  --image_width 5760 \
  --image_height 2880
'''
