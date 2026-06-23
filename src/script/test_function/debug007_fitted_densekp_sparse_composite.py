from __future__ import annotations

import argparse
import os
import re
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
from lib_pano.fitting_io.densekp_filter import load_densekp_filter
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
    parser.add_argument("--segment", type=str, nargs="+", required=True)
    parser.add_argument("--bbox_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--src_cam", type=str, default=None)
    parser.add_argument("--tgt_cam", type=str, default=None)
    parser.add_argument(
        "--directions",
        choices=["both", "cam4_to_cam2", "cam2_to_cam4"],
        default="both",
        help="Which direction(s) to process.",
    )
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
    parser.add_argument(
        "--dir_erode_px_cam2",
        type=int,
        default=5,
        help="Erosion px for cam2 along the bottom boundary (downward direction).",
    )
    parser.add_argument(
        "--dir_erode_px_cam4",
        type=int,
        default=5,
        help="Erosion px for cam4 along the top boundary (upward direction).",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# segment / direction helpers
# ---------------------------------------------------------------------------
def parse_segment_range(segment: str) -> tuple[int, int]:
    parts = segment.split("_")
    if len(parts) != 2:
        raise ValueError(f"segment must match start_end, got: {segment}")
    start_frame, end_frame = int(parts[0]), int(parts[1])
    if start_frame > end_frame:
        raise ValueError(f"segment start must be <= end: {segment}")
    return start_frame, end_frame


def get_direction_pairs(args) -> list[tuple[str, str, str]]:
    if args.src_cam is not None and args.tgt_cam is not None:
        return [(args.src_cam, args.tgt_cam, f"{args.src_cam}_to_{args.tgt_cam}")]

    if args.directions == "both":
        return [
            ("cam4", "cam2", "cam4_to_cam2"),
            ("cam2", "cam4", "cam2_to_cam4"),
        ]
    if args.directions == "cam4_to_cam2":
        return [("cam4", "cam2", "cam4_to_cam2")]
    if args.directions == "cam2_to_cam4":
        return [("cam2", "cam4", "cam2_to_cam4")]
    raise ValueError(f"Unsupported directions: {args.directions}")


def discover_segments(result_dir: Path, cam: str = "cam2") -> list[str]:
    """Scan result_dir/{cam}/seg_*_seq.pkl and return sorted segment strings."""
    pattern = re.compile(r"^seg_(\d+)_(\d+)_seq\.pkl$")
    scan_dir = result_dir / cam
    entries = []
    for p in scan_dir.glob("seg_*_seq.pkl"):
        m = pattern.match(p.name)
        if m is None:
            continue
        start = int(m.group(1))
        end = int(m.group(2))
        entries.append((start, end, f"{start}_{end}"))
    entries.sort(key=lambda x: (x[0], x[1]))
    return [s for _, _, s in entries]


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
    cam2_crop: np.ndarray,
    cam4_crop: np.ndarray,
    src_warp_crop: np.ndarray,
    tgt_filled_prior: np.ndarray,
    src_warp_filled_prior: np.ndarray,
    cam2_kps: np.ndarray,
    cam4_kps: np.ndarray,
    crop_offset: tuple[int, int],
    panel_w: int,
    src_cam: str,
    tgt_cam: str,
) -> np.ndarray:
    """Assemble a 2x3 debug panel.

    Row 1: cam2 | cam4 | {src_cam}_warp
    Row 2: overlay | {tgt_cam}_filled_prior | {src_cam}_warp_filled_prior
    """
    cam2_kp = draw_keypoints(cam2_crop, cam2_kps, crop_offset, (0, 255, 0), radius=4)
    cam4_kp = draw_keypoints(cam4_crop, cam4_kps, crop_offset, (0, 255, 0), radius=4)

    tgt_crop = cam2_crop if tgt_cam == "cam2" else cam4_crop
    overlay = cv2.addWeighted(tgt_crop, 0.5, src_warp_crop, 0.5, 0)

    warp_label = f"{src_cam}_warp"
    tgt_prior_label = f"{tgt_cam}_filled_prior"
    src_warp_prior_label = f"{src_cam}_warp_filled_prior"

    panels = [
        resize_to_width(cam2_kp, panel_w),
        resize_to_width(cam4_kp, panel_w),
        resize_to_width(src_warp_crop, panel_w),
        resize_to_width(overlay, panel_w),
        resize_to_width(tgt_filled_prior, panel_w),
        resize_to_width(src_warp_filled_prior, panel_w),
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
    labels = [
        "cam2",
        "cam4",
        warp_label,
        "overlay",
        tgt_prior_label,
        src_warp_prior_label,
    ]

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
# directional erosion
# ---------------------------------------------------------------------------
def directional_erode_valid_mask(
    valid_mask: np.ndarray,
    direction: str,
    ksize: int,
) -> np.ndarray:
    """Erode valid mask from one boundary only.

    direction="up":    top boundary eroded inward (cam4).
    direction="down":  bottom boundary eroded inward (cam2).
    """
    if ksize <= 1:
        return valid_mask
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, ksize))
    if direction == "down":
        anchor = (0, 0)           # anchor at top → erodes bottom
    elif direction == "up":
        anchor = (0, ksize - 1)   # anchor at bottom → erodes top
    else:
        raise ValueError(f"Unknown direction: {direction}")
    eroded = cv2.erode(valid_mask.astype(np.uint8), kernel, anchor=anchor)
    return eroded > 0


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
    dir_erode_px_cam2: int = 5,
    dir_erode_px_cam4: int = 5,
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

    # src_warp_filled_prior: warped_src valid -> warped_src, else target
    src_warp_filled_prior = tgt_crop.copy()
    src_warp_filled_prior[valid_mask] = warped_crop[valid_mask]

    # tgt_filled_prior: target valid -> target, else warped_src
    valid_target_mask = np.any(tgt_crop > black_thresh, axis=2)
    tgt_filled_prior = warped_crop.copy()
    tgt_filled_prior[valid_target_mask] = tgt_crop[valid_target_mask]

    # Directional-eroded composite: target priority, warp fills gaps
    tgt_dir = "down" if tgt_cam == "cam2" else "up"
    warp_dir = "down" if src_cam == "cam2" else "up"
    tgt_erode_px = dir_erode_px_cam2 if tgt_cam == "cam2" else dir_erode_px_cam4
    warp_erode_px = dir_erode_px_cam2 if src_cam == "cam2" else dir_erode_px_cam4

    tgt_valid_directional = directional_erode_valid_mask(valid_target_mask, tgt_dir, tgt_erode_px)
    warp_valid_directional = directional_erode_valid_mask(valid_mask, warp_dir, warp_erode_px)

    directional_filled_prior = tgt_crop.copy()
    fill_mask = (~tgt_valid_directional) & warp_valid_directional
    directional_filled_prior[fill_mask] = warped_crop[fill_mask]

    # Prepare cam2/cam4 crops and keypoints for debug panel
    if src_cam == "cam2":
        cam2_crop = src_crop
        cam4_crop = tgt_crop
        cam2_kps = src_kps
        cam4_kps = tgt_kps
    else:
        cam2_crop = tgt_crop
        cam4_crop = src_crop
        cam2_kps = tgt_kps
        cam4_kps = src_kps

    # Build debug panel
    panel = make_debug_panel(
        cam2_crop=cam2_crop,
        cam4_crop=cam4_crop,
        src_warp_crop=warped_crop,
        tgt_filled_prior=tgt_filled_prior,
        src_warp_filled_prior=src_warp_filled_prior,
        cam2_kps=cam2_kps,
        cam4_kps=cam4_kps,
        crop_offset=(x1c, y1c),
        panel_w=panel_w,
        src_cam=src_cam,
        tgt_cam=tgt_cam,
    )

    out_path = output_dir / f"{frame_idx:05d}_debug.jpg"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), panel)
    print(f"[{frame_idx:05d}] saved {out_path}")

    # Save directional filled prior
    dfp_dir = output_dir / "directional_filled_prior"
    dfp_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(
        str(dfp_dir / f"{frame_idx:05d}.jpg"),
        directional_filled_prior,
        [cv2.IMWRITE_JPEG_QUALITY, 95],
    )

    return True


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main() -> None:
    args = parse_args()

    image_root = Path(args.image_root)
    result_dir = Path(args.result_dir)

    if len(args.segment) == 1 and args.segment[0].lower() == "all":
        segments = discover_segments(result_dir, cam="cam2")
        if not segments:
            raise FileNotFoundError(f"No seg_*_seq.pkl found under {result_dir / 'cam2'}")
        print(f"Discovered {len(segments)} segments:")
        for s in segments:
            print(f"  {s}")
    else:
        segments = args.segment
    bbox_dir = Path(args.bbox_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    erp_w = args.image_width
    erp_h = args.image_height

    # Load DenseKP regressor and SMPLX forwarder (once)
    print("Loading DenseKP regressor ...")
    R = load_densekp_regressor()
    print(f"  R shape: {R.shape}")

    print("Loading SMPLX forwarder ...")
    forwarder = SMPLXForwarder(gender="neutral")

    # Load DenseKP ID filter if provided (once)
    id_mask = None
    if args.densekp_filter is not None:
        id_mask = load_densekp_filter(args.densekp_filter)
        print(f"DenseKP filter loaded: {id_mask.sum()} / {len(id_mask)} points kept")

    # Determine which direction(s) to process
    direction_pairs = get_direction_pairs(args)

    grand_ok = 0
    grand_frames = 0

    for segment in segments:
        start_frame, end_frame = parse_segment_range(segment)
        frame_indices = list(range(start_frame, end_frame + 1))

        print()
        print("*" * 60)
        print(f"Segment: {segment}  ({start_frame} - {end_frame})")
        print("*" * 60)

        for src_cam, tgt_cam, direction_name in direction_pairs:
            direction_output_dir = output_dir / direction_name
            direction_output_dir.mkdir(parents=True, exist_ok=True)

            print()
            print("=" * 60)
            print(f"Direction: {direction_name}")
            print(f"Source: {src_cam}, Target: {tgt_cam}")
            print(f"Output: {direction_output_dir}")
            print("=" * 60)

            # Load bbox dicts
            bbox_dict_src = load_bbox_dict(bbox_dir / f"{src_cam}_bboxes.json")
            bbox_dict_tgt = load_bbox_dict(bbox_dir / f"{tgt_cam}_bboxes.json")

            # Load SMPL-X fitting results
            result_pkl_src = result_dir / src_cam / f"seg_{segment}_seq.pkl"
            result_pkl_tgt = result_dir / tgt_cam / f"seg_{segment}_seq.pkl"
            result_src = _load_pickle(result_pkl_src)
            result_tgt = _load_pickle(result_pkl_tgt)
            params_src = unpack_smplx_result(result_src)
            params_tgt = unpack_smplx_result(result_tgt)

            total_frames = min(len(frame_indices), len(params_src), len(params_tgt))
            print(f"Frames: {start_frame}-{end_frame}, usable: {total_frames}")
            print(f"Image size: {erp_w}x{erp_h}")
            print(f"TPS regularization: {args.tps_lambda}")
            print()

            # Select projection functions
            project_fn_src = project_cam4 if src_cam == "cam4" else project_cam2
            project_fn_tgt = project_cam4 if tgt_cam == "cam4" else project_cam2

            n_ok = 0
            for idx in range(total_frames):
                frame_idx = frame_indices[idx]

                # Forward SMPLX params to get vertices for src and tgt cameras
                verts_src, _, _ = forwarder.forward_param_to_numpy(params_src[idx])
                verts_tgt, _, _ = forwarder.forward_param_to_numpy(params_tgt[idx])

                # DenseKP 3D points [138, 3]
                densekp_3d_src = R @ verts_src
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
                    output_dir=direction_output_dir,
                    erp_h=erp_h,
                    erp_w=erp_w,
                    src_cam=src_cam,
                    tgt_cam=tgt_cam,
                    tps_lambda=args.tps_lambda,
                    pad=args.pad,
                    panel_w=args.panel_size,
                    black_thresh=args.black_thresh,
                    src_kps_2d=src_kps_2d,
                    tgt_kps_2d=tgt_kps_2d,
                    dir_erode_px_cam2=args.dir_erode_px_cam2,
                    dir_erode_px_cam4=args.dir_erode_px_cam4,
                )
                if ok:
                    n_ok += 1

            print(f"\nDone direction {direction_name}: {n_ok}/{total_frames}")
            grand_ok += n_ok
            grand_frames += total_frames

    print(f"\n{'=' * 60}")
    print(f"All segments, all directions done. {grand_ok}/{grand_frames} frames processed successfully.")


if __name__ == "__main__":
    main()


r'''
# Specify segments explicitly
python src/script/test_function/debug007_fitted_densekp_sparse_composite.py \
  --image_root data_process/Output/20260311_riding/image/equirect_rotate \
  --result_dir data_process/Output/20260311_riding/fitting/dual_cam_new_sil \
  --segment 513_518 704_711 \
  --bbox_dir data_process/Output/20260311_riding/image/equirect_crop_rotate \
  --output_dir data_process/Output/20260311_riding/tps_result/kp_sil_fitted \
  --directions both \
  --image_width 5760 \
  --image_height 2880 \
  --dir_erode_px_cam2 30 \
  --dir_erode_px_cam4 20

# Discover all segments automatically
python src/script/test_function/debug007_fitted_densekp_sparse_composite.py \
  --image_root data_process/Output/20260311_riding/image/equirect_rotate \
  --result_dir data_process/Output/20260311_riding/fitting/dual_cam_dense_filter_all \
  --segment all \
  --bbox_dir data_process/Output/20260311_riding/image/equirect_crop_rotate \
  --output_dir data_process/Output/20260311_riding/tps_result/kp_densekp_fitted_filter_all \
  --directions cam2_to_cam4 \
  --image_width 5760 \
  --image_height 2880 \
  --dir_erode_px_cam2 30 \
  --dir_erode_px_cam4 20
  
python src/script/test_function/debug007_fitted_densekp_sparse_composite.py \
  --image_root data_process/Output/20260320_riding_dataset2/image/equirect_rotate \
  --result_dir data_process/Output/20260320_riding_dataset2/fitting/dual_cam_dense_filter_all \
  --segment all \
  --bbox_dir data_process/Output/20260320_riding_dataset2/image/equirect_crop_rotate \
  --output_dir data_process/Output/20260320_riding_dataset2/tps_result/kp_densekp_fitted_filter_all \
  --directions both \
  --image_width 5760 \
  --image_height 2880 \
  --dir_erode_px_cam2 30 \
  --dir_erode_px_cam4 20
'''
