#!/usr/bin/env python3
"""
Export fitted DenseKP 138 points as cross-view sparse correspondence JSON.

Output per-frame JSON files:
  {output_root}/data/{dataset}/smpl_corr/erp_corr_json/{frame:05d}.json
  {output_root}/data/{dataset}/smpl_corr/fisheye_corr_json/{frame:05d}.json
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import config_script  # noqa: F401 — ensure src/ on sys.path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import cv2

PROJECT_ROOT = Path(config_script.PROJECT_ROOT)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
SF_ROOT = PROJECT_ROOT / "tools" / "SingleFitting"
if str(SF_ROOT) not in sys.path:
    sys.path.insert(0, str(SF_ROOT))

from tools.fisheye_extractor.config import DS_PARAMS, DS_PARAMS_CAM2, DS_PARAMS_CAM4
from tools.fisheye_extractor.ds_model import project as ds_project
from lib_pano.human_vis.smplx_forward_vis import SMPLXForwarder, _load_pickle, unpack_smplx_result

# ---------------------------------------------------------------------------
# constants
# ---------------------------------------------------------------------------
ROTATION_MATRIX = np.array([1, 0, 0, 0, 0, 1, 0, -1, 0], dtype=np.float64).reshape(3, 3)
ERP_W = 5760
ERP_H = 2880
FISHEYE_W = 2880
FISHEYE_H = 2880
NUM_DENSEKP = 138

OUTPUT_ROOT_DEFAULT = PROJECT_ROOT / "Circular_Video" / "01_send" / "20260611_smpl_corr"

# ---------------------------------------------------------------------------
# export configuration
# ---------------------------------------------------------------------------
EXPORT_CONFIG = [
    {
        "dataset": "20260311_riding",
        "camera_group": "old",
        "items": [
            {"result_segment": "513_518", "export_start": 513, "export_end": 517},
            {"result_segment": "704_711", "export_start": 704, "export_end": 711},
            {"result_segment": "1217_1225", "export_start": 1217, "export_end": 1225},
        ],
    },
    {
        "dataset": "20260320_riding_dataset1",
        "camera_group": "new",
        "items": [
            {"result_segment": "886_895", "export_start": 886, "export_end": 895},
        ],
    },
    {
        "dataset": "20260320_riding_dataset2",
        "camera_group": "new",
        "items": [
            {"result_segment": "519_531", "export_start": 519, "export_end": 525},
            {"result_segment": "1769_1771", "export_start": 1769, "export_end": 1771},
            {"result_segment": "2850_2853", "export_start": 2851, "export_end": 2853},
        ],
    },
]

# ---------------------------------------------------------------------------
# argparse
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export fitted DenseKP 138 to JSON correspondences.")
    parser.add_argument(
        "--output_root",
        type=str,
        default=str(OUTPUT_ROOT_DEFAULT),
        help="Root output directory.",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        nargs="*",
        default=None,
        help="Optional subset of datasets to process.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# DenseKP regressor
# ---------------------------------------------------------------------------
def load_densekp_regressor() -> np.ndarray:
    npy_path = PROJECT_ROOT / "tools/CameraHMR/data/train-eval-utils/densekp_regressor_R.npy"
    if npy_path.exists():
        return np.load(npy_path).astype(np.float32)

    import pickle
    import ctypes
    import torch as _torch

    downsample_path = PROJECT_ROOT / "tools/CameraHMR/data/train-eval-utils/downsample_mat.pkl"
    smplx2smpl_path = PROJECT_ROOT / "tools/CameraHMR/data/train-eval-utils/smplx2smpl.pkl"

    with open(downsample_path, "rb") as f:
        ds = pickle.load(f)
    ds_dense = ds.to_dense() if hasattr(ds, "is_sparse") and ds.is_sparse else ds
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
    import ctypes
    import torch as _torch
    if not t.is_cpu:
        t = t.cpu()
    t = t.contiguous()
    nbytes = t.numel() * t.element_size()
    ptr = t.data_ptr()
    buf = (ctypes.c_char * nbytes).from_address(ptr)
    dtype_map = {
        _torch.float32: np.float32,
        _torch.float64: np.float64,
        _torch.int64: np.int64,
        _torch.int32: np.int32,
    }
    np_dtype = dtype_map.get(t.dtype, np.float32)
    return np.frombuffer(bytes(buf), dtype=np_dtype).reshape(tuple(t.shape)).copy()


# ---------------------------------------------------------------------------
# projection helpers (numpy)
# ---------------------------------------------------------------------------
def project_erp_np(points_3d: np.ndarray, W: int, H: int) -> np.ndarray:
    x = points_3d[..., 0]
    y = points_3d[..., 1]
    z = points_3d[..., 2]
    r = np.sqrt(x * x + y * y + z * z)
    r = np.clip(r, 1e-6, None)
    lon = np.arctan2(x, z)
    lat = np.arcsin(np.clip(-y / r, -1.0 + 1e-6, 1.0 - 1e-6))
    u = (lon / (2.0 * np.pi) + 0.5) * W
    v = (0.5 - lat / np.pi) * H
    return np.stack([u, v], axis=-1).astype(np.float32)


def camera_to_world(points_cam: np.ndarray) -> np.ndarray:
    return np.asarray(points_cam, dtype=np.float64) @ ROTATION_MATRIX.T


def densekp_to_erp_2d(densekp_3d: np.ndarray, W: int, H: int) -> np.ndarray:
    world = camera_to_world(densekp_3d)
    return project_erp_np(world, W, H)


def erp_2d_to_fisheye_2d(
    erp_pts: np.ndarray,
    erp_w: int,
    erp_h: int,
    ds_params: dict,
    cam_id: str,
) -> np.ndarray:
    u = erp_pts[:, 0].astype(np.float64)
    v = erp_pts[:, 1].astype(np.float64)
    lon = (u / erp_w) * 2.0 * np.pi - np.pi
    lat = (v / erp_h) * np.pi - np.pi / 2.0
    cos_lat = np.cos(lat)
    sin_lat = np.sin(lat)
    sin_lon = np.sin(lon)
    cos_lon = np.cos(lon)
    ray_x = cos_lat * sin_lon
    ray_y = sin_lat
    ray_z = cos_lat * cos_lon
    rays = np.stack([ray_x, ray_y, ray_z], axis=-1)
    if cam_id == "cam4":
        R_ext_cam4 = np.array([-1, 0, 0, 0, -1, 0, 0, 0, 1], dtype=np.float64).reshape(3, 3)
        rays = rays @ R_ext_cam4.T
    rays_cam = rays @ ROTATION_MATRIX.T
    pts, _ = ds_project(rays_cam, ds_params)
    return np.asarray(pts, dtype=np.float32).reshape(-1, 2)


# ---------------------------------------------------------------------------
# main export logic
# ---------------------------------------------------------------------------
def select_camera_params(camera_group: str) -> tuple[dict, dict]:
    if camera_group == "new":
        return DS_PARAMS_CAM2, DS_PARAMS_CAM4
    return DS_PARAMS, DS_PARAMS


def _copy_image_if_exists(
    src_root: Path,
    dst_root: Path,
    subpath: str,
    cam_id: str,
    frame_idx: int,
) -> bool:
    """Copy a single background image. Tries .jpg then .png. Returns True on success."""
    for suffix in (".jpg", ".png"):
        src = src_root / subpath / cam_id / f"{frame_idx:05d}{suffix}"
        if src.exists():
            dst = dst_root / subpath / cam_id
            dst.mkdir(parents=True, exist_ok=True)
            if not (dst / src.name).exists():
                shutil.copy2(str(src), str(dst / src.name))
            return True
    print(f"  [WARN] missing {subpath}/{cam_id}/{frame_idx:05d} image")
    return False


# ---------------------------------------------------------------------------
# valid mask helpers
# ---------------------------------------------------------------------------
def _find_bg_image(bg_dir: Path, cam_id: str, frame_idx: int) -> Path | None:
    for suffix in (".jpg", ".png"):
        p = bg_dir / cam_id / f"{frame_idx:05d}{suffix}"
        if p.exists():
            return p
    return None


def _compute_valid_mask(bg_path: Path | None, black_thresh: int = 5, erode_px: int = 10) -> np.ndarray | None:
    """Generate an eroded valid-region mask from the background image.

    Only border-connected black regions are treated as invalid (e.g. fisheye
    circular black borders, ERP top/bottom black bands).  Dark clothing / hair /
    shadows that do not touch the image border are kept valid.

    Returns a bool array (H, W), or None if the image is missing / unreadable.
    """
    if bg_path is None:
        return None
    img = cv2.imread(str(bg_path), cv2.IMREAD_COLOR)
    if img is None:
        return None

    h, w = img.shape[:2]

    # black pixels: max RGB <= black_thresh
    black = np.max(img, axis=2) <= black_thresh

    # Pad with a 1-pixel black border so that flood fill from (0,0) can reach
    # image-border black pixels.
    #   padded = 0  for padding + image black pixels
    #   padded = 255 for image non-black pixels
    padded = np.zeros((h + 2, w + 2), dtype=np.uint8)
    padded[1:-1, 1:-1] = np.where(black, 0, 255).astype(np.uint8)

    # Flood-fill from (0,0) with 128 — fills all 0-valued pixels connected to
    # the padding border, i.e. border-connected black regions.
    cv2.floodFill(padded, None, (0, 0), 128)

    # After fill:
    #   128 = border-connected black (invalid)
    #   0   = non-border-connected black (e.g. clothing — keep valid)
    #   255 = non-black (valid)
    border_black = padded[1:-1, 1:-1] == 128
    valid_mask = ~border_black

    # Erode valid region inward to add a margin from the black boundary
    kernel = np.ones((erode_px * 2 + 1, erode_px * 2 + 1), np.uint8)
    eroded = cv2.erode(valid_mask.astype(np.uint8) * 255, kernel)
    return eroded > 0


def _check_points_in_mask(pts_2d: np.ndarray, valid_mask: np.ndarray | None) -> np.ndarray:
    """Return int32 array [N] — 1 if the point is inside the eroded mask, 0 otherwise.

    If valid_mask is None (missing bg), all points are marked 0.
    """
    if valid_mask is None:
        return np.zeros(len(pts_2d), dtype=np.int32)
    h, w = valid_mask.shape
    result = np.ones(len(pts_2d), dtype=np.int32)
    for i in range(len(pts_2d)):
        x, y = int(round(pts_2d[i, 0])), int(round(pts_2d[i, 1]))
        if x < 0 or x >= w or y < 0 or y >= h or not valid_mask[y, x]:
            result[i] = 0
    return result


def export_segment(
    dataset: str,
    result_segment: str,
    export_start: int,
    export_end: int,
    output_root: Path,
    camera_group: str,
    forwarder: SMPLXForwarder,
    R: np.ndarray,
) -> tuple[int, int]:
    """Export one segment. Returns (n_erp, n_fisheye)."""
    data_root = PROJECT_ROOT / "data_process/Output" / dataset
    fitting_dir = data_root / "fitting/dual_cam_dense_filter_all"
    densekp_dir = data_root / "camerahmr/densekp_npz"

    export_seg_name = f"seg_{export_start}_{export_end}"

    pkl_cam2 = fitting_dir / "cam2" / f"seg_{result_segment}_seq.pkl"
    pkl_cam4 = fitting_dir / "cam4" / f"seg_{result_segment}_seq.pkl"

    if not pkl_cam2.exists():
        print(f"  [WARN] missing {pkl_cam2}")
        return 0, 0
    if not pkl_cam4.exists():
        print(f"  [WARN] missing {pkl_cam4}")
        return 0, 0

    result_cam2 = _load_pickle(pkl_cam2)
    result_cam4 = _load_pickle(pkl_cam4)
    params_cam2 = unpack_smplx_result(result_cam2)
    params_cam4 = unpack_smplx_result(result_cam4)

    ds_params_cam2, ds_params_cam4 = select_camera_params(camera_group)

    frame_indices = list(range(export_start, export_end + 1))
    result_start = int(pkl_cam2.stem.split("_")[1])
    available_frames = len(params_cam2)
    result_frames = list(range(result_start, result_start + available_frames))

    # Flat output: no segment subdir
    erp_output_dir = output_root / "data" / dataset / "smpl_corr" / "erp_corr_json"
    fisheye_output_dir = output_root / "data" / dataset / "smpl_corr" / "fisheye_corr_json"
    erp_output_dir.mkdir(parents=True, exist_ok=True)
    fisheye_output_dir.mkdir(parents=True, exist_ok=True)

    n_erp = 0
    n_fisheye = 0
    valid_sum = 0

    erp_bg_dir = data_root / "image" / "equirect_rotate"

    for frame_idx in frame_indices:
        if frame_idx not in result_frames:
            print(f"  [WARN] frame {frame_idx:05d} not in result segment {result_segment}, skipping")
            continue

        param_idx = frame_idx - result_start

        verts_cam2, _, _ = forwarder.forward_param_to_numpy(params_cam2[param_idx])
        verts_cam4, _, _ = forwarder.forward_param_to_numpy(params_cam4[param_idx])

        dkp3d_cam2 = R @ verts_cam2
        dkp3d_cam4 = R @ verts_cam4

        erp_cam2 = densekp_to_erp_2d(dkp3d_cam2, ERP_W, ERP_H)
        erp_cam4 = densekp_to_erp_2d(dkp3d_cam4, ERP_W, ERP_H)
        fisheye_cam2 = erp_2d_to_fisheye_2d(erp_cam2, ERP_W, ERP_H, ds_params_cam2, "cam2")
        fisheye_cam4 = erp_2d_to_fisheye_2d(erp_cam4, ERP_W, ERP_H, ds_params_cam4, "cam4")

        npz_cam2 = densekp_dir / "cam2" / f"{frame_idx:05d}_0.npz"
        npz_cam4 = densekp_dir / "cam4" / f"{frame_idx:05d}_0.npz"

        val_cam2 = np.ones(NUM_DENSEKP, dtype=np.float32)
        val_cam4 = np.ones(NUM_DENSEKP, dtype=np.float32)

        if npz_cam2.exists():
            data_c2 = np.load(npz_cam2)
            val_cam2 = data_c2["keypoint_value"].astype(np.float32)
        else:
            print(f"  [WARN] missing {npz_cam2}, using default value 1.0")

        if npz_cam4.exists():
            data_c4 = np.load(npz_cam4)
            val_cam4 = data_c4["keypoint_value"].astype(np.float32)
        else:
            print(f"  [WARN] missing {npz_cam4}, using default value 1.0")

        values = np.minimum(val_cam2, val_cam4).tolist()

        erp_pairs = [
            [float(erp_cam2[i, 0]), float(erp_cam2[i, 1]),
             float(erp_cam4[i, 0]), float(erp_cam4[i, 1])]
            for i in range(NUM_DENSEKP)
        ]
        fisheye_pairs = [
            [float(fisheye_cam2[i, 0]), float(fisheye_cam2[i, 1]),
             float(fisheye_cam4[i, 0]), float(fisheye_cam4[i, 1])]
            for i in range(NUM_DENSEKP)
        ]

        # --- valid (based on ERP images) ---
        mask_erp_c2 = _compute_valid_mask(_find_bg_image(erp_bg_dir, "cam2", frame_idx))
        mask_erp_c4 = _compute_valid_mask(_find_bg_image(erp_bg_dir, "cam4", frame_idx))
        if mask_erp_c2 is None:
            print(f"  [WARN] missing cam2 ERP bg for frame {frame_idx}, marking all points invalid")
        if mask_erp_c4 is None:
            print(f"  [WARN] missing cam4 ERP bg for frame {frame_idx}, marking all points invalid")
        valid_erp_c2 = _check_points_in_mask(erp_cam2, mask_erp_c2)
        valid_erp_c4 = _check_points_in_mask(erp_cam4, mask_erp_c4)
        valid = (valid_erp_c2 & valid_erp_c4).tolist()
        n_valid = sum(valid)
        valid_sum += n_valid

        print(f"  [frame {frame_idx:05d}] valid={n_valid}/{NUM_DENSEKP}")

        erp_json = {
            "frame_idx": frame_idx,
            "segment_name": export_seg_name,
            "coord_type": "erp",
            "image_width": ERP_W,
            "image_height": ERP_H,
            "point_ids": list(range(NUM_DENSEKP)),
            "pairs": erp_pairs,
            "values": values,
            "valid": valid,
        }
        with (erp_output_dir / f"{frame_idx:05d}.json").open("w", encoding="utf-8") as f:
            json.dump(erp_json, f, ensure_ascii=False, indent=2)
        n_erp += 1

        fisheye_json = {
            "frame_idx": frame_idx,
            "segment_name": export_seg_name,
            "coord_type": "fisheye",
            "image_width": FISHEYE_W,
            "image_height": FISHEYE_H,
            "point_ids": list(range(NUM_DENSEKP)),
            "pairs": fisheye_pairs,
            "values": values,
            "valid": valid,
        }
        with (fisheye_output_dir / f"{frame_idx:05d}.json").open("w", encoding="utf-8") as f:
            json.dump(fisheye_json, f, ensure_ascii=False, indent=2)
        n_fisheye += 1

        # Copy background images
        img_dst_root = output_root / "data" / dataset / "image"
        _copy_image_if_exists(data_root, img_dst_root, "image/equirect_rotate", "cam2", frame_idx)
        _copy_image_if_exists(data_root, img_dst_root, "image/equirect_rotate", "cam4", frame_idx)
        _copy_image_if_exists(data_root, img_dst_root, "image/fisheye", "cam2", frame_idx)
        _copy_image_if_exists(data_root, img_dst_root, "image/fisheye", "cam4", frame_idx)

    return n_erp, n_fisheye, valid_sum


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    configs = EXPORT_CONFIG
    if args.datasets:
        configs = [c for c in configs if c["dataset"] in args.datasets]
        if not configs:
            print(f"[ERROR] no matching datasets: {args.datasets}")
            return

    print("Loading DenseKP regressor ...")
    R = load_densekp_regressor()
    print(f"  R shape: {R.shape}")

    print("Loading SMPLX forwarder ...")
    forwarder = SMPLXForwarder(gender="neutral")
    print()

    grand_erp = 0
    grand_fisheye = 0
    summary: list[dict] = []

    for cfg in configs:
        dataset = cfg["dataset"]
        camera_group = cfg["camera_group"]
        print(f"[DATASET] {dataset} (camera: {camera_group})")
        dataset_erp = 0
        dataset_fisheye = 0
        dataset_details = []

        for item in cfg["items"]:
            result_segment = item["result_segment"]
            export_start = item["export_start"]
            export_end = item["export_end"]
            seg_name = f"seg_{export_start}_{export_end}"
            print(f"  exporting {seg_name} (from result seg_{result_segment}) ...")

            n_erp, n_fisheye, valid_sum = export_segment(
                dataset=dataset,
                result_segment=result_segment,
                export_start=export_start,
                export_end=export_end,
                output_root=output_root,
                camera_group=camera_group,
                forwarder=forwarder,
                R=R,
            )
            dataset_erp += n_erp
            dataset_fisheye += n_fisheye
            mean_valid = valid_sum / max(n_erp, 1)
            dataset_details.append(
                f"  {seg_name}: exported {n_erp} ERP json, {n_fisheye} fisheye json"
                f" | valid mean {mean_valid / NUM_DENSEKP:.1%} ({int(mean_valid)}/{NUM_DENSEKP})"
            )

        grand_erp += dataset_erp
        grand_fisheye += dataset_fisheye
        summary.append({
            "dataset": dataset,
            "details": dataset_details,
            "erp": dataset_erp,
            "fisheye": dataset_fisheye,
        })

    print()
    print("[SUMMARY]")
    for s in summary:
        print(f"dataset: {s['dataset']}")
        for d in s["details"]:
            print(d)
    print(f"total: {grand_erp} ERP json, {grand_fisheye} fisheye json")


if __name__ == "__main__":
    main()


r'''
# Export all datasets
python src/script/test_function/debug052_export_fitted_densekp138_json.py \
  --output_root Circular_Video/01_send/20260611_smpl_corr

# Export single dataset
    python src/script/test_function/debug052_export_fitted_densekp138_json.py \
    --output_root Circular_Video/01_send/20260611_smpl_corr \
    --datasets 20260311_riding
'''
