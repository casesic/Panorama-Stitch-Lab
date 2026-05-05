from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import config_script


os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
PROJECT_ROOT = Path(config_script.PROJECT_ROOT)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
SF_ROOT = PROJECT_ROOT / "tools" / "SingleFitting"
if str(SF_ROOT) not in sys.path:
    sys.path.insert(0, str(SF_ROOT))

from tools.fisheye_extractor.config import DS_PARAMS, DS_PARAMS_CAM2, DS_PARAMS_CAM4
from tools.fisheye_extractor.ds_model import project as ds_project
from lib_pano.human3d.smplx_model_io import get_smplx_faces_from_model, load_smplx_model
from lib_pano.human3d.smplx_parts import (
    get_smplx_hand_vertex_ids,
    load_exclude_vertex_ids,
    merge_exclude_vertex_ids,
)
from lib_pano.human3d.smplx_sequence import build_segment_index, get_frame_vertices


R_CAM2_NP = np.array([1, 0, 0, 0, 0, 1, 0, -1, 0], dtype=np.float64).reshape(3, 3)
R_CAM4_PROJ_NP = R_CAM2_NP
IDENTITY_NP = np.eye(3, dtype=np.float64)
ROTATION_MATRIX = R_CAM2_NP
R_EXTRINSIC_CAM2 = np.eye(3, dtype=np.float64)
R_EXTRINSIC_CAM4 = np.array(
    [-1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 1.0],
    dtype=np.float64,
).reshape(3, 3)
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build sparse no-hand SMPLX correspondences between cam2 and cam4.")
    parser.add_argument("--fitting_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--start_frame", type=int, default=None)
    parser.add_argument("--end_frame", type=int, default=None)
    parser.add_argument("--erp_width", type=int, default=5760)
    parser.add_argument("--erp_height", type=int, default=2880)
    parser.add_argument("--fisheye_width", type=int, default=2880)
    parser.add_argument("--fisheye_height", type=int, default=2880)
    parser.add_argument("--camera", choices=("old", "new"), default="old")
    parser.add_argument("--apply_extrinsic", action="store_true")
    parser.add_argument("--exclude_vertex_ids", type=str, default=None)
    return parser.parse_args()


def select_camera_params(camera_group: str) -> tuple[dict[str, float], dict[str, float]]:
    if camera_group == "new":
        return DS_PARAMS_CAM2, DS_PARAMS_CAM4
    return DS_PARAMS, DS_PARAMS


def select_frame_indices(
    segment_index: dict[int, tuple[Path, Path, int]],
    start_frame: int | None,
    end_frame: int | None,
) -> list[int]:
    if start_frame is not None and end_frame is not None:
        return list(range(start_frame, end_frame + 1))

    frame_indices = sorted(segment_index.keys())
    if start_frame is not None:
        frame_indices = [frame_idx for frame_idx in frame_indices if frame_idx >= start_frame]
    if end_frame is not None:
        frame_indices = [frame_idx for frame_idx in frame_indices if frame_idx <= end_frame]
    return frame_indices


def projected_in_bounds(pts_2d: np.ndarray, image_width: int, image_height: int) -> np.ndarray:
    finite = np.isfinite(pts_2d).all(axis=1)
    return finite & (pts_2d[:, 0] >= 0) & (pts_2d[:, 0] < image_width) & (pts_2d[:, 1] >= 0) & (pts_2d[:, 1] < image_height)


def project_vertices_to_erp(verts_unified: np.ndarray, erp_width: int, erp_height: int) -> tuple[np.ndarray, np.ndarray]:
    from flow.visibility import project_verts_to_rotated_erp

    pts_erp_all = project_verts_to_rotated_erp(verts_unified, IDENTITY_NP, erp_h=erp_height, erp_w=erp_width)
    return pts_erp_all, projected_in_bounds(pts_erp_all, erp_width, erp_height)


def project_vertices_to_fisheye(
    verts_unified: np.ndarray,
    cam_id: str,
    camera_group: str,
    apply_extrinsic: bool,
) -> np.ndarray:
    cam2_params, cam4_params = select_camera_params(camera_group)
    ds_params = cam2_params if cam_id == "cam2" else cam4_params
    r_extrinsic = (R_EXTRINSIC_CAM2 if cam_id == "cam2" else R_EXTRINSIC_CAM4) if apply_extrinsic else None

    rays = np.asarray(verts_unified, dtype=np.float64).reshape(-1, 3)
    rays_fisheye = rays @ r_extrinsic.T @ ROTATION_MATRIX.T if r_extrinsic is not None else rays @ ROTATION_MATRIX.T
    pts_fisheye_all, _ = ds_project(rays_fisheye, ds_params)
    return np.asarray(pts_fisheye_all, dtype=np.float32).reshape(-1, 2)


def build_matched_pairs(
    pts_cam2_all: np.ndarray,
    pts_cam4_all: np.ndarray,
    mesh_visible_mask_cam2: np.ndarray,
    mesh_visible_mask_cam4: np.ndarray,
    valid_mask_cam2: np.ndarray,
    valid_mask_cam4: np.ndarray,
    exclude_vertex_ids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    usable_cam2 = mesh_visible_mask_cam2 & valid_mask_cam2
    usable_cam4 = mesh_visible_mask_cam4 & valid_mask_cam4
    matched_vertex_ids = np.where(usable_cam2 & usable_cam4)[0].astype(np.int64)
    if len(exclude_vertex_ids) > 0:
        matched_vertex_ids = matched_vertex_ids[~np.isin(matched_vertex_ids, exclude_vertex_ids)]

    pairs = np.concatenate([pts_cam2_all[matched_vertex_ids], pts_cam4_all[matched_vertex_ids]], axis=1).astype(np.float32, copy=False)
    return matched_vertex_ids, pairs


def save_corr_npz(
    path: Path,
    frame_idx: int,
    segment_name: str,
    image_width: int,
    image_height: int,
    matched_vertex_ids: np.ndarray,
    pairs: np.ndarray,
) -> None:
    np.savez_compressed(
        path,
        frame_idx=np.asarray(frame_idx, dtype=np.int64),
        segment_name=np.asarray(segment_name),
        image_width=np.asarray(image_width, dtype=np.int64),
        image_height=np.asarray(image_height, dtype=np.int64),
        matched_vertex_ids=matched_vertex_ids.astype(np.int64, copy=False),
        pairs=pairs.astype(np.float32, copy=False),
    )


def process_frame(
    frame_idx: int,
    segment_name: str,
    verts_c2: np.ndarray,
    verts_c4: np.ndarray,
    faces: np.ndarray,
    output_dir: Path,
    erp_width: int,
    erp_height: int,
    fisheye_width: int,
    fisheye_height: int,
    camera_group: str,
    apply_extrinsic: bool,
    exclude_vertex_ids: np.ndarray,
) -> None:
    from flow.visibility import get_visible_vertices_erp

    verts_unified_c2 = verts_c2 @ R_CAM2_NP.T
    verts_unified_c4 = verts_c4 @ R_CAM4_PROJ_NP.T

    pts_erp_cam2_all, valid_erp_cam2 = project_vertices_to_erp(verts_unified_c2, erp_width, erp_height)
    pts_erp_cam4_all, valid_erp_cam4 = project_vertices_to_erp(verts_unified_c4, erp_width, erp_height)
    _, mesh_visible_cam2 = get_visible_vertices_erp(verts_unified_c2, faces, IDENTITY_NP, erp_height, erp_width, roi_bbox=None)
    _, mesh_visible_cam4 = get_visible_vertices_erp(verts_unified_c4, faces, IDENTITY_NP, erp_height, erp_width, roi_bbox=None)

    erp_vertex_ids, erp_pairs = build_matched_pairs(
        pts_erp_cam2_all,
        pts_erp_cam4_all,
        mesh_visible_cam2,
        mesh_visible_cam4,
        valid_erp_cam2,
        valid_erp_cam4,
        exclude_vertex_ids,
    )

    pts_fisheye_cam2_all = project_vertices_to_fisheye(verts_unified_c2, "cam2", camera_group, apply_extrinsic)
    pts_fisheye_cam4_all = project_vertices_to_fisheye(verts_unified_c4, "cam4", camera_group, apply_extrinsic)
    fisheye_vertex_ids = erp_vertex_ids.copy()
    fisheye_pairs = np.concatenate(
        [pts_fisheye_cam2_all[fisheye_vertex_ids], pts_fisheye_cam4_all[fisheye_vertex_ids]],
        axis=1,
    ).astype(np.float32, copy=False)

    frame_key = f"{frame_idx:05d}"
    save_corr_npz(output_dir / "erp_corr" / f"{frame_key}_erp_corr.npz", frame_idx, segment_name, erp_width, erp_height, erp_vertex_ids, erp_pairs)
    save_corr_npz(
        output_dir / "fisheye_corr" / f"{frame_key}_fisheye_corr.npz",
        frame_idx,
        segment_name,
        fisheye_width,
        fisheye_height,
        fisheye_vertex_ids,
        fisheye_pairs,
    )
    print(f"[frame {frame_key}] erp={len(erp_vertex_ids)} fisheye={len(fisheye_vertex_ids)}")


def main() -> None:
    args = parse_args()
    fitting_dir = Path(args.fitting_dir)
    output_dir = Path(args.output_dir)
    (output_dir / "erp_corr").mkdir(parents=True, exist_ok=True)
    (output_dir / "fisheye_corr").mkdir(parents=True, exist_ok=True)

    smplx_model = load_smplx_model(gender="neutral")
    faces = get_smplx_faces_from_model(smplx_model)
    hand_vertex_ids = get_smplx_hand_vertex_ids(smplx_model, PROJECT_ROOT)
    user_exclude_ids = load_exclude_vertex_ids(args.exclude_vertex_ids)
    exclude_vertex_ids = merge_exclude_vertex_ids(hand_vertex_ids, user_exclude_ids)

    segment_index = build_segment_index(fitting_dir)
    frame_indices = select_frame_indices(segment_index, args.start_frame, args.end_frame)
    segment_cache: dict[Path, np.ndarray] = {}

    for frame_idx in frame_indices:
        pkl_path_c2, pkl_path_c4, segment_start = segment_index[frame_idx]
        verts_c2 = get_frame_vertices(pkl_path_c2, frame_idx, segment_start, smplx_model, segment_cache)
        verts_c4 = get_frame_vertices(pkl_path_c4, frame_idx, segment_start, smplx_model, segment_cache)

        process_frame(
            frame_idx=frame_idx,
            segment_name=pkl_path_c2.name,
            verts_c2=verts_c2,
            verts_c4=verts_c4,
            faces=faces,
            output_dir=output_dir,
            erp_width=args.erp_width,
            erp_height=args.erp_height,
            fisheye_width=args.fisheye_width,
            fisheye_height=args.fisheye_height,
            camera_group=args.camera,
            apply_extrinsic=args.apply_extrinsic,
            exclude_vertex_ids=exclude_vertex_ids,
        )


if __name__ == "__main__":
    main()

r'''
python src/script/test_function/debug045_build_sparse_smpl_correspondence.py `
  --fitting_dir data_process/Output/20260311_riding/fitting/dual_cam_sil `
  --output_dir data_process/Output/20260311_riding/sparse_smpl_corr_debug045 `
  --start_frame 513 `
  --end_frame 518 `
  --camera old `
  --apply_extrinsic

python src/script/test_function/debug045_build_sparse_smpl_correspondence.py `
  --fitting_dir data_process/Output/20260311_riding/fitting/dual_cam_sil `
  --output_dir data_process/Output/20260311_riding/sparse_smpl_corr_debug045_nohand_filtered `
  --start_frame 513 `
  --end_frame 518 `
  --camera old `
  --apply_extrinsic `
  --exclude_vertex_ids data_process/Output/20260311_riding/exclude_vertex_ids.txt
'''
