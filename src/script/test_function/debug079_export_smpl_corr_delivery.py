#!/usr/bin/env python3
"""统一正式 exporter：四类跨视角人体稀疏点对（densekp_raw / densekp_fitting /
sam3dbody_raw / sam3dbody_fitting）→ 极简 ERP / fisheye JSON。

纯 CPU numpy / cv2 / json + torch（仅 fitting forward）。复用公共核心
``smpl_corr_delivery`` 的全部几何与筛选函数。
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path

# ---------------------------------------------------------------------------
# path bootstrap — 确保 src/ 与 PROJECT_ROOT 可用
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np

from lib_pano.correspondence.smpl_corr_delivery import (
    build_border_connected_valid_mask,
    densekp_to_erp_2d,
    erp_2d_to_fisheye_2d,
    find_image_path,
    get_camera_group_for_dataset,
    load_densekp_regressor,
    load_erp_image_checked,
    load_mhr_to_smplx_mapping,
    mhr_recon_to_smplx_surface_3d,
    project_pinhole_to_upright,
    select_camera_params,
    select_erp_valid_pairs,
    smplx_surface_to_densekp3d,
    write_minimal_corr_json,
)
from lib_pano.fitting_io.densekp_io import load_densekp_result
from lib_pano.human3d.crop_upright_coord import (
    resolve_dataset_rotate_rules,
    resolve_rotate_rule,
    upright_points_to_equirect_rotate,
)
from lib_pano.human3d.sam3dbody_io import load_sam3dbody_recon
from lib_pano.pipeline.bbox_io import load_bbox_dict

# ===================================================================
# 默认路径
# ===================================================================

DATA_ROOT_DEFAULT = PROJECT_ROOT / "data_process" / "Output"
BBOX_SUBDIR_DEFAULT = "image/equirect_crop_rotate"
ERP_IMAGE_SUBDIR = "image/equirect_rotate"
DENSEKP_SUBDIR = "camerahmr/densekp_npz"
SAM3DBODY_RECON_SUBDIR = "sam3dbody/recon"
MAPPING_DEFAULT = (
    PROJECT_ROOT
    / "tools/MHR/tools/mhr_smpl_conversion/assets/mhr2smplx_mapping.npz"
)
ROTATE_RULES_CONFIG = PROJECT_ROOT / "configs" / "rotate_upright_rules.json"

METHODS = frozenset(
    {"densekp_raw", "densekp_fitting", "sam3dbody_raw", "sam3dbody_fitting"}
)

# ===================================================================
# helpers
# ===================================================================


@dataclass(frozen=True)
class SegmentSpec:
    name: str
    start_frame: int
    end_frame: int
    frame_indices: list[int]


def parse_segment_specs(segment_args: list[str]) -> list[SegmentSpec]:
    """Parse start_end strings → SegmentSpec list, with overlap detection."""
    specs: list[SegmentSpec] = []
    seen: dict[int, str] = {}
    for seg in segment_args:
        parts = seg.split("_")
        if len(parts) != 2:
            raise ValueError(f"Invalid segment: {seg!r}")
        start, end = int(parts[0]), int(parts[1])
        if start > end:
            raise ValueError(f"Segment start > end: {seg!r}")
        frames = list(range(start, end + 1))
        for f in frames:
            if f in seen:
                raise ValueError(
                    f"Overlapping frame {f} in segment '{seg}' "
                    f"(already present in segment '{seen[f]}')"
                )
            seen[f] = seg
        specs.append(SegmentSpec(name=seg, start_frame=start, end_frame=end, frame_indices=frames))
    return specs


def check_fisheye_finite(fisheye_pts: np.ndarray, tag: str) -> None:
    """fisheye 坐标必须全部 finite；NaN/Inf 时 raise。"""
    if not np.isfinite(fisheye_pts).all():
        nan_count = (~np.isfinite(fisheye_pts)).sum()
        raise ValueError(
            f"{tag}: fisheye coords contain {nan_count} non-finite values"
        )


def _resolve_fitting_pkl(
    pkl_dir: Path, seg_start: int, seg_end: int, cam_label: str
) -> tuple[Path, int]:
    """Find pkl whose start_end range fully contains [seg_start, seg_end].

    Returns (pkl_path, pkl_start_frame).
    """
    import re

    pat = re.compile(r"^seg_(\d+)_(\d+)_seq\.pkl$")
    for p in sorted(pkl_dir.glob("seg_*_seq.pkl")):
        m = pat.match(p.name)
        if not m:
            continue
        ps, pe = int(m.group(1)), int(m.group(2))
        if ps <= seg_start and seg_end <= pe:
            return p, ps
    raise FileNotFoundError(
        f"No pkl found in {pkl_dir} whose range contains "
        f"[{seg_start}, {seg_end}] for {cam_label}"
    )


# ===================================================================
# fitting forward helper
# ===================================================================


def _load_fitting_forwarder():
    """延迟加载 SMPLXForwarder（仅 fitting method 需要）。"""
    from lib_pano.human_vis.smplx_forward_vis import (
        SMPLXForwarder,
        _load_pickle,
        unpack_smplx_result,
    )

    class _FittingLoader:
        def __init__(self):
            self.forwarder = SMPLXForwarder(gender="neutral")

        def load_segment_params(
            self, pkl_path: Path, segment_start: int
        ) -> tuple[list, int]:
            """加载 pkl，返回 (params_list, seg_start_frame)。"""
            result = _load_pickle(pkl_path)
            params = unpack_smplx_result(result)
            return params, segment_start

        def forward_to_vertices(self, param) -> np.ndarray:
            verts, _, _ = self.forwarder.forward_param_to_numpy(param)
            return verts  # [10475, 3] camera-frame

    return _FittingLoader()


# ===================================================================
# per-method export
# ===================================================================


def export_densekp_raw(
    frame_idx: int,
    densekp_dir: Path,
    bbox_dir: Path,
    erp_image_dir: Path,
    rotate_rule: str,
    output_base: Path,
) -> int:
    """densekp_raw 单帧导出 → 返回保留点数。"""
    cam2_bboxes = load_bbox_dict(bbox_dir / "cam2_bboxes.json")
    cam4_bboxes = load_bbox_dict(bbox_dir / "cam4_bboxes.json")

    bbox_c2 = cam2_bboxes.get(str(frame_idx))
    bbox_c4 = cam4_bboxes.get(str(frame_idx))
    if bbox_c2 is None:
        raise FileNotFoundError(f"bbox cam2 frame {frame_idx} not found in {bbox_dir}")
    if bbox_c4 is None:
        raise FileNotFoundError(f"bbox cam4 frame {frame_idx} not found in {bbox_dir}")

    result_c2 = load_densekp_result(
        frame_idx,
        densekp_dir,
        "cam2",
        5760,
        2880,
        crop_bbox=bbox_c2,
        rotate_rule=rotate_rule,
    )
    result_c4 = load_densekp_result(
        frame_idx,
        densekp_dir,
        "cam4",
        5760,
        2880,
        crop_bbox=bbox_c4,
        rotate_rule=rotate_rule,
    )
    if result_c2 is None or result_c4 is None:
        raise FileNotFoundError(
            f"densekp npz missing for frame {frame_idx}"
        )

    erp_c2 = result_c2.keypoints_2d  # [138, 2]
    erp_c4 = result_c4.keypoints_2d  # [138, 2]
    if erp_c2.shape != (138, 2) or erp_c4.shape != (138, 2):
        raise ValueError(
            f"densekp shape cam2={erp_c2.shape} cam4={erp_c4.shape}, expected (138, 2)"
        )

    # ERP valid masks
    img_c2 = load_erp_image_checked(erp_image_dir, "cam2", frame_idx)
    img_c4 = load_erp_image_checked(erp_image_dir, "cam4", frame_idx)
    mask_c2 = build_border_connected_valid_mask(img_c2)
    mask_c4 = build_border_connected_valid_mask(img_c4)

    keep = select_erp_valid_pairs(erp_c2, erp_c4, mask_c2, mask_c4)
    n_keep = int(keep.sum())

    # erp → fisheye for ALL 138 points
    camera_group = get_camera_group_for_dataset(_dataset_name)
    ds_c2, ds_c4 = select_camera_params(camera_group)
    fisheye_c2 = erp_2d_to_fisheye_2d(erp_c2, 5760, 2880, ds_c2, "cam2")
    fisheye_c4 = erp_2d_to_fisheye_2d(erp_c4, 5760, 2880, ds_c4, "cam4")

    # keep rows only
    erp_pairs = np.column_stack([erp_c2, erp_c4])[keep]
    fisheye_pairs = np.column_stack([fisheye_c2, fisheye_c4])[keep]

    check_fisheye_finite(fisheye_pairs, f"densekp_raw frame {frame_idx}")

    out_erp = output_base / "erp_corr_json"
    out_fisheye = output_base / "fisheye_corr_json"
    erp_path = out_erp / f"{frame_idx:05d}.json"
    fisheye_path = out_fisheye / f"{frame_idx:05d}.json"

    write_minimal_corr_json(erp_path, frame_idx, 5760, 2880, erp_pairs)
    write_minimal_corr_json(fisheye_path, frame_idx, 2880, 2880, fisheye_pairs)
    return n_keep


def export_fitting(
    frame_idx: int,
    params_cam2,
    params_cam4,
    param_idx: int,
    regressor: np.ndarray,
    erp_image_dir: Path,
    output_base: Path,
    loader,
) -> int:
    """densekp_fitting / sam3dbody_fitting 单帧导出 → 返回保留点数。"""
    verts_c2 = loader.forward_to_vertices(params_cam2[param_idx])
    verts_c4 = loader.forward_to_vertices(params_cam4[param_idx])

    dkp3d_c2 = regressor @ verts_c2  # [138, 10475] @ [10475, 3] → [138, 3]
    dkp3d_c4 = regressor @ verts_c4

    erp_c2 = densekp_to_erp_2d(dkp3d_c2, 5760, 2880)
    erp_c4 = densekp_to_erp_2d(dkp3d_c4, 5760, 2880)

    img_c2 = load_erp_image_checked(erp_image_dir, "cam2", frame_idx)
    img_c4 = load_erp_image_checked(erp_image_dir, "cam4", frame_idx)
    mask_c2 = build_border_connected_valid_mask(img_c2)
    mask_c4 = build_border_connected_valid_mask(img_c4)

    keep = select_erp_valid_pairs(erp_c2, erp_c4, mask_c2, mask_c4)
    n_keep = int(keep.sum())

    camera_group = get_camera_group_for_dataset(_dataset_name)
    ds_c2, ds_c4 = select_camera_params(camera_group)
    fisheye_c2 = erp_2d_to_fisheye_2d(erp_c2, 5760, 2880, ds_c2, "cam2")
    fisheye_c4 = erp_2d_to_fisheye_2d(erp_c4, 5760, 2880, ds_c4, "cam4")

    erp_pairs = np.column_stack([erp_c2, erp_c4])[keep]
    fisheye_pairs = np.column_stack([fisheye_c2, fisheye_c4])[keep]

    check_fisheye_finite(fisheye_pairs, f"fitting frame {frame_idx}")

    out_erp = output_base / "erp_corr_json"
    out_fisheye = output_base / "fisheye_corr_json"
    erp_path = out_erp / f"{frame_idx:05d}.json"
    fisheye_path = out_fisheye / f"{frame_idx:05d}.json"

    write_minimal_corr_json(erp_path, frame_idx, 5760, 2880, erp_pairs)
    write_minimal_corr_json(fisheye_path, frame_idx, 2880, 2880, fisheye_pairs)
    return n_keep


def export_sam3dbody_raw(
    frame_idx: int,
    recon_dir: Path,
    bbox_dir: Path,
    erp_image_dir: Path,
    mapping_path: Path,
    regressor: np.ndarray,
    rotate_rule: str,
    output_base: Path,
) -> int:
    """sam3dbody_raw 单帧导出 → 返回保留点数。"""
    cam2_bboxes = load_bbox_dict(bbox_dir / "cam2_bboxes.json")
    cam4_bboxes = load_bbox_dict(bbox_dir / "cam4_bboxes.json")

    triangle_ids, baryc_coords = load_mhr_to_smplx_mapping(mapping_path)

    n_keep = 0
    erp_pairs_list = []
    fisheye_pairs_list = []

    for cam_id, cam_bboxes in [("cam2", cam2_bboxes), ("cam4", cam4_bboxes)]:
        bbox = cam_bboxes.get(str(frame_idx))
        if bbox is None:
            raise FileNotFoundError(
                f"bbox {cam_id} frame {frame_idx} not found in {bbox_dir}"
            )

        npz_path = recon_dir / cam_id / f"{frame_idx:05d}.npz"
        if not npz_path.exists():
            raise FileNotFoundError(f"SAM3D recon npz missing: {npz_path}")

        recon = load_sam3dbody_recon(npz_path)
        image_h, image_w = recon.image_shape[:2]

        surface_cam = mhr_recon_to_smplx_surface_3d(
            recon.pred_vertices,
            recon.pred_cam_t,
            recon.faces,
            triangle_ids,
            baryc_coords,
        )  # [10475, 3]
        dkp3d_cam = smplx_surface_to_densekp3d(surface_cam, regressor)  # [138, 3]
        upright_2d, geom_valid = project_pinhole_to_upright(
            dkp3d_cam, float(recon.focal_length), image_h, image_w
        )  # [138, 2]

        # geometric_valid=False → set to NaN, excluded by ERP finite later
        upright_2d[~geom_valid] = np.nan

        erp_2d = upright_points_to_equirect_rotate(
            upright_2d, bbox, frame_idx, rotate_rule=rotate_rule
        )

        if cam_id == "cam2":
            erp_c2 = erp_2d
        else:
            erp_c4 = erp_2d

    # Build ERP masks
    img_c2 = load_erp_image_checked(erp_image_dir, "cam2", frame_idx)
    img_c4 = load_erp_image_checked(erp_image_dir, "cam4", frame_idx)
    mask_c2 = build_border_connected_valid_mask(img_c2)
    mask_c4 = build_border_connected_valid_mask(img_c4)

    keep = select_erp_valid_pairs(erp_c2, erp_c4, mask_c2, mask_c4)
    n_keep = int(keep.sum())

    # fisheye for all 138
    camera_group = get_camera_group_for_dataset(_dataset_name)
    ds_c2, ds_c4 = select_camera_params(camera_group)
    fisheye_c2 = erp_2d_to_fisheye_2d(erp_c2, 5760, 2880, ds_c2, "cam2")
    fisheye_c4 = erp_2d_to_fisheye_2d(erp_c4, 5760, 2880, ds_c4, "cam4")

    erp_pairs = np.column_stack([erp_c2, erp_c4])[keep]
    fisheye_pairs = np.column_stack([fisheye_c2, fisheye_c4])[keep]

    check_fisheye_finite(fisheye_pairs, f"sam3dbody_raw frame {frame_idx}")

    out_erp = output_base / "erp_corr_json"
    out_fisheye = output_base / "fisheye_corr_json"
    erp_path = out_erp / f"{frame_idx:05d}.json"
    fisheye_path = out_fisheye / f"{frame_idx:05d}.json"

    write_minimal_corr_json(erp_path, frame_idx, 5760, 2880, erp_pairs)
    write_minimal_corr_json(fisheye_path, frame_idx, 2880, 2880, fisheye_pairs)
    return n_keep


# ===================================================================
# dry-run
# ===================================================================


def dry_run_path(path: Path, label: str) -> bool:
    """检查路径是否存在，打印状态。返回 True = 存在。"""
    status = "EXISTS" if path.exists() else "MISSING"
    print(f"  [{status}] {label}: {path}")
    return path.exists()


def dry_run_method(args, method: str) -> bool:
    """dry-run 指定 method，返回 True 表示所有必要路径存在。"""
    print(f"\n{'=' * 60}")
    print(f"[DRY-RUN] method={method}  dataset={args.dataset}")
    print(f"  delivery_root: {args.delivery_root}")
    print(f"  camera_group: {get_camera_group_for_dataset(args.dataset)}")

    dataset_root = Path(args.data_root) / args.dataset
    delivery_root = Path(args.delivery_root)

    segment_specs = parse_segment_specs(args.segment)

    all_ok = True

    # common paths
    bbox_dir = Path(args.bbox_dir) if args.bbox_dir else (dataset_root / BBOX_SUBDIR_DEFAULT)
    erp_image_dir = (
        Path(args.erp_image_dir) if args.erp_image_dir else (dataset_root / ERP_IMAGE_SUBDIR)
    )
    print(f"  bbox_dir: {bbox_dir}")
    print(f"  erp_image_dir: {erp_image_dir}")

    if method in ("densekp_raw",):
        densekp_dir = (
            Path(args.densekp_dir) if args.densekp_dir else (dataset_root / DENSEKP_SUBDIR)
        )
        print(f"  densekp_dir: {densekp_dir}")

        for spec in segment_specs:
            for fid in spec.frame_indices:
                ok = True
                ok &= dry_run_path(densekp_dir / "cam2" / f"{fid:05d}_0.npz", f"densekp cam2 {fid:05d}")
                ok &= dry_run_path(densekp_dir / "cam4" / f"{fid:05d}_0.npz", f"densekp cam4 {fid:05d}")
                ok &= dry_run_path(bbox_dir / "cam2_bboxes.json", "cam2 bbox json")
                ok &= dry_run_path(bbox_dir / "cam4_bboxes.json", "cam4 bbox json")
                ok &= dry_run_path(erp_image_dir, f"ERP image root")
                erp_out = delivery_root / "data" / args.dataset / "smpl_corr" / method / "erp_corr_json" / f"{fid:05d}.json"
                fisheye_out = delivery_root / "data" / args.dataset / "smpl_corr" / method / "fisheye_corr_json" / f"{fid:05d}.json"
                print(f"    → ERP JSON:   {erp_out}")
                print(f"    → fisheye JSON: {fisheye_out}")
                if not ok:
                    all_ok = False

    elif method in ("densekp_fitting", "sam3dbody_fitting"):
        if method == "densekp_fitting":
            fitting_dir = (
                Path(args.densekp_fitting_dir)
                if args.densekp_fitting_dir
                else (dataset_root / "fitting/dual_cam_dense_filter_all")
            )
        else:
            fitting_dir = (
                Path(args.sam3dbody_fitting_dir)
                if args.sam3dbody_fitting_dir
                else (dataset_root / "fitting/dual_cam_sam3d_dense")
            )
        print(f"  fitting_dir: {fitting_dir}")

        for spec in segment_specs:
            print(f"\n  segment={spec.name}")
            try:
                pkl_c2, pkl_start = _resolve_fitting_pkl(
                    fitting_dir / "cam2", spec.start_frame, spec.end_frame, "cam2"
                )
                pkl_c4, _ = _resolve_fitting_pkl(
                    fitting_dir / "cam4", spec.start_frame, spec.end_frame, "cam4"
                )
            except FileNotFoundError as e:
                print(f"  [MISSING] {e}")
                all_ok = False
                continue
            ok = True
            ok &= dry_run_path(pkl_c2, f"pkl cam2: {pkl_c2}")
            ok &= dry_run_path(pkl_c4, f"pkl cam4: {pkl_c4}")
            ok &= dry_run_path(erp_image_dir, "ERP image root")
            if not ok:
                all_ok = False
            print(
                f"  frames: {spec.start_frame}..{spec.end_frame} "
                f"({len(spec.frame_indices)} frames)"
            )
            for fid in spec.frame_indices:
                erp_out = delivery_root / "data" / args.dataset / "smpl_corr" / method / "erp_corr_json" / f"{fid:05d}.json"
                fisheye_out = delivery_root / "data" / args.dataset / "smpl_corr" / method / "fisheye_corr_json" / f"{fid:05d}.json"
                print(f"    frame {fid:05d} → ERP: {erp_out}")
                print(f"              → fisheye: {fisheye_out}")

    elif method in ("sam3dbody_raw",):
        recon_dir = (
            Path(args.sam3dbody_recon_dir)
            if args.sam3dbody_recon_dir
            else (dataset_root / SAM3DBODY_RECON_SUBDIR)
        )
        mapping_path = (
            Path(args.mhr_to_smplx_mapping)
            if args.mhr_to_smplx_mapping
            else MAPPING_DEFAULT
        )
        print(f"  recon_dir: {recon_dir}")
        print(f"  mapping_path: {mapping_path}")

        ok = True
        ok &= dry_run_path(mapping_path, "MHR→SMPLX mapping")
        for spec in segment_specs:
            for fid in spec.frame_indices:
                ok &= dry_run_path(recon_dir / "cam2" / f"{fid:05d}.npz", f"SAM3D recon cam2 {fid:05d}")
                ok &= dry_run_path(recon_dir / "cam4" / f"{fid:05d}.npz", f"SAM3D recon cam4 {fid:05d}")
                ok &= dry_run_path(bbox_dir / "cam2_bboxes.json", "cam2 bbox json")
                ok &= dry_run_path(bbox_dir / "cam4_bboxes.json", "cam4 bbox json")
                ok &= dry_run_path(erp_image_dir, "ERP image root")
                erp_out = delivery_root / "data" / args.dataset / "smpl_corr" / method / "erp_corr_json" / f"{fid:05d}.json"
                fisheye_out = delivery_root / "data" / args.dataset / "smpl_corr" / method / "fisheye_corr_json" / f"{fid:05d}.json"
                print(f"    → ERP JSON:   {erp_out}")
                print(f"    → fisheye JSON: {fisheye_out}")
                if not ok:
                    all_ok = False

    if not all_ok:
        print("\n[DRY-RUN] FAIL: some required paths are missing")
    else:
        print("\n[DRY-RUN] all required paths exist")
    return all_ok


# ===================================================================
# main dispatch
# ===================================================================

# module-global for dataset name (used by per-frame helpers)
_dataset_name: str = ""


def main() -> None:
    global _dataset_name

    parser = argparse.ArgumentParser(
        description="Export SMPL cross-view sparse correspondences to minimal JSON."
    )
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--method", type=str, nargs="+", required=True)
    parser.add_argument("--segment", type=str, nargs="+", default=[])
    parser.add_argument("--delivery_root", type=str, required=True)
    parser.add_argument("--dry_run", action="store_true", default=False)
    parser.add_argument("--data_root", type=str, default=str(DATA_ROOT_DEFAULT))
    parser.add_argument("--summary_path", type=str, default=None)

    # path overrides
    parser.add_argument("--densekp_dir", type=str, default=None)
    parser.add_argument("--densekp_fitting_dir", type=str, default=None)
    parser.add_argument("--sam3dbody_fitting_dir", type=str, default=None)
    parser.add_argument("--sam3dbody_recon_dir", type=str, default=None)
    parser.add_argument("--bbox_dir", type=str, default=None)
    parser.add_argument("--mhr_to_smplx_mapping", type=str, default=None)
    parser.add_argument("--erp_image_dir", type=str, default=None)

    args = parser.parse_args()

    # validate
    if args.dataset not in ("20260311_riding", "20260320_riding_dataset1", "20260320_riding_dataset2"):
        parser.error(f"Unknown dataset: {args.dataset}")
    for m in args.method:
        if m not in METHODS:
            parser.error(f"Unknown method: {m}. Choose from: {', '.join(sorted(METHODS))}")

    if not args.segment:
        parser.error("--segment is required")

    segment_specs = parse_segment_specs(args.segment)
    frame_indices: list[int] = []
    for spec in segment_specs:
        frame_indices.extend(spec.frame_indices)

    _dataset_name = args.dataset

    # dry-run
    if args.dry_run:
        all_ok = True
        for method in args.method:
            ok = dry_run_method(args, method)
            if not ok:
                all_ok = False
        sys.exit(0 if all_ok else 1)

    # === real export =================================================
    dataset_root = Path(args.data_root) / args.dataset
    delivery_root = Path(args.delivery_root)
    for method in args.method:
        print(f"\n{'=' * 60}")
        print(f"[EXPORT] method={method}  dataset={args.dataset}")

        output_base = delivery_root / "data" / args.dataset / "smpl_corr" / method
        bbox_dir = Path(args.bbox_dir) if args.bbox_dir else (dataset_root / BBOX_SUBDIR_DEFAULT)
        erp_image_dir = (
            Path(args.erp_image_dir) if args.erp_image_dir else (dataset_root / ERP_IMAGE_SUBDIR)
        )

        # rotate rules (only needed for densekp_raw and sam3dbody_raw)
        rotate_rules = resolve_dataset_rotate_rules(
            str(ROTATE_RULES_CONFIG), str(bbox_dir)
        )

        n_total = 0
        keep_counts: list[int] = []

        if method == "densekp_raw":
            densekp_dir = (
                Path(args.densekp_dir) if args.densekp_dir else (dataset_root / DENSEKP_SUBDIR)
            )
            for fid in frame_indices:
                rotate_str = resolve_rotate_rule(fid, rotate_rules)
                n = export_densekp_raw(
                    fid, densekp_dir, bbox_dir, erp_image_dir,
                    rotate_str, output_base,
                )
                keep_counts.append(n)
                n_total += 1
                print(f"[densekp_raw] frame {fid:05d} → {n} pairs kept")

        elif method in ("densekp_fitting", "sam3dbody_fitting"):
            if method == "densekp_fitting":
                fitting_dir = (
                    Path(args.densekp_fitting_dir)
                    if args.densekp_fitting_dir
                    else (dataset_root / "fitting/dual_cam_dense_filter_all")
                )
            else:
                fitting_dir = (
                    Path(args.sam3dbody_fitting_dir)
                    if args.sam3dbody_fitting_dir
                    else (dataset_root / "fitting/dual_cam_sam3d_dense")
                )

            regressor = load_densekp_regressor()
            loader = _load_fitting_forwarder()

            for spec in segment_specs:
                pkl_c2, pkl_start = _resolve_fitting_pkl(
                    fitting_dir / "cam2", spec.start_frame, spec.end_frame, "cam2"
                )
                pkl_c4, _ = _resolve_fitting_pkl(
                    fitting_dir / "cam4", spec.start_frame, spec.end_frame, "cam4"
                )

                params_c2, seg_start = loader.load_segment_params(pkl_c2, pkl_start)
                params_c4, _ = loader.load_segment_params(pkl_c4, pkl_start)

                for fid in spec.frame_indices:
                    param_idx = fid - pkl_start
                    if param_idx < 0 or param_idx >= len(params_c2):
                        raise IndexError(
                            f"[{method}] segment={spec.name} frame={fid}: "
                            f"param_idx={param_idx} out of range "
                            f"(params_len={len(params_c2)})"
                        )
                    n = export_fitting(
                        fid, params_c2, params_c4, param_idx,
                        regressor, erp_image_dir, output_base, loader,
                    )
                    keep_counts.append(n)
                    n_total += 1
                    print(
                        f"[{method}] frame {fid:05d} → {n} pairs kept"
                    )

        elif method == "sam3dbody_raw":
            recon_dir = (
                Path(args.sam3dbody_recon_dir)
                if args.sam3dbody_recon_dir
                else (dataset_root / SAM3DBODY_RECON_SUBDIR)
            )
            mapping_path = (
                Path(args.mhr_to_smplx_mapping)
                if args.mhr_to_smplx_mapping
                else MAPPING_DEFAULT
            )
            regressor = load_densekp_regressor()

            for fid in frame_indices:
                rotate_str = resolve_rotate_rule(fid, rotate_rules)
                n = export_sam3dbody_raw(
                    fid, recon_dir, bbox_dir, erp_image_dir,
                    mapping_path, regressor, rotate_str, output_base,
                )
                keep_counts.append(n)
                n_total += 1
                print(f"[sam3dbody_raw] frame {fid:05d} → {n} pairs kept")

        # summary
        if keep_counts:
            print(f"\n[DONE] {n_total} frames, keep counts: min={min(keep_counts)} max={max(keep_counts)} mean={np.mean(keep_counts):.1f}")

        if args.summary_path:
            import json as _json

            summary = {
                "dataset": args.dataset,
                "method": method,
                "frames_processed": n_total,
                "keep_counts": keep_counts,
            }
            Path(args.summary_path).parent.mkdir(parents=True, exist_ok=True)
            with open(args.summary_path, "w", encoding="utf-8") as f:
                _json.dump(summary, f, ensure_ascii=False, indent=2)
            print(f"summary written to {args.summary_path}")


if __name__ == "__main__":
    main()


r'''
# A. 单段 dry-run：
python src/script/test_function/debug079_export_smpl_corr_delivery.py \
  --dataset 20260311_riding \
  --method densekp_raw \
  --segment 513_518 \
  --delivery_root /tmp/smpl_corr_test \
  --dry_run

# C. 多段正式导出：
python src/script/test_function/debug079_export_smpl_corr_delivery.py \
  --dataset 20260320_riding_dataset2 \
  --method densekp_raw densekp_fitting sam3dbody_raw sam3dbody_fitting \
  --segment 519_528 529_531 \
  --delivery_root Circular_Video/01_send/20260625_smpl_corr \
'''
