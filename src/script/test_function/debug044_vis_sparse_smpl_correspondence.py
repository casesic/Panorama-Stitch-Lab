from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

import config_script

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")


def _ensure_paths() -> Path:
    repo_root = Path(__file__).resolve().parents[3]
    sf_root = repo_root / "tools" / "SingleFitting"
    if str(sf_root) not in sys.path:
        sys.path.insert(0, str(sf_root))
    return repo_root


REPO_ROOT = _ensure_paths()

from flow.visibility import (  # noqa: E402
    get_smplx_batch,
    get_visible_vertices_erp,
    project_verts_to_rotated_erp,
)
from lib_dynamic_hvip.smpl.smpl_lib import forward_smplx_tensor_batch  # noqa: E402
from single_fitting.debug_shared import load_pickle  # noqa: E402


R_CAM2 = torch.tensor([1, 0, 0, 0, 0, 1, 0, -1, 0], dtype=torch.float32).reshape(3, 3)
R_CAM4 = torch.tensor([-1, 0, 0, 0, 1, 0, 0, 0, -1], dtype=torch.float32).reshape(3, 3)
R_CAM4_PROJ = R_CAM2

R_CAM2_NP = R_CAM2.numpy()
R_CAM4_PROJ_NP = R_CAM4_PROJ.numpy()
IDENTITY_NP = np.eye(3, dtype=np.float64)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize sparse SMPLX vertex-id correspondences between cam2 and cam4 in full ERP coordinates."
    )
    parser.add_argument("--fitting_dir", type=str, required=True, help="Fitting output root containing cam2/ and cam4/.")
    parser.add_argument("--output_dir", type=str, required=True, help="Output root directory.")
    parser.add_argument("--start_frame", type=int, default=None, help="Start frame index, inclusive.")
    parser.add_argument("--end_frame", type=int, default=None, help="End frame index, inclusive.")
    parser.add_argument("--image_width", type=int, default=5760, help="Full ERP width.")
    parser.add_argument("--image_height", type=int, default=2880, help="Full ERP height.")
    parser.add_argument("--visibility_mode", choices=("visible", "all"), default="visible")
    parser.add_argument("--max_lines", type=int, default=300, help="Maximum correspondence lines drawn in overlay_lines.")
    parser.add_argument("--sample_mode", choices=("uniform", "random"), default="uniform")
    parser.add_argument("--seed", type=int, default=1234, help="Random seed used when --sample_mode random.")
    parser.add_argument("--point_radius", type=int, default=2, help="Point radius on black visualization canvases.")
    parser.add_argument("--line_thickness", type=int, default=1, help="Line thickness in overlay_lines.")
    parser.add_argument("--vis_scale", type=float, default=1.0, help="Output visualization scale. NPZ coordinates remain full ERP.")
    parser.add_argument(
        "--save_npz",
        action="store_true",
        help="Save per-frame sparse correspondence npz.",
    )
    return parser.parse_args()


def build_segment_index(fitting_dir: Path) -> tuple[dict[int, tuple[Path, Path, int]], list[str]]:
    cam2_dir = fitting_dir / "cam2"
    cam4_dir = fitting_dir / "cam4"
    pattern = re.compile(r"^seg_(\d+)_(\d+)_seq\.pkl$")
    segment_index: dict[int, tuple[Path, Path, int]] = {}
    missing_cam4_segments: list[str] = []

    for pkl_path_c2 in sorted(cam2_dir.glob("seg_*_*_seq.pkl")):
        match = pattern.match(pkl_path_c2.name)
        if match is None:
            continue
        start = int(match.group(1))
        end = int(match.group(2))
        pkl_path_c4 = cam4_dir / pkl_path_c2.name
        if not pkl_path_c4.exists():
            missing_cam4_segments.append(pkl_path_c2.name)
            print(f"Warning: missing cam4 segment for {pkl_path_c2.name}, skipping segment")
            continue
        for frame_idx in range(start, end + 1):
            segment_index[frame_idx] = (pkl_path_c2, pkl_path_c4, start)

    return segment_index, missing_cam4_segments


def discover_frame_indices(fitting_dir: Path) -> list[int]:
    cam2_dir = fitting_dir / "cam2"
    pattern = re.compile(r"^seg_(\d+)_(\d+)_seq\.pkl$")
    frame_indices: set[int] = set()
    for pkl_path in cam2_dir.glob("seg_*_*_seq.pkl"):
        match = pattern.match(pkl_path.name)
        if match is None:
            continue
        start = int(match.group(1))
        end = int(match.group(2))
        frame_indices.update(range(start, end + 1))
    return sorted(frame_indices)


def load_all_verts(
    pkl_path: Path,
    model,
    n_expand_beta: int | None = None,
) -> np.ndarray:
    smplx_param = load_pickle(pkl_path)
    n_frame = n_expand_beta if n_expand_beta is not None else smplx_param.num_batch
    with torch.no_grad():
        verts_t, _, _ = forward_smplx_tensor_batch(
            smplx_model=model,
            global_orient=smplx_param.orient,
            body_pose=smplx_param.body_pose,
            left_hand_pose=smplx_param.left_hand_pose,
            right_hand_pose=smplx_param.right_hand_pose,
            betas=smplx_param.beta.unsqueeze(0).expand(n_frame, -1),
            transl=smplx_param.transl,
            scale=smplx_param.scale.expand(n_frame),
        )
    return verts_t.detach().cpu().numpy()


def projected_in_bounds(pts_2d: np.ndarray, image_width: int, image_height: int) -> np.ndarray:
    finite = np.isfinite(pts_2d).all(axis=1)
    return (
        finite
        & (pts_2d[:, 0] >= 0)
        & (pts_2d[:, 0] < image_width)
        & (pts_2d[:, 1] >= 0)
        & (pts_2d[:, 1] < image_height)
    )


def sample_vertex_ids(vertex_ids: np.ndarray, max_lines: int, sample_mode: str, seed: int) -> np.ndarray:
    if max_lines <= 0 or len(vertex_ids) == 0:
        return np.empty((0,), dtype=np.int64)
    if len(vertex_ids) <= max_lines:
        return vertex_ids.astype(np.int64, copy=True)
    if sample_mode == "uniform":
        sample_indices = np.linspace(0, len(vertex_ids) - 1, max_lines, dtype=np.int64)
        return vertex_ids[sample_indices].astype(np.int64, copy=True)

    rng = np.random.default_rng(seed)
    sample_indices = rng.choice(len(vertex_ids), size=max_lines, replace=False)
    sample_indices.sort()
    return vertex_ids[sample_indices].astype(np.int64, copy=True)


def make_scaled_shape(image_width: int, image_height: int, vis_scale: float) -> tuple[int, int]:
    scaled_w = max(1, int(round(image_width * vis_scale)))
    scaled_h = max(1, int(round(image_height * vis_scale)))
    return scaled_w, scaled_h


def draw_points_canvas(
    pts_2d: np.ndarray,
    vertex_ids: np.ndarray,
    image_width: int,
    image_height: int,
    vis_scale: float,
    point_radius: int,
    color: tuple[int, int, int] = (0, 255, 0),
) -> np.ndarray:
    scaled_w, scaled_h = make_scaled_shape(image_width, image_height, vis_scale)
    canvas = np.zeros((scaled_h, scaled_w, 3), dtype=np.uint8)
    radius = max(1, int(round(point_radius * vis_scale)))

    if len(vertex_ids) == 0:
        return canvas

    draw_pts = np.rint(pts_2d[vertex_ids] * vis_scale).astype(np.int32)
    for u, v in draw_pts:
        if 0 <= u < scaled_w and 0 <= v < scaled_h:
            cv2.circle(canvas, (int(u), int(v)), radius, color, -1, lineType=cv2.LINE_AA)
    return canvas


def draw_overlay_lines_canvas(
    pts_2d_cam2: np.ndarray,
    pts_2d_cam4: np.ndarray,
    visible_ids_cam2: np.ndarray,
    visible_ids_cam4: np.ndarray,
    line_vertex_ids: np.ndarray,
    image_width: int,
    image_height: int,
    vis_scale: float,
    point_radius: int,
    line_thickness: int,
) -> np.ndarray:
    scaled_w, scaled_h = make_scaled_shape(image_width, image_height, vis_scale)
    canvas = np.zeros((scaled_h, scaled_w, 3), dtype=np.uint8)
    radius = max(1, int(round(point_radius * vis_scale)))
    thickness = max(1, int(round(line_thickness * vis_scale)))

    cam2_pts = np.rint(pts_2d_cam2[visible_ids_cam2] * vis_scale).astype(np.int32)
    for u, v in cam2_pts:
        if 0 <= u < scaled_w and 0 <= v < scaled_h:
            cv2.circle(canvas, (int(u), int(v)), radius, (0, 255, 0), -1, lineType=cv2.LINE_AA)

    cam4_pts = np.rint(pts_2d_cam4[visible_ids_cam4] * vis_scale).astype(np.int32)
    for u, v in cam4_pts:
        if 0 <= u < scaled_w and 0 <= v < scaled_h:
            cv2.circle(canvas, (int(u), int(v)), radius, (0, 255, 255), -1, lineType=cv2.LINE_AA)

    for vertex_id in line_vertex_ids:
        p2 = pts_2d_cam2[vertex_id]
        p4 = pts_2d_cam4[vertex_id]
        p2_draw = (int(round(p2[0] * vis_scale)), int(round(p2[1] * vis_scale)))
        p4_draw = (int(round(p4[0] * vis_scale)), int(round(p4[1] * vis_scale)))

        cv2.line(canvas, p2_draw, p4_draw, (220, 220, 220), thickness, lineType=cv2.LINE_AA)
        cv2.circle(canvas, p2_draw, radius, (0, 255, 0), -1, lineType=cv2.LINE_AA)
        cv2.circle(canvas, p4_draw, radius, (0, 255, 255), -1, lineType=cv2.LINE_AA)

    return canvas


def process_frame(
    frame_idx: int,
    segment_name: str,
    verts_c2: np.ndarray,
    verts_c4: np.ndarray,
    faces: np.ndarray,
    output_dir: Path,
    image_width: int,
    image_height: int,
    visibility_mode: str,
    max_lines: int,
    sample_mode: str,
    seed: int,
    point_radius: int,
    line_thickness: int,
    vis_scale: float,
    save_npz: bool,
) -> dict:
    verts_unified_c2 = verts_c2 @ R_CAM2_NP.T
    verts_unified_c4 = verts_c4 @ R_CAM4_PROJ_NP.T

    pts_2d_cam2 = project_verts_to_rotated_erp(
        verts_unified_c2,
        IDENTITY_NP,
        erp_h=image_height,
        erp_w=image_width,
    )
    pts_2d_cam4 = project_verts_to_rotated_erp(
        verts_unified_c4,
        IDENTITY_NP,
        erp_h=image_height,
        erp_w=image_width,
    )

    if visibility_mode == "visible":
        _, visible_mask_c2 = get_visible_vertices_erp(
            verts_world=verts_unified_c2,
            faces=faces,
            rotation_matrix=IDENTITY_NP,
            erp_h=image_height,
            erp_w=image_width,
            roi_bbox=None,
        )
        _, visible_mask_c4 = get_visible_vertices_erp(
            verts_world=verts_unified_c4,
            faces=faces,
            rotation_matrix=IDENTITY_NP,
            erp_h=image_height,
            erp_w=image_width,
            roi_bbox=None,
        )
    else:
        visible_mask_c2 = projected_in_bounds(pts_2d_cam2, image_width, image_height)
        visible_mask_c4 = projected_in_bounds(pts_2d_cam4, image_width, image_height)

    visible_ids_cam2 = np.where(visible_mask_c2)[0].astype(np.int64)
    visible_ids_cam4 = np.where(visible_mask_c4)[0].astype(np.int64)
    common_visible_ids = np.intersect1d(visible_ids_cam2, visible_ids_cam4, assume_unique=True).astype(np.int64)
    line_vertex_ids = sample_vertex_ids(common_visible_ids, max_lines, sample_mode, seed)

    cam2_points = draw_points_canvas(
        pts_2d=pts_2d_cam2,
        vertex_ids=visible_ids_cam2,
        image_width=image_width,
        image_height=image_height,
        vis_scale=vis_scale,
        point_radius=point_radius,
    )
    cam4_points = draw_points_canvas(
        pts_2d=pts_2d_cam4,
        vertex_ids=visible_ids_cam4,
        image_width=image_width,
        image_height=image_height,
        vis_scale=vis_scale,
        point_radius=point_radius,
    )
    overlay_lines = draw_overlay_lines_canvas(
        pts_2d_cam2=pts_2d_cam2,
        pts_2d_cam4=pts_2d_cam4,
        visible_ids_cam2=visible_ids_cam2,
        visible_ids_cam4=visible_ids_cam4,
        line_vertex_ids=line_vertex_ids,
        image_width=image_width,
        image_height=image_height,
        vis_scale=vis_scale,
        point_radius=point_radius,
        line_thickness=line_thickness,
    )

    frame_key = f"{frame_idx:05d}"
    cv2.imwrite(str(output_dir / "cam2_points" / f"{frame_key}.png"), cam2_points)
    cv2.imwrite(str(output_dir / "cam4_points" / f"{frame_key}.png"), cam4_points)
    cv2.imwrite(str(output_dir / "overlay_lines" / f"{frame_key}.png"), overlay_lines)

    if save_npz:
        np.savez_compressed(
            output_dir / "frame_data" / f"{frame_key}_sparse_corr.npz",
            visible_ids_cam2=visible_ids_cam2,
            visible_ids_cam4=visible_ids_cam4,
            common_visible_ids=common_visible_ids,
            line_vertex_ids_drawn=line_vertex_ids,
            pts_cam2_all=pts_2d_cam2,
            pts_cam4_all=pts_2d_cam4,
            pts_cam2_visible=pts_2d_cam2[visible_ids_cam2],
            pts_cam4_visible=pts_2d_cam4[visible_ids_cam4],
            pts_cam2_common=pts_2d_cam2[common_visible_ids],
            pts_cam4_common=pts_2d_cam4[common_visible_ids],
            pts_line_cam2=pts_2d_cam2[line_vertex_ids],
            pts_line_cam4=pts_2d_cam4[line_vertex_ids],
        )

    result = {
        "valid": True,
        "segment": segment_name,
        "visible_count_cam2": int(len(visible_ids_cam2)),
        "visible_count_cam4": int(len(visible_ids_cam4)),
        "common_visible_count": int(len(common_visible_ids)),
        "drawn_points_cam2": int(len(visible_ids_cam2)),
        "drawn_points_cam4": int(len(visible_ids_cam4)),
        "drawn_lines": int(len(line_vertex_ids)),
    }
    return result


def make_failure_frame(reason: str, segment_name: str | None = None) -> dict:
    result = {
        "valid": False,
        "reason": reason,
        "visible_count_cam2": 0,
        "visible_count_cam4": 0,
        "common_visible_count": 0,
        "drawn_points_cam2": 0,
        "drawn_points_cam4": 0,
        "drawn_lines": 0,
    }
    if segment_name is not None:
        result["segment"] = segment_name
    return result


def main() -> None:
    args = parse_args()
    if args.image_width <= 0 or args.image_height <= 0:
        raise ValueError("--image_width and --image_height must be positive")
    if args.vis_scale <= 0:
        raise ValueError("--vis_scale must be positive")
    if args.max_lines < 0:
        raise ValueError("--max_lines must be non-negative")
    if args.point_radius <= 0:
        raise ValueError("--point_radius must be positive")
    if args.line_thickness <= 0:
        raise ValueError("--line_thickness must be positive")

    fitting_dir = Path(args.fitting_dir)
    output_dir = Path(args.output_dir)
    for subdir in ("cam2_points", "cam4_points", "overlay_lines", "frame_data"):
        (output_dir / subdir).mkdir(parents=True, exist_ok=True)

    smplx_model = get_smplx_batch(gender="neutral")
    faces = smplx_model.faces.numpy() if hasattr(smplx_model.faces, "numpy") else np.asarray(smplx_model.faces)
    faces = np.asarray(faces, dtype=int)

    segment_index, missing_cam4_segments = build_segment_index(fitting_dir)
    frame_indices = discover_frame_indices(fitting_dir)
    if args.start_frame is not None or args.end_frame is not None:
        if args.start_frame is not None and args.end_frame is not None:
            frame_indices = list(range(args.start_frame, args.end_frame + 1))
        else:
            if args.start_frame is not None:
                frame_indices = [frame_idx for frame_idx in frame_indices if frame_idx >= args.start_frame]
            if args.end_frame is not None:
                frame_indices = [frame_idx for frame_idx in frame_indices if frame_idx <= args.end_frame]
    else:
        frame_indices = sorted(set(frame_indices) | set(segment_index.keys()))

    meta_frames: dict[str, dict] = {}
    summary_counts: dict[str, int] = {
        "missing_segment": 0,
        "frame_offset_out_of_range": 0,
        "error": 0,
    }
    _segment_cache: dict[Path, np.ndarray] = {}

    for frame_idx in frame_indices:
        frame_key = f"{frame_idx:05d}"
        segment_info = segment_index.get(frame_idx)
        if segment_info is None:
            meta_frames[frame_key] = make_failure_frame("no_segment")
            summary_counts["missing_segment"] += 1
            print(f"[frame {frame_key}] no_segment")
            continue

        pkl_path_c2, pkl_path_c4, segment_start = segment_info
        frame_offset = frame_idx - segment_start
        try:
            if pkl_path_c2 not in _segment_cache:
                _segment_cache[pkl_path_c2] = load_all_verts(pkl_path_c2, smplx_model)
            if pkl_path_c4 not in _segment_cache:
                _segment_cache[pkl_path_c4] = load_all_verts(pkl_path_c4, smplx_model)

            verts_all_c2 = _segment_cache[pkl_path_c2]
            verts_all_c4 = _segment_cache[pkl_path_c4]
            if frame_offset < 0 or frame_offset >= len(verts_all_c2) or frame_offset >= len(verts_all_c4):
                meta_frames[frame_key] = make_failure_frame(
                    "frame_offset_out_of_range",
                    segment_name=pkl_path_c2.name,
                )
                summary_counts["frame_offset_out_of_range"] += 1
                print(f"[frame {frame_key}] frame_offset_out_of_range")
                continue

            meta_frames[frame_key] = process_frame(
                frame_idx=frame_idx,
                segment_name=pkl_path_c2.name,
                verts_c2=verts_all_c2[frame_offset],
                verts_c4=verts_all_c4[frame_offset],
                faces=faces,
                output_dir=output_dir,
                image_width=args.image_width,
                image_height=args.image_height,
                visibility_mode=args.visibility_mode,
                max_lines=args.max_lines,
                sample_mode=args.sample_mode,
                seed=args.seed,
                point_radius=args.point_radius,
                line_thickness=args.line_thickness,
                vis_scale=args.vis_scale,
                save_npz=args.save_npz,
            )
            print(
                f"[frame {frame_key}] ok "
                f"cam2={meta_frames[frame_key]['visible_count_cam2']} "
                f"cam4={meta_frames[frame_key]['visible_count_cam4']} "
                f"common={meta_frames[frame_key]['common_visible_count']} "
                f"lines={meta_frames[frame_key]['drawn_lines']}"
            )
        except Exception as exc:
            meta_frames[frame_key] = make_failure_frame(
                f"error: {exc}",
                segment_name=pkl_path_c2.name,
            )
            summary_counts["error"] += 1
            print(f"[frame {frame_key}] error: {exc}")

    valid_frames = sum(1 for item in meta_frames.values() if item["valid"])
    failed_frames = len(meta_frames) - valid_frames
    meta = {
        "meta": {
            "fitting_dir": str(fitting_dir),
            "image_width": args.image_width,
            "image_height": args.image_height,
            "visibility_mode": args.visibility_mode,
            "max_lines": args.max_lines,
            "sample_mode": args.sample_mode,
            "seed": args.seed,
            "point_radius": args.point_radius,
            "line_thickness": args.line_thickness,
            "vis_scale": args.vis_scale,
            "save_npz": bool(args.save_npz),
            "missing_cam4_segments": missing_cam4_segments,
            "outputs": {
                "cam2_points": "single-camera cam2 visible vertices on a black full ERP canvas",
                "cam4_points": "single-camera cam4 visible vertices on a black full ERP canvas",
                "overlay_lines": "all cam2/cam4 visible points plus sampled common visible correspondence lines on the same full ERP canvas",
                "frame_data": "full ERP sparse correspondence arrays; coordinates are not affected by vis_scale",
            },
        },
        "frames": meta_frames,
        "summary": {
            "total_frames": len(meta_frames),
            "valid_frames": valid_frames,
            "failed_frames": failed_frames,
            "missing_segment": summary_counts["missing_segment"],
            "missing_cam4_segment": len(missing_cam4_segments),
            "frame_offset_out_of_range": summary_counts["frame_offset_out_of_range"],
            "error": summary_counts["error"],
        },
    }

    meta_path = output_dir / "sparse_corr_meta.json"
    with meta_path.open("w", encoding="utf-8") as file:
        json.dump(meta, file, ensure_ascii=False, indent=2)

    print(f"valid frames: {valid_frames}")
    print(f"failed frames: {failed_frames}")
    print(f"missing_cam4_segment: {len(missing_cam4_segments)}")
    print(f"saved meta to: {meta_path}")


if __name__ == "__main__":
    main()


r'''
python src/script/test_function/debug044_vis_sparse_smpl_correspondence.py `
  --fitting_dir data_process/Output/20260311_riding/fitting/dual_cam_new_sil `
  --output_dir data_process/Output/20260311_riding/sparse_smpl_corr/dual_cam_new_sil `
  --start_frame 513 `
  --end_frame 518 `
  --image_width 5760 `
  --image_height 2880 `
  --visibility_mode visible `
  --max_lines 300 `
  --sample_mode uniform `
  --save_npz

Outputs:
  cam2_points / cam4_points: single-camera visible vertices.
  overlay_lines: all visible points plus sampled correspondence lines on the same full ERP canvas.
'''
