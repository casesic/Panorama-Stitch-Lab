from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import os
import config_script
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')

def _ensure_paths() -> Path:
    repo_root = Path(__file__).resolve().parents[3]
    sf_root = repo_root / "tools" / "SingleFitting"
    if str(sf_root) not in sys.path:
        sys.path.insert(0, str(sf_root))
    return repo_root


REPO_ROOT = _ensure_paths()

from flow.visibility import (
    compute_semantic_flow,
    get_visible_vertices_erp,
    load_smplx_faces_and_verts,
    project_verts_to_rotated_erp,
)
ROTATION_MATRIX = np.array([1, 0, 0, 0, 0, 1, 0, -1, 0], dtype=np.float64).reshape(3, 3)
R_CAM4_NP = np.array([-1, 0, 0, 0, 1, 0, 0, 0, -1], dtype=np.float64).reshape(3, 3)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch semantic flow generation from DualCamERPFitter results.")
    parser.add_argument("--fitting_dir", type=str, required=True, help="DualCamERPFitter output root containing cam2/ and cam4/")
    parser.add_argument("--image_dir", type=str, required=True, help="Equirect image root containing cam2/ and cam4/")
    parser.add_argument("--output_dir", type=str, required=True, help="Output root directory.")
    parser.add_argument("--start_frame", type=int, required=True, help="Start frame index, inclusive.")
    parser.add_argument("--end_frame", type=int, required=True, help="End frame index, inclusive.")
    parser.add_argument("--save_vis", action="store_true", help="Save visualization results.")
    return parser.parse_args()


def find_segment_pkl(fitting_dir: Path, frame_idx: int):
    cam2_dir = fitting_dir / "cam2"
    for pkl_path_c2 in sorted(cam2_dir.glob("seg_*_*_seq.pkl")):
        stem = pkl_path_c2.stem
        parts = stem.split("_")
        if len(parts) != 4:
            continue
        _, start_s, end_s, _ = parts
        start = int(start_s)
        end = int(end_s)
        if start <= frame_idx <= end:
            pkl_path_c4 = fitting_dir / "cam4" / pkl_path_c2.name
            return pkl_path_c2, pkl_path_c4, start
    return None, None, None


def load_verts_for_frame(pkl_path: Path, frame_idx: int, segment_start: int, gender: str):
    verts_all, faces = load_smplx_faces_and_verts(str(pkl_path), gender=gender)
    frame_offset = frame_idx - segment_start
    if frame_offset < 0 or frame_offset >= len(verts_all):
        raise IndexError(f"Frame {frame_idx} out of range for segment {pkl_path.name}")
    return verts_all[frame_offset], faces


def compute_human_bbox(pts_list, erp_h, erp_w, padding=30):
    valid_pts = []
    for pts in pts_list:
        if pts is None or len(pts) == 0:
            continue
        pts = np.asarray(pts, dtype=np.float64)
        finite_mask = np.isfinite(pts).all(axis=1)
        if np.any(finite_mask):
            valid_pts.append(pts[finite_mask])

    if not valid_pts:
        return (0, 0, erp_w, erp_h)

    all_pts = np.concatenate(valid_pts, axis=0)
    x1 = max(0, int(np.floor(all_pts[:, 0].min())) - padding)
    y1 = max(0, int(np.floor(all_pts[:, 1].min())) - padding)
    x2 = min(erp_w, int(np.ceil(all_pts[:, 0].max())) + padding)
    y2 = min(erp_h, int(np.ceil(all_pts[:, 1].max())) + padding)
    return (x1, y1, x2, y2)


def visualize_mesh_on_erp(img, pts_2d, color=(0, 255, 0), radius=2):
    canvas = img.copy()
    if pts_2d is None or len(pts_2d) == 0:
        return canvas

    h, w = canvas.shape[:2]
    pts_px = np.rint(np.asarray(pts_2d, dtype=np.float64)).astype(np.int32)
    for u, v in pts_px:
        if 0 <= u < w and 0 <= v < h:
            cv2.circle(canvas, (u, v), radius, color, -1)
    return canvas


def visualize_flow(flow_roi, confidence_roi):
    flow_mag = np.linalg.norm(flow_roi, axis=2)
    flow_angle = np.arctan2(flow_roi[..., 1], flow_roi[..., 0])

    hsv = np.zeros((flow_roi.shape[0], flow_roi.shape[1], 3), dtype=np.uint8)
    hsv[..., 0] = ((flow_angle + np.pi) / (2 * np.pi) * 179).astype(np.uint8)
    hsv[..., 1] = 255
    max_mag = max(float(flow_mag.max()), 1e-6)
    hsv[..., 2] = (np.clip(flow_mag / max_mag, 0.0, 1.0) * 255).astype(np.uint8)
    flow_color = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    flow_color[confidence_roi == 0] = 0

    flow_arrows = np.zeros((flow_roi.shape[0], flow_roi.shape[1], 3), dtype=np.uint8)
    step = 20
    for r in range(0, flow_roi.shape[0], step):
        for c in range(0, flow_roi.shape[1], step):
            conf = int(confidence_roi[r, c])
            if conf == 0:
                continue
            dx, dy = flow_roi[r, c]
            color = (0, 255, 0) if conf == 2 else (255, 0, 0)
            cv2.arrowedLine(
                flow_arrows,
                (c, r),
                (int(round(c + dx)), int(round(r + dy))),
                color,
                1,
                tipLength=0.3,
            )

    return flow_color, flow_arrows


def process_frame(
    frame_idx: int,
    verts_c2: np.ndarray,
    verts_c4: np.ndarray,
    faces: np.ndarray,
    image_dir: Path,
    output_dir: Path,
    save_vis: bool,
):
    image_path_c2 = image_dir / "cam2" / f"{frame_idx:05d}.jpg"
    image_path_c4 = image_dir / "cam4" / f"{frame_idx:05d}.jpg"
    img_cam2 = cv2.imread(str(image_path_c2), cv2.IMREAD_COLOR) if image_path_c2.exists() else None
    img_cam4 = cv2.imread(str(image_path_c4), cv2.IMREAD_COLOR) if image_path_c4.exists() else None

    if img_cam2 is None and img_cam4 is None:
        return {"frame_idx": frame_idx, "valid": False, "reason": "missing_both"}
    if img_cam2 is None:
        return {"frame_idx": frame_idx, "valid": False, "reason": "missing_cam2"}
    if img_cam4 is None:
        return {"frame_idx": frame_idx, "valid": False, "reason": "missing_cam4"}

    verts_unified_c2 = verts_c2
    verts_unified_c4 = verts_c4 @ R_CAM4_NP.T
    verts_unified_c4 = verts_c4

    erp_h, erp_w = img_cam2.shape[:2]
    pts_2d_cam2 = project_verts_to_rotated_erp(verts_unified_c2, ROTATION_MATRIX, erp_h, erp_w)
    pts_2d_cam4 = project_verts_to_rotated_erp(verts_unified_c4, ROTATION_MATRIX, erp_h, erp_w)
    human_bbox = compute_human_bbox([pts_2d_cam2, pts_2d_cam4], erp_h, erp_w)

    _, visible_mask_c2 = get_visible_vertices_erp(
        verts_world=verts_unified_c2,
        faces=faces,
        rotation_matrix=ROTATION_MATRIX,
        erp_h=erp_h,
        erp_w=erp_w,
    )
    _, visible_mask_c4 = get_visible_vertices_erp(
        verts_world=verts_unified_c4,
        faces=faces,
        rotation_matrix=ROTATION_MATRIX,
        erp_h=erp_h,
        erp_w=erp_w,
    )

    flow_c2c4, conf_c2c4 = compute_semantic_flow(
        verts_world_A=verts_unified_c4,
        verts_world_B=verts_unified_c2,
        faces=faces,
        erp_h=erp_h,
        erp_w=erp_w,
        rotation_matrix=ROTATION_MATRIX,
        roi_bbox=human_bbox,
        visible_mask_B=visible_mask_c2,
    )
    flow_c4c2, conf_c4c2 = compute_semantic_flow(
        verts_world_A=verts_unified_c2,
        verts_world_B=verts_unified_c4,
        faces=faces,
        erp_h=erp_h,
        erp_w=erp_w,
        rotation_matrix=ROTATION_MATRIX,
        roi_bbox=human_bbox,
        visible_mask_B=visible_mask_c4,
    )

    frame_output_dir = output_dir / f"frame_{frame_idx:05d}"
    frame_output_dir.mkdir(parents=True, exist_ok=True)
    np.save(frame_output_dir / "semantic_flow_c2_to_c4.npy", flow_c2c4)
    np.save(frame_output_dir / "semantic_flow_confidence_c2_to_c4.npy", conf_c2c4)
    np.save(frame_output_dir / "semantic_flow_c4_to_c2.npy", flow_c4c2)
    np.save(frame_output_dir / "semantic_flow_confidence_c4_to_c2.npy", conf_c4c2)
    np.save(frame_output_dir / "human_bbox.npy", np.asarray(human_bbox, dtype=np.int32))

    if save_vis:
        mesh_cam2 = visualize_mesh_on_erp(img_cam2, pts_2d_cam2)
        mesh_cam4 = visualize_mesh_on_erp(img_cam4, pts_2d_cam4)
        flow_color_c2c4, flow_arrows_c2c4 = visualize_flow(flow_c2c4, conf_c2c4)
        flow_color_c4c2, flow_arrows_c4c2 = visualize_flow(flow_c4c2, conf_c4c2)
        cv2.imwrite(str(frame_output_dir / "mesh_on_erp_cam2.jpg"), mesh_cam2, [cv2.IMWRITE_JPEG_QUALITY, 95])
        cv2.imwrite(str(frame_output_dir / "mesh_on_erp_cam4.jpg"), mesh_cam4, [cv2.IMWRITE_JPEG_QUALITY, 95])
        cv2.imwrite(str(frame_output_dir / "flow_color_c2_to_c4.png"), flow_color_c2c4)
        cv2.imwrite(str(frame_output_dir / "flow_arrows_c2_to_c4.png"), flow_arrows_c2c4)
        cv2.imwrite(str(frame_output_dir / "flow_color_c4_to_c2.png"), flow_color_c4c2)
        cv2.imwrite(str(frame_output_dir / "flow_arrows_c4_to_c2.png"), flow_arrows_c4c2)

    return {
        "frame_idx": frame_idx,
        "valid": True,
        "reason": "ok",
        "human_bbox": list(human_bbox),
        "flow_pixels_c2_to_c4": int((conf_c2c4 > 0).sum()),
        "flow_pixels_c4_to_c2": int((conf_c4c2 > 0).sum()),
    }


def main() -> None:
    args = parse_args()

    fitting_dir = Path(args.fitting_dir)
    image_dir = Path(args.image_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    meta = []
    for frame_idx in range(args.start_frame, args.end_frame + 1):
        pkl_path_c2, pkl_path_c4, segment_start = find_segment_pkl(fitting_dir, frame_idx)
        if pkl_path_c2 is None or pkl_path_c4 is None or segment_start is None:
            result = {"frame_idx": frame_idx, "valid": False, "reason": "no_segment"}
            meta.append(result)
            print(f"[frame {frame_idx:05d}] no_segment")
            continue

        try:
            verts_c2, faces = load_verts_for_frame(pkl_path_c2, frame_idx, segment_start, gender="neutral")
            verts_c4, faces_cam4 = load_verts_for_frame(pkl_path_c4, frame_idx, segment_start, gender="neutral")
            if not np.array_equal(faces, faces_cam4):
                raise ValueError("SMPL-X faces mismatch between cam2 and cam4 PKL inputs")
            result = process_frame(
                frame_idx=frame_idx,
                verts_c2=verts_c2,
                verts_c4=verts_c4,
                faces=faces,
                image_dir=image_dir,
                output_dir=output_dir,
                save_vis=args.save_vis,
            )
        except Exception as exc:
            result = {"frame_idx": frame_idx, "valid": False, "reason": f"error: {exc}"}

        meta.append(result)
        print(f"[frame {frame_idx:05d}] {result['reason']}")

    meta_path = output_dir / "flow_meta.json"
    with meta_path.open("w", encoding="utf-8") as file:
        json.dump(meta, file, ensure_ascii=False, indent=2)

    valid_count = sum(1 for item in meta if item["valid"])
    no_segment_count = sum(1 for item in meta if item["reason"] == "no_segment")
    missing_cam2 = sum(1 for item in meta if item["reason"] == "missing_cam2")
    missing_cam4 = sum(1 for item in meta if item["reason"] == "missing_cam4")
    missing_both = sum(1 for item in meta if item["reason"] == "missing_both")

    print(f"valid frames: {valid_count}")
    print(f"no_segment: {no_segment_count}")
    print(f"missing_cam2: {missing_cam2}")
    print(f"missing_cam4: {missing_cam4}")
    print(f"missing_both: {missing_both}")
    print(f"saved meta to: {meta_path}")


if __name__ == "__main__":
    main()

r'''
python src\script\test_function\debug031_batch_semantic_flow.py `
  --fitting_dir data_process\Output\20260311_riding\fitting\dual_cam `
  --image_dir data_process\Output\20260311_riding\image\equirect_rotate `
  --output_dir data_process\Output\20260311_riding\semantic_flow `
  --start_frame 513 `
  --end_frame 519 `
  --save_vis
'''
