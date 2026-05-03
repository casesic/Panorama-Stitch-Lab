from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import cv2
import numpy as np
import os
import torch
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
    get_smplx_batch,
    get_visible_vertices_erp,
    project_verts_to_rotated_erp,
)
from lib_dynamic_hvip.smpl.smpl_lib import forward_smplx_tensor_batch
from single_fitting.debug_shared import load_pickle
R_CAM2 = torch.tensor([1, 0, 0, 0, 0, 1, 0, -1, 0], dtype=torch.float32).reshape(3, 3)
R_CAM4 = torch.tensor([-1, 0, 0, 0, 1, 0, 0, 0, -1], dtype=torch.float32).reshape(3, 3)
R_CAM4_PROJ = R_CAM2   # 用于 cam4 投影

R_CAM2_NP = R_CAM2.numpy()
R_CAM4_PROJ_NP = R_CAM4_PROJ.numpy()
IDENTITY_NP = np.eye(3, dtype=np.float64)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch semantic flow generation from DualCamERPFitter results.")
    parser.add_argument("--fitting_dir", type=str, required=True, help="DualCamERPFitter output root containing cam2/ and cam4/")
    parser.add_argument("--image_dir", type=str, required=True, help="Equirect image root containing cam2/ and cam4/")
    parser.add_argument("--output_dir", type=str, required=True, help="Output root directory.")
    parser.add_argument("--invalid_json", type=str, default=None, help="Path to invalid.json from debug030.")
    parser.add_argument("--start_frame", type=int, required=False, default=None, help="Start frame index, inclusive.")
    parser.add_argument("--end_frame", type=int, required=False, default=None, help="End frame index, inclusive.")
    parser.add_argument("--save_vis", action="store_true", help="Save visualization results.")
    parser.add_argument(
        "--directions",
        choices=("both", "c2_to_c4", "c4_to_c2"),
        default="both",
        help="Semantic flow direction(s) to compute.",
    )
    parser.add_argument("--image_width", type=int, default=5760, help="ERP image width used when --save_vis is not set.")
    parser.add_argument("--image_height", type=int, default=2880, help="ERP image height used when --save_vis is not set.")
    parser.add_argument("--bbox_dir", type=str, default=None, help="Optional crop bbox directory containing cam2/cam4 bbox json.")
    parser.add_argument("--bbox_padding", type=int, default=100, help="Padding added to bbox union when --bbox_dir is provided.")
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


def build_segment_index(fitting_dir: Path) -> dict[int, tuple[Path, Path, int]]:
    cam2_dir = fitting_dir / "cam2"
    cam4_dir = fitting_dir / "cam4"
    pattern = re.compile(r"^seg_(\d+)_(\d+)_seq\.pkl$")
    segment_index: dict[int, tuple[Path, Path, int]] = {}

    for pkl_path_c2 in sorted(cam2_dir.glob("seg_*_*_seq.pkl")):
        match = pattern.match(pkl_path_c2.name)
        if match is None:
            continue
        start = int(match.group(1))
        end = int(match.group(2))
        pkl_path_c4 = cam4_dir / pkl_path_c2.name
        if not pkl_path_c4.exists():
            print(f"Warning: missing cam4 segment for {pkl_path_c2.name}, skipping segment")
            continue
        for frame_idx in range(start, end + 1):
            segment_index[frame_idx] = (pkl_path_c2, pkl_path_c4, start)

    return segment_index


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


def load_invalid_frames(invalid_json_path: Path | None) -> set[int]:
    if invalid_json_path is None or not invalid_json_path.exists():
        return set()

    with invalid_json_path.open("r", encoding="utf-8") as file:
        invalid_json = json.load(file)

    cam2_frames = invalid_json.get("cam2", {}).get("equirect", [])
    cam4_frames = invalid_json.get("cam4", {}).get("equirect", [])
    return set(cam2_frames) | set(cam4_frames)


def load_bbox_dict(json_path: Path) -> dict[str, list[int]]:
    with json_path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, dict):
        raise ValueError(f"bbox json must contain a dict: {json_path}")
    return data


def discover_frame_indices(fitting_dir: Path) -> list[int]:
    cam2_dir = fitting_dir / "cam2"
    pattern = re.compile(r"^seg_(\d+)_(\d+)_seq\.pkl$")
    frame_indices: set[int] = set()
    for pkl_path in cam2_dir.glob("seg_*_*_seq.pkl"):
        match = pattern.match(pkl_path.name)
        if match is None:
            continue
        start_frame = int(match.group(1))
        end_frame = int(match.group(2))
        frame_indices.update(range(start_frame, end_frame + 1))
    return sorted(frame_indices)


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


def compute_human_bbox_from_crop_bboxes(
    frame_idx: int,
    cam2_bboxes: dict[str, list[int]] | None,
    cam4_bboxes: dict[str, list[int]] | None,
    erp_h: int,
    erp_w: int,
    padding: int,
) -> tuple[int, int, int, int] | None:
    if cam2_bboxes is None and cam4_bboxes is None:
        return None

    bbox_items = []
    key = str(frame_idx)
    if cam2_bboxes is not None and key in cam2_bboxes:
        bbox_items.append(cam2_bboxes[key])
    if cam4_bboxes is not None and key in cam4_bboxes:
        bbox_items.append(cam4_bboxes[key])
    if not bbox_items:
        return None

    bboxes = np.asarray(bbox_items, dtype=np.float64).reshape(-1, 4)
    x1 = max(0, int(np.floor(bboxes[:, 0].min())) - padding)
    y1 = max(0, int(np.floor(bboxes[:, 1].min())) - padding)
    x2 = min(erp_w, int(np.ceil(bboxes[:, 2].max())) + padding)
    y2 = min(erp_h, int(np.ceil(bboxes[:, 3].max())) + padding)
    if x2 <= x1 or y2 <= y1:
        return None
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
    directions: str,
    erp_w: int,
    erp_h: int,
    cam2_bboxes: dict[str, list[int]] | None,
    cam4_bboxes: dict[str, list[int]] | None,
    bbox_padding: int,
):
    img_cam2 = None
    img_cam4 = None
    if save_vis:
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
        erp_h, erp_w = img_cam2.shape[:2]

    verts_unified_c2 = verts_c2 @ R_CAM2_NP.T
    verts_unified_c4 = verts_c4 @ R_CAM4_PROJ_NP.T

    human_bbox = compute_human_bbox_from_crop_bboxes(
        frame_idx=frame_idx,
        cam2_bboxes=cam2_bboxes,
        cam4_bboxes=cam4_bboxes,
        erp_h=erp_h,
        erp_w=erp_w,
        padding=bbox_padding,
    )

    pts_2d_cam2 = None
    pts_2d_cam4 = None
    need_projected_points = save_vis or human_bbox is None
    if need_projected_points:
        pts_2d_cam2 = project_verts_to_rotated_erp(verts_unified_c2, IDENTITY_NP, erp_h, erp_w)
        pts_2d_cam4 = project_verts_to_rotated_erp(verts_unified_c4, IDENTITY_NP, erp_h, erp_w)

    if human_bbox is None:
        human_bbox = compute_human_bbox([pts_2d_cam2, pts_2d_cam4], erp_h, erp_w)

    flow_c2c4 = None
    conf_c2c4 = None
    flow_c4c2 = None
    conf_c4c2 = None
    compute_c2_to_c4 = directions in ("both", "c2_to_c4")
    compute_c4_to_c2 = directions in ("both", "c4_to_c2")

    if compute_c2_to_c4:
        _, visible_mask_c2 = get_visible_vertices_erp(
            verts_world=verts_unified_c2,
            faces=faces,
            rotation_matrix=IDENTITY_NP,
            erp_h=erp_h,
            erp_w=erp_w,
            roi_bbox=human_bbox,
        )
        flow_c2c4, conf_c2c4 = compute_semantic_flow(
            verts_world_A=verts_unified_c4,
            verts_world_B=verts_unified_c2,
            faces=faces,
            erp_h=erp_h,
            erp_w=erp_w,
            rotation_matrix=IDENTITY_NP,
            roi_bbox=human_bbox,
            visible_mask_B=visible_mask_c2,
        )

    if compute_c4_to_c2:
        _, visible_mask_c4 = get_visible_vertices_erp(
            verts_world=verts_unified_c4,
            faces=faces,
            rotation_matrix=IDENTITY_NP,
            erp_h=erp_h,
            erp_w=erp_w,
            roi_bbox=human_bbox,
        )
        flow_c4c2, conf_c4c2 = compute_semantic_flow(
            verts_world_A=verts_unified_c2,
            verts_world_B=verts_unified_c4,
            faces=faces,
            erp_h=erp_h,
            erp_w=erp_w,
            rotation_matrix=IDENTITY_NP,
            roi_bbox=human_bbox,
            visible_mask_B=visible_mask_c4,
        )

    frame_output_dir = output_dir / f"frame_{frame_idx:05d}"
    frame_output_dir.mkdir(parents=True, exist_ok=True)
    if compute_c2_to_c4:
        np.save(frame_output_dir / "semantic_flow_c2_to_c4.npy", flow_c2c4)
        np.save(frame_output_dir / "semantic_flow_confidence_c2_to_c4.npy", conf_c2c4)
    if compute_c4_to_c2:
        np.save(frame_output_dir / "semantic_flow_c4_to_c2.npy", flow_c4c2)
        np.save(frame_output_dir / "semantic_flow_confidence_c4_to_c2.npy", conf_c4c2)
    np.save(frame_output_dir / "human_bbox.npy", np.asarray(human_bbox, dtype=np.int32))

    if save_vis:
        mesh_cam2 = visualize_mesh_on_erp(img_cam2, pts_2d_cam2)
        mesh_cam4 = visualize_mesh_on_erp(img_cam4, pts_2d_cam4)
        cv2.imwrite(str(frame_output_dir / "mesh_on_erp_cam2.jpg"), mesh_cam2, [cv2.IMWRITE_JPEG_QUALITY, 95])
        cv2.imwrite(str(frame_output_dir / "mesh_on_erp_cam4.jpg"), mesh_cam4, [cv2.IMWRITE_JPEG_QUALITY, 95])
        if compute_c2_to_c4:
            flow_color_c2c4, flow_arrows_c2c4 = visualize_flow(flow_c2c4, conf_c2c4)
            cv2.imwrite(str(frame_output_dir / "flow_color_c2_to_c4.png"), flow_color_c2c4)
            cv2.imwrite(str(frame_output_dir / "flow_arrows_c2_to_c4.png"), flow_arrows_c2c4)
        if compute_c4_to_c2:
            flow_color_c4c2, flow_arrows_c4c2 = visualize_flow(flow_c4c2, conf_c4c2)
            cv2.imwrite(str(frame_output_dir / "flow_color_c4_to_c2.png"), flow_color_c4c2)
            cv2.imwrite(str(frame_output_dir / "flow_arrows_c4_to_c2.png"), flow_arrows_c4c2)

    return {
        "frame_idx": frame_idx,
        "valid": True,
        "reason": "ok",
        "human_bbox": list(human_bbox),
        "flow_pixels_c2_to_c4": int((conf_c2c4 > 0).sum()) if conf_c2c4 is not None else None,
        "flow_pixels_c4_to_c2": int((conf_c4c2 > 0).sum()) if conf_c4c2 is not None else None,
    }


def main() -> None:
    args = parse_args()

    fitting_dir = Path(args.fitting_dir)
    image_dir = Path(args.image_dir)
    output_dir = Path(args.output_dir)
    invalid_frames = load_invalid_frames(Path(args.invalid_json) if args.invalid_json else None)
    output_dir.mkdir(parents=True, exist_ok=True)
    smplx_model = get_smplx_batch(gender="neutral")
    faces = smplx_model.faces.numpy() if hasattr(smplx_model.faces, "numpy") else np.asarray(smplx_model.faces)
    faces = np.asarray(faces, dtype=int)
    segment_index = build_segment_index(fitting_dir)
    frame_indices = sorted(segment_index.keys())
    if args.start_frame is not None:
        frame_indices = [i for i in frame_indices if i >= args.start_frame]
    if args.end_frame is not None:
        frame_indices = [i for i in frame_indices if i <= args.end_frame]

    cam2_bboxes = None
    cam4_bboxes = None
    if args.bbox_dir is not None:
        bbox_dir = Path(args.bbox_dir)
        cam2_bbox_path = bbox_dir / "cam2_bboxes.json"
        cam4_bbox_path = bbox_dir / "cam4_bboxes.json"
        if cam2_bbox_path.exists():
            cam2_bboxes = load_bbox_dict(cam2_bbox_path)
        else:
            print(f"Warning: cam2 bbox json not found: {cam2_bbox_path}")
        if cam4_bbox_path.exists():
            cam4_bboxes = load_bbox_dict(cam4_bbox_path)
        else:
            print(f"Warning: cam4 bbox json not found: {cam4_bbox_path}")

    meta = []
    _segment_cache: dict[Path, np.ndarray] = {}
    for frame_idx in frame_indices:
        if frame_idx in invalid_frames:
            meta.append({"frame_idx": frame_idx, "valid": False, "reason": "invalid_detection"})
            print(f"[frame {frame_idx:05d}] invalid_detection, skipping")
            continue

        segment_info = segment_index.get(frame_idx)
        if segment_info is None:
            result = {"frame_idx": frame_idx, "valid": False, "reason": "no_segment"}
            meta.append(result)
            print(f"[frame {frame_idx:05d}] no_segment")
            continue
        pkl_path_c2, pkl_path_c4, segment_start = segment_info

        try:
            if pkl_path_c2 not in _segment_cache:
                _segment_cache[pkl_path_c2] = load_all_verts(pkl_path_c2, smplx_model)
            if pkl_path_c4 not in _segment_cache:
                _segment_cache[pkl_path_c4] = load_all_verts(pkl_path_c4, smplx_model)
            frame_offset = frame_idx - segment_start
            verts_all_c2 = _segment_cache[pkl_path_c2]
            verts_all_c4 = _segment_cache[pkl_path_c4]
            if frame_offset < 0 or frame_offset >= len(verts_all_c2):
                raise IndexError(f"Frame {frame_idx} out of range for segment {pkl_path_c2.name}")
            if frame_offset < 0 or frame_offset >= len(verts_all_c4):
                raise IndexError(f"Frame {frame_idx} out of range for segment {pkl_path_c4.name}")
            verts_c2 = verts_all_c2[frame_offset]
            verts_c4 = verts_all_c4[frame_offset]
            result = process_frame(
                frame_idx=frame_idx,
                verts_c2=verts_c2,
                verts_c4=verts_c4,
                faces=faces,
                image_dir=image_dir,
                output_dir=output_dir,
                save_vis=args.save_vis,
                directions=args.directions,
                erp_w=args.image_width,
                erp_h=args.image_height,
                cam2_bboxes=cam2_bboxes,
                cam4_bboxes=cam4_bboxes,
                bbox_padding=args.bbox_padding,
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
  --end_frame 518 `
  --invalid_json data_process\Output\20260311_riding\vis\dual_erp\invalid.json `
  --save_vis
'''

r'''
python src\script\test_function\debug031_batch_semantic_flow.py `
  --fitting_dir data_process\Output\20260320_riding_dataset1\fitting\dual_cam `
  --image_dir data_process\Output\20260320_riding_dataset1\image\equirect_rotate `
  --output_dir data_process\Output\20260320_riding_dataset1\semantic_flow `
  --invalid_json data_process\Output\20260320_riding_dataset1\vis\dual_erp_no_sil\invalid.json `
  --save_vis `
  --start_frame 890 `
  --end_frame 890
'''

# 这块要改, 要不然要一直改padding参数 现在先改为500, 以后可能直接用前面得到的bbox结果, 这样就不需要建立human_bbox了
