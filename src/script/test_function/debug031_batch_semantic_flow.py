from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import os
import config_script
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')

from lib_pano.pipeline.bbox_io import load_bbox_dict
from lib_pano.pipeline.paths import ensure_single_fitting_path
from lib_pano.human3d.smplx_model_io import load_smplx_faces, load_smplx_model
from lib_pano.human3d.smplx_sequence import build_segment_index, get_frame_vertices
from lib_pano.flow.semantic_flow import (
    compute_flow_bbox,
    compute_flow_bbox_from_crop_bboxes,
    compute_semantic_flow_bidirectional,
    project_vertices_to_erp,
)
from lib_pano.flow.semantic_flow_io import save_flow_meta_json, save_semantic_flow_frame_npy
from lib_pano.flow.semantic_flow_vis import save_flow_debug_images

def _ensure_paths() -> Path:
    return ensure_single_fitting_path()


REPO_ROOT = _ensure_paths()

R_CAM2 = np.asarray([1, 0, 0, 0, 0, 1, 0, -1, 0], dtype=np.float32).reshape(3, 3)
R_CAM4 = np.asarray([-1, 0, 0, 0, 1, 0, 0, 0, -1], dtype=np.float32).reshape(3, 3)
R_CAM4_PROJ = R_CAM2   # 用于 cam4 投影

R_CAM2_NP = R_CAM2
R_CAM4_PROJ_NP = R_CAM4_PROJ
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


def load_invalid_frames(invalid_json_path: Path | None) -> set[int]:
    if invalid_json_path is None or not invalid_json_path.exists():
        return set()

    with invalid_json_path.open("r", encoding="utf-8") as file:
        invalid_json = json.load(file)

    cam2_frames = invalid_json.get("cam2", {}).get("equirect", [])
    cam4_frames = invalid_json.get("cam4", {}).get("equirect", [])
    return set(cam2_frames) | set(cam4_frames)


def main() -> None:
    args = parse_args()

    fitting_dir = Path(args.fitting_dir)
    image_dir = Path(args.image_dir)
    output_dir = Path(args.output_dir)
    invalid_frames = load_invalid_frames(Path(args.invalid_json) if args.invalid_json else None)
    output_dir.mkdir(parents=True, exist_ok=True)
    smplx_model = load_smplx_model(gender="neutral")
    faces = load_smplx_faces(smplx_model)
    segment_index = build_segment_index(fitting_dir, warn_missing_pair=True)
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
            verts_c2 = get_frame_vertices(pkl_path_c2, frame_idx, segment_start, smplx_model, _segment_cache)
            verts_c4 = get_frame_vertices(pkl_path_c4, frame_idx, segment_start, smplx_model, _segment_cache)

            img_cam2 = None
            img_cam4 = None
            if args.save_vis:
                image_path_c2 = image_dir / "cam2" / f"{frame_idx:05d}.jpg"
                image_path_c4 = image_dir / "cam4" / f"{frame_idx:05d}.jpg"
                img_cam2 = cv2.imread(str(image_path_c2), cv2.IMREAD_COLOR) if image_path_c2.exists() else None
                img_cam4 = cv2.imread(str(image_path_c4), cv2.IMREAD_COLOR) if image_path_c4.exists() else None

                if img_cam2 is None and img_cam4 is None:
                    result = {"frame_idx": frame_idx, "valid": False, "reason": "missing_both"}
                    meta.append(result)
                    print(f"[frame {frame_idx:05d}] {result['reason']}")
                    continue
                if img_cam2 is None:
                    result = {"frame_idx": frame_idx, "valid": False, "reason": "missing_cam2"}
                    meta.append(result)
                    print(f"[frame {frame_idx:05d}] {result['reason']}")
                    continue
                if img_cam4 is None:
                    result = {"frame_idx": frame_idx, "valid": False, "reason": "missing_cam4"}
                    meta.append(result)
                    print(f"[frame {frame_idx:05d}] {result['reason']}")
                    continue

                erp_h, erp_w = img_cam2.shape[:2]
            else:
                erp_w = args.image_width
                erp_h = args.image_height

            verts_unified_c2 = verts_c2 @ R_CAM2_NP.T
            verts_unified_c4 = verts_c4 @ R_CAM4_PROJ_NP.T
            human_bbox = compute_flow_bbox_from_crop_bboxes(
                frame_idx=frame_idx,
                cam2_bboxes=cam2_bboxes,
                cam4_bboxes=cam4_bboxes,
                erp_h=erp_h,
                erp_w=erp_w,
                padding=args.bbox_padding,
            )

            pts_2d_cam2 = None
            pts_2d_cam4 = None
            if args.save_vis or human_bbox is None:
                pts_2d_cam2 = project_vertices_to_erp(verts_unified_c2, IDENTITY_NP, erp_h, erp_w)
                pts_2d_cam4 = project_vertices_to_erp(verts_unified_c4, IDENTITY_NP, erp_h, erp_w)
            if human_bbox is None:
                human_bbox = compute_flow_bbox([pts_2d_cam2, pts_2d_cam4], erp_h, erp_w)

            flow_results = compute_semantic_flow_bidirectional(
                verts_unified_c2=verts_unified_c2,
                verts_unified_c4=verts_unified_c4,
                faces=faces,
                erp_h=erp_h,
                erp_w=erp_w,
                rotation_matrix=IDENTITY_NP,
                roi_bbox=human_bbox,
                directions=args.directions,
            )
            frame_output_dir = output_dir / f"frame_{frame_idx:05d}"
            save_semantic_flow_frame_npy(frame_output_dir, human_bbox, flow_results)

            if args.save_vis:
                save_flow_debug_images(
                    frame_output_dir=frame_output_dir,
                    img_cam2=img_cam2,
                    img_cam4=img_cam4,
                    pts_2d_cam2=pts_2d_cam2,
                    pts_2d_cam4=pts_2d_cam4,
                    flow_results=flow_results,
                )

            c2_to_c4 = flow_results.get("c2_to_c4")
            c4_to_c2 = flow_results.get("c4_to_c2")
            result = {
                "frame_idx": frame_idx,
                "valid": True,
                "reason": "ok",
                "human_bbox": list(human_bbox),
                "flow_pixels_c2_to_c4": int((c2_to_c4[1] > 0).sum()) if c2_to_c4 is not None else None,
                "flow_pixels_c4_to_c2": int((c4_to_c2[1] > 0).sum()) if c4_to_c2 is not None else None,
            }
        except Exception as exc:
            result = {"frame_idx": frame_idx, "valid": False, "reason": f"error: {exc}"}

        meta.append(result)
        print(f"[frame {frame_idx:05d}] {result['reason']}")

    meta_path = output_dir / "flow_meta.json"
    save_flow_meta_json(meta_path, meta)

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
python src/script/test_function/debug031_batch_semantic_flow.py \
  --fitting_dir data_process/Output/20260311_riding/fitting/dual_cam_dense \
  --image_dir data_process/Output/20260311_riding/image/equirect_rotate \
  --output_dir data_process/Output/20260311_riding/semantic_flow/dual_cam_dense 
'''

r'''
python src/script/test_function/debug031_batch_semantic_flow.py \
  --fitting_dir data_process/Output/20260320_riding_dataset1/fitting/dual_cam \
  --image_dir data_process/Output/20260320_riding_dataset1/image/equirect_rotate \
  --output_dir data_process/Output/20260320_riding_dataset1/semantic_flow \
  --invalid_json data_process/Output/20260320_riding_dataset1/vis/dual_erp_no_sil/invalid.json \
  --save_vis \
  --start_frame 890 \
  --end_frame 890
'''

