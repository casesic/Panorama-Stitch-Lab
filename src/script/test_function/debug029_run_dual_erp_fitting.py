from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm

import config_script
from batch_utils import load_reviewed_frame_indices, split_frame_segments
from lib_pano.pipeline.bbox_io import load_bbox_dict as load_bbox_json
from lib_pano.pipeline.invalid_io import load_invalid_frames, load_problem_frames
from lib_pano.pipeline.paths import ensure_single_fitting_path

def _ensure_paths() -> Path:
    repo_root = ensure_single_fitting_path()
    return repo_root


REPO_ROOT = _ensure_paths()

from lib_pano.fitting_io.fitting_result_io import save_segment_result
from lib_pano.fitting_io.hmr_init import (
    build_hmr_init_for_frame,
    build_segment_smplx_init,
    try_load_smplx_init,
)
from lib_pano.fitting_io.densekp_io import load_densekp_result
from lib_pano.fitting_io.sam3d_dense_io import load_sam3d_dense_result
from lib_pano.fitting_io.mask_io import get_mask_paths, is_seg_valid
from lib_pano.fitting_io.rtm_io import (
    infer_keypoints_format_and_dim,
    is_rtm_det_valid,
    load_det_result,
)
from lib_pano.fitting_io.segments import split_long_segments, split_reconstructable_subsegments
from lib_pano.human_fitting import DualCamERPFitter
from lib_pano.human_fitting.constraints import prepare_silhouette_data


R_CAM4 = torch.tensor([-1, 0, 0, 0, 1, 0, 0, 0, -1], dtype=torch.float32).reshape(3, 3)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch-run DualCamERPFitter on reviewed equirect frame segments.")
    parser.add_argument("--reviewed", type=str, required=True, help="Path to reviewed.json.")
    parser.add_argument(
        "--rtm_dir",
        type=str,
        required=True,
        help="frame_cache root directory containing cam2/ and cam4/ subdirectories",
    )
    parser.add_argument(
        "--hmr_dir",
        type=str,
        required=True,
        help="HMR pkl root directory containing cam2/ and cam4/ subdirectories",
    )
    parser.add_argument("--output_dir", type=str, required=True, help="Output root directory.")
    parser.add_argument(
        "--rtm_invalid_json",
        type=str,
        default=None,
        help="Optional path to RTM invalid.json from debug026/review.",
    )
    parser.add_argument(
        "--seg_invalid_json",
        type=str,
        default=None,
        help="Optional path to segmentation invalid.json from debug038/review.",
    )
    parser.add_argument(
        "--hmr_invalid_json",
        type=str,
        default=None,
        help="Optional path to HMR init invalid.json from debug024/review.",
    )
    parser.add_argument(
        "--seg_dir",
        type=str,
        required=True,
        help="Segmentation mask root directory containing cam2/ and cam4/ subdirectories.",
    )
    parser.add_argument(
        "--bbox_dir",
        type=str,
        required=True,
        help="BBox json directory containing cam2_bboxes.json and cam4_bboxes.json.",
    )
    parser.add_argument(
        "--use_silhouette_loss",
        action="store_true",
        default=True,
        help="启用轮廓损失",
    )
    parser.add_argument(
        "--no_silhouette_loss",
        dest="use_silhouette_loss",
        action="store_false",
        help="禁用轮廓损失",
    )
    parser.add_argument(
        "--image_width",
        type=int,
        default=5760,
        help="Equirect image width.",
    )
    parser.add_argument(
        "--image_height",
        type=int,
        default=2880,
        help="Equirect image height.",
    )
    parser.add_argument(
        "--start_frame",
        type=int,
        default=None,
        help="[legacy] Only process segments whose end frame is >= this frame index.",
    )
    parser.add_argument(
        "--end_frame",
        type=int,
        default=None,
        help="[legacy] Only process segments whose start frame is <= this frame index.",
    )
    parser.add_argument(
        "--segments",
        type=str,
        default=None,
        help='Explicit segments, e.g. "513_518" or "513_518,704_711". Overrides --start_frame/--end_frame.',
    )
    parser.add_argument(
        "--max_segment_len",
        type=int,
        default=0,
        help="将超过此长度的段切成子段，0表示不限制（默认0）。",
    )
    parser.add_argument(
        "--densekp_dir",
        type=str,
        default=None,
        help="Optional DenseKP npz root directory containing cam2/ and cam4/ subdirectories.",
    )
    parser.add_argument(
        "--sam3d_dense_dir",
        type=str,
        default=None,
        help=(
            "Optional SAM 3D Body dense SMPL-X observation root "
            "containing cam2/ and cam4/ subdirectories."
        ),
    )
    parser.add_argument(
        "--densekp_filter",
        type=str,
        default=None,
        help="Optional JSON path for DenseKP ID filter (keep_ids / drop_ids).",
    )
    parser.add_argument(
        "--mask_fill_stage3",
        action="store_true",
        help="Enable Stage 3 mask fill refine (requires --image_root).",
    )
    parser.add_argument(
        "--image_root",
        type=str,
        default=None,
        help="Equirect image root (required when --mask_fill_stage3 is set).",
    )
    return parser.parse_args()


def parse_segments_arg(segments_str: str) -> list[list[int]]:
    """Parse --segments string like "513_518" or "513_518,704_711".

    Returns a list of frame-index lists, one per segment.
    """
    result: list[list[int]] = []
    for item in segments_str.split(","):
        item = item.strip()
        if not item:
            continue
        parts = item.split("_")
        if len(parts) != 2:
            raise ValueError(f"Invalid segment format: {item!r}. Expected start_end, e.g. 513_518.")
        start = int(parts[0])
        end = int(parts[1])
        if start > end:
            raise ValueError(f"Segment start must be <= end, got: {item!r}.")
        result.append(list(range(start, end + 1)))
    if not result:
        raise ValueError(f"No valid segment entries found in: {segments_str!r}")
    return result


def main() -> None:
    args = parse_args()

    reviewed_path = Path(args.reviewed)
    rtm_dir = Path(args.rtm_dir)
    hmr_dir = Path(args.hmr_dir)
    output_dir = Path(args.output_dir)
    seg_dir = Path(args.seg_dir)
    bbox_dir = Path(args.bbox_dir)
    rtm_invalid_path = Path(args.rtm_invalid_json) if args.rtm_invalid_json is not None else None
    seg_invalid_path = Path(args.seg_invalid_json) if args.seg_invalid_json is not None else None
    hmr_invalid_path = Path(args.hmr_invalid_json) if args.hmr_invalid_json is not None else None
    densekp_dir = Path(args.densekp_dir) if args.densekp_dir is not None else None
    sam3d_dense_dir = (
        Path(args.sam3d_dense_dir)
        if args.sam3d_dense_dir is not None
        else None
    )

    frame_indices = load_reviewed_frame_indices(reviewed_path)
    reviewed_frame_set = set(frame_indices)

    if args.segments is not None and (args.start_frame is not None or args.end_frame is not None):
        print("[ERROR] --segments cannot be used together with --start_frame/--end_frame.")
        return

    if args.segments is not None:
        specified = parse_segments_arg(args.segments)
        print(f"[INFO] using explicit segments: {args.segments}")

        segments = []
        for seg in specified:
            seg_key = f"{seg[0]}_{seg[-1]}"
            missing = [idx for idx in seg if idx not in reviewed_frame_set]
            if missing:
                print(f"[WARN] segment {seg_key} has frames not in reviewed.json: {missing}")
            filtered = [idx for idx in seg if idx in reviewed_frame_set]
            if not filtered:
                print(f"[WARN] segment {seg_key} has no reviewed frames, skipping")
                continue
            if filtered != seg:
                filtered_key = f"{filtered[0]}_{filtered[-1]}"
                print(f"[INFO] actual segment after reviewed filter: {seg_key} -> {filtered_key}")
            segments.append(filtered)

        if not segments:
            print("[ERROR] no reviewed frames in any specified segment, exiting.")
            return
    else:
        segments = split_frame_segments(frame_indices)
        if args.max_segment_len > 0:
            segments = split_long_segments(segments, args.max_segment_len)
            print(f"切段后共 {len(segments)} 段，最大段长 {args.max_segment_len}")

        if args.start_frame is not None:
            segments = [frame_idxs for frame_idxs in segments if frame_idxs[-1] >= args.start_frame]
        if args.end_frame is not None:
            segments = [frame_idxs for frame_idxs in segments if frame_idxs[0] <= args.end_frame]

    rtm_invalid_frames = load_invalid_frames(rtm_invalid_path, label="rtm_invalid_json", recurse_all=False)
    seg_invalid_frames = load_invalid_frames(seg_invalid_path, label="seg_invalid_json", recurse_all=False)
    hmr_invalid_frames = load_invalid_frames(hmr_invalid_path, label="hmr_invalid_json", recurse_all=False)
    problem_frames = {
        "rtm": sorted(load_problem_frames(rtm_invalid_path, label="rtm_invalid_json", recurse_all=False)),
        "seg": sorted(load_problem_frames(seg_invalid_path, label="seg_invalid_json", recurse_all=False)),
        "hmr": sorted(load_problem_frames(hmr_invalid_path, label="hmr_invalid_json", recurse_all=False)),
    }
    cam2_bboxes = load_bbox_json(bbox_dir / "cam2_bboxes.json")
    cam4_bboxes = load_bbox_json(bbox_dir / "cam4_bboxes.json")
    default_keypoints_format_c2, num_keypoints_c2 = infer_keypoints_format_and_dim(frame_indices, rtm_dir, "cam2")
    default_keypoints_format_c4, num_keypoints_c4 = infer_keypoints_format_and_dim(frame_indices, rtm_dir, "cam4")

    fitter = DualCamERPFitter(
        gender="neutral",
        R_cam4=R_CAM4,
        use_silhouette_loss=args.use_silhouette_loss,
        densekp_filter_path=args.densekp_filter,
        enable_mask_fill_stage3=args.mask_fill_stage3,
        image_root=args.image_root,
        seg_dir=args.seg_dir,
        bbox_dir=args.bbox_dir,
    )
    sil_device = "cuda" if torch.cuda.is_available() else "cpu"

    frame_status: dict[str, dict[str, Any]] = {}
    segment_summary: list[dict[str, Any]] = []
    init_counter: Counter[str] = Counter()
    fitting_subsegment_count = 0

    for frame_idxs in tqdm(segments, desc="DualCamERPFitter", unit="segment"):
        segment_key = f"{frame_idxs[0]}_{frame_idxs[-1]}"
        direct_hmr_c2 = {
            frame_idx: try_load_smplx_init(frame_idx, hmr_dir, hmr_invalid_frames, "cam2")
            for frame_idx in frame_idxs
        }
        direct_hmr_c4 = {
            frame_idx: try_load_smplx_init(frame_idx, hmr_dir, hmr_invalid_frames, "cam4")
            for frame_idx in frame_idxs
        }

        reconstructable_by_frame: dict[int, bool] = {}
        for frame_idx in frame_idxs:
            rtm_valid_c2 = is_rtm_det_valid(frame_idx, rtm_dir, rtm_invalid_frames, "cam2")
            rtm_valid_c4 = is_rtm_det_valid(frame_idx, rtm_dir, rtm_invalid_frames, "cam4")
            rtm_valid = rtm_valid_c2 or rtm_valid_c4
            seg_valid = is_seg_valid(frame_idx, seg_dir, seg_invalid_frames)
            hmr_valid = direct_hmr_c2.get(frame_idx) is not None or direct_hmr_c4.get(frame_idx) is not None
            densekp_c2 = load_densekp_result(frame_idx, densekp_dir, "cam2", args.image_width, args.image_height, cam2_bboxes.get(str(frame_idx))) if densekp_dir is not None else None
            densekp_c4 = load_densekp_result(frame_idx, densekp_dir, "cam4", args.image_width, args.image_height, cam4_bboxes.get(str(frame_idx))) if densekp_dir is not None else None
            densekp_valid = bool(
                (densekp_c2 is not None and densekp_c2.valid.any())
                or (densekp_c4 is not None and densekp_c4.valid.any())
            )
            reconstructable = rtm_valid or seg_valid
            reconstructable_by_frame[frame_idx] = reconstructable
            frame_status[str(frame_idx)] = {
                "rtm_valid": rtm_valid,
                "seg_valid": seg_valid,
                "hmr_valid": hmr_valid,
                "densekp_valid": densekp_valid,
                "reconstructable": reconstructable,
                "status": "rebuilt" if reconstructable else "skipped_both_invalid",
                "init": "none",
                "segment": segment_key,
                "subsegment": None,
            }

        subsegments = split_reconstructable_subsegments(frame_idxs, reconstructable_by_frame)
        fitting_subsegment_count += len(subsegments)
        segment_summary.append(
            {
                "segment": segment_key,
                "frame_count": len(frame_idxs),
                "subsegments": [f"{subseg[0]}_{subseg[-1]}" for subseg in subsegments],
                "rebuilt": sum(1 for idx in frame_idxs if reconstructable_by_frame[idx]),
                "skipped_both_invalid": sum(1 for idx in frame_idxs if not reconstructable_by_frame[idx]),
                "rtm_valid": sum(1 for idx in frame_idxs if frame_status[str(idx)]["rtm_valid"]),
                "seg_valid": sum(1 for idx in frame_idxs if frame_status[str(idx)]["seg_valid"]),
                "hmr_valid": sum(1 for idx in frame_idxs if frame_status[str(idx)]["hmr_valid"]),
                "densekp_valid": sum(1 for idx in frame_idxs if frame_status[str(idx)]["densekp_valid"]),
            }
        )

        for sub_frame_idxs in subsegments:
            subsegment_key = f"{sub_frame_idxs[0]}_{sub_frame_idxs[-1]}"
            det_list_c2 = [
                load_det_result(
                    frame_idx,
                    rtm_dir,
                    rtm_invalid_frames,
                    default_keypoints_format=default_keypoints_format_c2,
                    num_keypoints=num_keypoints_c2,
                    cam_id="cam2",
                )
                for frame_idx in sub_frame_idxs
            ]
            det_list_c4 = [
                load_det_result(
                    frame_idx,
                    rtm_dir,
                    rtm_invalid_frames,
                    default_keypoints_format=default_keypoints_format_c4,
                    num_keypoints=num_keypoints_c4,
                    cam_id="cam4",
                )
                for frame_idx in sub_frame_idxs
            ]

            smplx_list_c2, smplx_list_c4 = [], []
            for frame_idx in sub_frame_idxs:
                smplx_c2, init_c2 = build_hmr_init_for_frame(frame_idx, frame_idxs, direct_hmr_c2)
                smplx_c4, init_c4 = build_hmr_init_for_frame(frame_idx, frame_idxs, direct_hmr_c4)
                smplx_list_c2.append(smplx_c2)
                smplx_list_c4.append(smplx_c4)
                init_sources = {init_c2, init_c4}
                if "hmr" in init_sources:
                    init_source = "hmr"
                elif "interpolated" in init_sources:
                    init_source = "interpolated"
                elif "neighbor" in init_sources:
                    init_source = "neighbor"
                else:
                    init_source = "tpose"
                frame_status[str(frame_idx)]["init"] = init_source
                frame_status[str(frame_idx)]["subsegment"] = subsegment_key
                init_counter[init_source] += 1

            smplx_init_c2, smplx_init_c4 = build_segment_smplx_init(smplx_list_c2, smplx_list_c4)

            if densekp_dir is not None:
                densekp_list_c2 = [
                    load_densekp_result(frame_idx, densekp_dir, "cam2", args.image_width, args.image_height, crop_bbox=cam2_bboxes.get(str(frame_idx)),)
                    for frame_idx in sub_frame_idxs
                ]
                densekp_list_c4 = [
                    load_densekp_result(frame_idx, densekp_dir, "cam4", args.image_width, args.image_height, crop_bbox=cam4_bboxes.get(str(frame_idx)),)
                    for frame_idx in sub_frame_idxs
                ]
            else:
                densekp_list_c2 = None
                densekp_list_c4 = None

            if sam3d_dense_dir is not None:
                sam3d_dense_list_c2 = [
                    load_sam3d_dense_result(frame_idx, sam3d_dense_dir, "cam2", args.image_width, args.image_height, crop_bbox=cam2_bboxes.get(str(frame_idx)),)
                    for frame_idx in sub_frame_idxs
                ]
                sam3d_dense_list_c4 = [
                    load_sam3d_dense_result(frame_idx, sam3d_dense_dir, "cam4", args.image_width, args.image_height, crop_bbox=cam4_bboxes.get(str(frame_idx)),)
                    for frame_idx in sub_frame_idxs
                ]
            else:
                sam3d_dense_list_c2 = None
                sam3d_dense_list_c4 = None

            boundary_pts_list_c2, boundary_pts_list_c4 = [], []
            bbox_list_c2, bbox_list_c4 = [], []
            for frame_idx in sub_frame_idxs:
                mask_c2, mask_c4 = get_mask_paths(frame_idx, seg_dir)
                use_seg_frame = frame_status[str(frame_idx)]["seg_valid"] and args.use_silhouette_loss
                boundary_pts_list_c2.append(
                    prepare_silhouette_data(mask_c2, device=sil_device)
                    if use_seg_frame and mask_c2.exists()
                    else None
                )
                boundary_pts_list_c4.append(
                    prepare_silhouette_data(mask_c4, device=sil_device)
                    if use_seg_frame and mask_c4.exists()
                    else None
                )
                bbox_list_c2.append(cam2_bboxes.get(str(frame_idx)))
                bbox_list_c4.append(cam4_bboxes.get(str(frame_idx)))

            valid_frame_count_c2 = sum(1 for det in det_list_c2 if bool(det.flag_valid))
            valid_frame_count_c4 = sum(1 for det in det_list_c4 if bool(det.flag_valid))

            result_c2, result_c4, shared_beta = fitter.fit(
                cam2_dets=det_list_c2,
                cam4_dets=det_list_c4,
                image_size=(args.image_width, args.image_height),
                smplx_init_c2=smplx_init_c2,
                smplx_init_c4=smplx_init_c4,
                boundary_pts_list_c2=boundary_pts_list_c2,
                boundary_pts_list_c4=boundary_pts_list_c4,
                bbox_list_c2=bbox_list_c2,
                bbox_list_c4=bbox_list_c4,
                densekp_list_c2=densekp_list_c2,
                densekp_list_c4=densekp_list_c4,
                sam3d_dense_list_c2=sam3d_dense_list_c2,
                sam3d_dense_list_c4=sam3d_dense_list_c4,
                output_dir=output_dir / subsegment_key if args.mask_fill_stage3 else None,
                frame_indices=sub_frame_idxs,
            )
            save_segment_result(
                result_c2=result_c2,
                result_c4=result_c4,
                shared_beta=shared_beta,
                output_dir=output_dir,
                frame_idxs=sub_frame_idxs,
                valid_frame_count_c2=valid_frame_count_c2,
                valid_frame_count_c4=valid_frame_count_c4,
            )

    status_values = list(frame_status.values())
    global_summary = {
        "original_segments": len(segments),
        "fitting_subsegments": fitting_subsegment_count,
        "skipped_both_invalid_frames": sum(1 for item in status_values if item["status"] == "skipped_both_invalid"),
        "rebuilt_frames": sum(1 for item in status_values if item["status"] == "rebuilt"),
        "rtm_valid": sum(1 for item in status_values if item["rtm_valid"]),
        "rtm_invalid": sum(1 for item in status_values if not item["rtm_valid"]),
        "seg_valid": sum(1 for item in status_values if item["seg_valid"]),
        "seg_invalid": sum(1 for item in status_values if not item["seg_valid"]),
        "hmr_valid": sum(1 for item in status_values if item["hmr_valid"]),
        "hmr_invalid": sum(1 for item in status_values if not item["hmr_valid"]),
        "densekp_valid": sum(1 for item in status_values if item["densekp_valid"]),
        "densekp_invalid": sum(1 for item in status_values if not item["densekp_valid"]),
        "init_sources": {
            "hmr": init_counter.get("hmr", 0),
            "interpolated": init_counter.get("interpolated", 0),
            "neighbor": init_counter.get("neighbor", 0),
            "tpose": init_counter.get("tpose", 0),
        },
        "problem_frames": problem_frames,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "fitting_summary.json").open("w", encoding="utf-8") as file:
        json.dump(
            {
                "global_summary": global_summary,
                "segment_summary": segment_summary,
                "frame_status": frame_status,
            },
            file,
            ensure_ascii=False,
            indent=2,
        )

    print("Fitting summary:")
    print(json.dumps(global_summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()


r'''
python src/script/test_function/debug029_run_dual_erp_fitting.py \
  --reviewed data_process/Input/20260320_riding/dataset2/review.json \
  --rtm_dir data_process/Output/20260320_riding_dataset2/rtm_cache/equirect_rotate_rtmw \
  --hmr_dir data_process/Output/20260320_riding_dataset2/hmr_cache \
  --seg_dir data_process/Output/20260320_riding_dataset2/seg/equirect_crop_rotate_sam3 \
  --output_dir data_process/Output/20260320_riding_dataset2/fitting/dual_cam_dense_filter_all \
  --rtm_invalid_json data_process/Output/20260320_riding_dataset2/rtm_cache/equirect_rotate_rtmw/invalid.json \
  --seg_invalid_json data_process/Output/20260320_riding_dataset2/seg/invalid.json \
  --hmr_invalid_json data_process/Output/20260320_riding_dataset2/hmr_cache/invalid.json \
  --bbox_dir data_process/Output/20260320_riding_dataset2/image/equirect_crop_rotate \
  --image_root data_process/Output/20260320_riding_dataset2/image/equirect_rotate \
  --densekp_dir data_process/Output/20260320_riding_dataset2/camerahmr/densekp_npz \
  --max_segment_len 10 \
  --segments 216_226,519_531,695_704,1231_1240,1241_1250,1251_1263,1555_1561,1769_1771,2850_2853,3110_3118,3580_3592,3819_3831 \
  --no_silhouette_loss

# Multiple segments
python src/script/test_function/debug029_run_dual_erp_fitting.py \
  --reviewed data_process/Input/20260311_riding/review.json \
  --rtm_dir data_process/Output/20260311_riding/rtm_cache/equirect_rotate_rtmw \
  --hmr_dir data_process/Output/20260311_riding/hmr_cache \
  --seg_dir data_process/Output/20260311_riding/seg/equirect_crop_rotate_sam3 \
  --output_dir data_process/Output/20260311_riding/fitting/dual_cam_sam3d_dense \
  --rtm_invalid_json data_process/Output/20260311_riding/rtm_cache/equirect_rotate_rtmw/invalid.json \
  --seg_invalid_json data_process/Output/20260311_riding/seg/invalid.json \
  --hmr_invalid_json data_process/Output/20260311_riding/hmr_cache/invalid.json \
  --bbox_dir data_process/Output/20260311_riding/image/equirect_crop_rotate \
  --image_root data_process/Output/20260311_riding/image/equirect_rotate \
  --densekp_dir data_process/Output/20260311_riding/camerahmr/densekp_npz \
  --sam3d_dense_dir data_process/Output/20260311_riding/sam3dbody/smplx_dense \
  --max_segment_len 10 \
  --segments 513_518,704_711 \
  --no_silhouette_loss

# Standard fitting with silhouette loss, single segment
python src/script/test_function/debug029_run_dual_erp_fitting.py \
  --reviewed data_process/Input/20260320_riding/dataset1/review.json \
  --rtm_dir data_process/Output/20260320_riding_dataset1/rtm_cache/equirect_rotate \
  --hmr_dir data_process/Output/20260320_riding_dataset1/hmr_cache \
  --output_dir data_process/Output/20260320_riding_dataset1/fitting/dual_cam_sil_886_895 \
  --rtm_invalid_json data_process/Output/20260320_riding_dataset1/invalid_detections.json \
  --seg_dir data_process/Output/20260320_riding_dataset1/seg/equirect_crop_rotate_sam3 \
  --bbox_dir data_process/Output/20260320_riding_dataset1/image/equirect_crop_rotate \
  --segments 886_895 \
  --use_silhouette_loss
  
0320_riding_dataset1: segments 886_895,1143_1151,3741_3745,5103_5121,5352_5363,5501_5506
'''
