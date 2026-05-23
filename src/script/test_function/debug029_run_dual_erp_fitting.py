from __future__ import annotations

import argparse
from collections import Counter
import json
import pickle
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

from lib_dynamic_hvip.fitting.smplx_seq_forward_module import SMPLX_Seq_Forward_Module
from lib_pano.human3d.keypoints_detector import Result_Single_Human_Single_Frame
from single_fitting.fitting import DualCamERPFitter, init_smplx_from_smpl54_seq, slice_smplx_seq


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
        help="Only process segments whose end frame is >= this frame index.",
    )
    parser.add_argument(
        "--end_frame",
        type=int,
        default=None,
        help="Only process segments whose start frame is <= this frame index.",
    )
    parser.add_argument(
        "--max_segment_len",
        type=int,
        default=0,
        help="将超过此长度的段切成子段，0表示不限制（默认0）。",
    )
    return parser.parse_args()


def split_long_segments(
    segments: list[list[int]],
    max_len: int,
) -> list[list[int]]:
    """将超过 max_len 帧的段切成等长子段。"""
    if max_len <= 0:
        return segments
    result: list[list[int]] = []
    for seg in segments:
        for i in range(0, len(seg), max_len):
            result.append(seg[i:i + max_len])
    return result


def split_reconstructable_subsegments(frame_indices: list[int], reconstructable: dict[int, bool]) -> list[list[int]]:
    subsegments: list[list[int]] = []
    current: list[int] = []
    for frame_idx in frame_indices:
        if reconstructable.get(frame_idx, False):
            if current and frame_idx != current[-1] + 1:
                subsegments.append(current)
                current = []
            current.append(frame_idx)
        elif current:
            subsegments.append(current)
            current = []
    if current:
        subsegments.append(current)
    return subsegments


def infer_keypoints_format_and_dim(frame_indices: list[int], rtm_dir: Path, cam_id: str) -> tuple[dict[str, int], int]:
    cam_suffix = "c2" if cam_id == "cam2" else "c4"
    for frame_idx in frame_indices:
        pkl_path = rtm_dir / cam_id / f"rtm_s{frame_idx:05d}_{cam_suffix}.pkl"
        if not pkl_path.exists():
            continue
        with pkl_path.open("rb") as file:
            seq = pickle.load(file)
        keypoints_format = dict(seq.keypoints_format)
        num_keypoints = int(seq.array_keypoints.shape[1])
        return keypoints_format, num_keypoints
    raise RuntimeError(f"Failed to infer keypoints_format from RTM pkls for {cam_id} reviewed frames")


def _build_smpl_init_from_single_frame_pkl(pkl_path: Path):
    import importlib

    import torch

    smpl_mod = importlib.import_module("lib_dynamic_hvip.fitting.smpl_seq_forward_module")
    SMPLFitter = smpl_mod.SMPL_Seq_Forward_Module

    with pkl_path.open("rb") as file:
        src = pickle.load(file)

    if isinstance(src, SMPLFitter.SMPL_Param_Seq):
        return src
    if not isinstance(src, dict):
        raise TypeError(f"Unsupported HMR pkl object type: {type(src)}")

    num_batch = int(src["num_batch"])
    dim_shape = int(src["beta"].shape[-1])
    gender = str(src.get("gender", "neutral"))

    smpl_init = SMPLFitter.SMPL_Param_Seq(dim_shape=dim_shape, gender=gender, num_batch=num_batch)
    smpl_init.orient = src["orient"].clone()
    smpl_init.transl = src["transl"].clone()
    smpl_init.pose = src["pose"].clone()
    smpl_init.beta = src["beta"].clone().reshape(-1)
    smpl_init.scale = src["scale"].clone().reshape(-1)[:1]
    smpl_init.valid = src.get("valid", torch.ones(num_batch, dtype=torch.bool)).clone()
    return smpl_init


def load_det_result(
    frame_idx: int,
    rtm_dir: Path,
    invalid_frames: set[int],
    default_keypoints_format: dict[str, int],
    num_keypoints: int,
    cam_id: str,
) -> Result_Single_Human_Single_Frame:
    zero_keypoints = None
    if num_keypoints > 0:
        import numpy as np

        zero_keypoints = np.zeros((num_keypoints, 3), dtype=np.float32)

    if frame_idx in invalid_frames:
        return Result_Single_Human_Single_Frame(
            bbox=None,
            bbox_score=0.0,
            keypoints=zero_keypoints,
            keypoints_format=default_keypoints_format,
        )

    cam_suffix = "c2" if cam_id == "cam2" else "c4"
    pkl_path = rtm_dir / cam_id / f"rtm_s{frame_idx:05d}_{cam_suffix}.pkl"
    if not pkl_path.exists():
        return Result_Single_Human_Single_Frame(
            bbox=None,
            bbox_score=0.0,
            keypoints=zero_keypoints,
            keypoints_format=default_keypoints_format,
        )

    try:
        with pkl_path.open("rb") as file:
            seq = pickle.load(file)
    except Exception:
        return Result_Single_Human_Single_Frame(
            bbox=None,
            bbox_score=0.0,
            keypoints=zero_keypoints,
            keypoints_format=default_keypoints_format,
        )

    is_valid = bool(seq.array_valid[0])
    det = Result_Single_Human_Single_Frame(
        bbox=tuple(seq.array_bbox_h1h2w1w2[0].tolist()) if is_valid else None,
        bbox_score=float(seq.array_bbox_score[0]),
        keypoints=seq.array_keypoints[0].copy() if is_valid else zero_keypoints.copy(),
        keypoints_format=dict(seq.keypoints_format),
    )
    return det


def rtm_pkl_path(frame_idx: int, rtm_dir: Path, cam_id: str) -> Path:
    cam_suffix = "c2" if cam_id == "cam2" else "c4"
    return rtm_dir / cam_id / f"rtm_s{frame_idx:05d}_{cam_suffix}.pkl"


def is_rtm_det_valid(frame_idx: int, rtm_dir: Path, invalid_frames: set[int], cam_id: str) -> bool:
    if frame_idx in invalid_frames:
        return False
    pkl_path = rtm_pkl_path(frame_idx, rtm_dir, cam_id)
    if not pkl_path.exists():
        return False
    try:
        with pkl_path.open("rb") as file:
            seq = pickle.load(file)
        return bool(seq.array_valid[0])
    except Exception:
        return False


def get_mask_paths(frame_idx: int, seg_dir: Path) -> tuple[Path, Path]:
    return (
        seg_dir / "cam2" / "mask" / f"{frame_idx:05d}.png",
        seg_dir / "cam4" / "mask" / f"{frame_idx:05d}.png",
    )


def is_seg_valid(frame_idx: int, seg_dir: Path, invalid_frames: set[int]) -> bool:
    if frame_idx in invalid_frames:
        return False
    mask_c2, mask_c4 = get_mask_paths(frame_idx, seg_dir)
    return mask_c2.exists() or mask_c4.exists()


def load_smplx_init(frame_idx: int, hmr_dir: Path, cam_id: str):
    cam_suffix = "c2" if cam_id == "cam2" else "c4"
    pkl_path = hmr_dir / cam_id / f"hmr_s{frame_idx:05d}_{cam_suffix}.pkl"
    if not pkl_path.exists():
        return None
    
    smpl_seq = _build_smpl_init_from_single_frame_pkl(pkl_path)
    return init_smplx_from_smpl54_seq(smpl_seq, [0])


def try_load_smplx_init(frame_idx: int, hmr_dir: Path, hmr_invalid_frames: set[int], cam_id: str):
    if frame_idx in hmr_invalid_frames:
        return None
    try:
        return load_smplx_init(frame_idx, hmr_dir, cam_id)
    except Exception:
        return None


def blend_smplx_init(left, right, alpha: float):
    if left is None:
        return right
    if right is None:
        return left

    alpha = float(max(0.0, min(1.0, alpha)))
    out = slice_smplx_seq(left, [0])
    for attr in ("orient", "transl", "body_pose", "left_hand_pose", "right_hand_pose"):
        if hasattr(out, attr) and hasattr(left, attr) and hasattr(right, attr):
            getattr(out, attr)[0] = (1.0 - alpha) * getattr(left, attr)[0] + alpha * getattr(right, attr)[0]
    for attr in ("beta", "scale"):
        if hasattr(out, attr) and hasattr(left, attr) and hasattr(right, attr):
            setattr(out, attr, (1.0 - alpha) * getattr(left, attr).clone() + alpha * getattr(right, attr).clone())
    return out


def build_hmr_init_for_frame(frame_idx: int, original_segment: list[int], direct_by_frame: dict[int, Any]):
    direct = direct_by_frame.get(frame_idx)
    if direct is not None:
        return direct, "hmr"

    left_frames = [idx for idx in original_segment if idx < frame_idx and direct_by_frame.get(idx) is not None]
    right_frames = [idx for idx in original_segment if idx > frame_idx and direct_by_frame.get(idx) is not None]
    left_idx = left_frames[-1] if left_frames else None
    right_idx = right_frames[0] if right_frames else None

    if left_idx is not None and right_idx is not None:
        denom = max(1, right_idx - left_idx)
        alpha = (frame_idx - left_idx) / denom
        return blend_smplx_init(direct_by_frame[left_idx], direct_by_frame[right_idx], alpha), "interpolated"
    if left_idx is not None:
        return slice_smplx_seq(direct_by_frame[left_idx], [0]), "neighbor"
    if right_idx is not None:
        return slice_smplx_seq(direct_by_frame[right_idx], [0]), "neighbor"
    return None, "tpose"


def build_segment_smplx_init(smplx_list_c2, smplx_list_c4):
    def _build_single(smplx_list):
        fallback = next((item for item in smplx_list if item is not None), None)
        if fallback is None:
            return None

        if len(smplx_list) == 1 and smplx_list[0] is not None:
            return slice_smplx_seq(smplx_list[0], [0])

        n = len(smplx_list)
        out = SMPLX_Seq_Forward_Module.SMPLX_Param_Seq(num_batch=n, gender="neutral")
        for i, smplx_item in enumerate(smplx_list):
            src = smplx_item if smplx_item is not None else fallback
            out.orient[i] = src.orient[0]
            out.transl[i] = src.transl[0]
            out.body_pose[i] = src.body_pose[0]
            out.left_hand_pose[i] = src.left_hand_pose[0]
            out.right_hand_pose[i] = src.right_hand_pose[0]
        out.beta = fallback.beta.clone()
        out.scale = fallback.scale.clone()
        return out

    return _build_single(smplx_list_c2), _build_single(smplx_list_c4)


def save_segment_result(
    result_c2,
    result_c4,
    shared_beta: torch.Tensor,
    output_dir: Path,
    frame_idxs: list[int],
    valid_frame_count_c2: int,
    valid_frame_count_c4: int,
) -> None:
    start = frame_idxs[0]
    end = frame_idxs[-1]
    shared_beta_list = [float(x) for x in shared_beta.detach().cpu().tolist()]

    for cam_id, result, valid_frame_count in (
        ("cam2", result_c2, valid_frame_count_c2),
        ("cam4", result_c4, valid_frame_count_c4),
    ):
        cam_dir = output_dir / cam_id
        cam_dir.mkdir(parents=True, exist_ok=True)

        seq_path = cam_dir / f"seg_{start}_{end}_seq.pkl"
        meta_path = cam_dir / f"seg_{start}_{end}_meta.json"

        with seq_path.open("wb") as file:
            pickle.dump(result, file)

        meta = {
            "camera": cam_id,
            "start": start,
            "end": end,
            "frame_idxs": frame_idxs,
            "valid_frame_count": valid_frame_count,
            "total_frame_count": len(frame_idxs),
            "shared_beta": shared_beta_list,
        }
        with meta_path.open("w", encoding="utf-8") as file:
            json.dump(meta, file, ensure_ascii=False, indent=2)


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

    frame_indices = load_reviewed_frame_indices(reviewed_path)
    segments = split_frame_segments(frame_indices)
    if args.max_segment_len > 0:
        segments = split_long_segments(segments, args.max_segment_len)
        print(f"切段后共 {len(segments)} 段，最大段长 {args.max_segment_len}")
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

    if args.start_frame is not None:
        segments = [frame_idxs for frame_idxs in segments if frame_idxs[-1] >= args.start_frame]
    if args.end_frame is not None:
        segments = [frame_idxs for frame_idxs in segments if frame_idxs[0] <= args.end_frame]

    fitter = DualCamERPFitter(
        gender="neutral",
        R_cam4=R_CAM4,
        use_silhouette_loss=args.use_silhouette_loss,
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
            reconstructable = rtm_valid or seg_valid
            reconstructable_by_frame[frame_idx] = reconstructable
            frame_status[str(frame_idx)] = {
                "rtm_valid": rtm_valid,
                "seg_valid": seg_valid,
                "hmr_valid": hmr_valid,
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
            boundary_pts_list_c2, boundary_pts_list_c4 = [], []
            bbox_list_c2, bbox_list_c4 = [], []
            for frame_idx in sub_frame_idxs:
                mask_c2, mask_c4 = get_mask_paths(frame_idx, seg_dir)
                use_seg_frame = frame_status[str(frame_idx)]["seg_valid"] and args.use_silhouette_loss
                boundary_pts_list_c2.append(
                    DualCamERPFitter._prepare_silhouette_data(mask_c2, device=sil_device)
                    if use_seg_frame and mask_c2.exists()
                    else None
                )
                boundary_pts_list_c4.append(
                    DualCamERPFitter._prepare_silhouette_data(mask_c4, device=sil_device)
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
  --reviewed data_process/Input/20260311_riding/review.json \
  --rtm_dir data_process/Output/20260311_riding/rtm_cache/equirect_rotate_rtmw \
  --hmr_dir data_process/Output/20260311_riding/hmr_cache \
  --seg_dir data_process/Output/20260311_riding/seg/equirect_crop_rotate_sam3 \
  --output_dir data_process/Output/20260311_riding/fitting/dual_cam_sil_new_513_518 \
  --rtm_invalid_json data_process/Output/20260311_riding/rtm_cache/equirect_rotate_rtmw/invalid.json \
  --seg_invalid_json data_process/Output/20260311_riding/seg/invalid.json \
  --hmr_invalid_json data_process/Output/20260311_riding/hmr_cache/invalid.json \
  --bbox_dir data_process/Output/20260311_riding/image/equirect_crop_rotate \
  --max_segment_len 10 \
  --start_frame 513 --end_frame 518 \
  --use_silhouette_loss

python src/script/test_function/debug029_run_dual_erp_fitting.py `
  --reviewed data_process/Input/20260320_riding/dataset1/review.json `
  --rtm_dir data_process/Output/20260320_riding_dataset1/rtm_cache/equirect_rotate `
  --hmr_dir data_process/Output/20260320_riding_dataset1/hmr_cache `
  --output_dir data_process/Output/20260320_riding_dataset1/fitting/dual_cam_sil_886_895 `
  --rtm_invalid_json data_process/Output/20260320_riding_dataset1/invalid_detections.json `
  --seg_dir data_process/Output/20260320_riding_dataset1/seg/equirect_crop_rotate_sam3 `
  --bbox_dir data_process/Output/20260320_riding_dataset1/image/equirect_crop_rotate `
  --start_frame 886 --end_frame 895 `
  --use_silhouette_loss
'''
