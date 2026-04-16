from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm

import config_script


def _ensure_paths() -> Path:
    repo_root = Path(__file__).resolve().parents[3]
    sf_root = repo_root / "tools" / "SingleFitting"
    if str(sf_root) not in sys.path:
        sys.path.insert(0, str(sf_root))
    return repo_root


REPO_ROOT = _ensure_paths()

from lib_dynamic_hvip.fitting.smplx_seq_forward_module import SMPLX_Seq_Forward_Module
from lib_pano.human3d.keypoints_detector import Result_Single_Human_Single_Frame
from single_fitting.fitting import ERPFitter, init_smplx_from_smpl54_seq, slice_smplx_seq


ROTATION_MATRIX = torch.tensor(
    [1, 0, 0, 0, 0, 1, 0, -1, 0], dtype=torch.float32
).reshape(3, 3)


class RotatedERPFitter(ERPFitter):
    @staticmethod
    def _project(points_3d: torch.Tensor, W: int, H: int) -> torch.Tensor:
        import math

        R = ROTATION_MATRIX.to(points_3d.device, dtype=points_3d.dtype)
        pts = points_3d @ R.T
        x, y, z = pts[..., 0], pts[..., 1], pts[..., 2]
        r = torch.norm(pts, dim=-1).clamp_min(1e-6)
        lon = torch.atan2(x, z)
        lat = torch.asin((-y / r).clamp(-1 + 1e-6, 1 - 1e-6))
        u = (lon / (2.0 * math.pi) + 0.5) * W
        v = (0.5 - lat / math.pi) * H
        return torch.stack([u, v], dim=-1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch-run ERPFitter on reviewed cam2 equirect frames.")
    parser.add_argument("--reviewed", type=str, required=True, help="Path to reviewed.json.")
    parser.add_argument(
        "--rtm_dir",
        type=str,
        required=True,
        help="frame_cache root directory containing cam2 RTM pkls.",
    )
    parser.add_argument(
        "--hmr_dir",
        type=str,
        required=True,
        help="4D-Humans pkl root directory containing cam2 HMR pkls.",
    )
    parser.add_argument("--output_dir", type=str, required=True, help="Output root directory.")
    parser.add_argument(
        "--invalid_detections",
        type=str,
        default=None,
        help="Optional path to invalid_detections.json.",
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
    return parser.parse_args()


def load_reviewed_frame_indices(reviewed_path: Path) -> list[int]:
    with reviewed_path.open("r", encoding="utf-8") as file:
        reviewed = json.load(file)

    if not isinstance(reviewed, list):
        raise ValueError("reviewed.json must contain a list")

    frame_indices: set[int] = set()
    for item in reviewed:
        if not isinstance(item, dict):
            raise ValueError("Each reviewed entry must be a dict")
        if "frame_idx" not in item:
            raise ValueError("Each reviewed entry must contain frame_idx")
        frame_indices.add(int(item["frame_idx"]))

    return sorted(frame_indices)


def split_frame_segments(frame_indices: list[int], gap_threshold: int = 2) -> list[list[int]]:
    if not frame_indices:
        return []

    segments: list[list[int]] = [[frame_indices[0]]]
    for frame_idx in frame_indices[1:]:
        if frame_idx - segments[-1][-1] > gap_threshold:
            segments.append([frame_idx])
        else:
            segments[-1].append(frame_idx)
    return segments


def _extract_frame_indices_from_obj(obj: Any) -> set[int]:
    frame_indices: set[int] = set()

    if isinstance(obj, int):
        frame_indices.add(int(obj))
        return frame_indices

    if isinstance(obj, list):
        for item in obj:
            frame_indices.update(_extract_frame_indices_from_obj(item))
        return frame_indices

    if isinstance(obj, dict):
        if "frame_idx" in obj:
            frame_indices.add(int(obj["frame_idx"]))
        for key in (
            "invalid_frames",
            "frame_idxs",
            "frames",
            "invalid_detections",
            "cam2",
        ):
            if key in obj:
                frame_indices.update(_extract_frame_indices_from_obj(obj[key]))
        return frame_indices

    return frame_indices


def load_invalid_frames(invalid_path: Path | None) -> set[int]:
    if invalid_path is None:
        return set()
    if not invalid_path.exists():
        raise FileNotFoundError(f"invalid_detections file not found: {invalid_path}")

    with invalid_path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    return _extract_frame_indices_from_obj(data)


def infer_keypoints_format_and_dim(frame_indices: list[int], rtm_dir: Path) -> tuple[dict[str, int], int]:
    for frame_idx in frame_indices:
        pkl_path = rtm_dir / "cam2" / f"rtm_s{frame_idx:05d}_c2.pkl"
        if not pkl_path.exists():
            continue
        with pkl_path.open("rb") as file:
            seq = pickle.load(file)
        keypoints_format = dict(seq.keypoints_format)
        num_keypoints = int(seq.array_keypoints.shape[1])
        return keypoints_format, num_keypoints
    raise RuntimeError("Failed to infer keypoints_format from RTM pkls for reviewed frames")


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
    smpl_init.beta = src["beta"].clone()
    smpl_init.scale = src["scale"].clone()
    smpl_init.valid = src.get("valid", torch.ones(num_batch, dtype=torch.bool)).clone()
    return smpl_init


def load_det_result(
    frame_idx: int,
    rtm_dir: Path,
    invalid_frames: set[int],
    default_keypoints_format: dict[str, int],
    num_keypoints: int,
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

    pkl_path = rtm_dir / "cam2" / f"rtm_s{frame_idx:05d}_c2.pkl"
    if not pkl_path.exists():
        return Result_Single_Human_Single_Frame(
            bbox=None,
            bbox_score=0.0,
            keypoints=zero_keypoints,
            keypoints_format=default_keypoints_format,
        )

    with pkl_path.open("rb") as file:
        seq = pickle.load(file)

    is_valid = bool(seq.array_valid[0])
    det = Result_Single_Human_Single_Frame(
        bbox=tuple(seq.array_bbox_h1h2w1w2[0].tolist()) if is_valid else None,
        bbox_score=float(seq.array_bbox_score[0]),
        keypoints=seq.array_keypoints[0].copy() if is_valid else zero_keypoints.copy(),
        keypoints_format=dict(seq.keypoints_format),
    )
    return det


def load_smplx_init(frame_idx: int, hmr_dir: Path):
    pkl_path = hmr_dir / "cam2" / f"hmr_s{frame_idx}_pose.pkl"
    if not pkl_path.exists():
        return None

    smpl_seq = _build_smpl_init_from_single_frame_pkl(pkl_path)
    return init_smplx_from_smpl54_seq(smpl_seq, [0])


def build_segment_smplx_init(smplx_list):
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


def save_segment_result(
    result,
    output_dir: Path,
    frame_idxs: list[int],
    valid_frame_count: int,
) -> None:
    start = frame_idxs[0]
    end = frame_idxs[-1]
    cam2_dir = output_dir / "cam2"
    cam2_dir.mkdir(parents=True, exist_ok=True)

    seq_path = cam2_dir / f"seg_{start}_{end}_seq.pkl"
    meta_path = cam2_dir / f"seg_{start}_{end}_meta.json"

    with seq_path.open("wb") as file:
        pickle.dump(result, file)

    meta = {
        "start": start,
        "end": end,
        "frame_idxs": frame_idxs,
        "valid_frame_count": valid_frame_count,
        "total_frame_count": len(frame_idxs),
    }
    with meta_path.open("w", encoding="utf-8") as file:
        json.dump(meta, file, ensure_ascii=False, indent=2)


def main() -> None:
    args = parse_args()

    reviewed_path = Path(args.reviewed)
    rtm_dir = Path(args.rtm_dir)
    hmr_dir = Path(args.hmr_dir)
    output_dir = Path(args.output_dir)
    invalid_path = Path(args.invalid_detections) if args.invalid_detections is not None else None

    frame_indices = load_reviewed_frame_indices(reviewed_path)
    segments = split_frame_segments(frame_indices)
    invalid_frames = load_invalid_frames(invalid_path)
    default_keypoints_format, num_keypoints = infer_keypoints_format_and_dim(frame_indices, rtm_dir)

    if args.start_frame is not None:
        segments = [frame_idxs for frame_idxs in segments if frame_idxs[-1] >= args.start_frame]
    if args.end_frame is not None:
        segments = [frame_idxs for frame_idxs in segments if frame_idxs[0] <= args.end_frame]

    fitter = RotatedERPFitter(gender="neutral")

    for frame_idxs in tqdm(segments, desc="ERPFitter cam2", unit="segment"):
        det_list = [
            load_det_result(
                frame_idx,
                rtm_dir,
                invalid_frames,
                default_keypoints_format=default_keypoints_format,
                num_keypoints=num_keypoints,
            )
            for frame_idx in frame_idxs
        ]
        smplx_list = [load_smplx_init(frame_idx, hmr_dir) for frame_idx in frame_idxs]
        smplx_init = build_segment_smplx_init(smplx_list)

        valid_frame_count = sum(1 for det in det_list if bool(det.flag_valid))
        if valid_frame_count == 0:
            print(
                f"Warning: skip segment {frame_idxs[0]}-{frame_idxs[-1]} because no valid detections."
            )
            continue

        result = fitter.fit(
            detection_results_list=det_list,
            image_size=(args.image_width, args.image_height),
            smpl_param_init=smplx_init,
        )
        save_segment_result(result, output_dir, frame_idxs, valid_frame_count)


if __name__ == "__main__":
    main()

'''
python src\script\test_function\debug023_run_erp_fitting.py `
  --reviewed data_process\Input\20260311_riding\review.json `
  --rtm_dir data_process\Output\20260311_riding\rtm_cache\equirect `
  --hmr_dir data_process\Output\20260311_riding\hmr_cache `
  --output_dir data_process\Output\20260311_riding\fitting `
  --invalid_detections data_process\Output\20260311_riding\invalid_detections.json `
  --start_frame 513 --end_frame 519
'''
