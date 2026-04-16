from __future__ import annotations

import argparse
import importlib
import json
import pickle
import sys
from pathlib import Path

import cv2 as cv
import numpy as np
from tqdm import tqdm
import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import config_script
from batch_utils import load_reviewed_entries

PROJECT_ROOT = Path(config_script.PROJECT_ROOT)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lib_pano.human3d.keypoints_detector import (
    Result_Single_Human_Single_Frame,
    WholePoseDetector,
    draw_keypoints,
)


def _ensure_paths() -> Path:
    """Ensure the SingleFitting package is importable."""
    repo_root = Path(__file__).resolve().parents[3]
    sf_root = repo_root / "tools" / "SingleFitting"
    if str(sf_root) not in sys.path:
        sys.path.insert(0, str(sf_root))
    fisheye_root = repo_root / "tools" / "fisheye_extractor"
    if str(fisheye_root) not in sys.path:
        sys.path.insert(0, str(fisheye_root))
    return repo_root


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run WholePoseDetector on equirect crops and map results back to full ERP coordinates."
    )
    parser.add_argument("--reviewed", type=str, required=True, help="Path to reviewed.json")
    parser.add_argument(
        "--crop_dir",
        type=str,
        required=True,
        help="ERP crop root directory containing cam2/, cam4/, cam2_bboxes.json and cam4_bboxes.json",
    )
    parser.add_argument("--output_dir", type=str, required=True, help="Output root directory")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device for RTMLib inference, cuda or cpu",
    )
    return parser.parse_args()


def _build_multi_frame_det_sequence(det_results: list):
    """Build the detection sequence object expected by SingleFitting."""
    base_engine_mod = importlib.import_module("lib_dynamic_hvip.detection.engine.base_engine")
    Base_Detect_Engine = base_engine_mod.Base_Detect_Engine

    if not det_results:
        raise RuntimeError("det_results is empty")

    num_frames = len(det_results)
    bbox = np.zeros((num_frames, 4), dtype=np.int32)
    score = np.zeros((num_frames,), dtype=np.float32)
    kps = np.zeros((num_frames, 133, 3), dtype=np.float32)
    valid = np.zeros((num_frames,), dtype=np.bool_)

    keypoints_format = None
    for i, det_result in enumerate(det_results):
        is_valid = bool(getattr(det_result, "flag_valid", False))
        valid[i] = is_valid
        if not is_valid:
            continue

        if keypoints_format is None:
            keypoints_format = det_result.keypoints_format

        bbox_i = np.asarray(det_result.bbox, dtype=np.int32).reshape(-1)
        if bbox_i.size >= 4:
            bbox[i] = bbox_i[:4]
        score[i] = float(det_result.bbox_score)

        kps_i = getattr(det_result, "keypoints", None)
        if kps_i is not None:
            kps_i = np.asarray(kps_i, dtype=np.float32)
            if kps_i.ndim == 2:
                n = min(133, kps_i.shape[0])
                c = min(3, kps_i.shape[1])
                kps[i, :n, :c] = kps_i[:n, :c]

    if keypoints_format is None:
        raise RuntimeError("No valid detection result with keypoints_format found")

    return Base_Detect_Engine.Result_Single_Human_Sequence(
        array_valid=valid,
        array_bbox_h1h2w1w2=bbox,
        array_bbox_score=score,
        array_keypoints=kps,
        keypoints_format=keypoints_format,
        flag_with_keypoints=True,
    )


def _build_single_frame_det_seq(det_result):
    """Wrap one detection result as a single-frame sequence."""
    return _build_multi_frame_det_sequence([det_result])


def _load_bbox_dict(json_path: Path) -> dict[str, list[int]]:
    with json_path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, dict):
        raise ValueError(f"bbox json must contain a dict: {json_path}")
    return data


def _offset_detection_to_full_erp(
    det_result: Result_Single_Human_Single_Frame,
    offset_x: int,
    offset_y: int,
) -> Result_Single_Human_Single_Frame:
    ymin, ymax, xmin, xmax = det_result.bbox
    bbox_full = (
        int(ymin + offset_y),
        int(ymax + offset_y),
        int(xmin + offset_x),
        int(xmax + offset_x),
    )

    keypoints_full = None
    if det_result.keypoints is not None:
        keypoints_full = np.asarray(det_result.keypoints, dtype=np.float32).copy()
        keypoints_full[:, 0] += float(offset_x)
        keypoints_full[:, 1] += float(offset_y)

    return Result_Single_Human_Single_Frame(
        bbox=bbox_full,
        bbox_score=float(det_result.bbox_score),
        keypoints=keypoints_full,
        keypoints_format=det_result.keypoints_format,
        flag_with_keypoints=det_result.flag_with_keypoints,
    )


def _process_single_camera(
    detector: WholePoseDetector,
    frame_idx: int,
    camera_name: str,
    camera_suffix: str,
    crop_dir: Path,
    output_dir: Path,
    vis_dir: Path,
    bbox_dict: dict[str, list[int]],
) -> None:
    bbox_key = str(frame_idx)
    bbox_xyxy = bbox_dict.get(bbox_key)
    if bbox_xyxy is None:
        print(f"Warning: bbox not found, skipping {camera_name} frame {frame_idx:05d}")
        return

    crop_path = crop_dir / camera_name / f"{frame_idx:05d}.jpg"
    crop_image = cv.imread(str(crop_path))
    if crop_image is None:
        print(f"Warning: crop image not found, skipping {camera_name} frame {frame_idx:05d}")
        return

    det_result = detector.detect_single_image(crop_image)
    if not det_result.flag_valid or det_result.bbox is None:
        print(f"Warning: no valid detection for {camera_name} frame {frame_idx:05d}")
        return

    x1, y1, x2, y2 = [int(v) for v in bbox_xyxy]
    det_result_full = _offset_detection_to_full_erp(det_result, x1, y1)
    det_seq = _build_single_frame_det_seq(det_result_full)

    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / f"rtm_s{frame_idx:05d}_{camera_suffix}.pkl").open("wb") as file:
        pickle.dump(det_seq, file)

    vis_dir.mkdir(parents=True, exist_ok=True)
    vis_image = draw_keypoints(crop_image, det_result, draw_face=True, draw_hands=True)
    cv.rectangle(vis_image, (0, 0), (crop_image.shape[1] - 1, crop_image.shape[0] - 1), (255, 0, 0), 2)
    cv.imwrite(str(vis_dir / f"vis_{frame_idx:05d}_{camera_suffix}.jpg"), vis_image)


def main() -> None:
    _ensure_paths()
    args = parse_args()

    reviewed_path = Path(args.reviewed)
    crop_dir = Path(args.crop_dir)
    output_dir = Path(args.output_dir)

    reviewed_pairs = load_reviewed_entries(reviewed_path)
    cam2_bboxes = _load_bbox_dict(crop_dir / "cam2_bboxes.json")
    cam4_bboxes = _load_bbox_dict(crop_dir / "cam4_bboxes.json")
    detector = WholePoseDetector(device=args.device)

    total_steps = len(reviewed_pairs) * 2
    with tqdm(total=total_steps, desc="Running WholePoseDetector on ERP crops", unit="frame") as progress:
        for _, frame_idx in reviewed_pairs:
            _process_single_camera(
                detector=detector,
                frame_idx=frame_idx,
                camera_name="cam2",
                camera_suffix="c2",
                crop_dir=crop_dir,
                output_dir=output_dir / "cam2",
                vis_dir=output_dir / "vis" / "cam2",
                bbox_dict=cam2_bboxes,
            )
            progress.update(1)

            _process_single_camera(
                detector=detector,
                frame_idx=frame_idx,
                camera_name="cam4",
                camera_suffix="c4",
                crop_dir=crop_dir,
                output_dir=output_dir / "cam4",
                vis_dir=output_dir / "vis" / "cam4",
                bbox_dict=cam4_bboxes,
            )
            progress.update(1)


if __name__ == "__main__":
    main()
