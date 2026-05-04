from __future__ import annotations

import argparse
import importlib
import os
import pickle
import sys
from pathlib import Path

import cv2 as cv
import numpy as np
from tqdm import tqdm

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import config_script
from batch_utils import load_reviewed_entries
from lib_pano.pipeline.bbox_io import load_bbox_dict
from lib_pano.pipeline.invalid_io import load_invalid_frames
from lib_pano.pipeline.paths import ensure_single_fitting_path

from lib_pano.human3d.keypoints_detector import (
    Result_Single_Human_Single_Frame,
    draw_keypoints,
)
from lib_pano.human3d.skeletion_constant import COCO_133_KEYPOINTS
from mmpose.apis import inference_topdown, init_model

try:
    from mmpose.visualization import PoseLocalVisualizer
except Exception:
    PoseLocalVisualizer = None

try:
    from mmpose.structures import merge_data_samples
except Exception:
    try:
        from mmpose.structures.utils import merge_data_samples
    except Exception:
        merge_data_samples = None


def _ensure_paths() -> Path:
    """Ensure the SingleFitting package is importable."""
    repo_root = ensure_single_fitting_path()
    fisheye_root = repo_root / "tools" / "fisheye_extractor"
    if str(fisheye_root) not in sys.path:
        sys.path.insert(0, str(fisheye_root))
    return repo_root


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run MMPose RTMW on equirect crops and map results back to full ERP coordinates."
    )
    parser.add_argument("--reviewed", type=str, required=True, help="Path to reviewed.json")
    parser.add_argument(
        "--crop_dir",
        type=str,
        required=True,
        help="ERP crop root directory containing cam2/, cam4/, cam2_bboxes.json and cam4_bboxes.json",
    )
    parser.add_argument("--output_dir", type=str, required=True, help="Output root directory")
    parser.add_argument("--pose_config", type=str, required=True, help="MMPose RTMW config path")
    parser.add_argument("--pose_checkpoint", type=str, required=True, help="MMPose RTMW checkpoint path")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device for MMPose inference, for example cuda:0 or cpu",
    )
    parser.add_argument(
        "--invalid_json",
        type=str,
        default=None,
        help="Optional invalid.json containing invalid_frames to skip.",
    )
    parser.add_argument(
        "--verbose_warnings",
        action="store_true",
        help="Print per-frame warnings instead of only summary.",
    )
    parser.add_argument(
        "--disable_mmpose_vis",
        action="store_true",
        help="Disable MMPose visualizer and use project draw_keypoints fallback.",
    )
    parser.add_argument(
        "--kpt_thr",
        type=float,
        default=0.3,
        help="Keypoint confidence threshold for visualization only.",
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


def _get_crop_bbox(bbox_dict: dict[str, list[int]], frame_idx: int) -> list[int] | None:
    bbox_xyxy = bbox_dict.get(str(frame_idx))
    if bbox_xyxy is None:
        bbox_xyxy = bbox_dict.get(f"{frame_idx:05d}")
    return bbox_xyxy


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


def _as_numpy(value, dtype=np.float32) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=dtype)


def _pose_result_to_detection(
    pose_result,
    crop_h: int,
    crop_w: int,
    frame_idx: int,
    camera_name: str,
    verbose_warnings: bool,
) -> Result_Single_Human_Single_Frame | None:
    pred_instances = getattr(pose_result, "pred_instances", None)
    if pred_instances is None or not hasattr(pred_instances, "keypoints"):
        return None

    kpts = _as_numpy(pred_instances.keypoints, dtype=np.float32)
    if kpts.ndim == 3:
        kpts = kpts[0]
    if kpts.ndim != 2 or kpts.shape[1] < 2:
        return None
    kpts = kpts[:, :2]

    scores_raw = getattr(pred_instances, "keypoints_visible", None)
    if scores_raw is None:
        scores = np.ones((kpts.shape[0],), dtype=np.float32)
    else:
        scores = _as_numpy(scores_raw, dtype=np.float32)
        if scores.ndim == 2:
            scores = scores[0]
        scores = scores.reshape(-1)

    if kpts.shape[0] != 133 and verbose_warnings:
        print(
            f"Warning: {camera_name} frame {frame_idx:05d} has "
            f"{kpts.shape[0]} keypoints, expected 133"
        )

    keypoints = np.zeros((133, 3), dtype=np.float32)
    n = min(133, kpts.shape[0], scores.shape[0])
    if n <= 0:
        return None
    keypoints[:n, :2] = kpts[:n]
    keypoints[:n, 2] = scores[:n]

    return Result_Single_Human_Single_Frame(
        bbox=(0, crop_h - 1, 0, crop_w - 1),
        bbox_score=1.0,
        keypoints=keypoints,
        keypoints_format=COCO_133_KEYPOINTS,
        flag_with_keypoints=True,
    )


def _save_mmpose_visualization(
    visualizer,
    pose_results: list,
    crop_image: np.ndarray,
    out_file: Path,
    camera_name: str,
    frame_idx: int,
    kpt_thr: float,
    verbose_warnings: bool,
) -> bool:
    if visualizer is None:
        return False

    try:
        if merge_data_samples is None:
            raise RuntimeError("merge_data_samples is unavailable")

        data_sample = merge_data_samples(pose_results)
        image_rgb = cv.cvtColor(crop_image, cv.COLOR_BGR2RGB)
        visualizer.add_datasample(
            name=f"{camera_name}_{frame_idx:05d}",
            image=image_rgb,
            data_sample=data_sample,
            draw_gt=False,
            draw_bbox=True,
            show=False,
            wait_time=0,
            out_file=str(out_file),
            kpt_thr=kpt_thr,
        )
        return True
    except Exception as exc:
        if verbose_warnings:
            print(
                f"Warning: MMPose visualizer failed for {camera_name} "
                f"frame {frame_idx:05d}: {exc}"
            )
        return False


def _save_fallback_visualization(
    crop_image: np.ndarray,
    det_result_crop: Result_Single_Human_Single_Frame,
    out_file: Path,
    kpt_thr: float,
) -> None:
    crop_h, crop_w = crop_image.shape[:2]
    vis_image = draw_keypoints(
        crop_image,
        det_result_crop,
        conf_thresh=kpt_thr,
        draw_face=True,
        draw_hands=True,
    )
    cv.rectangle(vis_image, (0, 0), (crop_w - 1, crop_h - 1), (255, 0, 0), 2)
    cv.imwrite(str(out_file), vis_image)


def _process_single_camera(
    pose_model,
    visualizer,
    frame_idx: int,
    camera_name: str,
    camera_suffix: str,
    crop_dir: Path,
    output_dir: Path,
    vis_dir: Path,
    bbox_dict: dict[str, list[int]],
    verbose_warnings: bool,
    kpt_thr: float,
) -> str:
    crop_bbox = _get_crop_bbox(bbox_dict, frame_idx)
    if crop_bbox is None:
        if verbose_warnings:
            print(f"Warning: bbox not found, skipping {camera_name} frame {frame_idx:05d}")
        return "missing_bbox"

    crop_path = crop_dir / camera_name / f"{frame_idx:05d}.jpg"
    crop_image = cv.imread(str(crop_path))
    if crop_image is None:
        if verbose_warnings:
            print(f"Warning: crop image not found, skipping {camera_name} frame {frame_idx:05d}")
        return "missing_crop"

    crop_h, crop_w = crop_image.shape[:2]
    bbox_xyxy = np.array([[0, 0, crop_w - 1, crop_h - 1]], dtype=np.float32)
    pose_results = inference_topdown(
        pose_model,
        crop_image,
        bboxes=bbox_xyxy,
    )
    if not pose_results:
        if verbose_warnings:
            print(f"Warning: no valid detection for {camera_name} frame {frame_idx:05d}")
        return "no_detection"

    det_result_crop = _pose_result_to_detection(
        pose_results[0],
        crop_h,
        crop_w,
        frame_idx,
        camera_name,
        verbose_warnings,
    )
    if det_result_crop is None or not det_result_crop.flag_valid or det_result_crop.bbox is None:
        if verbose_warnings:
            print(f"Warning: no valid detection for {camera_name} frame {frame_idx:05d}")
        return "no_detection"

    x1, y1, x2, y2 = [int(v) for v in crop_bbox]
    det_result_full = _offset_detection_to_full_erp(det_result_crop, x1, y1)
    det_seq = _build_single_frame_det_seq(det_result_full)

    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / f"rtm_s{frame_idx:05d}_{camera_suffix}.pkl").open("wb") as file:
        pickle.dump(det_seq, file)

    vis_dir.mkdir(parents=True, exist_ok=True)
    vis_path = vis_dir / f"vis_{frame_idx:05d}_{camera_suffix}.jpg"
    if not _save_mmpose_visualization(
        visualizer,
        pose_results,
        crop_image,
        vis_path,
        camera_name,
        frame_idx,
        kpt_thr,
        verbose_warnings,
    ):
        _save_fallback_visualization(crop_image, det_result_crop, vis_path, kpt_thr)

    return "ok"


def _init_visualizer(args: argparse.Namespace, pose_model):
    if args.disable_mmpose_vis:
        return None
    if PoseLocalVisualizer is None:
        print("Warning: MMPose PoseLocalVisualizer is unavailable; falling back to draw_keypoints.")
        return None

    try:
        visualizer = PoseLocalVisualizer(name="rtmw_visualizer")
        if hasattr(pose_model, "dataset_meta"):
            visualizer.set_dataset_meta(pose_model.dataset_meta)
        return visualizer
    except Exception as exc:
        print(f"Warning: failed to initialize MMPose visualizer: {exc}")
        return None


def main() -> None:
    _ensure_paths()
    args = parse_args()

    reviewed_path = Path(args.reviewed)
    crop_dir = Path(args.crop_dir)
    output_dir = Path(args.output_dir)
    invalid_path = Path(args.invalid_json) if args.invalid_json is not None else None

    reviewed_pairs = load_reviewed_entries(reviewed_path)
    cam2_bboxes = load_bbox_dict(crop_dir / "cam2_bboxes.json")
    cam4_bboxes = load_bbox_dict(crop_dir / "cam4_bboxes.json")
    invalid_frames = load_invalid_frames(invalid_path, encoding="utf-8-sig")
    print(f"loaded invalid frames: {len(invalid_frames)}")

    pose_model = init_model(
        config=args.pose_config,
        checkpoint=args.pose_checkpoint,
        device=args.device,
    )
    visualizer = _init_visualizer(args, pose_model)

    stats = {
        "cam2": {"ok": 0, "missing_bbox": 0, "missing_crop": 0, "no_detection": 0, "skipped_invalid": 0},
        "cam4": {"ok": 0, "missing_bbox": 0, "missing_crop": 0, "no_detection": 0, "skipped_invalid": 0},
    }

    total_steps = len(reviewed_pairs) * 2
    with tqdm(total=total_steps, desc="Running RTMW-MMPose on ERP crops", unit="frame") as progress:
        for _, frame_idx in reviewed_pairs:
            if frame_idx in invalid_frames:
                stats["cam2"]["skipped_invalid"] += 1
                stats["cam4"]["skipped_invalid"] += 1
                progress.update(2)
                continue

            status_c2 = _process_single_camera(
                pose_model=pose_model,
                visualizer=visualizer,
                frame_idx=frame_idx,
                camera_name="cam2",
                camera_suffix="c2",
                crop_dir=crop_dir,
                output_dir=output_dir / "cam2",
                vis_dir=output_dir / "vis" / "cam2",
                bbox_dict=cam2_bboxes,
                verbose_warnings=args.verbose_warnings,
                kpt_thr=args.kpt_thr,
            )
            stats["cam2"][status_c2] += 1
            progress.update(1)

            status_c4 = _process_single_camera(
                pose_model=pose_model,
                visualizer=visualizer,
                frame_idx=frame_idx,
                camera_name="cam4",
                camera_suffix="c4",
                crop_dir=crop_dir,
                output_dir=output_dir / "cam4",
                vis_dir=output_dir / "vis" / "cam4",
                bbox_dict=cam4_bboxes,
                verbose_warnings=args.verbose_warnings,
                kpt_thr=args.kpt_thr,
            )
            stats["cam4"][status_c4] += 1
            progress.update(1)

    print("Detection summary:")
    for camera_name, camera_stats in stats.items():
        print(f"{camera_name}:")
        for key, value in camera_stats.items():
            print(f"  {key}: {value}")


if __name__ == "__main__":
    main()


r'''
python src\script\test_function\debug027_run_rtmw_mmpose.py `
  --reviewed data_process\Input\20260311_riding\review.json `
  --crop_dir data_process\Output\20260311_riding\image\equirect_crop_rotate `
  --output_dir data_process\Output\20260311_riding\rtm_cache\equirect_rotate_rtmw `
  --invalid_json data_process\Output\20260311_riding\invalid.json `
  --pose_config tools\mmpose\configs\wholebody_2d_keypoint\rtmpose\cocktail14\rtmw-l_8xb320-270e_cocktail14-384x288.py `
  --pose_checkpoint D:\models\rtmw\rtmw-dw-x-l_simcc-cocktail14_270e-384x288-20231122.pth `
  --device cuda:0 `
  --kpt_thr 0.57

如果要和rtmlib比对, 只比对都有的点的位置, 这样子可以判断哪个方法更好

python src\script\test_function\debug027_run_rtmw_mmpose.py `
  --reviewed data_process\Input\20260320_riding\dataset1\review.json `
  --crop_dir data_process\Output\20260320_riding_dataset1\image\equirect_crop_rotate `
  --output_dir data_process\Output\20260320_riding_dataset1\rtm_cache\equirect_rotate_rtmw `
  --invalid_json data_process\Output\20260320_riding_dataset1\invalid.json `
  --pose_config tools\mmpose\configs\wholebody_2d_keypoint\rtmpose\cocktail14\rtmw-l_8xb320-270e_cocktail14-384x288.py `
  --pose_checkpoint D:\models\rtmw\rtmw-dw-x-l_simcc-cocktail14_270e-384x288-20231122.pth `
  --device cuda:0 `
  --kpt_thr 0.57

If MMPose visualizer has issues, add:

  --disable_mmpose_vis

To show more low-confidence keypoints, use --kpt_thr 0.1 or --kpt_thr 0.2.
To show only more reliable keypoints, use --kpt_thr 0.5.
'''
