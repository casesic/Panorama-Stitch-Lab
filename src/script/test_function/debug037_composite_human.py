from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt
from tqdm import tqdm

from batch_utils import load_reviewed_entries


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Composite segmented humans between ERP cameras using semantic flow."
    )
    parser.add_argument(
        "--seg_dir",
        type=str,
        required=True,
        help="Segmentation root containing cam2/mask, cam2/rgba, cam4/mask, cam4/rgba.",
    )
    parser.add_argument(
        "--flow_dir",
        type=str,
        required=True,
        help="Semantic flow root with frame_{idx:05d}/ subdirectories.",
    )
    parser.add_argument(
        "--image_dir",
        type=str,
        required=True,
        help="Full ERP image root containing cam2/ and cam4/.",
    )
    parser.add_argument(
        "--bbox_dir",
        type=str,
        required=True,
        help="Directory containing cam2_bboxes.json and cam4_bboxes.json.",
    )
    parser.add_argument("--output_dir", type=str, required=True, help="Output root directory.")
    parser.add_argument("--reviewed", type=str, required=True, help="Path to reviewed.json.")
    parser.add_argument(
        "--start",
        type=int,
        default=None,
        help="起始帧索引（包含），不指定则从头开始",
    )
    parser.add_argument(
        "--end",
        type=int,
        default=None,
        help="结束帧索引（包含），不指定则处理到末尾",
    )
    return parser.parse_args()


def load_bbox_dict(json_path: Path) -> dict[str, list[int]]:
    with json_path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, dict):
        raise ValueError(f"bbox json must contain a dict: {json_path}")
    return data


def clamp_int(value: int, low: int, high: int) -> int:
    return max(low, min(high, value))


def place_rgba_on_black(
    rgba_path: Path,
    bbox: list[int],
    erp_h: int,
    erp_w: int,
) -> np.ndarray | None:
    rgba = cv2.imread(str(rgba_path), cv2.IMREAD_UNCHANGED)
    if rgba is None:
        return None
    if rgba.ndim != 3 or rgba.shape[2] != 4:
        return None

    x1, y1, x2, y2 = [int(v) for v in bbox]
    clip_x1 = clamp_int(x1, 0, erp_w)
    clip_y1 = clamp_int(y1, 0, erp_h)
    clip_x2 = clamp_int(x2, 0, erp_w)
    clip_y2 = clamp_int(y2, 0, erp_h)
    if clip_x1 >= clip_x2 or clip_y1 >= clip_y2:
        return None

    rgba_h, rgba_w = rgba.shape[:2]
    src_x2 = min(rgba_w, clip_x2 - clip_x1)
    src_y2 = min(rgba_h, clip_y2 - clip_y1)
    if src_x2 <= 0 or src_y2 <= 0:
        return None

    rgba = rgba[:src_y2, :src_x2]
    result = np.zeros((erp_h, erp_w, 3), dtype=np.uint8)
    alpha_mask = rgba[..., 3] > 0
    if not np.any(alpha_mask):
        return result

    region = result[clip_y1:clip_y1 + src_y2, clip_x1:clip_x1 + src_x2]
    region[alpha_mask] = rgba[..., :3][alpha_mask]
    return result


def blend_two_humans(warp: np.ndarray, target_origin: np.ndarray) -> np.ndarray:
    if warp.shape != target_origin.shape:
        raise ValueError("warp and target_origin must have the same shape")

    target_valid = np.any(target_origin > 0, axis=2)
    top_boundary_per_col = np.where(
        target_valid.any(axis=0),
        target_valid.argmax(axis=0),
        target_valid.shape[0],
    ).astype(np.int32)

    rows = np.arange(target_valid.shape[0], dtype=np.int32)[:, None]
    boundary_mask = rows < top_boundary_per_col[None, :]

    warp_valid = np.any(warp > 0, axis=2)
    completion_mask = warp_valid & (~target_valid) & boundary_mask

    result = target_origin.copy()
    result[completion_mask] = warp[completion_mask]
    return result


def composite_direction(
    frame_idx: int,
    source_camera: str,
    target_camera: str,
    flow_name: str,
    seg_dir: Path,
    flow_dir: Path,
    image_dir: Path,
    source_bboxes: dict[str, list[int]],
    target_bboxes: dict[str, list[int]],
    human_bbox: list[int],
 ) -> np.ndarray | None:
    bbox_key = str(frame_idx)
    source_bbox = source_bboxes.get(bbox_key)
    target_bbox = target_bboxes.get(bbox_key)
    if source_bbox is None or target_bbox is None:
        print(f"Warning: missing bbox, skipping frame {frame_idx:05d} for {source_camera}_to_{target_camera}")
        return None

    frame_flow_dir = flow_dir / f"frame_{frame_idx:05d}"
    flow_path = frame_flow_dir / flow_name
    confidence_name = flow_name.replace(".npy", "").replace("semantic_flow", "semantic_flow_confidence") + ".npy"
    confidence_path = frame_flow_dir / confidence_name
    rgba_path = seg_dir / source_camera / "rgba" / f"{frame_idx:05d}.png"
    target_image_path = image_dir / target_camera / f"{frame_idx:05d}.jpg"

    if not flow_path.exists() or not confidence_path.exists():
        print(f"Warning: missing flow files, skipping frame {frame_idx:05d} for {source_camera}_to_{target_camera}")
        return None

    flow = np.load(flow_path)
    confidence = np.load(confidence_path)
    source_rgba = cv2.imread(str(rgba_path), cv2.IMREAD_UNCHANGED)
    target_bgr = cv2.imread(str(target_image_path), cv2.IMREAD_COLOR)

    if source_rgba is None:
        print(f"Warning: source rgba not found, skipping {source_camera} frame {frame_idx:05d}")
        return None
    if source_rgba.ndim != 3 or source_rgba.shape[2] != 4:
        print(f"Warning: source rgba is not BGRA, skipping {source_camera} frame {frame_idx:05d}")
        return None
    if target_bgr is None:
        print(f"Warning: target ERP image not found, skipping {target_camera} frame {frame_idx:05d}")
        return None

    flow_h, flow_w = flow.shape[:2]
    rgba_h, rgba_w = source_rgba.shape[:2]
    work_h = min(flow_h, rgba_h, confidence.shape[0])
    work_w = min(flow_w, rgba_w, confidence.shape[1])
    if work_h <= 0 or work_w <= 0:
        print(f"Warning: empty work region, skipping frame {frame_idx:05d} for {source_camera}_to_{target_camera}")
        return None

    flow = flow[:work_h, :work_w]
    confidence = confidence[:work_h, :work_w]
    source_rgba = source_rgba[:work_h, :work_w]

    source_x1, source_y1, _, _ = [int(v) for v in source_bbox]
    hx1, hy1, _, _ = [int(v) for v in human_bbox]
    _target_x1, _target_y1, _, _ = [int(v) for v in target_bbox]
    erp_h, erp_w = target_bgr.shape[:2]

    roi_y1, roi_x1 = hy1, hx1
    roi_h, roi_w = flow.shape[:2]
    roi_y2 = roi_y1 + roi_h
    roi_x2 = roi_x1 + roi_w
    clip_y1 = max(0, roi_y1)
    clip_x1 = max(0, roi_x1)
    clip_y2 = min(erp_h, roi_y2)
    clip_x2 = min(erp_w, roi_x2)
    if clip_y1 >= clip_y2 or clip_x1 >= clip_x2:
        print(f"Warning: ROI out of image bounds, skipping frame {frame_idx:05d} for {source_camera}_to_{target_camera}")
        return None

    flow_y1 = clip_y1 - roi_y1
    flow_x1 = clip_x1 - roi_x1
    flow_y2 = flow_y1 + (clip_y2 - clip_y1)
    flow_x2 = flow_x1 + (clip_x2 - clip_x1)

    full_mask = np.zeros((erp_h, erp_w), dtype=bool)
    full_flow = np.zeros((erp_h, erp_w, 2), dtype=np.float32)
    full_mask[clip_y1:clip_y2, clip_x1:clip_x2] = confidence[flow_y1:flow_y2, flow_x1:flow_x2] > 0
    full_flow[clip_y1:clip_y2, clip_x1:clip_x2] = flow[flow_y1:flow_y2, flow_x1:flow_x2].astype(np.float32)

    decay_radius = 200.0
    pad = int(decay_radius) + 1
    ys, xs = np.where(full_mask)
    if len(ys) == 0:
        return np.zeros((erp_h, erp_w, 3), dtype=np.uint8)

    roi_y1_edt = max(0, int(ys.min()) - pad)
    roi_y2_edt = min(erp_h, int(ys.max()) + pad + 1)
    roi_x1_edt = max(0, int(xs.min()) - pad)
    roi_x2_edt = min(erp_w, int(xs.max()) + pad + 1)

    mask_roi = full_mask[roi_y1_edt:roi_y2_edt, roi_x1_edt:roi_x2_edt]
    flow_roi = full_flow[roi_y1_edt:roi_y2_edt, roi_x1_edt:roi_x2_edt]

    dist_roi, nearest_indices_roi = distance_transform_edt(
        ~mask_roi, return_distances=True, return_indices=True
    )
    nearest_y_roi = nearest_indices_roi[0]
    nearest_x_roi = nearest_indices_roi[1]
    propagated_flow_roi = flow_roi[nearest_y_roi, nearest_x_roi]
    weight_roi = np.clip(1.0 - dist_roi / decay_radius, 0.0, 1.0).astype(np.float32)
    flow_with_decay_roi = propagated_flow_roi * weight_roi[..., None]
    flow_with_decay_roi[..., 0] = cv2.GaussianBlur(flow_with_decay_roi[..., 0], (51, 51), 0)
    flow_with_decay_roi[..., 1] = cv2.GaussianBlur(flow_with_decay_roi[..., 1], (51, 51), 0)
    flow_with_decay_roi[mask_roi] = flow_roi[mask_roi]

    flow_with_decay = np.zeros((erp_h, erp_w, 2), dtype=np.float32)
    flow_with_decay[roi_y1_edt:roi_y2_edt, roi_x1_edt:roi_x2_edt] = flow_with_decay_roi

    src_roi_y1, src_roi_x1 = source_y1, source_x1
    src_roi_h, src_roi_w = source_rgba.shape[:2]
    src_roi_y2 = src_roi_y1 + src_roi_h
    src_roi_x2 = src_roi_x1 + src_roi_w
    src_clip_y1 = max(0, src_roi_y1)
    src_clip_x1 = max(0, src_roi_x1)
    src_clip_y2 = min(erp_h, src_roi_y2)
    src_clip_x2 = min(erp_w, src_roi_x2)
    if src_clip_y1 >= src_clip_y2 or src_clip_x1 >= src_clip_x2:
        print(f"Warning: source RGBA ROI out of image bounds, skipping frame {frame_idx:05d} for {source_camera}_to_{target_camera}")
        return None

    src_y1 = src_clip_y1 - src_roi_y1
    src_x1 = src_clip_x1 - src_roi_x1
    src_y2 = src_y1 + (src_clip_y2 - src_clip_y1)
    src_x2 = src_x1 + (src_clip_x2 - src_clip_x1)

    source_rgba_full = np.zeros((erp_h, erp_w, 4), dtype=np.uint8)
    source_rgba_full[src_clip_y1:src_clip_y2, src_clip_x1:src_clip_x2] = source_rgba[src_y1:src_y2, src_x1:src_x2]

    grid_x, grid_y = np.meshgrid(np.arange(erp_w, dtype=np.float32), np.arange(erp_h, dtype=np.float32))
    map_x = np.clip(grid_x + flow_with_decay[..., 0], 0.0, float(erp_w - 1))
    map_y = np.clip(grid_y + flow_with_decay[..., 1], 0.0, float(erp_h - 1))
    warped_rgba = cv2.remap(
        source_rgba_full,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0, 0),
    )

    result_bgr = np.zeros((erp_h, erp_w, 3), dtype=np.uint8)
    warped_alpha = warped_rgba[..., 3] > 0
    result_bgr[warped_alpha] = warped_rgba[..., :3][warped_alpha]
    return result_bgr


def process_frame(
    frame_idx: int,
    seg_dir: Path,
    flow_dir: Path,
    image_dir: Path,
    cam2_bboxes: dict[str, list[int]],
    cam4_bboxes: dict[str, list[int]],
    output_dir: Path,
    erp_h: int,
    erp_w: int,
) -> None:
    bbox_key = str(frame_idx)
    cam2_bbox = cam2_bboxes.get(bbox_key)
    cam4_bbox = cam4_bboxes.get(bbox_key)
    frame_flow_dir = flow_dir / f"frame_{frame_idx:05d}"
    human_bbox_path = frame_flow_dir / "human_bbox.npy"

    if not human_bbox_path.exists():
        print(f"Warning: missing human_bbox, skipping frame {frame_idx:05d}")
        return

    human_bbox = np.load(human_bbox_path).tolist()

    cam2_origin: np.ndarray | None = None
    cam4_origin: np.ndarray | None = None
    if cam2_bbox is None:
        print(f"Warning: missing cam2 bbox for frame {frame_idx:05d}")
    else:
        cam2_origin = place_rgba_on_black(seg_dir / "cam2" / "rgba" / f"{frame_idx:05d}.png", cam2_bbox, erp_h, erp_w)
        if cam2_origin is None:
            print(f"Warning: failed to place cam2 rgba for frame {frame_idx:05d}")

    if cam4_bbox is None:
        print(f"Warning: missing cam4 bbox for frame {frame_idx:05d}")
    else:
        cam4_origin = place_rgba_on_black(seg_dir / "cam4" / "rgba" / f"{frame_idx:05d}.png", cam4_bbox, erp_h, erp_w)
        if cam4_origin is None:
            print(f"Warning: failed to place cam4 rgba for frame {frame_idx:05d}")

    warped_cam2 = composite_direction(
        frame_idx=frame_idx,
        source_camera="cam2",
        target_camera="cam4",
        flow_name="semantic_flow_c2_to_c4.npy",
        seg_dir=seg_dir,
        flow_dir=flow_dir,
        image_dir=image_dir,
        source_bboxes=cam2_bboxes,
        target_bboxes=cam4_bboxes,
        human_bbox=human_bbox,
    )
    if warped_cam2 is None:
        print(f"Warning: failed to compose cam2_to_cam4 for frame {frame_idx:05d}")

    warped_cam4 = composite_direction(
        frame_idx=frame_idx,
        source_camera="cam4",
        target_camera="cam2",
        flow_name="semantic_flow_c4_to_c2.npy",
        seg_dir=seg_dir,
        flow_dir=flow_dir,
        image_dir=image_dir,
        source_bboxes=cam4_bboxes,
        target_bboxes=cam2_bboxes,
        human_bbox=human_bbox,
    )
    if warped_cam4 is None:
        print(f"Warning: failed to compose cam4_to_cam2 for frame {frame_idx:05d}")

    if warped_cam2 is not None:
        warp_dir = output_dir / "cam2_to_cam4" / "warp"
        warp_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(warp_dir / f"{frame_idx:05d}.jpg"), warped_cam2)
    if cam2_origin is not None:
        cam2_origin_dir = output_dir / "cam2_to_cam4" / "cam2_origin"
        cam2_origin_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(cam2_origin_dir / f"{frame_idx:05d}.jpg"), cam2_origin)
    if cam4_origin is not None:
        cam4_origin_dir = output_dir / "cam2_to_cam4" / "cam4_origin"
        cam4_origin_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(cam4_origin_dir / f"{frame_idx:05d}.jpg"), cam4_origin)
    if warped_cam2 is not None and cam4_origin is not None:
        blend_dir = output_dir / "cam2_to_cam4" / "blend"
        blend_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(
            str(blend_dir / f"{frame_idx:05d}.jpg"),
            blend_two_humans(warp=warped_cam2, target_origin=cam4_origin),
        )

    if warped_cam4 is not None:
        warp_dir = output_dir / "cam4_to_cam2" / "warp"
        warp_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(warp_dir / f"{frame_idx:05d}.jpg"), warped_cam4)
    if cam4_origin is not None:
        cam4_origin_dir = output_dir / "cam4_to_cam2" / "cam4_origin"
        cam4_origin_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(cam4_origin_dir / f"{frame_idx:05d}.jpg"), cam4_origin)
    if cam2_origin is not None:
        cam2_origin_dir = output_dir / "cam4_to_cam2" / "cam2_origin"
        cam2_origin_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(cam2_origin_dir / f"{frame_idx:05d}.jpg"), cam2_origin)
    if warped_cam4 is not None and cam2_origin is not None:
        blend_dir = output_dir / "cam4_to_cam2" / "blend"
        blend_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(
            str(blend_dir / f"{frame_idx:05d}.jpg"),
            blend_two_humans(warp=warped_cam4, target_origin=cam2_origin),
        )


def infer_erp_size(frame_indices: list[int], image_dir: Path) -> tuple[int, int]:
    for frame_idx in frame_indices:
        for camera_name in ("cam2", "cam4"):
            image_path = image_dir / camera_name / f"{frame_idx:05d}.jpg"
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image is None:
                continue
            return image.shape[0], image.shape[1]
    raise RuntimeError("Failed to infer ERP image size from input images")


def main() -> None:
    args = parse_args()

    seg_dir = Path(args.seg_dir)
    flow_dir = Path(args.flow_dir)
    image_dir = Path(args.image_dir)
    bbox_dir = Path(args.bbox_dir)
    output_dir = Path(args.output_dir)
    reviewed_path = Path(args.reviewed)

    reviewed_pairs = load_reviewed_entries(reviewed_path)
    frame_indices = [frame_idx for _, frame_idx in reviewed_pairs]
    if args.start is not None or args.end is not None:
        start = args.start if args.start is not None else frame_indices[0]
        end = args.end if args.end is not None else frame_indices[-1]
        frame_indices = [i for i in frame_indices if start <= i <= end]
        print(f"过滤后处理帧数: {len(frame_indices)}，范围 [{start}, {end}]")
    cam2_bboxes = load_bbox_dict(bbox_dir / "cam2_bboxes.json")
    cam4_bboxes = load_bbox_dict(bbox_dir / "cam4_bboxes.json")
    erp_h, erp_w = infer_erp_size(frame_indices, image_dir)

    with tqdm(frame_indices, desc="Compositing humans", unit="frame") as progress:
        for frame_idx in progress:
            process_frame(
                frame_idx=frame_idx,
                seg_dir=seg_dir,
                flow_dir=flow_dir,
                image_dir=image_dir,
                cam2_bboxes=cam2_bboxes,
                cam4_bboxes=cam4_bboxes,
                output_dir=output_dir,
                erp_h=erp_h,
                erp_w=erp_w,
            )


if __name__ == "__main__":
    main()


r'''
python src\script\test_function\debug037_composite_human.py `
  --seg_dir data_process\Output\20260311_riding\seg\equirect_rotate_crop `
  --flow_dir data_process\Output\20260311_riding\semantic_flow `
  --image_dir data_process\Output\20260311_riding\image\equirect_rotate `
  --bbox_dir data_process\Output\20260311_riding\image\equirect_rotate_crop `
  --output_dir data_process\Output\20260311_riding\composite_human `
  --reviewed data_process\Input\20260311_riding\review.json `
'''
