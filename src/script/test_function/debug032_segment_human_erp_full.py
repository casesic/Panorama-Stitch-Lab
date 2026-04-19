from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm
import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")


MASK_ALPHA = 0.5
SEG_COLOR = (0, 120, 255)


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
        description="Segment humans from full ERP images and save mask, RGBA, and visualization outputs."
    )
    parser.add_argument(
        "--image_dir",
        type=str,
        required=True,
        help="Full ERP image root directory containing cam2/ and cam4/ subdirectories",
    )
    parser.add_argument(
        "--semantic_flow_dir",
        type=str,
        required=True,
        help="debug031 output root directory containing frame_{idx:05d}/human_bbox.npy",
    )
    parser.add_argument("--output_dir", type=str, required=True, help="Output root directory")
    parser.add_argument("--start_frame", type=int, required=True, help="Start frame index (inclusive)")
    parser.add_argument("--end_frame", type=int, required=True, help="End frame index (inclusive)")
    parser.add_argument(
        "--sam_model",
        type=str,
        default="sam2_b.pt",
        help='SAM2 model name, default "sam2_b.pt"',
    )
    parser.add_argument(
        "--yolo_model",
        type=str,
        default="yolo11x-seg.pt",
        help='YOLO11-seg model name, default "yolo11x-seg.pt"',
    )
    return parser.parse_args()


def render_segmentation(
    image_bgr: np.ndarray,
    mask: np.ndarray,
    prompt_bbox: list[float] | None = None,
    alpha: float = MASK_ALPHA,
    color: tuple[int, int, int] = SEG_COLOR,
) -> np.ndarray:
    h, w = image_bgr.shape[:2]
    if mask.shape != (h, w):
        mask = cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)

    seg = image_bgr.copy()
    overlay = seg.copy()
    overlay[mask] = color
    seg = cv2.addWeighted(overlay, alpha, seg, 1 - alpha, 0)

    contours, _ = cv2.findContours(mask.astype(np.uint8) * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(seg, contours, -1, (0, 255, 255), 2)
    if prompt_bbox is not None:
        x1, y1, x2, y2 = [int(v) for v in prompt_bbox]
        cv2.rectangle(seg, (x1, y1), (x2, y2), (0, 255, 0), 2)
    return seg


def load_models(sam_model_name: str, yolo_model_name: str):
    from ultralytics import SAM, YOLO

    return SAM(sam_model_name), YOLO(yolo_model_name)


def _result_masks_to_bool_list(result, image_shape) -> list[np.ndarray]:
    if result.masks is None or result.masks.data is None or len(result.masks.data) == 0:
        return []

    masks = result.masks.data.detach().cpu().numpy()
    h, w = image_shape[:2]
    out: list[np.ndarray] = []
    for mask in masks:
        if mask.shape != (h, w):
            mask = cv2.resize(mask.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
        out.append(mask > 0.5)
    return out


def _pick_largest_mask(
    masks: list[np.ndarray],
    class_ids: list[int] | None = None,
    target_class_id: int | None = None,
) -> np.ndarray | None:
    best_mask = None
    best_area = -1
    for idx, mask in enumerate(masks):
        if class_ids is not None and target_class_id is not None and class_ids[idx] != target_class_id:
            continue
        area = int(mask.sum())
        if area > best_area:
            best_area = area
            best_mask = mask
    return best_mask


def _sample_points_from_mask(mask: np.ndarray, n_points: int, seed: int) -> list[list[int]]:
    coords = np.argwhere(mask)
    if len(coords) == 0 or n_points <= 0:
        return []
    rng = np.random.default_rng(seed)
    count = min(n_points, len(coords))
    picked = coords[rng.choice(len(coords), size=count, replace=False)]
    return [[int(x), int(y)] for y, x in picked]


def segment_with_yolo_sam_refine(
    sam_model,
    yolo_model,
    image_crop_bgr: np.ndarray,
) -> tuple[np.ndarray | None, list[list[int]], list[int]]:
    PERSON_CLASS_ID = 0
    BICYCLE_CLASS_ID = 1

    yolo_results = yolo_model(image_crop_bgr, verbose=False)
    if not yolo_results:
        return None, [], []

    result = yolo_results[0]
    masks = _result_masks_to_bool_list(result, image_crop_bgr.shape)
    if not masks:
        return None, [], []

    if result.boxes is not None and result.boxes.cls is not None:
        class_ids = [int(x) for x in result.boxes.cls.detach().cpu().numpy().tolist()]
    else:
        class_ids = [0] * len(masks)

    rough_person_mask = _pick_largest_mask(masks, class_ids=class_ids, target_class_id=PERSON_CLASS_ID)
    if rough_person_mask is None:
        rough_person_mask = _pick_largest_mask(masks)
    if rough_person_mask is None:
        return None, [], []

    positive_points = _sample_points_from_mask(rough_person_mask, n_points=10, seed=int(rough_person_mask.sum()) + 17)
    bicycle_mask = _pick_largest_mask(masks, class_ids=class_ids, target_class_id=BICYCLE_CLASS_ID)
    negative_points = _sample_points_from_mask(bicycle_mask, n_points=6, seed=int(rough_person_mask.sum())) if bicycle_mask is not None else []
    points = positive_points + negative_points
    labels = [1] * len(positive_points) + [0] * len(negative_points)

    ys, xs = np.where(rough_person_mask)
    if len(xs) == 0 or len(ys) == 0:
        return None, [], []
    x, y, w, h = cv2.boundingRect(np.stack([xs, ys], axis=1).astype(np.int32))
    person_bbox = [int(x), int(y), int(x + w), int(y + h)]

    image_rgb = cv2.cvtColor(image_crop_bgr, cv2.COLOR_BGR2RGB)
    sam_results = sam_model(
        image_rgb,
        bboxes=[person_bbox],
        points=[points],
        labels=[labels],
        verbose=False,
    )
    if not sam_results:
        return rough_person_mask, points, labels

    refined_masks = _result_masks_to_bool_list(sam_results[0], image_crop_bgr.shape)
    refined_mask = _pick_largest_mask(refined_masks)
    if refined_mask is None:
        refined_mask = rough_person_mask
    return refined_mask, points, labels


def load_human_bbox(semantic_flow_dir: Path, frame_idx: int) -> tuple[int, int, int, int] | None:
    bbox_path = semantic_flow_dir / f"frame_{frame_idx:05d}" / "human_bbox.npy"
    if not bbox_path.exists():
        return None

    bbox = np.load(str(bbox_path))
    bbox = np.asarray(bbox).reshape(-1)
    if bbox.size < 4:
        return None
    x1, y1, x2, y2 = [int(round(float(v))) for v in bbox[:4]]
    return x1, y1, x2, y2


def _process_camera_frame(
    sam_model,
    yolo_model,
    image_dir: Path,
    semantic_flow_dir: Path,
    output_dir: Path,
    cam_id: str,
    frame_idx: int,
) -> None:
    image_path = image_dir / cam_id / f"{frame_idx:05d}.jpg"
    image_bgr = cv2.imread(str(image_path))
    if image_bgr is None:
        print(f"Warning: full ERP image not found, skipping {cam_id} frame {frame_idx:05d}")
        return

    bbox_xyxy = load_human_bbox(semantic_flow_dir, frame_idx)
    if bbox_xyxy is None:
        print(f"Warning: human bbox not found, skipping {cam_id} frame {frame_idx:05d}")
        return

    h, w = image_bgr.shape[:2]
    x1, y1, x2, y2 = bbox_xyxy
    x1 = max(0, min(w, int(x1)))
    y1 = max(0, min(h, int(y1)))
    x2 = max(0, min(w, int(x2)))
    y2 = max(0, min(h, int(y2)))
    if x2 <= x1 or y2 <= y1:
        print(f"Warning: invalid human bbox, skipping {cam_id} frame {frame_idx:05d}")
        return

    image_crop_bgr = image_bgr[y1:y2, x1:x2]
    crop_mask, negative_points, negative_labels = segment_with_yolo_sam_refine(
        sam_model,
        yolo_model,
        image_crop_bgr,
    )
    if crop_mask is None:
        print(f"Warning: failed to segment human, skipping {cam_id} frame {frame_idx:05d}")
        return

    mask = np.zeros((h, w), dtype=bool)
    mask[y1:y2, x1:x2] = crop_mask

    mask_dir = output_dir / cam_id / "mask"
    rgba_dir = output_dir / cam_id / "rgba"
    vis_dir = output_dir / cam_id / "vis"
    mask_dir.mkdir(parents=True, exist_ok=True)
    rgba_dir.mkdir(parents=True, exist_ok=True)
    vis_dir.mkdir(parents=True, exist_ok=True)

    mask_uint8 = mask.astype(np.uint8) * 255
    cv2.imwrite(str(mask_dir / f"{frame_idx:05d}.png"), mask_uint8)

    rgba_image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2BGRA)
    rgba_image[~mask, 3] = 0
    cv2.imwrite(str(rgba_dir / f"{frame_idx:05d}.png"), rgba_image)

    vis_image = render_segmentation(image_bgr, mask, prompt_bbox=[x1, y1, x2, y2])
    for (px, py), label in zip(negative_points, negative_labels):
        color = (0, 0, 255) if label == 0 else (0, 255, 0)
        cv2.circle(vis_image, (int(px + x1), int(py + y1)), 4, color, -1)
    cv2.imwrite(
        str(vis_dir / f"{frame_idx:05d}.jpg"),
        vis_image,
        [cv2.IMWRITE_JPEG_QUALITY, 92],
    )


def main() -> None:
    _ensure_paths()
    args = parse_args()

    if args.end_frame < args.start_frame:
        raise ValueError("--end_frame must be >= --start_frame")

    image_dir = Path(args.image_dir)
    semantic_flow_dir = Path(args.semantic_flow_dir)
    output_dir = Path(args.output_dir)
    sam_model, yolo_model = load_models(args.sam_model, args.yolo_model)

    frame_indices = list(range(args.start_frame, args.end_frame + 1))
    total_steps = len(frame_indices) * 2

    with tqdm(total=total_steps, desc="Segmenting full ERP frames", unit="frame") as progress:
        for frame_idx in frame_indices:
            _process_camera_frame(
                sam_model=sam_model,
                yolo_model=yolo_model,
                image_dir=image_dir,
                semantic_flow_dir=semantic_flow_dir,
                output_dir=output_dir,
                cam_id="cam2",
                frame_idx=frame_idx,
            )
            progress.update(1)

            _process_camera_frame(
                sam_model=sam_model,
                yolo_model=yolo_model,
                image_dir=image_dir,
                semantic_flow_dir=semantic_flow_dir,
                output_dir=output_dir,
                cam_id="cam4",
                frame_idx=frame_idx,
            )
            progress.update(1)


if __name__ == "__main__":
    main()

r'''
python src\script\test_function\debug032_segment_human_erp_full.py `
  --image_dir data_process\Output\20260311_riding\image\equirect_rotate `
  --semantic_flow_dir data_process\Output\20260311_riding\semantic_flow `
  --output_dir data_process\Output\20260311_riding\seg_erp_full `
  --start_frame 513 `
  --end_frame 513 `
  --sam_model sam2_b.pt `
  --yolo_model yolo11x-seg.pt
'''
