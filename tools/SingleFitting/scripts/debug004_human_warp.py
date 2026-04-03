import os

import cv2
import numpy as np
from scipy import ndimage


ROTATED_CAM0_PATH = (
    r"data_process\Output\20260311_riding\preview\auto_rotated_seam\debug_sampled_cam_0_auto_rot.png"
)
ROTATED_CAM1_PATH = (
    r"data_process\Output\20260311_riding\preview\auto_rotated_seam\debug_sampled_cam_1_auto_rot.png"
)
FLOW_ROI_PATH = (
    r"data_process\Output\20260311_riding\semantic_flow\semantic_flow_roi.npy"
)
CONFIDENCE_ROI_PATH = (
    r"data_process\Output\20260311_riding\semantic_flow\semantic_flow_confidence_roi.npy"
)
ROI_BBOX_PATH = r"data_process\Output\20260311_riding\semantic_flow\roi_bbox.txt"
HUMAN_BBOX_PATH = r"data_process\Output\20260311_riding\semantic_flow\human_bbox.npy"
OUTPUT_DIR = r"data_process\Output\20260311_riding\semantic_flow"
DECAY_RADIUS = 120.0
FEATHER_PX = 15


def load_bbox_txt(path: str) -> np.ndarray:
    bbox = np.loadtxt(path, dtype=np.int32)
    if bbox.shape != (4,):
        raise ValueError(f"Invalid bbox txt shape at {path}: {bbox.shape}")
    return bbox


def validate_bbox(name: str, bbox: np.ndarray, image_shape: tuple[int, int, int]) -> tuple[int, int, int, int]:
    if bbox.shape != (4,):
        raise ValueError(f"{name} shape must be (4,), got {bbox.shape}")

    x1, y1, x2, y2 = [int(v) for v in bbox.tolist()]
    height, width = image_shape[:2]
    if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
        raise ValueError(
            f"{name} out of image bounds: {(x1, y1, x2, y2)} for image size {(width, height)}"
        )
    return x1, y1, x2, y2


def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    rotated_cam0 = cv2.imread(ROTATED_CAM0_PATH)
    rotated_cam1 = cv2.imread(ROTATED_CAM1_PATH)
    semantic_flow_roi = np.load(FLOW_ROI_PATH)
    semantic_confidence_roi = np.load(CONFIDENCE_ROI_PATH)
    roi_bbox = load_bbox_txt(ROI_BBOX_PATH)
    human_bbox = np.load(HUMAN_BBOX_PATH).astype(np.int32)

    if rotated_cam0 is None:
        raise FileNotFoundError(f"Failed to read canvas image: {ROTATED_CAM0_PATH}")
    if rotated_cam1 is None:
        raise FileNotFoundError(f"Failed to read source image: {ROTATED_CAM1_PATH}")
    if rotated_cam0.shape != rotated_cam1.shape:
        raise ValueError(
            f"Canvas/source size mismatch: {rotated_cam0.shape} vs {rotated_cam1.shape}"
        )

    roi_x1, roi_y1, roi_x2, roi_y2 = validate_bbox("roi_bbox", roi_bbox, rotated_cam0.shape)
    human_x1, human_y1, human_x2, human_y2 = validate_bbox(
        "human_bbox", human_bbox, rotated_cam0.shape
    )

    roi_h, roi_w = semantic_flow_roi.shape[:2]
    if semantic_flow_roi.shape != (roi_y2 - roi_y1, roi_x2 - roi_x1, 2):
        raise ValueError(
            "semantic_flow_roi shape mismatch: "
            f"{semantic_flow_roi.shape} vs expected {(roi_y2 - roi_y1, roi_x2 - roi_x1, 2)}"
        )
    if semantic_confidence_roi.shape != (roi_h, roi_w):
        raise ValueError(
            "semantic_confidence_roi shape mismatch: "
            f"{semantic_confidence_roi.shape} vs expected {(roi_h, roi_w)}"
        )

    overlap_x1 = max(human_x1, roi_x1)
    overlap_y1 = max(human_y1, roi_y1)
    overlap_x2 = min(human_x2, roi_x2)
    overlap_y2 = min(human_y2, roi_y2)
    if overlap_x1 >= overlap_x2 or overlap_y1 >= overlap_y2:
        raise ValueError("human_bbox does not overlap roi_bbox")

    human_w = human_x2 - human_x1
    human_h = human_y2 - human_y1
    flow_full = np.zeros((human_h, human_w, 2), dtype=np.float32)
    confidence_full = np.zeros((human_h, human_w), dtype=semantic_confidence_roi.dtype)

    roi_local_x1 = overlap_x1 - roi_x1
    roi_local_y1 = overlap_y1 - roi_y1
    roi_local_x2 = overlap_x2 - roi_x1
    roi_local_y2 = overlap_y2 - roi_y1

    human_local_x1 = overlap_x1 - human_x1
    human_local_y1 = overlap_y1 - human_y1
    human_local_x2 = overlap_x2 - human_x1
    human_local_y2 = overlap_y2 - human_y1

    flow_full[human_local_y1:human_local_y2, human_local_x1:human_local_x2] = (
        semantic_flow_roi[roi_local_y1:roi_local_y2, roi_local_x1:roi_local_x2].astype(np.float32)
    )
    confidence_full[human_local_y1:human_local_y2, human_local_x1:human_local_x2] = (
        semantic_confidence_roi[roi_local_y1:roi_local_y2, roi_local_x1:roi_local_x2]
    )

    human_mask = confidence_full > 0
    if not human_mask.any():
        raise ValueError("No valid human flow pixels found inside human_bbox")

    print(f"flow dx mean: {flow_full[..., 0].mean():.2f}, std: {flow_full[..., 0].std():.2f}")
    print(f"flow dy mean: {flow_full[..., 1].mean():.2f}, std: {flow_full[..., 1].std():.2f}")
    print(f"人体区域 flow dx mean: {flow_full[human_mask, 0].mean():.2f}")
    print(f"人体区域 flow dy mean: {flow_full[human_mask, 1].mean():.2f}")
    
    # For each pixel, fetch the nearest valid human-flow pixel index.
    dist_to_human, nearest_indices = ndimage.distance_transform_edt(
        ~human_mask,
        return_distances=True,
        return_indices=True,
    )
    nearest_y = nearest_indices[0]
    nearest_x = nearest_indices[1]
    propagated_flow = flow_full[nearest_y, nearest_x]

    # Preserve human flow as-is, while background inherits nearest valid flow
    # and fades quadratically with distance.
    weight = np.clip(1.0 - dist_to_human / DECAY_RADIUS, 0.0, 1.0).astype(np.float32) ** 2
    flow_with_decay = propagated_flow * weight[..., None]
    flow_with_decay[..., 0] = cv2.GaussianBlur(flow_with_decay[..., 0], (51, 51), 0)
    flow_with_decay[..., 1] = cv2.GaussianBlur(flow_with_decay[..., 1], (51, 51), 0)
    flow_with_decay[human_mask] = flow_full[human_mask]

    human_crop_canvas = rotated_cam0[human_y1:human_y2, human_x1:human_x2].copy()
    human_crop_source = rotated_cam1[human_y1:human_y2, human_x1:human_x2].copy()

    grid_x, grid_y = np.meshgrid(
        np.arange(human_w, dtype=np.float32),
        np.arange(human_h, dtype=np.float32),
    )
    map_x = grid_x + flow_with_decay[..., 0]
    map_y = grid_y + flow_with_decay[..., 1]

    semantic_warped_human = cv2.remap(
        human_crop_source,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )

    mask = np.ones((human_h, human_w), dtype=np.float32)

    mask3 = mask[..., None]
    human_crop_original = rotated_cam1[human_y1:human_y2, human_x1:human_x2].copy()
    
    blended = semantic_warped_human.astype(np.float32) * mask3 + human_crop_original.astype(
        np.float32
    ) * (1.0 - mask3)

    source_preprocessed = rotated_cam1.copy()
    source_preprocessed[human_y1:human_y2, human_x1:human_x2] = blended.astype(np.uint8)
    human_overlay = cv2.addWeighted(human_crop_canvas, 0.5, semantic_warped_human, 0.5, 0.0)

    valid_rows = human_crop_source.sum(axis=(1, 2)) > 10
    y_threshold_local = int(np.where(valid_rows)[0].min()) if valid_rows.any() else human_h
    region = source_preprocessed[human_y1:human_y2, human_x1:human_x2].copy()
    fill_mask = np.zeros((human_h, human_w), dtype=np.uint8)
    fill_mask[y_threshold_local:, :] = 255
    black_pixels = ((region.sum(axis=2) < 10).astype(np.uint8)) * 255
    final_mask = cv2.bitwise_and(fill_mask, black_pixels)
    if np.any(final_mask):
        filled = cv2.inpaint(region, final_mask, inpaintRadius=10, flags=cv2.INPAINT_TELEA)
        source_preprocessed[human_y1:human_y2, human_x1:human_x2] = filled

    cv2.imwrite(os.path.join(OUTPUT_DIR, "diag_human_crop_canvas.png"), human_crop_canvas)
    cv2.imwrite(os.path.join(OUTPUT_DIR, "diag_human_crop_source.png"), human_crop_source)
    cv2.imwrite(os.path.join(OUTPUT_DIR, "diag_human_warped.png"), semantic_warped_human)
    cv2.imwrite(os.path.join(OUTPUT_DIR, "diag_human_overlay.png"), human_overlay)
    cv2.imwrite(os.path.join(OUTPUT_DIR, "diag_inpaint_mask.png"), final_mask)
    cv2.imwrite(os.path.join(OUTPUT_DIR, "diag_source_preprocessed.png"), source_preprocessed)

    print(f"rotated_cam0 shape: {rotated_cam0.shape}")
    print(f"rotated_cam1 shape: {rotated_cam1.shape}")
    print(f"roi_bbox: {(roi_x1, roi_y1, roi_x2, roi_y2)}")
    print(f"human_bbox: {(human_x1, human_y1, human_x2, human_y2)}")
    print(f"overlap_bbox: {(overlap_x1, overlap_y1, overlap_x2, overlap_y2)}")
    print(f"human crop size: {(human_w, human_h)}")
    print(f"human mask pixels: {int(human_mask.sum())}")
    if human_mask.any():
        flow_mag = np.linalg.norm(flow_with_decay[human_mask > 0], axis=1)
        print(
            "flow_with_decay magnitude: "
            f"min={float(flow_mag.min()):.3f}, "
            f"max={float(flow_mag.max()):.3f}, "
            f"mean={float(flow_mag.mean()):.3f}"
        )
    print(f"saved: {os.path.join(OUTPUT_DIR, 'diag_human_crop_canvas.png')}")
    print(f"saved: {os.path.join(OUTPUT_DIR, 'diag_human_crop_source.png')}")
    print(f"saved: {os.path.join(OUTPUT_DIR, 'diag_human_warped.png')}")
    print(f"saved: {os.path.join(OUTPUT_DIR, 'diag_human_overlay.png')}")
    print(f"saved: {os.path.join(OUTPUT_DIR, 'diag_inpaint_mask.png')}")
    print(f"saved: {os.path.join(OUTPUT_DIR, 'diag_source_preprocessed.png')}")


if __name__ == "__main__":
    main()
