import os

import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt

'''
只要valid_mask的移动, 不是valid_mask的统统不移动
'''

ROTATED_CAM0_PATH = (
    r"data_process\Output\20260311_riding\segmentation\preview\auto_rotated\debug_sampled_cam_0_auto_rot.png"
)
ROTATED_CAM1_PATH = (
    r"data_process\Output\20260311_riding\segmentation\preview\auto_rotated\debug_sampled_cam_1_auto_rot.png"
)
FLOW_ROI_PATH = (
    r"data_process\Output\20260311_riding\segmentation\semantic_flow\semantic_flow_roi.npy"
)
CONFIDENCE_ROI_PATH = (
    r"data_process\Output\20260311_riding\segmentation\semantic_flow\semantic_flow_confidence_roi.npy"
)
ROI_BBOX_PATH = r"data_process\Output\20260311_riding\segmentation\semantic_flow\roi_bbox.txt"
OUTPUT_DIR = r"data_process\Output\20260311_riding\segmentation\semantic_flow"
DECAY_RADIUS = 200.0


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


def normalize_to_uint8(image: np.ndarray) -> np.ndarray:
    image = image.astype(np.float32)
    max_value = float(image.max())
    if max_value <= 0.0:
        return np.zeros(image.shape, dtype=np.uint8)
    return np.clip(image / max_value * 255.0, 0.0, 255.0).astype(np.uint8)


def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    rotated_cam0 = cv2.imread(ROTATED_CAM0_PATH)
    rotated_cam1 = cv2.imread(ROTATED_CAM1_PATH)
    semantic_flow_roi = np.load(FLOW_ROI_PATH)
    semantic_confidence_roi = np.load(CONFIDENCE_ROI_PATH)
    roi_bbox = load_bbox_txt(ROI_BBOX_PATH)

    if rotated_cam0 is None:
        raise FileNotFoundError(f"Failed to read canvas image: {ROTATED_CAM0_PATH}")
    if rotated_cam1 is None:
        raise FileNotFoundError(f"Failed to read source image: {ROTATED_CAM1_PATH}")
    if rotated_cam0.shape != rotated_cam1.shape:
        raise ValueError(
            f"Canvas/source size mismatch: {rotated_cam0.shape} vs {rotated_cam1.shape}"
        )

    roi_x1, roi_y1, roi_x2, roi_y2 = validate_bbox("roi_bbox", roi_bbox, rotated_cam1.shape)
    roi_h = roi_y2 - roi_y1
    roi_w = roi_x2 - roi_x1
    if semantic_flow_roi.shape != (roi_h, roi_w, 2):
        raise ValueError(
            "semantic_flow_roi shape mismatch: "
            f"{semantic_flow_roi.shape} vs expected {(roi_h, roi_w, 2)}"
        )
    if semantic_confidence_roi.shape != (roi_h, roi_w):
        raise ValueError(
            "semantic_confidence_roi shape mismatch: "
            f"{semantic_confidence_roi.shape} vs expected {(roi_h, roi_w)}"
        )

    erp_h, erp_w = rotated_cam1.shape[:2]
    full_human_mask = np.zeros((erp_h, erp_w), dtype=bool)
    full_human_mask[roi_y1:roi_y2, roi_x1:roi_x2] = semantic_confidence_roi > 0

    full_flow = np.zeros((erp_h, erp_w, 2), dtype=np.float32)
    full_flow[roi_y1:roi_y2, roi_x1:roi_x2] = semantic_flow_roi.astype(np.float32)

    if not full_human_mask.any():
        raise ValueError("No valid semantic flow pixels found inside roi_bbox")

    dist_full, nearest_indices = distance_transform_edt(
        ~full_human_mask,
        return_distances=True,
        return_indices=True,
    )
    nearest_y = nearest_indices[0]
    nearest_x = nearest_indices[1]
    propagated_flow = full_flow[nearest_y, nearest_x]

    weight = np.clip(1.0 - dist_full / DECAY_RADIUS, 0.0, 1.0).astype(np.float32)
    flow_with_decay = propagated_flow * weight[..., None]
    flow_with_decay[..., 0] = cv2.GaussianBlur(flow_with_decay[..., 0], (51, 51), 0)
    flow_with_decay[..., 1] = cv2.GaussianBlur(flow_with_decay[..., 1], (51, 51), 0)
    flow_with_decay[full_human_mask] = full_flow[full_human_mask]

    grid_x, grid_y = np.meshgrid(
        np.arange(erp_w, dtype=np.float32),
        np.arange(erp_h, dtype=np.float32),
    )
    map_x = grid_x + flow_with_decay[..., 0]
    map_y = grid_y + flow_with_decay[..., 1]
    source_preprocessed = cv2.remap(
        rotated_cam1,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )

    source_preprocessed_valid_mask = (
        (source_preprocessed.sum(axis=2) > 0).astype(np.uint8) * 255
    )
    source_preprocessed_overlay = cv2.addWeighted(
        rotated_cam0,
        0.5,
        source_preprocessed,
        0.5,
        0.0,
    )
    flow_magnitude = np.linalg.norm(flow_with_decay, axis=2)
    diag_flow_magnitude = normalize_to_uint8(flow_magnitude)
    diag_flow_weight = np.clip(weight * 255.0, 0.0, 255.0).astype(np.uint8)

    cv2.imwrite(os.path.join(OUTPUT_DIR, "source_preprocessed.png"), source_preprocessed)
    cv2.imwrite(
        os.path.join(OUTPUT_DIR, "source_preprocessed_valid_mask.png"),
        source_preprocessed_valid_mask,
    )
    cv2.imwrite(
        os.path.join(OUTPUT_DIR, "source_preprocessed_overlay.png"),
        source_preprocessed_overlay,
    )
    cv2.imwrite(os.path.join(OUTPUT_DIR, "flow_magnitude.png"), diag_flow_magnitude)
    cv2.imwrite(os.path.join(OUTPUT_DIR, "flow_weight.png"), diag_flow_weight)

    print(f"rotated_cam0 shape: {rotated_cam0.shape}")
    print(f"rotated_cam1 shape: {rotated_cam1.shape}")
    print(f"roi_bbox: {(roi_x1, roi_y1, roi_x2, roi_y2)}")
    print(f"erp size: {(erp_w, erp_h)}")
    print(f"valid flow pixels: {int(full_human_mask.sum())}")
    print(f"valid warped pixels: {int((source_preprocessed_valid_mask > 0).sum())}")
    print(f"flow magnitude max: {float(flow_magnitude.max()):.3f}")
    print(f"saved: {os.path.join(OUTPUT_DIR, 'source_preprocessed.png')}")
    print(f"saved: {os.path.join(OUTPUT_DIR, 'source_preprocessed_valid_mask.png')}")
    print(f"saved: {os.path.join(OUTPUT_DIR, 'source_preprocessed_overlay.png')}")
    print(f"saved: {os.path.join(OUTPUT_DIR, 'flow_magnitude.png')}")
    print(f"saved: {os.path.join(OUTPUT_DIR, 'flow_weight.png')}")


if __name__ == "__main__":
    main()
