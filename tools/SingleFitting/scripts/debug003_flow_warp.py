import os
from pathlib import Path

import cv2
import numpy as np


shard_path = r"data_process\Output\20260311_riding\preview\auto_rotated_seam\debug_sampled_cam_1_auto_rot.png"
canvas_path = r"data_process\Output\20260311_riding\preview\auto_rotated_seam\debug_sampled_cam_0_auto_rot.png"
flow_roi_path = r"data_process\Output\20260311_riding\semantic_flow\semantic_flow_roi.npy"
confidence_roi_path = r"data_process\Output\20260311_riding\semantic_flow\semantic_flow_confidence_roi.npy"
roi_bbox_path = r"data_process\Output\20260311_riding\semantic_flow\roi_bbox.txt"
output_dir = r"data_process\Output\20260311_riding\semantic_flow"

os.makedirs(output_dir, exist_ok=True)

source = cv2.imread(shard_path)
canvas = cv2.imread(canvas_path)
flow_roi = np.load(flow_roi_path)
confidence_roi = np.load(confidence_roi_path)
roi_bbox = np.loadtxt(roi_bbox_path, dtype=np.int32)

if source is None:
    raise FileNotFoundError(f"Failed to read source: {shard_path}")
if canvas is None:
    raise FileNotFoundError(f"Failed to read canvas: {canvas_path}")

print(f"source shape: {source.shape}")
print(f"canvas shape: {canvas.shape}")
print(f"flow_roi shape: {flow_roi.shape}")
print(f"confidence_roi shape: {confidence_roi.shape}")
print(f"roi_bbox: {roi_bbox.tolist()}")

x1, y1, x2, y2 = roi_bbox.tolist()
if roi_bbox.shape != (4,):
    raise ValueError(f"Invalid roi_bbox shape: {roi_bbox.shape}, expected (4,)")
if not (0 <= x1 < x2 <= canvas.shape[1] and 0 <= y1 < y2 <= canvas.shape[0]):
    raise ValueError(f"roi_bbox out of canvas bounds: {(x1, y1, x2, y2)}")
if source.shape[:2] != canvas.shape[:2]:
    raise ValueError(
        f"source/canvas size mismatch: {source.shape[:2]} vs {canvas.shape[:2]}"
    )

crop_source = source[y1:y2, x1:x2].copy()
crop_canvas = canvas[y1:y2, x1:x2].copy()

roi_h, roi_w = flow_roi.shape[:2]
assert crop_source.shape[:2] == (roi_h, roi_w), (
    f"crop_source size mismatch: got {crop_source.shape[:2]}, expected {(roi_h, roi_w)}"
)
assert crop_canvas.shape[:2] == (roi_h, roi_w), (
    f"crop_canvas size mismatch: got {crop_canvas.shape[:2]}, expected {(roi_h, roi_w)}"
)
assert confidence_roi.shape == (roi_h, roi_w), (
    f"confidence_roi size mismatch: got {confidence_roi.shape}, expected {(roi_h, roi_w)}"
)

grid_x, grid_y = np.meshgrid(
    np.arange(roi_w, dtype=np.float32),
    np.arange(roi_h, dtype=np.float32),
)
map_x = grid_x + flow_roi[..., 0].astype(np.float32)
map_y = grid_y + flow_roi[..., 1].astype(np.float32)
print(
    f"map_x range: [{float(map_x.min()):.2f}, {float(map_x.max()):.2f}], "
    f"map_y range: [{float(map_y.min()):.2f}, {float(map_y.max()):.2f}]"
)

semantic_warped_crop = cv2.remap(
    crop_source,
    map_x,
    map_y,
    interpolation=cv2.INTER_LINEAR,
    borderMode=cv2.BORDER_CONSTANT,
    borderValue=0,
)

mesh_mask = confidence_roi > 0
warped_valid_mask = semantic_warped_crop.sum(axis=2) > 10
valid_mask = mesh_mask & warped_valid_mask
print(f"mesh_mask pixels: {int(mesh_mask.sum())}")
print(f"warped_valid_mask pixels: {int(warped_valid_mask.sum())}")
print(f"valid_mask pixels: {int(valid_mask.sum())}")

result = canvas.copy()
result_roi = result[y1:y2, x1:x2]
result_roi[valid_mask] = semantic_warped_crop[valid_mask]
replaced_path = os.path.join(output_dir, "debug003_flow_warp_replace_human.png")
cv2.imwrite(replaced_path, result)
print(f"saved replaced result: {replaced_path}")
