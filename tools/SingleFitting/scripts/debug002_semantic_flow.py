import numpy as np
import cv2
import os
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

SRC_ROOT = Path(__file__).resolve().parents[3]  # 到 Panorama-stitch-lab 根目录
sys.path.insert(0, str(SRC_ROOT / "src"))
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
from single_fitting.debug_shared import DS_PARAMS, R_CAM4, erp_project_consistent
from flow.visibility import (
    load_smplx_faces_and_verts,
    get_visible_vertices,
    compute_semantic_flow,
)
from lib_pano.flow.roi_process import (
    extract_overlap_roi,
    soft_overlap_from_masks,
)

pkl_cam2 = "tools/SingleFitting/outputs/fisheye_smplx_cam2/fisheye_cam2_s0_i0_0_seq.pkl"
pkl_cam4 = "tools/SingleFitting/outputs/fisheye_smplx_cam4/fisheye_cam4_s0_i0_0_seq.pkl"
# 旋转后的经纬图和 valid mask（debug012 的输出）
rotated_erp_cam0_path = "data_process\\Output\\20260311_riding\\segmentation\\preview\\auto_rotated\\debug_sampled_cam_0_auto_rot.png"
rotated_erp_cam1_path = "data_process\\Output\\20260311_riding\\segmentation\\preview\\auto_rotated\\debug_sampled_cam_1_auto_rot.png"
rotated_mask_cam0_path = "data_process\\Output\\20260311_riding\\segmentation\\preview\\auto_rotated\\debug_valid_mask_cam_0_auto_rot.png"
rotated_mask_cam1_path = "data_process\\Output\\20260311_riding\\segmentation\\preview\\auto_rotated\\debug_valid_mask_cam_1_auto_rot.png"
# 未旋转的 valid mask（debug004 的输出，用于计算 rotation_matrix）
mask_cam0_path = "data_process\\Output\\20260311_riding\\segmentation\\preview\\debug_valid_mask_cam_0.png"
mask_cam1_path = "data_process\\Output\\20260311_riding\\segmentation\\preview\\debug_valid_mask_cam_1.png"
output_dir = "data_process\\Output\\20260311_riding\\segmentation\\semantic_flow"
os.makedirs(output_dir, exist_ok=True)

verts_cam2, faces = load_smplx_faces_and_verts(pkl_cam2)
verts_cam4, _ = load_smplx_faces_and_verts(pkl_cam4)
verts_cam2 = verts_cam2[0]  # 取第 0 帧
verts_cam4 = verts_cam4[0]
print(f"verts_cam2: {verts_cam2.shape}, verts_cam4: {verts_cam4.shape}")

verts_world_A = verts_cam2  # cam2 = 世界坐标系
verts_world_B = verts_cam4 @ R_CAM4  # cam4 转世界，T=0
print(f"verts_world_A range: {verts_world_A.min(axis=0)} ~ {verts_world_A.max(axis=0)}")
print(f"verts_world_B range: {verts_world_B.min(axis=0)} ~ {verts_world_B.max(axis=0)}")


_, visible_mask_B = get_visible_vertices(verts_cam4, faces, DS_PARAMS, 2880, 2880)
print(f"cam4 visible vertices: {visible_mask_B.sum()}")

mask_0 = cv2.imread(mask_cam0_path, cv2.IMREAD_GRAYSCALE)
mask_1 = cv2.imread(mask_cam1_path, cv2.IMREAD_GRAYSCALE)
rotation_matrix = np.array([1, 0, 0, 0, 0, 1, 0, -1, 0]).reshape(3, 3)
print(f"rotation_matrix:\n{rotation_matrix}")


rotated_cam0 = cv2.imread(rotated_erp_cam0_path)
rotated_cam1 = cv2.imread(rotated_erp_cam1_path)
rotated_mask0 = cv2.imread(rotated_mask_cam0_path, cv2.IMREAD_GRAYSCALE)
rotated_mask1 = cv2.imread(rotated_mask_cam1_path, cv2.IMREAD_GRAYSCALE)
erp_h, erp_w = rotated_cam0.shape[:2]
print(f"ERP size: {erp_w} x {erp_h}")


overlap_mask = soft_overlap_from_masks(
    (rotated_mask0 > 0).astype(np.uint8) * 255,
    (rotated_mask1 > 0).astype(np.uint8) * 255,
    extend_y=180,
)
_, _, roi_bbox = extract_overlap_roi(rotated_cam1, rotated_cam0, overlap_mask)
x1, y1, x2, y2 = roi_bbox
print(f"roi_bbox: {roi_bbox}, size: {x2-x1} x {y2-y1}")
with open(os.path.join(output_dir, "roi_bbox.txt"), "w") as f:
    f.write(f"{x1} {y1} {x2} {y2}\n")
    
# 不旋转直接投影
pts_2d_no_rot = erp_project_consistent(verts_world_A, erp_w, erp_h)
diag_no_rot = cv2.imread("data_process\\Output\\20260311_riding\\segmentation\\preview\\debug_sampled_cam_0.png")  # 未旋转的 cam0
if diag_no_rot is not None:
    print(f"Loaded diag_no_rot image: {diag_no_rot.shape}")
    for u, v in pts_2d_no_rot:
        iu, iv = int(round(u)), int(round(v))
        if 0 <= iu < erp_w and 0 <= iv < erp_h:
            cv2.circle(diag_no_rot, (iu, iv), 2, (0, 255, 0), -1)
    cv2.imwrite(os.path.join(output_dir, "diag_mesh_on_erp_no_rot.png"), diag_no_rot)

# 诊断：检查 mesh 投影到旋转后经纬图的位置
from flow.visibility import project_verts_to_rotated_erp
pts_2d_A_erp = project_verts_to_rotated_erp(verts_world_A, rotation_matrix, erp_h, erp_w)
pts_2d_B_erp = project_verts_to_rotated_erp(verts_world_B, rotation_matrix, erp_h, erp_w)
print(f"cam2 verts on rotated ERP: u=[{pts_2d_A_erp[:,0].min():.1f}, {pts_2d_A_erp[:,0].max():.1f}], v=[{pts_2d_A_erp[:,1].min():.1f}, {pts_2d_A_erp[:,1].max():.1f}]")
print(f"cam4 verts on rotated ERP: u=[{pts_2d_B_erp[:,0].min():.1f}, {pts_2d_B_erp[:,0].max():.1f}], v=[{pts_2d_B_erp[:,1].min():.1f}, {pts_2d_B_erp[:,1].max():.1f}]")
print(f"ROI: y=[{y1}, {y2}]")

# 诊断：在旋转后经纬图上画 cam2 mesh 的投影
diag_img = rotated_cam0.copy()
for u, v in pts_2d_A_erp:
    iu, iv = int(round(u)), int(round(v))
    if 0 <= iu < erp_w and 0 <= iv < erp_h:
        cv2.circle(diag_img, (iu, iv), 2, (0, 255, 0), -1)
cv2.imwrite(os.path.join(output_dir, "diag_mesh_on_erp.png"), diag_img)

diag_img_cam4 = rotated_cam1.copy()
for u, v in pts_2d_B_erp:
    iu, iv = int(round(u)), int(round(v))
    if 0 <= iu < erp_w and 0 <= iv < erp_h:
        cv2.circle(diag_img_cam4, (iu, iv), 2, (0, 255, 0), -1)
cv2.imwrite(os.path.join(output_dir, "diag_mesh_on_erp_cam4.png"), diag_img_cam4)

flow_roi, confidence_roi= compute_semantic_flow(
    verts_world_A=verts_world_A,
    verts_world_B=verts_world_B,
    faces=faces,
    erp_h=erp_h,
    erp_w=erp_w,
    rotation_matrix=rotation_matrix,
    roi_bbox=roi_bbox,
    visible_mask_B=visible_mask_B,
)
print(f"flow_roi shape: {flow_roi.shape}")
print(f"confidence_roi shape: {confidence_roi.shape}")
print(f"pixels with flow (confidence>0): {(confidence_roi > 0).sum()}")
print(f"high confidence (=2): {(confidence_roi == 2).sum()}")
print(f"low confidence (=1): {(confidence_roi == 1).sum()}")
np.save(os.path.join(output_dir, "semantic_flow_roi.npy"), flow_roi)
np.save(os.path.join(output_dir, "semantic_flow_confidence_roi.npy"), confidence_roi)
print(f"saved ROI flow npy: {os.path.join(output_dir, 'semantic_flow_roi.npy')}")
print(f"saved ROI confidence npy: {os.path.join(output_dir, 'semantic_flow_confidence_roi.npy')}")
# flow 统计（只看有 flow 的像素）
has_flow = confidence_roi > 0
if has_flow.any():
    flow_mag = np.linalg.norm(flow_roi[has_flow], axis=1)
    print(f"flow magnitude: min={flow_mag.min():.2f}, max={flow_mag.max():.2f}, mean={flow_mag.mean():.2f}")
    
flow_mag_map = np.linalg.norm(flow_roi, axis=2)
flow_angle = np.arctan2(flow_roi[..., 1], flow_roi[..., 0])
# angle :H, [0, 179] for OpenCV HSV hue range
hsv = np.zeros((flow_roi.shape[0], flow_roi.shape[1], 3), dtype=np.uint8)
hsv[..., 0] = ((flow_angle + np.pi) / (2 * np.pi) * 179).astype(np.uint8) # [0, 179] for OpenCV HSV hue range
hsv[..., 1] = 255  # S: 255 for full saturation
max_mag = max(flow_mag_map.max(), 1e-6)
hsv[..., 2] = (np.clip(flow_mag_map / max_mag, 0, 1) * 255).astype(np.uint8)  # V: flow magnitude normalized to [0, 255]
flow_vis = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
flow_vis[confidence_roi == 0] = 0
cv2.imwrite(os.path.join(output_dir, "semantic_flow_color.png"), flow_vis)

conf_vis = np.zeros((confidence_roi.shape[0], confidence_roi.shape[1], 3), dtype=np.uint8)
conf_vis[confidence_roi == 1] = [255, 0, 0]  # 蓝
conf_vis[confidence_roi == 2] = [0, 255, 0]  # 绿
cv2.imwrite(os.path.join(output_dir, "semantic_flow_confidence.png"), conf_vis)

crop_canvas = rotated_cam0[y1:y2, x1:x2].copy()
arrow_vis = crop_canvas.copy()
step = 20
for r in range(0, flow_roi.shape[0], step):
    for c in range(0, flow_roi.shape[1], step):
        if confidence_roi[r, c] == 0:
            continue
        dx, dy = flow_roi[r, c]
        color = (0, 255, 0) if confidence_roi[r, c] == 2 else (255, 0, 0)
        cv2.arrowedLine(arrow_vis, (c, r), (int(c + dx), int(r + dy)), color, 1, tipLength=0.3)
cv2.imwrite(os.path.join(output_dir, "semantic_flow_arrows.png"), arrow_vis)

all_pts_2d = np.concatenate([pts_2d_A_erp, pts_2d_B_erp], axis=0)
padding_x = 20
padding_y = 10
human_x1 = max(0, int(np.floor(all_pts_2d[:, 0].min())) - padding_x)
human_y1 = max(0, int(np.floor(all_pts_2d[:, 1].min())) - padding_y)
human_x2 = min(erp_w, int(np.ceil(all_pts_2d[:, 0].max())) + padding_x)
human_y2 = min(erp_h, int(np.ceil(all_pts_2d[:, 1].max())) + padding_y)
human_bbox = (human_x1, human_y1, human_x2, human_y2)
print(f"human_bbox: {human_bbox}")

np.save(os.path.join(output_dir, "human_bbox.npy"), np.array(human_bbox, dtype=np.int32))

diag_human_bbox = rotated_cam0.copy()
cv2.rectangle(
    diag_human_bbox,
    (human_x1, human_y1),
    (human_x2 - 1, human_y2 - 1),
    (0, 0, 255),
    2,
)
cv2.imwrite(os.path.join(output_dir, "diag_human_bbox.png"), diag_human_bbox)
