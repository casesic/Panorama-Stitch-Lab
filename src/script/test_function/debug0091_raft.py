import cv2
import config_script

from lib_pano.flow.raft_align import RAFTAligner
from lib_pano.flow.pixelstitch_align import PixelStitchAligner
# from lib_pano.flow.gmflow_align import GMFlowAligner
from lib_pano.io.flow_utils import blend_priority, get_roi_overlap_mask, extract_overlap_roi, compose_final_panorama
from lib_pano.vis.visualize_features import visualize_optical_flow_color, visualize_optical_flow_arrows
import lib_pano.local_setting as ls
import os
import numpy as np
from lib_pano.flow.mesh_warp_utils import MeshGridWarper

# 1. 准备全景图层
print("Initializing...")
source_pano_path = os.path.join(ls.DATA_OUTPUT_DIR, 'preview', 'debug_sampled_cam_1.png') 
target_pano_path = os.path.join(ls.DATA_OUTPUT_DIR, 'preview', 'debug_sampled_cam_0.png')

# 读取完整的原始图层
# 注意：这两个图大部分区域可能都是黑的，只有各自负责的视场有内容
source_pano_full = cv2.imread(source_pano_path)
target_pano_full = cv2.imread(target_pano_path)

if source_pano_full is None or target_pano_full is None:
    print("Error: Images not found.")
    exit()

# 2. 提取 overlap crop
print("Extracting overlap ROI...")
overlap_mask = get_roi_overlap_mask(target_pano_full, source_pano_full)
crop_source, crop_target, roi_bbox = extract_overlap_roi(source_pano_full, target_pano_full, overlap_mask)
print("Overlap ROI BBox:", roi_bbox)  # (x_min, y_min, x_max, y_max)
h_crop, w_crop = crop_source.shape[:2]  # 获取 ROI Crop 的实际大小
h_full, w_full = source_pano_full.shape[:2]
cv2.imwrite(os.path.join(ls.DATA_OUTPUT_DIR, 'test_raft_result', 'crop_source.png'), crop_source)
cv2.imwrite(os.path.join(ls.DATA_OUTPUT_DIR, 'test_raft_result', 'crop_target.png'), crop_target)
cv2.imwrite(os.path.join(ls.DATA_OUTPUT_DIR, 'test_raft_result', 'overlap_mask.png'), overlap_mask)

# =========================================================================
# 3. 对齐处理 (Alignment)
# =========================================================================

USE_PIXEL_STITCH = False # 切换是否使用 PixelStitch

if USE_PIXEL_STITCH:
    print("Aligning overlap crop using PixelStitch...")
    aligner = PixelStitchAligner()
else:
    print("Aligning overlap crop using RAFT...")
    aligner = RAFTAligner()

MASK_PEOPLE = False
DILATE_PIXELS = 30 

align_result = aligner.align_with_masking(
    crop_source, 
    crop_target, 
    return_flow=True,
    mask_people=MASK_PEOPLE,
    dilate_pixels=DILATE_PIXELS,
    mask_threshold=0.2,
    bidirectional=False # 开启双向光流一致性检查
)

valid_mask = None
if len(align_result) == 5:
    aligned_crop_source, flow, valid_mask, src_remove_people, dst_remove_people = align_result
elif len(align_result) == 4:
    aligned_crop_source, flow, src_remove_people, dst_remove_people = align_result
else:
    raise ValueError("Unexpected return shape from align_with_masking")

cv2.imwrite(os.path.join(ls.DATA_OUTPUT_DIR, 'test_raft_result', 'debug_aligned_crop.png'), aligned_crop_source)

if valid_mask is not None:
    # Visualize validity mask
    # valid_mask is 0/1 float, scale to 0-255
    mask_vis = (valid_mask * 255).astype(np.uint8)
    cv2.imwrite(os.path.join(ls.DATA_OUTPUT_DIR, 'test_raft_result', 'debug_flow_validity_mask.png'), mask_vis)
    print("Saved validity mask.")

visualize_optical_flow_color(
    flow,
    os.path.join(ls.DATA_OUTPUT_DIR, 'test_raft_result', 'debug_flow_color.png')
)
visualize_optical_flow_arrows(
    image=crop_source,
    flow=flow,
    save_path=os.path.join(ls.DATA_OUTPUT_DIR, 'test_raft_result', 'debug_flow_arrows.png')
)

# 3.5 应用 Mesh Grid Warp (Global Consistency)
print("Applying Global Mesh Warp for consistency...")

# 初始化 Warper
mesh_warper = MeshGridWarper(h_full, w_full, grid_size=32)

# 计算全局 Mesh Flow
# 注意：flow 是 (h_crop, w_crop, 2)，roi_bbox 是 (x_min, y_min, x_max, y_max)
# 传入 valid_mask 以过滤不可靠的光流（例如遮挡区域）
map_x, map_y = mesh_warper.compute_mesh_flow_from_roi(flow, roi_bbox, validity_mask=valid_mask)

# 对 全图 Source 进行 Warp
warped_source_full = mesh_warper.warp_image(source_pano_full, map_x, map_y)
cv2.imwrite(os.path.join(ls.DATA_OUTPUT_DIR, 'test_raft_result', 'debug_warped_source_full.png'), warped_source_full)

# 从 Warped Full Source 中重新提取 ROI Crop，用于后续融合
# 这样保证 Overlap 区域和 Non-Overlap 区域的形变是连续的
roi_x_min, roi_y_min, roi_x_max, roi_y_max = roi_bbox
warped_crop_source = warped_source_full[roi_y_min:roi_y_max+1, roi_x_min:roi_x_max+1]

# Debug Output: Blend the warped crop with target crop to verify local alignment
print("Saving debug_crop_blend.png ...")
debug_crop_blend = blend_priority(
    warped_crop_source, 
    crop_target, 
    use_target_mask=False,
    erode_kernel_size=3 
)
cv2.imwrite(os.path.join(ls.DATA_OUTPUT_DIR, 'test_raft_result', 'debug_crop_blend.png'), debug_crop_blend)

# =========================================================================
# 4. 直接融合全景图 (Global Blending)
# 既然已经有了全局Warp的结果，直接将 Warped Full Source 和 Target Full 进行融合
# 这样可以避免手动拼接 ROI 带来的缝隙问题，且能利用 blend_priority 里的去灰边逻辑处理全局边缘
# =========================================================================
print("Composing final panorama (Global Blend)...")

# 注意：全图融合时 use_target_mask 必须为 False，否则 Source 中非 Overlap 的区域会被裁掉
final_pano = blend_priority(
    warped_source_full, 
    target_pano_full, 
    use_target_mask=False,
    save_cleaned_source_path=os.path.join(ls.DATA_OUTPUT_DIR, 'test_raft_result', 'debug_warped_source_full_cleaned.png'),
    erode_kernel_size=3 
)

# =========================================================================

# 保存结果
output_dir = os.path.join(ls.DATA_OUTPUT_DIR, 'test_raft_result')
os.makedirs(output_dir, exist_ok=True)

final_path = os.path.join(output_dir, 'final_composed_result.png')
cv2.imwrite(final_path, final_pano)
print(f"Final composed panorama saved to: {final_path}")

print("Done.") 