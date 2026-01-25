import cv2
import config_script

from lib_pano.flow.raft_align import RAFTAligner
from lib_pano.flow.gmflow_align import GMFlowAligner
from lib_pano.io.flow_utils import blend_priority, get_roi_overlap_mask, extract_overlap_roi, compose_final_panorama
from lib_pano.vis.visualize_features import visualize_optical_flow_color, visualize_optical_flow_arrows
import lib_pano.local_setting as ls
import os
import numpy as np

# Select Alignment Method
USE_GMFLOW = False# Set True to use GMFlow, False to use RAFT
DO_ALIGNMENT = True # Set True to perform alignment, False to just overlay
USE_MESH_WARP = True # Set True to apply global mesh warp, False for local alignment only

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

if DO_ALIGNMENT:
    # 2.5 & 3. 计算光流对齐 (带人像Mask选项)
    print(f"Aligning overlap crop using {'GMFlow' if USE_GMFLOW else 'RAFT'}...")

    if USE_GMFLOW:
        try:
            aligner = GMFlowAligner()
        except ImportError as e:
            print(f"Error initializing GMFlow: {e}")
            print("Falling back to RAFT...")
            USE_GMFLOW = False
            aligner = RAFTAligner()
    else:
        aligner = RAFTAligner()

    MASK_PEOPLE = True
    DILATE_PIXELS = 30 

    aligned_crop_source, flow, src_remove_people, dst_remove_people = aligner.align_with_masking(
        crop_source, 
        crop_target, 
        return_flow=True,
        mask_people=MASK_PEOPLE,
        dilate_pixels=DILATE_PIXELS,
        mask_threshold=0.2
    )

    cv2.imwrite(os.path.join(ls.DATA_OUTPUT_DIR, 'test_raft_result', 'debug_src_remove_people.png'), src_remove_people)
    cv2.imwrite(os.path.join(ls.DATA_OUTPUT_DIR, 'test_raft_result', 'debug_dst_remove_people.png'), dst_remove_people)
    cv2.imwrite(os.path.join(ls.DATA_OUTPUT_DIR, 'test_raft_result', 'debug_aligned_crop.png'), aligned_crop_source)

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
    
    # 默认（未开启Mesh Warp时）：全图不形变，ROI crop使用局部对齐结果
    warped_source_full = source_pano_full
    warped_crop_source = aligned_crop_source 

    if USE_MESH_WARP:
        from lib_pano.flow.mesh_warp_utils import MeshGridWarper

        # 初始化 Warper
        mesh_warper = MeshGridWarper(h_full, w_full, grid_size=32)

        # 计算全局 Mesh Flow
        # 注意：flow 是 (h_crop, w_crop, 2)，roi_bbox 是 (x_min, y_min, x_max, y_max)
        map_x, map_y = mesh_warper.compute_mesh_flow_from_roi(flow, roi_bbox)

        # 对 全图 Source 进行 Warp
        warped_source_full = mesh_warper.warp_image(source_pano_full, map_x, map_y)
        cv2.imwrite(os.path.join(ls.DATA_OUTPUT_DIR, 'test_raft_result', 'debug_warped_source_full.png'), warped_source_full)

        # 从 Warped Full Source 中重新提取 ROI Crop，用于后续融合
        # 这样保证 Overlap 区域和 Non-Overlap 区域的形变是连续的
        roi_x_min, roi_y_min, roi_x_max, roi_y_max = roi_bbox
        warped_crop_source = warped_source_full[roi_y_min:roi_y_max+1, roi_x_min:roi_x_max+1]
    else:
        print("Skipping Mesh Warp (using local RAFT alignment only)...")

else:
    print("Skipping alignment (Just overlaying)...")
    warped_source_full = source_pano_full
    warped_crop_source = crop_source # 直接使用原始 ROI

# 4. 在 Crop 区域内部进行融合 (Target 与 对齐后的 Source)
# 使用 warped_crop_source 替代原先的 aligned_crop_source
blended_crop = blend_priority(warped_crop_source, crop_target)
cv2.imwrite(os.path.join(ls.DATA_OUTPUT_DIR, 'test_raft_result', 'debug_blended_crop.png'), blended_crop)
# =========================================================================
print("Composing final panorama...")

final_pano = compose_final_panorama(
    target_pano=target_pano_full,
    source_pano=warped_source_full, # 传入 Warped Source
    blended_roi=blended_crop,
    roi_bbox=roi_bbox,
    overlap_mask=overlap_mask
)

# =========================================================================

# 保存结果
output_dir = os.path.join(ls.DATA_OUTPUT_DIR, 'test_raft_result')
os.makedirs(output_dir, exist_ok=True)

final_path = os.path.join(output_dir, 'final_composed_result.png')
cv2.imwrite(final_path, final_pano)
print(f"Final composed panorama saved to: {final_path}")

print("Done.") 