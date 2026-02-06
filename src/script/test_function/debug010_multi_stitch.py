
import cv2
import config_script
from lib_pano.flow.raft_align import RAFTAligner
from lib_pano.flow.mesh_warp_utils import MeshGridWarper
from lib_pano.io.flow_utils import blend_priority, get_roi_overlap_mask, extract_overlap_roi, compose_final_panorama
import lib_pano.local_setting as ls
import os
import numpy as np

def load_projected_image(idx):
    """Load the pre-projected (global warp) images."""
    path = os.path.join(ls.DATA_OUTPUT_DIR, 'preview', f'debug_sampled_cam_{idx}.png')
    img = cv2.imread(path)
    if img is None:
        print(f"[Warning] Could not load {path}")
        return None
    return img

def stitch_cumulative(canvas, new_shard, aligner, step_idx, use_mesh_warp=True, distribute_warp=True):
    """
    Aligns new_shard to canvas using RAFT with masking, then matches and blends.
    Returns the updated canvas.
    Args:
        distribute_warp (bool): If True, warp both Canvas and Shard towards a middle ground (Bidirectional).
                                If False, only warp Shard towards Canvas (Unidirectional).
    """
    if new_shard is None:
        return canvas
        
    print(f"\n[Step {step_idx}] Processing overlap...")
    
    # 1. Compute overlap mask between the accumulated canvas and the new image shard
    overlap_mask = get_roi_overlap_mask(canvas, new_shard)
    nz_pixels = cv2.countNonZero(overlap_mask)
    print(f"  Overlap pixels: {nz_pixels}")
    
    if nz_pixels < 100:
        print("  Overlap too small. Simply overlaying (max).")
        return cv2.max(canvas, new_shard)

    # 2. Extract Rectangular ROI covering the overlap
    # crop_shard: The part of new_shard we want to align
    # crop_canvas: The corresponding part of the current canvas (reference)
    crop_shard, crop_canvas, roi_bbox = extract_overlap_roi(new_shard, canvas, overlap_mask)
    x1, y1, x2, y2 = roi_bbox
    print(f"  ROI BBox: {roi_bbox} (W={x2-x1}, H={y2-y1})")
    
    # Debug saving
    debug_dir = os.path.join(ls.DATA_OUTPUT_DIR, 'test_raft_result', 'debug_steps')
    os.makedirs(debug_dir, exist_ok=True)
    cv2.imwrite(os.path.join(debug_dir, f'debug_step{step_idx}_overlap_mask.png'), overlap_mask)
    cv2.imwrite(os.path.join(debug_dir, f'step{step_idx}_crop_shard.png'), crop_shard)
    cv2.imwrite(os.path.join(debug_dir, f'step{step_idx}_crop_canvas.png'), crop_canvas)

    # 3. Align the shard ROI to the canvas ROI using RAFT (with Person Masking)
    # This morphs crop_shard to look like crop_canvas
    print("  Running RAFT alignment with Masking...")
    # We need flow if we are doing mesh warp
    return_flow = use_mesh_warp 
    use_bidirectional = True # Enable bidirectional flow consistency check
    
    result_tuple = aligner.align_with_masking(
        img_source=crop_shard, 
        img_target=crop_canvas, 
        return_flow=return_flow,
        mask_people=False,      # ENABLE MASKING to ignore moving people
        dilate_pixels=30,      # DILATE to cover handheld objects/shadows
        mask_threshold=0.2,
        bidirectional=use_bidirectional
    )
    
    valid_mask = None
    if return_flow:
        if use_bidirectional:
            aligned_crop_shard, flow_roi, valid_mask, src_run, dst_run = result_tuple
            # Visualize mask for debug
            if valid_mask is not None:
                mask_vis = (valid_mask * 255).astype(np.uint8)
                cv2.imwrite(os.path.join(debug_dir, f'step{step_idx}_flow_validity_mask.png'), mask_vis)
        else:
            aligned_crop_shard, flow_roi, src_run, dst_run = result_tuple
    else:
        aligned_crop_shard, src_run, dst_run = result_tuple

    cv2.imwrite(os.path.join(debug_dir, f'step{step_idx}_aligned_shard.png'), aligned_crop_shard)

    # Prepare for Composition
    warped_shard_full = new_shard # Default: no global warp
    crop_to_blend = aligned_crop_shard # Default: use locally aligned cop
    
    # Store the canvas to be used for blending (it might be warped below)
    canvas_to_blend = canvas

    if use_mesh_warp and return_flow:
        h_full, w_full = canvas.shape[:2]
        mesh_warper = MeshGridWarper(h_full, w_full, grid_size=32)
        
        if distribute_warp:
            print("  Applying Global Mesh Warp (Bidirectional / Distributed)...")
            
            # Compute global displacement map
            # map_x, map_y maps SOURCE -> TARGET (Standard RAFT)
            # But we want to distribute the warp.
            # 1. Get the full field flow (Displacement D where Target(x) ~= Source(x + D))
            _, _, full_flow = mesh_warper.compute_mesh_flow_from_roi(flow_roi, roi_bbox, validity_mask=valid_mask, return_flow=True)
            
            # 2. Distribute Flow
            # We want to warp Canvas (Target) by +0.5 * D (Forward Warp direction Target->Mid)
            # We want to warp Shard (Source) by -0.5 * D ??
            
            # Let's think about coordinates.
            # Mid Coordinate m.
            # Target coordinate x_t.
            # Source coordinate x_s.
            # x_s = x_t + D(x_t)
            # We define m = x_t + 0.5 * D(x_t).
            
            # Warp Shard (Source) -> Mid:
            # We need a map for remap(Source, map_shard). map_shard(m) should point to x_s.
            # x_s = m + 0.5 * D(x_t).
            # Since we only have D(x_t), we approximate D(x_t) approx D(m).
            # map_shard = grid + 0.5 * D
            
            # Warp Canvas (Target) -> Mid:
            # We need a map for remap(Target, map_canvas). map_canvas(m) should point to x_t.
            # x_t = m - 0.5 * D(x_t).
            # map_canvas = grid - 0.5 * D
            
            flow_mid = full_flow * 0.5
            
            grid_y, grid_x = np.mgrid[0:h_full, 0:w_full].astype(np.float32)
            
            # Map for Canvas: T -> Mid
            map_x_canvas = grid_x - flow_mid[..., 0]
            map_y_canvas = grid_y - flow_mid[..., 1]
            
            # Map for Shard: S -> Mid
            # Note: map_x normally is x + flow. Here flow describes x_t -> x_s displacement.
            # Source is at grid + full_flow relative to target.
            # We want Source at grid + 0.5 * full_flow relative to Mid ??
            # No.
            # map_shard(m) = x_s.
            # x_s = x_t + D.
            # m = x_t + 0.5 D.
            # x_s = m + 0.5 D.
            map_x_shard = grid_x + flow_mid[..., 0]
            map_y_shard = grid_y + flow_mid[..., 1]
            
            # Warp BOTH
            canvas_to_blend = cv2.remap(canvas, map_x_canvas, map_y_canvas, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
            warped_shard_full = cv2.remap(new_shard, map_x_shard, map_y_shard, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
            
            cv2.imwrite(os.path.join(debug_dir, f'step{step_idx}_warped_canvas_full.png'), canvas_to_blend)

        else:
            print("  Applying Global Mesh Warp (Unidirectional: Shard -> Canvas)...")
            # Unidirectional: Only warp the new shard (Source) to match the Canvas (Target)
            # The Canvas remains fixed.
            map_x, map_y = mesh_warper.compute_mesh_flow_from_roi(flow_roi, roi_bbox, validity_mask=valid_mask)
            
            warped_shard_full = mesh_warper.warp_image(new_shard, map_x, map_y)
            # canvas_to_blend remains the original canvas
        
        cv2.imwrite(os.path.join(debug_dir, f'step{step_idx}_warped_shard_full.png'), warped_shard_full)
        
        # ROI adjustments?
        # Since we warped both globally, valid blending region is the intersection.
        # However, blend_priority uses the full images.
        # So we can just blend the globally warped images.
    else:
        print("  Skipping Mesh Warp (using local RAFT alignment only)...")

    print("  Composing integration (Global Blend)...")
    
    updated_canvas = blend_priority(
        warped_shard_full, 
        canvas_to_blend, 
        use_target_mask=False, # Do NOT mask by target, strictly use Source validity
        erode_kernel_size=3, 
        save_cleaned_source_path=os.path.join(debug_dir, f'step{step_idx}_warped_shard_cleaned.png')
    )
    
    return updated_canvas

if __name__ == "__main__":
    print("=== Cumulative Stitching 'Snowball' Test ===")
    USE_MESH_WARP = True # Global Toggle for Mesh Warping
    # 改为 True 启用双向光流分布（形变分摊），改为 False 使用单向形变（只变新图，画板不动）
    DISTRIBUTE_WARP = True 
    
    # Initialize Aligner
    aligner = RAFTAligner()
    
    # 1. Start with Cam 0
    print("Initializing Canvas with Cam 0...")
    canvas = load_projected_image(0)
    if canvas is None:
        print("Failed to start: Cam 0 missing.")
    else:
        # 2. Iterate through subsequent cameras
        # Adding Cam 1, then Cam 2...
        cam_indices = [1, 2] 
        
        for i, cam_idx in enumerate(cam_indices):
            print(f"\n--- Adding Cam {cam_idx} ---")
            new_img = load_projected_image(cam_idx)
            
            # UPDATE the canvas by stitching the new image onto it
            canvas = stitch_cumulative(canvas, new_img, aligner, step_idx=i+1, use_mesh_warp=USE_MESH_WARP, distribute_warp=DISTRIBUTE_WARP)
            
            # Save intermediate checkpoint
            out_path = os.path.join(ls.DATA_OUTPUT_DIR, 'test_raft_result', f'snowball_result_after_cam{cam_idx}.png')
            cv2.imwrite(out_path, canvas)
            print(f"Saved checkpoint: {out_path}")

    print("\nProcessing Complete.")
