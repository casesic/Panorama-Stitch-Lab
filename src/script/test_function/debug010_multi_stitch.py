
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

def stitch_cumulative(canvas, new_shard, aligner, step_idx, use_mesh_warp=True):
    """
    Aligns new_shard to canvas using RAFT with masking, then matches and blends.
    Returns the updated canvas.
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
    
    result_tuple = aligner.align_with_masking(
        img_source=crop_shard, 
        img_target=crop_canvas, 
        return_flow=return_flow,
        mask_people=False,      # ENABLE MASKING to ignore moving people
        dilate_pixels=30,      # DILATE to cover handheld objects/shadows
        mask_threshold=0.5
    )
    
    if return_flow:
        aligned_crop_shard, flow_roi, src_run, dst_run = result_tuple
    else:
        aligned_crop_shard, src_run, dst_run = result_tuple

    cv2.imwrite(os.path.join(debug_dir, f'step{step_idx}_aligned_shard.png'), aligned_crop_shard)

    # Prepare for Composition
    warped_shard_full = new_shard # Default: no global warp
    crop_to_blend = aligned_crop_shard # Default: use locally aligned cop

    if use_mesh_warp and return_flow:
        print("  Applying Global Mesh Warp...")
        h_full, w_full = canvas.shape[:2]
        mesh_warper = MeshGridWarper(h_full, w_full, grid_size=32)
        
        # Compute global displacement map
        map_x, map_y = mesh_warper.compute_mesh_flow_from_roi(flow_roi, roi_bbox)
        
        # Warp the full source image (new_shard)
        warped_shard_full = mesh_warper.warp_image(new_shard, map_x, map_y)
        cv2.imwrite(os.path.join(debug_dir, f'step{step_idx}_warped_shard_full.png'), warped_shard_full)
        
        # Re-extract the ROI from the GLOBALLY warped image to use for blending
        # This replaces the 'aligned_crop_shard' which was only locally warped
        # Valid slicing: [y_min : y_max+1, x_min : x_max+1]
        crop_to_blend = warped_shard_full[y1:y2+1, x1:x2+1] 
        
        # Double check size match 
        if crop_to_blend.shape[:2] != crop_canvas.shape[:2]:
            print(f"  Warning: Warped crop shape {crop_to_blend.shape} mismatch with canvas crop {crop_canvas.shape}. Resizing to fit.")
            crop_to_blend = cv2.resize(crop_to_blend, (crop_canvas.shape[1], crop_canvas.shape[0]))
    else:
        print("  Skipping Mesh Warp (using local RAFT alignment only)...")

    # 4. Blend the aligned shard into the canvas ROI
    # crop_canvas is the Reference. aligned_crop_shard is the Adjusted Overlay.
    # blend_priority typically takes (Source, Target). 
    # Here we treat the new image as 'Source' to replace/blend onto 'Target' Canvas.
    blended_roi = blend_priority(crop_to_blend, crop_canvas)
    cv2.imwrite(os.path.join(debug_dir, f'step{step_idx}_blended_roi.png'), blended_roi)

    # 5. Compose the full updated panorama
    # We take the original 'new_shard' (which contains the non-overlapping parts we want to keep)
    # and the 'canvas' (base), and update it with the 'blended_roi'.
    # Note: 'compose_final_panorama' logic usually:
    #   Output = Canvas (Target)
    #   ROI Area = Blended ROI
    #   Non-ROI Area of Shard = Added to output (if not black)
    
    print("  Composing integration...")
    updated_canvas = compose_final_panorama(
        target_pano=canvas,       # The accumulating result
        source_pano=warped_shard_full,    # The raw new image (provides the outlying pixels)
        blended_roi=blended_roi,  # The fixed confluence area
        roi_bbox=roi_bbox,
        overlap_mask=overlap_mask
    )
    
    return updated_canvas

def main():
    print("=== Cumulative Stitching 'Snowball' Test ===")
    USE_MESH_WARP = True # Global Toggle for Mesh Warping
    
    # Initialize Aligner
    aligner = RAFTAligner()
    
    # 1. Start with Cam 0
    print("Initializing Canvas with Cam 0...")
    canvas = load_projected_image(0)
    if canvas is None:
        print("Failed to start: Cam 0 missing.")
        return

    # 2. Iterate through subsequent cameras
    # Adding Cam 1, then Cam 2...
    cam_indices = [1, 2] 
    
    for i, cam_idx in enumerate(cam_indices):
        print(f"\n--- Adding Cam {cam_idx} ---")
        new_img = load_projected_image(cam_idx)
        
        # UPDATE the canvas by stitching the new image onto it
        canvas = stitch_cumulative(canvas, new_img, aligner, step_idx=i+1, use_mesh_warp=USE_MESH_WARP)
        
        # Save intermediate checkpoint
        out_path = os.path.join(ls.DATA_OUTPUT_DIR, 'test_raft_result', f'snowball_result_after_cam{cam_idx}.png')
        cv2.imwrite(out_path, canvas)
        print(f"Saved checkpoint: {out_path}")

    print("\nProcessing Complete.")

if __name__ == "__main__":
    main()
