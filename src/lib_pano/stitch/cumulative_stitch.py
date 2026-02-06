import cv2
import numpy as np
import os
from lib_pano.flow.mesh_warp_utils import MeshGridWarper
from lib_pano.io.flow_utils import blend_priority, get_roi_overlap_mask, extract_overlap_roi

def stitch_cumulative(canvas, new_shard, aligner, step_idx=0, use_mesh_warp=True, distribute_warp=True, debug_dir=None):
    """
    Aligns new_shard to canvas using RAFT with masking, then matches and blends.
    Returns the updated canvas.
    Args:
        canvas (np.ndarray): The current accumulated panorama.
        new_shard (np.ndarray): The new image to stitch onto the canvas.
        aligner: Instance of RAFTAligner or similar.
        step_idx (int): Step index for debug logging.
        use_mesh_warp (bool): Whether to use global mesh warping.
        distribute_warp (bool): If True, warp both Canvas and Shard towards a middle ground (Bidirectional).
                                If False, only warp Shard towards Canvas (Unidirectional).
        debug_dir (str): Directory to save debug images. If None, no debug images are saved.
    """
    if new_shard is None:
        return canvas
    
    # Ensure debug_dir exists if provided
    if debug_dir:
        os.makedirs(debug_dir, exist_ok=True)
        
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
    
    if debug_dir:
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
            if debug_dir and valid_mask is not None:
                mask_vis = (valid_mask * 255).astype(np.uint8)
                cv2.imwrite(os.path.join(debug_dir, f'step{step_idx}_flow_validity_mask.png'), mask_vis)
        else:
            aligned_crop_shard, flow_roi, src_run, dst_run = result_tuple
    else:
        aligned_crop_shard, src_run, dst_run = result_tuple

    if debug_dir:
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
            # Warp Shard (Source) -> Mid: map_shard = grid + 0.5 * D
            # Warp Canvas (Target) -> Mid: map_canvas = grid - 0.5 * D
            
            flow_mid = full_flow * 0.5
            
            grid_y, grid_x = np.mgrid[0:h_full, 0:w_full].astype(np.float32)
            
            # Map for Canvas: T -> Mid
            map_x_canvas = grid_x - flow_mid[..., 0]
            map_y_canvas = grid_y - flow_mid[..., 1]
            
            # Map for Shard: S -> Mid
            map_x_shard = grid_x + flow_mid[..., 0]
            map_y_shard = grid_y + flow_mid[..., 1]
            
            # Warp BOTH
            canvas_to_blend = cv2.remap(canvas, map_x_canvas, map_y_canvas, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
            warped_shard_full = cv2.remap(new_shard, map_x_shard, map_y_shard, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
            
            if debug_dir:
                cv2.imwrite(os.path.join(debug_dir, f'step{step_idx}_warped_canvas_full.png'), canvas_to_blend)

        else:
            print("  Applying Global Mesh Warp (Unidirectional: Shard -> Canvas)...")
            # Unidirectional: Only warp the new shard (Source) to match the Canvas (Target)
            map_x, map_y = mesh_warper.compute_mesh_flow_from_roi(flow_roi, roi_bbox, validity_mask=valid_mask)
            warped_shard_full = mesh_warper.warp_image(new_shard, map_x, map_y)
        
        if debug_dir:
            cv2.imwrite(os.path.join(debug_dir, f'step{step_idx}_warped_shard_full.png'), warped_shard_full)
        
    else:
        print("  Skipping Mesh Warp (using local RAFT alignment only)...")

    print("  Composing integration (Global Blend)...")
    
    save_cleaned_path = None
    if debug_dir:
        save_cleaned_path = os.path.join(debug_dir, f'step{step_idx}_warped_shard_cleaned.png')

    updated_canvas = blend_priority(
        warped_shard_full, 
        canvas_to_blend, 
        use_target_mask=False, # Do NOT mask by target, strictly use Source validity
        erode_kernel_size=3, 
        save_cleaned_source_path=save_cleaned_path
    )
    
    return updated_canvas
