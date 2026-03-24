import cv2
import numpy as np
import os
from typing import Optional, Any
from lib_pano.flow.mesh_warper import Mesh_Grid_Warper
from lib_pano.flow.roi_process import blend_priority, get_roi_overlap_mask, extract_overlap_roi

def stitch_cumulative(
    canvas: np.ndarray, 
    new_shard: np.ndarray, 
    aligner: Any, 
    step_idx: int = 0, 
    use_mesh_warp: bool = True, 
    distribute_warp: bool = True, 
    debug_dir: Optional[str] = None,
    mask_people: bool = True,
    dilate_pixels: int = 20,
    mask_threshold: float = 0.5,
    use_bidirectional: bool = True,
    blend_threshold: int = 1,
    blend_erode_kernel_size: int = 1,
    source_valid_mask: Optional[np.ndarray] = None,
    target_valid_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Aligns new_shard to canvas using RAFT with masking, then matches and blends.
    Returns the updated canvas.

    Args:
        canvas (np.ndarray): The current accumulated panorama.
        new_shard (np.ndarray): The new image to stitch onto the canvas.
        aligner (Any): Instance of RAFTAligner or similar (must implement `align_with_masking`).
        step_idx (int, optional): Step index for debug logging. Defaults to 0.
        use_mesh_warp (bool, optional): Whether to use global mesh warping. Defaults to True.
        distribute_warp (bool, optional): If True, warp both Canvas and Shard towards a middle ground (Bidirectional).
                                          If False, only warp Shard towards Canvas (Unidirectional). Defaults to True.
        debug_dir (str, optional): Directory to save debug images. If None, no debug images are saved. Defaults to None.
        mask_people (bool, optional): Whether to mask people before RAFT alignment.
        dilate_pixels (int, optional): Dilation radius for person mask.
        mask_threshold (float, optional): Person mask threshold.
        use_bidirectional (bool, optional): Whether to compute bidirectional flow validity.
        blend_threshold (int, optional): Grayscale threshold for source-valid mask in blending.
        blend_erode_kernel_size (int, optional): Erode size for blend mask; too large may remove dark clothes.

    Returns:
        np.ndarray: The updated canvas with the new shard stitched.
    """
    if new_shard is None:
        return canvas
    
    # Ensure debug_dir exists if provided
    if debug_dir:
        os.makedirs(debug_dir, exist_ok=True)
        
    print(f'\n[Step {step_idx}] Processing overlap...')
    
    # 1. Compute overlap mask between the accumulated canvas and the new image shard
    if source_valid_mask is not None and target_valid_mask is not None:
        src_m = (source_valid_mask > 0).astype(np.uint8) * 255
        tgt_m = (target_valid_mask > 0).astype(np.uint8) * 255
        if src_m.shape != tgt_m.shape:
            h, w = tgt_m.shape[:2]
            src_m = cv2.resize(src_m, (w, h), interpolation=cv2.INTER_NEAREST)
        overlap_mask = cv2.bitwise_and(src_m, tgt_m)
    else:
        overlap_mask = get_roi_overlap_mask(canvas, new_shard)
    nz_pixels = cv2.countNonZero(overlap_mask)
    print(f'  Overlap pixels: {nz_pixels}')
    
    # Check if overlap is sufficient for alignment.
    # If overlap is too small (< 100 pixels), feature matching/optical flow will likely fail.
    # In this case, we fallback to a simple maximum overlay.
    if nz_pixels < 100:
        print('  Overlap too small. Simply overlaying (max).')
        return cv2.max(canvas, new_shard)

    # 2. Extract Rectangular ROI covering the overlap
    # crop_shard: The part of new_shard we want to align
    # crop_canvas: The corresponding part of the current canvas (reference)
    crop_shard, crop_canvas, roi_bbox = extract_overlap_roi(new_shard, canvas, overlap_mask)
    x1, y1, x2, y2 = roi_bbox
    print(f'  ROI BBox: {roi_bbox} (W={x2-x1}, H={y2-y1})')
    
    if debug_dir:
        cv2.imwrite(os.path.join(debug_dir, f'debug_step{step_idx}_overlap_mask.png'), overlap_mask)
        cv2.imwrite(os.path.join(debug_dir, f'step{step_idx}_crop_shard.png'), crop_shard)
        cv2.imwrite(os.path.join(debug_dir, f'step{step_idx}_crop_canvas.png'), crop_canvas)

    # 3. Align the shard ROI to the canvas ROI using RAFT (with Person Masking)
    # This morphs crop_shard to look like crop_canvas
    print('  Running RAFT alignment with Masking...')
    # We need flow if we are doing mesh warp
    return_flow = use_mesh_warp 
    
    result_tuple = aligner.align_with_masking(
        img_source=crop_shard, 
        img_target=crop_canvas, 
        return_flow=return_flow,
        mask_people=mask_people,
        dilate_pixels=dilate_pixels,
        mask_threshold=mask_threshold,
        bidirectional=use_bidirectional
    )
    
    valid_mask: Optional[np.ndarray] = None
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
    warped_source_valid_full = source_valid_mask
    # crop_to_blend = aligned_crop_shard # Default: use locally aligned cop (Unused variable)
    
    # Store the canvas to be used for blending (it might be warped below)
    canvas_to_blend = canvas

    if use_mesh_warp and return_flow:
        h_full, w_full = canvas.shape[:2]
        mesh_warper = Mesh_Grid_Warper(h_full, w_full, grid_size=32)
        
        if distribute_warp:
            print('  Applying Global Mesh Warp (Bidirectional / Distributed)...')
            
            # 1. Get the full field flow (Displacement D where Target(x) ~= Source(x + D))
            _, _, full_flow = mesh_warper.compute_mesh_flow_from_roi(flow_roi, roi_bbox, validity_mask=valid_mask, return_flow=True)
            
            # 2. Distribute Flow
            # Mathematical Formulation:
            # Let D(x) be the flow vector at position x such that Source(x + D(x)) matches Target(x).
            # We want to warp both images towards a "middle appearance" to minimize distortion on both sides.
            # 
            # Mid_Warp_Target: Map_Canvas(x) = x - 0.5 * D(x)
            # Mid_Warp_Source: Map_Shard(x)  = x + 0.5 * D(x)
            # 
            # This balances the deformation:
            # Canvas is pulled "forward" by half the flow.
            # Shard is pulled "backward" by half the flow.
            
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
            if source_valid_mask is not None:
                warped_source_valid_full = cv2.remap(
                    source_valid_mask.astype(np.uint8),
                    map_x_shard,
                    map_y_shard,
                    interpolation=cv2.INTER_NEAREST,
                    borderMode=cv2.BORDER_CONSTANT,
                    borderValue=0,
                )
            
            if debug_dir:
                cv2.imwrite(os.path.join(debug_dir, f'step{step_idx}_warped_canvas_full.png'), canvas_to_blend)

        else:
            print('  Applying Global Mesh Warp (Unidirectional: Shard -> Canvas)...')
            # Unidirectional: Only warp the new shard (Source) to match the Canvas (Target)
            map_x, map_y = mesh_warper.compute_mesh_flow_from_roi(flow_roi, roi_bbox, validity_mask=valid_mask)
            warped_shard_full = mesh_warper.warp_image(new_shard, map_x, map_y)
            if source_valid_mask is not None:
                warped_source_valid_full = cv2.remap(
                    source_valid_mask.astype(np.uint8),
                    map_x,
                    map_y,
                    interpolation=cv2.INTER_NEAREST,
                    borderMode=cv2.BORDER_CONSTANT,
                    borderValue=0,
                )
        
        if debug_dir:
            cv2.imwrite(os.path.join(debug_dir, f'step{step_idx}_warped_shard_full.png'), warped_shard_full)
        
    else:
        print('  Skipping Mesh Warp (using local RAFT alignment only)...')

    print('  Composing integration (Global Blend)...')
    
    save_cleaned_path: Optional[str] = None
    if debug_dir:
        save_cleaned_path = os.path.join(debug_dir, f'step{step_idx}_warped_shard_cleaned.png')

    updated_canvas = blend_priority(
        warped_shard_full, 
        canvas_to_blend, 
        threshold=blend_threshold,
        use_target_mask=False, # Do NOT mask by target, strictly use Source validity
        erode_kernel_size=blend_erode_kernel_size,
        save_cleaned_source_path=save_cleaned_path,
        source_valid_mask=warped_source_valid_full,
    )
    # updated_canvas = blend_priority(
    #     warped_shard_full, 
    #     canvas_to_blend, 
    #     use_target_mask=False,
    #     erode_kernel_size=5,   # 凸包边界腐蚀 5px：去除鱼眼外圈灰边，不影响手臂等内部细节
    #     feather_radius=20,     # 窄过渡带（30px），仅在 shard 边缘做平滑
    #     save_cleaned_source_path=save_cleaned_path
    # )
    
    return updated_canvas
