import cv2
import numpy as np
import os
from typing import Optional, Any
from lib_pano.flow.mesh_warper import Mesh_Grid_Warper
from lib_pano.flow.roi_process import (
    blend_priority,
    extract_overlap_roi,
    soft_overlap_from_masks,
)


def visualize_flow_bgr(flow: np.ndarray) -> np.ndarray:
    magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1], angleInDegrees=False)
    hsv = np.zeros((flow.shape[0], flow.shape[1], 3), dtype=np.uint8)
    hsv[..., 0] = ((angle * 180.0 / np.pi) / 2.0).astype(np.uint8)
    hsv[..., 1] = 255
    if float(magnitude.max()) > 0.0:
        hsv[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

def stitch_cumulative(
    canvas: np.ndarray, 
    new_shard: np.ndarray, 
    aligner: Any, 
    source_valid_mask: np.ndarray,
    target_valid_mask: np.ndarray,
    debug_dir: str,
    step_idx: int = 0, 
    use_mesh_warp: bool = True, 
    distribute_warp: bool = True, 
    use_bidirectional: bool = True,
    blend_threshold: int = 1,
    blend_erode_kernel_size: int = 1,
    extend_y: int = 0,
    mesh_decay_radius: float = 6.0,
    mesh_max_propagation: Optional[float] = 8.0,
) -> np.ndarray:
    """
    Aligns new_shard to canvas using RAFT with masking, then matches and blends.
    Returns the updated canvas.
    """

    os.makedirs(debug_dir, exist_ok=True)
    print(f'\n[Step {step_idx}] Processing overlap...')
    
    # Compute overlap mask between the accumulated canvas and the new image shard
    src_m = (source_valid_mask > 0).astype(np.uint8) * 255
    tgt_m = (target_valid_mask > 0).astype(np.uint8) * 255
    if src_m.shape != tgt_m.shape:
        h, w = tgt_m.shape[:2]
        src_m = cv2.resize(src_m, (w, h), interpolation=cv2.INTER_NEAREST)
    overlap_mask = soft_overlap_from_masks(
        src_m,
        tgt_m,
        extend_y=extend_y,
    )
    
    nz_pixels = cv2.countNonZero(overlap_mask)
    print(f'  Overlap pixels: {nz_pixels}')
    
    # Check if overlap is sufficient for alignment.
    if nz_pixels < 100:
        print('  Overlap too small. Simply overlaying (max).')
        return cv2.max(canvas, new_shard)

    # Extract Rectangular ROI covering the overlap
    # crop_shard: The part of new_shard we want to align
    # crop_canvas: The corresponding part of the current canvas (reference)
    crop_shard, crop_canvas, roi_bbox = extract_overlap_roi(new_shard, canvas, overlap_mask)
    x1, y1, x2, y2 = roi_bbox
    print(f'  ROI BBox: (x1={x1}, y1={y1}, x2={x2}, y2={y2}), size: {x2-x1} x {y2-y1}')
    
    cv2.imwrite(os.path.join(debug_dir, f'debug_step{step_idx}_overlap_mask.png'), overlap_mask)
    cv2.imwrite(os.path.join(debug_dir, f'step{step_idx}_crop_shard.png'), crop_shard)
    cv2.imwrite(os.path.join(debug_dir, f'step{step_idx}_crop_canvas.png'), crop_canvas)
   
    print('  Running RAFT alignment with Masking...')
    return_flow = True
    
    result_tuple = aligner.align_with_masking(
        img_source=crop_shard, 
        img_target=crop_canvas, 
        return_flow=return_flow,
        bidirectional=use_bidirectional,
    )
    
    valid_mask: Optional[np.ndarray] = None
    
    if use_bidirectional:
        aligned_crop_shard, flow_roi, valid_mask, src_run, dst_run = result_tuple
        if valid_mask is not None:
            mask_vis = (valid_mask * 255).astype(np.uint8)
            cv2.imwrite(os.path.join(debug_dir, f'step{step_idx}_flow_validity_mask.png'), mask_vis)
    else:
        aligned_crop_shard, flow_roi, src_run, dst_run = result_tuple
    
        cv2.imwrite(os.path.join(debug_dir, f"step{step_idx}_flow_roi_vis.png"),visualize_flow_bgr(flow_roi),)
        cv2.imwrite(os.path.join(debug_dir, f'step{step_idx}_aligned_shard.png'), aligned_crop_shard)

    # Prepare for Composition
    warped_shard_full = new_shard # Default: no global warp
    warped_source_valid_full = source_valid_mask
    warped_target_valid_full = target_valid_mask
    flow_full_vis: Optional[np.ndarray] = None
    
    # Store the canvas to be used for blending (it might be warped below)
    canvas_to_blend = canvas

    if use_mesh_warp :
        h_full, w_full = canvas.shape[:2]
        mesh_warper = Mesh_Grid_Warper(
            h_full,
            w_full,
            grid_size=32,
            decay_radius=mesh_decay_radius,
            max_propagation=mesh_max_propagation,
        )
        
        if distribute_warp:
            print('  Applying Global Mesh Warp (Bidirectional / Distributed)...')
            
            # 1. Get the full field flow (Displacement D where Target(x) ~= Source(x + D))
            _, _, full_flow = mesh_warper.compute_mesh_flow_from_roi(flow_roi, roi_bbox, validity_mask=valid_mask, return_flow=True)
            flow_full_vis = full_flow
            
            # 2. Distribute Flow
    
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
            if target_valid_mask is not None:
                warped_target_valid_full = cv2.remap(
                    target_valid_mask.astype(np.uint8),
                    map_x_canvas,
                    map_y_canvas,
                    interpolation=cv2.INTER_NEAREST,
                    borderMode=cv2.BORDER_CONSTANT,
                    borderValue=0,
                )
            if debug_dir:
                cv2.imwrite(os.path.join(debug_dir, f'step{step_idx}_warped_canvas_full.png'), canvas_to_blend)

        else:
            print('  Applying Global Mesh Warp (Unidirectional: Shard -> Canvas)...')
            # Unidirectional: Only warp the new shard (Source) to match the Canvas (Target)
            map_x, map_y, flow_full = mesh_warper.compute_mesh_flow_from_roi(flow_roi, roi_bbox, validity_mask=valid_mask, return_flow=True)
            flow_full_vis = flow_full
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
            if flow_full_vis is not None:
                cv2.imwrite(
                    os.path.join(debug_dir, f"step{step_idx}_full_flow_vis.png"),
                    visualize_flow_bgr(flow_full_vis),
                )
            cv2.imwrite(os.path.join(debug_dir, f'step{step_idx}_warped_shard_full.png'), warped_shard_full)
        
    else:
        print('  Skipping Mesh Warp (using local RAFT alignment only)...')

    print('  Composing integration (Global Blend)...')
    
    save_cleaned_path: Optional[str] = None
    if debug_dir:
        save_cleaned_path = os.path.join(debug_dir, f'step{step_idx}_warped_shard_cleaned.png')

    if warped_source_valid_full is not None:
        blend_source_mask = (warped_source_valid_full > 0).astype(np.uint8) * 255
        if blend_erode_kernel_size > 0:
            kernel = np.ones((1, blend_erode_kernel_size), np.uint8)
            blend_source_mask = cv2.erode(blend_source_mask, kernel, iterations=1)
    else:
        gray_source = cv2.cvtColor(warped_shard_full, cv2.COLOR_BGR2GRAY)
        blend_source_mask = (gray_source > blend_threshold).astype(np.uint8) * 255

    updated_canvas = blend_priority(
        warped_shard_full, 
        canvas_to_blend, 
        threshold=blend_threshold,
        use_target_mask=False, # Do NOT mask by target, strictly use Source validity
        erode_kernel_size=blend_erode_kernel_size,
        save_cleaned_source_path=save_cleaned_path,
        source_valid_mask=warped_source_valid_full,
    )

    if warped_target_valid_full is not None:
        final_valid_mask = warped_target_valid_full.copy().astype(np.uint8)
    else:
        gray_target = cv2.cvtColor(canvas_to_blend, cv2.COLOR_BGR2GRAY)
        final_valid_mask = (gray_target > blend_threshold).astype(np.uint8) * 255
    final_valid_mask[blend_source_mask > 0] = 255
    if debug_dir:
        cv2.imwrite(os.path.join(debug_dir, f"step{step_idx}_final_valid_mask.png"), final_valid_mask)

    # Debug mask for seam artifacts:
    # 1) near seam band
    # 2) source/target disagree
    # 3) final is darker than both candidates
    gray_tgt = cv2.cvtColor(canvas_to_blend, cv2.COLOR_BGR2GRAY)

    src_valid_u8 = (blend_source_mask > 0).astype(np.uint8) * 255
    tgt_valid_u8 = (warped_target_valid_full > 0).astype(np.uint8) * 255 if warped_target_valid_full is not None else (
        (gray_tgt > blend_threshold).astype(np.uint8) * 255
    )
    overlap_valid = cv2.bitwise_and(src_valid_u8, tgt_valid_u8)

    seam_boundary = cv2.morphologyEx(src_valid_u8, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8))
    seam_band = cv2.dilate(seam_boundary, np.ones((15, 15), np.uint8), iterations=1)
    seam_band = cv2.bitwise_and(seam_band, overlap_valid)

    seam_band_u8 = (seam_band > 0).astype(np.uint8) * 255
    final_valid_u8 = (final_valid_mask > 0).astype(np.uint8) * 255
    seam_band_black = cv2.bitwise_not(seam_band_u8)
    merged_final_valid_mask = cv2.bitwise_and(final_valid_u8, seam_band_black)
    print(f"final_valid black pixels: {np.count_nonzero(final_valid_u8 == 0)}")
    print(f"seam_band_black black pixels: {np.count_nonzero(seam_band_black == 0)}")
    print(f"merged black pixels: {np.count_nonzero(merged_final_valid_mask == 0)}")
    inpaint_mask = (merged_final_valid_mask == 0).astype(np.uint8) * 255
    inpaint_mask_dilated = cv2.dilate(inpaint_mask, np.ones((3,3), np.uint8), iterations=2)
    if cv2.countNonZero(inpaint_mask) > 0:
        updated_canvas = cv2.inpaint(
            updated_canvas,
            inpaint_mask_dilated,
            inpaintRadius=2,
            flags=cv2.INPAINT_TELEA,
        )
    if debug_dir:
        cv2.imwrite(
            os.path.join(debug_dir, f"step{step_idx}_seam_band_black.png"),
            seam_band_black,
        )
        cv2.imwrite(
            os.path.join(debug_dir, f"step{step_idx}_final_valid_mask_merged.png"),
            merged_final_valid_mask,
        )
        cv2.imwrite(
            os.path.join(debug_dir, f"step{step_idx}_final_composed_filled.png"),
            updated_canvas,
        )
    return updated_canvas
