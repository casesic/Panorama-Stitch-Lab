"""光流拼接编排器：计算 ROI、跑 RAFT、融合语义先验、构建变形场、混合并修复接缝。"""

import cv2
import numpy as np
from pathlib import Path
from typing import Optional, Tuple
from lib_pano.flow.raft_flow import RAFTFlow
from lib_pano.flow.roi_process import (
    soft_overlap_from_masks,
    extract_overlap_roi,
    blend_priority,
    crop_mask_by_bbox,
    compute_landed_source_valid_mask,
    compute_photometric_mask,
    build_flow_seed_mask,
    build_flow_weight_map,
    crop_bbox_dense_semantic_to_roi,
    build_semantic_weight_map,
    combine_flow_with_semantic_prior,
)
from lib_pano.flow.warp_strategy import apply_warp, WarpStrategy, MeshConfig
from lib_pano.flow.seam_repair import repair_seam
from lib_pano.flow.debug_writer import DebugWriter


def stitch_with_flow(
    img_source: np.ndarray,
    img_target: np.ndarray,
    raft_flow: RAFTFlow,
    source_valid_mask: np.ndarray,
    target_valid_mask: np.ndarray,
    strategy: WarpStrategy = WarpStrategy.MESH_BIDIRECTIONAL,
    mesh_config: MeshConfig = MeshConfig(),
    bidirectional: bool = True,
    extend_y: int = 0,
    blend_threshold: int = 1,
    blend_erode_kernel_size: int = 1,
    use_seam_repair: bool = True,
    seam_band_radius: int = 5,
    inpaint_radius: int = 2,
    photometric_threshold: int = 30,
    photo_sigma: float = 60.0,
    landed_invalid_multiplier: float = 0.10,
    raft_invalid_multiplier: float = 0.35,
    flow_weight_min: float = 0.0,
    semantic_prior: Optional[dict] = None,
    semantic_weight_scale: float = 4.0,
    semantic_conf_thresh: float = 0.05,
    debug_dir: Optional[str] = None,
    debug_level: str = "minimal",
    save_full_flow_path: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """两相机光流拼接主流程：ROI→RAFT 光流→语义先验融合→变形→优先级混合→接缝修复，返回 (拼接图, 有效遮罩)。"""

    writer = DebugWriter(
        debug_dir=debug_dir,
        enabled=debug_dir is not None and debug_level != "none",
        debug_level=debug_level,
    )

    flow_src_m = (source_valid_mask > 0).astype(np.uint8) * 255
    flow_tgt_m = (target_valid_mask > 0).astype(np.uint8) * 255
    support_roi_mask = soft_overlap_from_masks(flow_src_m, flow_tgt_m, extend_y=extend_y)
    writer.write_mask('support_roi_mask.png', support_roi_mask)
    nz_pixels = cv2.countNonZero(support_roi_mask)

    if nz_pixels < 100:
        src_m = (source_valid_mask > 0).astype(np.uint8) * 255
        tgt_m = (target_valid_mask > 0).astype(np.uint8) * 255
        merged_mask = cv2.max(src_m, tgt_m)
        return cv2.max(img_source, img_target), merged_mask

    crop_source, crop_target, roi_bbox = extract_overlap_roi(
        img_source,
        img_target,
        support_roi_mask,
    )
    writer.write_crops(crop_source, crop_target)

    source_valid_roi = crop_mask_by_bbox(source_valid_mask, roi_bbox)
    target_valid_roi = crop_mask_by_bbox(target_valid_mask, roi_bbox)
    support_roi_roi = crop_mask_by_bbox(support_roi_mask, roi_bbox)
    writer.write_mask('source_valid_roi.png', source_valid_roi)
    writer.write_mask('target_valid_roi.png', target_valid_roi)

    flow_roi, raft_valid_mask = raft_flow.compute(crop_source, crop_target, bidirectional=bidirectional)
    writer.write_flow(flow_roi, raft_valid_mask)
    writer.write_alignment_debug(crop_source, crop_target, flow_roi)

    source_valid_at_landed = compute_landed_source_valid_mask(flow_roi, source_valid_roi)
    photometric_mask, warped_source_by_raft, photometric_error = compute_photometric_mask(
        crop_source,
        crop_target,
        flow_roi,
        threshold=photometric_threshold,
    )
    flow_seed_mask = build_flow_seed_mask(
        target_valid_roi=target_valid_roi,
        source_valid_at_landed=source_valid_at_landed,
        raft_valid_mask=raft_valid_mask,
        photometric_mask=photometric_mask,
    )
    flow_weight_map = build_flow_weight_map(
        target_valid_roi=target_valid_roi,
        source_valid_at_landed=source_valid_at_landed,
        raft_valid_mask=raft_valid_mask,
        photometric_error=photometric_error,
        photometric_mask=photometric_mask,
        support_roi_roi=support_roi_roi,
        photo_sigma=photo_sigma,
        landed_invalid_multiplier=landed_invalid_multiplier,
        raft_invalid_multiplier=raft_invalid_multiplier,
        min_weight=flow_weight_min,
    )

    writer.write_mask('source_valid_at_landed.png', source_valid_at_landed)
    if raft_valid_mask is not None:
        writer.write_mask('raft_valid_mask.png', raft_valid_mask)
    writer.write_mask('photometric_mask.png', photometric_mask)
    writer.write_mask('final_flow_seed_mask.png', flow_seed_mask)
    writer.write_mask('flow_weight_map.png', flow_weight_map)
    writer.write_image('warped_source_by_raft.png', warped_source_by_raft)

    semantic_flow_roi, semantic_conf_roi = crop_bbox_dense_semantic_to_roi(
        semantic_prior,
        roi_bbox,
    )
    semantic_weight_map = build_semantic_weight_map(
        semantic_conf_roi,
        semantic_weight_scale=semantic_weight_scale,
        semantic_conf_thresh=semantic_conf_thresh,
    )
    combined_flow_roi, combined_weight_map = combine_flow_with_semantic_prior(
        raft_flow_roi=flow_roi,
        raft_weight_map=flow_weight_map,
        semantic_flow_roi=semantic_flow_roi,
        semantic_weight_map=semantic_weight_map,
    )

    if semantic_prior is not None:
        writer.write_weight_map(
            "semantic_weight_map.png",
            semantic_weight_map,
            max_value=semantic_weight_scale,
        )
        writer.write_weight_map(
            "combined_weight_map.png",
            combined_weight_map,
            max_value=max(1.0, semantic_weight_scale),
        )
        if debug_level == "full":
            writer.write_flow_vis("semantic_flow_vis.png", semantic_flow_roi)
            writer.write_flow_vis("combined_flow_vis.png", combined_flow_roi)

    warp_result = apply_warp(
        img_source=img_source,
        img_target=img_target,
        flow_roi=combined_flow_roi,
        roi_bbox=roi_bbox,
        valid_mask=target_valid_roi,
        strategy=strategy,
        source_valid_mask=source_valid_mask,
        target_valid_mask=target_valid_mask,
        mesh_config=mesh_config,
        weight_map=combined_weight_map,
    )
    writer.write_warp_result(warp_result)

    if save_full_flow_path is not None:
        Path(save_full_flow_path).parent.mkdir(parents=True, exist_ok=True)
        np.save(save_full_flow_path, warp_result.full_flow.astype(np.float32))

    blended = blend_priority(
        image_source=warp_result.warped_source,
        image_target=warp_result.warped_target,
        threshold=blend_threshold,
        erode_kernel_size=blend_erode_kernel_size,
        source_valid_mask=warp_result.warped_source_mask,
    )

    if (
        use_seam_repair
        and warp_result.warped_source_mask is not None
        and warp_result.warped_target_mask is not None
    ):
        blended = repair_seam(
            image=blended,
            source_mask=warp_result.warped_source_mask,
            target_mask=warp_result.warped_target_mask,
            seam_band_radius=seam_band_radius,
            inpaint_radius=inpaint_radius,
        )

    if warp_result.warped_source_mask is not None:
        src_valid = (warp_result.warped_source_mask > 0).astype(np.uint8) * 255
    else:
        gray = cv2.cvtColor(warp_result.warped_source, cv2.COLOR_BGR2GRAY)
        src_valid = (gray > blend_threshold).astype(np.uint8) * 255

    if warp_result.warped_target_mask is not None:
        final_valid_mask = warp_result.warped_target_mask.copy().astype(np.uint8)
    else:
        gray = cv2.cvtColor(warp_result.warped_target, cv2.COLOR_BGR2GRAY)
        final_valid_mask = (gray > blend_threshold).astype(np.uint8) * 255

    final_valid_mask[src_valid > 0] = 255

    writer.write_final(blended, final_valid_mask)

    return blended, final_valid_mask
