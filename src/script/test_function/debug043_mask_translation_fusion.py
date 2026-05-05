from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import cv2

SRC_ROOT = Path(__file__).resolve().parents[2]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from lib_pano.pipeline.bbox_io import load_bbox_dict
from lib_pano.segmentation.mask_fusion_io import (
    discover_fusion_frames,
    ensure_fusion_output_dirs,
    get_bbox,
    save_original_fallback,
    validate_fusion_inputs,
    write_translation_meta,
)
from lib_pano.segmentation.mask_fusion_vis import draw_matches_vis, make_fusion_vis
from lib_pano.segmentation.mask_matching import empty_matches, estimate_translation, match_loftr, match_orb
from lib_pano.segmentation.mask_ops import fuse_dual_masks_with_translation

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - tqdm is only for CLI ergonomics.
    tqdm = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fuse dual-camera SAM3 crop masks with a coarse feature-match translation baseline. "
            "The output mask/ folders are compatible with debug029 --seg_dir."
        )
    )
    parser.add_argument("--crop_image_dir", type=str, required=True, help="Root containing cam2/00513.jpg and cam4/00513.jpg.")
    parser.add_argument("--seg_dir", type=str, required=True, help="SAM3 segmentation root containing camX/mask/00513.png.")
    parser.add_argument("--bbox_dir", type=str, required=True, help="Directory containing cam2_bboxes.json and cam4_bboxes.json.")
    parser.add_argument("--output_dir", type=str, required=True, help="Output root directory for fused masks and debug data.")
    parser.add_argument("--start_frame", type=int, default=None, help="Optional start frame, inclusive.")
    parser.add_argument("--end_frame", type=int, default=None, help="Optional end frame, inclusive.")
    parser.add_argument("--method", choices=("loftr", "orb"), default="loftr", help="Feature matching method.")
    parser.add_argument("--max_width", type=int, default=640, help="Max width for LoFTR resize.")
    parser.add_argument("--max_height", type=int, default=480, help="Max height for LoFTR resize.")
    parser.add_argument("--min_matches", type=int, default=20, help="Minimum raw matches for translation estimation.")
    parser.add_argument("--min_inliers", type=int, default=8, help="Minimum robust inliers for a valid translation.")
    parser.add_argument("--max_mad", type=float, default=25.0, help="Residual threshold around median translation.")
    parser.add_argument("--image_width", type=int, default=5760, help="Full equirect_rotate canvas width.")
    parser.add_argument("--image_height", type=int, default=2880, help="Full equirect_rotate canvas height.")
    parser.add_argument("--min_component_area", type=int, default=0, help="Optional removal of tiny mask components; 0 disables.")
    parser.add_argument("--close_kernel", type=int, default=0, help="Optional morphology close kernel size; 0 disables.")
    parser.add_argument("--save_vis", action="store_true", help="Save match and fusion visualizations.")
    return parser.parse_args()


def process_frame(
    frame_idx: int,
    args: argparse.Namespace,
    crop_image_dir: Path,
    seg_dir: Path,
    output_dir: Path,
    cam2_bboxes: dict[str, list[int]],
    cam4_bboxes: dict[str, list[int]],
    matcher_cache: dict[str, Any],
) -> dict[str, Any]:
    frame_key = f"{frame_idx:05d}"
    data, missing = validate_fusion_inputs(crop_image_dir, seg_dir, cam2_bboxes, cam4_bboxes, frame_idx)
    cam2_bbox = get_bbox(cam2_bboxes, frame_idx)
    cam4_bbox = get_bbox(cam4_bboxes, frame_idx)
    base_meta: dict[str, Any] = {
        "valid_translation": False,
        "dx_cam2_to_cam4": 0.0,
        "dy_cam2_to_cam4": 0.0,
        "match_count": 0,
        "inlier_count": 0,
        "inlier_ratio": 0.0,
        "mad": None,
        "reason": "ok",
        "cam2_bbox": cam2_bbox,
        "cam4_bbox": cam4_bbox,
    }
    if data is None:
        reason = "missing_input:" + ",".join(missing)
        save_original_fallback(seg_dir, output_dir, frame_idx, reason)
        base_meta["reason"] = reason
        return base_meta

    image2 = data["cam2"]["image"]
    image4 = data["cam4"]["image"]
    mask2_crop = data["cam2"]["mask"]
    mask4_crop = data["cam4"]["mask"]
    cam2_bbox = data["cam2"]["bbox"]
    cam4_bbox = data["cam4"]["bbox"]
    assert cam2_bbox is not None and cam4_bbox is not None

    try:
        if args.method == "loftr":
            pts2_crop, pts4_crop, confidence = match_loftr(image2, image4, args.max_width, args.max_height, matcher_cache)
            matcher_used = "loftr"
        else:
            pts2_crop, pts4_crop, confidence = match_orb(image2, image4)
            matcher_used = "orb"
    except Exception as exc:
        if args.method == "loftr":
            print(f"Warning: LoFTR failed for frame {frame_key}, trying ORB fallback: {exc}")
            pts2_crop, pts4_crop, confidence = match_orb(image2, image4)
            matcher_used = "orb_fallback"
        else:
            print(f"Warning: ORB failed for frame {frame_key}: {exc}")
            pts2_crop, pts4_crop, confidence = empty_matches()
            matcher_used = "orb_failed"

    translation = estimate_translation(
        pts_cam2_crop=pts2_crop,
        pts_cam4_crop=pts4_crop,
        cam2_bbox=cam2_bbox,
        cam4_bbox=cam4_bbox,
        min_matches=args.min_matches,
        min_inliers=args.min_inliers,
        max_mad=args.max_mad,
    )

    fused = fuse_dual_masks_with_translation(
        mask2_crop=mask2_crop,
        mask4_crop=mask4_crop,
        cam2_bbox=cam2_bbox,
        cam4_bbox=cam4_bbox,
        translation=translation,
        width=args.image_width,
        height=args.image_height,
        min_component_area=args.min_component_area,
        close_kernel=args.close_kernel,
    )

    cv2.imwrite(str(output_dir / "cam2" / "mask" / f"{frame_key}.png"), fused["mask2_completed_crop"])
    cv2.imwrite(str(output_dir / "cam4" / "mask" / f"{frame_key}.png"), fused["mask4_completed_crop"])
    cv2.imwrite(str(output_dir / "cam2" / "full_original_mask" / f"{frame_key}.png"), fused["full2_original"])
    cv2.imwrite(str(output_dir / "cam4" / "full_original_mask" / f"{frame_key}.png"), fused["full4_original"])
    cv2.imwrite(str(output_dir / "cam2" / "full_warped_from_cam4" / f"{frame_key}.png"), fused["full4_to_2"])
    cv2.imwrite(str(output_dir / "cam4" / "full_warped_from_cam2" / f"{frame_key}.png"), fused["full2_to_4"])
    cv2.imwrite(str(output_dir / "cam2" / "full_completed_mask" / f"{frame_key}.png"), fused["full2_completed"])
    cv2.imwrite(str(output_dir / "cam4" / "full_completed_mask" / f"{frame_key}.png"), fused["full4_completed"])

    if args.save_vis:
        vis2 = make_fusion_vis(image2, mask2_crop, fused["warped4_to_2_crop"], fused["mask2_completed_crop"])
        vis4 = make_fusion_vis(image4, mask4_crop, fused["warped2_to_4_crop"], fused["mask4_completed_crop"])
        cv2.imwrite(str(output_dir / "cam2" / "vis" / f"{frame_key}.jpg"), vis2)
        cv2.imwrite(str(output_dir / "cam4" / "vis" / f"{frame_key}.jpg"), vis4)
        match_vis = draw_matches_vis(image2, image4, pts2_crop, pts4_crop, confidence, translation["inlier_mask"])
        cv2.imwrite(str(output_dir / "matches" / f"{frame_key}.jpg"), match_vis)

    meta = dict(base_meta)
    meta.update({key: value for key, value in translation.items() if key != "inlier_mask"})
    meta["matcher_used"] = matcher_used
    meta["cam2_bbox"] = cam2_bbox
    meta["cam4_bbox"] = cam4_bbox
    return meta


def summarize(frames_meta: dict[str, dict[str, Any]]) -> dict[str, int]:
    summary = {
        "total_frames": len(frames_meta),
        "valid_translation": 0,
        "fallback_original_only": 0,
        "missing_input": 0,
        "low_matches": 0,
    }
    for frame_meta in frames_meta.values():
        reason = str(frame_meta.get("reason", ""))
        if frame_meta.get("valid_translation"):
            summary["valid_translation"] += 1
        else:
            summary["fallback_original_only"] += 1
        if reason.startswith("missing_input"):
            summary["missing_input"] += 1
        if reason == "low_matches":
            summary["low_matches"] += 1
    return summary


def main() -> None:
    args = parse_args()
    crop_image_dir = Path(args.crop_image_dir)
    seg_dir = Path(args.seg_dir)
    bbox_dir = Path(args.bbox_dir)
    output_dir = Path(args.output_dir)
    ensure_fusion_output_dirs(output_dir)

    cam2_bboxes = load_bbox_dict(bbox_dir / "cam2_bboxes.json", normalize_values=True)
    cam4_bboxes = load_bbox_dict(bbox_dir / "cam4_bboxes.json", normalize_values=True)
    frame_indices = discover_fusion_frames(crop_image_dir, seg_dir, cam2_bboxes, cam4_bboxes)
    if args.start_frame is not None:
        frame_indices = [idx for idx in frame_indices if idx >= args.start_frame]
    if args.end_frame is not None:
        frame_indices = [idx for idx in frame_indices if idx <= args.end_frame]

    if not frame_indices:
        print("Warning: no frames found after filtering.")

    frames_meta: dict[str, dict[str, Any]] = {}
    matcher_cache: dict[str, Any] = {}
    iterator = tqdm(frame_indices, desc="Fusing masks", unit="frame") if tqdm is not None else frame_indices
    for frame_idx in iterator:
        frames_meta[f"{frame_idx:05d}"] = process_frame(
            frame_idx=frame_idx,
            args=args,
            crop_image_dir=crop_image_dir,
            seg_dir=seg_dir,
            output_dir=output_dir,
            cam2_bboxes=cam2_bboxes,
            cam4_bboxes=cam4_bboxes,
            matcher_cache=matcher_cache,
        )

    meta = {
        "meta": {
            "crop_image_dir": str(crop_image_dir),
            "seg_dir": str(seg_dir),
            "bbox_dir": str(bbox_dir),
            "method": args.method,
            "image_width": args.image_width,
            "image_height": args.image_height,
            "note": "Coarse support/completed mask baseline; not semantic-flow ground truth.",
        },
        "frames": frames_meta,
        "summary": summarize(frames_meta),
    }
    write_translation_meta(output_dir, meta)

    print(f"Processed {len(frames_meta)} frames.")
    print(f"Valid translations: {meta['summary']['valid_translation']}")
    print(f"Saved fused seg_dir to: {output_dir}")


if __name__ == "__main__":
    main()


"""
Example:

python src/script/test_function/debug043_mask_translation_fusion.py `
  --crop_image_dir data_process/Output/20260311_riding/image/equirect_crop_rotate `
  --seg_dir data_process/Output/20260311_riding/seg/equirect_crop_rotate_sam3 `
  --bbox_dir data_process/Output/20260311_riding/image/equirect_crop_rotate `
  --output_dir data_process/Output/20260311_riding/seg/equirect_crop_rotate_sam3_fused_translation `
  --start_frame 513 `
  --end_frame 518 `
  --method loftr `
  --save_vis
"""
