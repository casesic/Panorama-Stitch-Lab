from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from batch_utils import load_reviewed_entries
from lib_pano.composite.human_blend import blend_two_humans
from lib_pano.composite.rgba_place import place_rgba_on_black
from lib_pano.composite.semantic_flow_warp import composite_direction
from lib_pano.pipeline.bbox_io import load_bbox_dict


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Composite segmented humans between ERP cameras using semantic flow."
    )
    parser.add_argument(
        "--seg_dir",
        type=str,
        required=True,
        help="Segmentation root containing cam2/mask, cam2/rgba, cam4/mask, cam4/rgba.",
    )
    parser.add_argument(
        "--flow_dir",
        type=str,
        required=True,
        help="Semantic flow root with frame_{idx:05d}/ subdirectories.",
    )
    parser.add_argument(
        "--image_dir",
        type=str,
        required=True,
        help="Full ERP image root containing cam2/ and cam4/.",
    )
    parser.add_argument(
        "--bbox_dir",
        type=str,
        required=True,
        help="Directory containing cam2_bboxes.json and cam4_bboxes.json.",
    )
    parser.add_argument("--output_dir", type=str, required=True, help="Output root directory.")
    parser.add_argument("--reviewed", type=str, required=True, help="Path to reviewed.json.")
    parser.add_argument(
        "--blend_direction",
        type=str,
        choices=("top", "bottom"),
        default="top",
        help="补背部方向：top=填充目标人物顶部以上区域，bottom=填充底部以下区域",
    )
    parser.add_argument(
        "--start",
        type=int,
        default=None,
        help="起始帧索引（包含），不指定则从头开始",
    )
    parser.add_argument(
        "--end",
        type=int,
        default=None,
        help="结束帧索引（包含），不指定则处理到末尾",
    )
    parser.add_argument(
        "--directions",
        choices=("both", "c2_to_c4", "c4_to_c2"),
        default="both",
        help="Composite direction(s) to process.",
    )
    return parser.parse_args()


def process_frame(
    frame_idx: int,
    seg_dir: Path,
    flow_dir: Path,
    cam2_bboxes: dict[str, list[int]],
    cam4_bboxes: dict[str, list[int]],
    output_dir: Path,
    erp_h: int,
    erp_w: int,
    blend_direction: str,
    directions: str,
) -> None:
    bbox_key = str(frame_idx)
    cam2_bbox = cam2_bboxes.get(bbox_key)
    cam4_bbox = cam4_bboxes.get(bbox_key)
    frame_flow_dir = flow_dir / f"frame_{frame_idx:05d}"
    human_bbox_path = frame_flow_dir / "human_bbox.npy"

    if not human_bbox_path.exists():
        print(f"Warning: missing human_bbox, skipping frame {frame_idx:05d}")
        return

    human_bbox = np.load(human_bbox_path).tolist()

    cam2_origin: np.ndarray | None = None
    cam4_origin: np.ndarray | None = None
    if cam2_bbox is None:
        print(f"Warning: missing cam2 bbox for frame {frame_idx:05d}")
    else:
        cam2_origin = place_rgba_on_black(seg_dir / "cam2" / "rgba" / f"{frame_idx:05d}.png", cam2_bbox, erp_h, erp_w)
        if cam2_origin is None:
            print(f"Warning: failed to place cam2 rgba for frame {frame_idx:05d}")

    if cam4_bbox is None:
        print(f"Warning: missing cam4 bbox for frame {frame_idx:05d}")
    else:
        cam4_origin = place_rgba_on_black(seg_dir / "cam4" / "rgba" / f"{frame_idx:05d}.png", cam4_bbox, erp_h, erp_w)
        if cam4_origin is None:
            print(f"Warning: failed to place cam4 rgba for frame {frame_idx:05d}")

    if directions in ("both", "c2_to_c4"):
        warped_cam2 = composite_direction(
            frame_idx=frame_idx,
            source_camera="cam2",
            target_camera="cam4",
            flow_name="semantic_flow_c2_to_c4.npy",
            seg_dir=seg_dir,
            flow_dir=flow_dir,
            source_bboxes=cam2_bboxes,
            target_bboxes=cam4_bboxes,
            human_bbox=human_bbox,
            erp_h=erp_h,
            erp_w=erp_w,
        )
        if warped_cam2 is None:
            print(f"Warning: failed to compose cam2_to_cam4 for frame {frame_idx:05d}")

        if warped_cam2 is not None:
            warp_dir = output_dir / "cam2_to_cam4" / "warp"
            warp_dir.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(warp_dir / f"{frame_idx:05d}.jpg"), warped_cam2)
        if cam2_origin is not None:
            cam2_origin_dir = output_dir / "cam2_to_cam4" / "cam2_origin"
            cam2_origin_dir.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(cam2_origin_dir / f"{frame_idx:05d}.jpg"), cam2_origin)
        if cam4_origin is not None:
            cam4_origin_dir = output_dir / "cam2_to_cam4" / "cam4_origin"
            cam4_origin_dir.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(cam4_origin_dir / f"{frame_idx:05d}.jpg"), cam4_origin)
        if warped_cam2 is not None and cam4_origin is not None:
            blend_dir = output_dir / "cam2_to_cam4" / "blend"
            blend_dir.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(
                str(blend_dir / f"{frame_idx:05d}.jpg"),
                blend_two_humans(warp=warped_cam2, target_origin=cam4_origin, direction=blend_direction),
            )

    if directions in ("both", "c4_to_c2"):
        warped_cam4 = composite_direction(
            frame_idx=frame_idx,
            source_camera="cam4",
            target_camera="cam2",
            flow_name="semantic_flow_c4_to_c2.npy",
            seg_dir=seg_dir,
            flow_dir=flow_dir,
            source_bboxes=cam4_bboxes,
            target_bboxes=cam2_bboxes,
            human_bbox=human_bbox,
            erp_h=erp_h,
            erp_w=erp_w,
        )
        if warped_cam4 is None:
            print(f"Warning: failed to compose cam4_to_cam2 for frame {frame_idx:05d}")

        if warped_cam4 is not None:
            warp_dir = output_dir / "cam4_to_cam2" / "warp"
            warp_dir.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(warp_dir / f"{frame_idx:05d}.jpg"), warped_cam4)
        if cam4_origin is not None:
            cam4_origin_dir = output_dir / "cam4_to_cam2" / "cam4_origin"
            cam4_origin_dir.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(cam4_origin_dir / f"{frame_idx:05d}.jpg"), cam4_origin)
        if cam2_origin is not None:
            cam2_origin_dir = output_dir / "cam4_to_cam2" / "cam2_origin"
            cam2_origin_dir.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(cam2_origin_dir / f"{frame_idx:05d}.jpg"), cam2_origin)
        if warped_cam4 is not None and cam2_origin is not None:
            blend_dir = output_dir / "cam4_to_cam2" / "blend"
            blend_dir.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(
                str(blend_dir / f"{frame_idx:05d}.jpg"),
                blend_two_humans(warp=warped_cam4, target_origin=cam2_origin, direction=blend_direction),
            )


def infer_erp_size(frame_indices: list[int], image_dir: Path) -> tuple[int, int]:
    for frame_idx in frame_indices:
        for camera_name in ("cam2", "cam4"):
            image_path = image_dir / camera_name / f"{frame_idx:05d}.jpg"
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image is None:
                continue
            return image.shape[0], image.shape[1]
    raise RuntimeError("Failed to infer ERP image size from input images")


def main() -> None:
    args = parse_args()

    seg_dir = Path(args.seg_dir)
    flow_dir = Path(args.flow_dir)
    image_dir = Path(args.image_dir)
    bbox_dir = Path(args.bbox_dir)
    output_dir = Path(args.output_dir)
    reviewed_path = Path(args.reviewed)

    reviewed_pairs = load_reviewed_entries(reviewed_path)
    frame_indices = [frame_idx for _, frame_idx in reviewed_pairs]
    if args.start is not None or args.end is not None:
        start = args.start if args.start is not None else frame_indices[0]
        end = args.end if args.end is not None else frame_indices[-1]
        frame_indices = [i for i in frame_indices if start <= i <= end]
        print(f"过滤后处理帧数: {len(frame_indices)}，范围 [{start}, {end}]")
    cam2_bboxes = load_bbox_dict(bbox_dir / "cam2_bboxes.json")
    cam4_bboxes = load_bbox_dict(bbox_dir / "cam4_bboxes.json")
    erp_h, erp_w = infer_erp_size(frame_indices, image_dir)

    with tqdm(frame_indices, desc="Compositing humans", unit="frame") as progress:
        for frame_idx in progress:
            process_frame(
                frame_idx=frame_idx,
                seg_dir=seg_dir,
                flow_dir=flow_dir,
                cam2_bboxes=cam2_bboxes,
                cam4_bboxes=cam4_bboxes,
                output_dir=output_dir,
                erp_h=erp_h,
                erp_w=erp_w,
                blend_direction=args.blend_direction,
                directions=args.directions,
            )


if __name__ == "__main__":
    main()


r'''
python src/script/test_function/debug037_composite_human.py \
  --seg_dir data_process/Output/20260311_riding/seg/equirect_crop_rotate_sam3 \
  --flow_dir data_process/Output/20260311_riding/semantic_flow/new_sil \
  --image_dir data_process/Output/20260311_riding/image/equirect_rotate \
  --bbox_dir data_process/Output/20260311_riding/image/equirect_crop_rotate \
  --output_dir data_process/Output/20260311_riding/composite_human/new_sil \
  --reviewed data_process/Input/20260311_riding/review.json \
  --blend_direction top \
  --directions c2_to_c4 
'''

r'''
python src/script/test_function/debug037_composite_human.py \
  --seg_dir data_process/Output/20260320_riding_dataset1/seg/equirect_crop_rotate_sam3 \
  --flow_dir data_process/Output/20260320_riding_dataset1/semantic_flow/new_sil \
  --image_dir data_process/Output/20260320_riding_dataset1/image/equirect_rotate \
  --bbox_dir data_process/Output/20260320_riding_dataset1/image/equirect_crop_rotate \
  --output_dir data_process/Output/20260320_riding_dataset1/composite_human/new_sil \
  --reviewed data_process/Input/20260320_riding/dataset1/review.json \
  --blend_direction bottom \
  --directions c4_to_c2
''' 
# 0311_riding top, 0320_riding_dataset1 bottom,  0320_riding_dataset1 bottom(不完善，可能要做改变) 
