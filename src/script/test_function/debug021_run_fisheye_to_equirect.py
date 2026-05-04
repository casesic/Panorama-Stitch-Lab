from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

import config_script

PROJECT_ROOT = Path(config_script.PROJECT_ROOT)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lib_pano.camera.fisheye_equirect import (
    OUTPUT_HEIGHT,
    OUTPUT_WIDTH,
    R_EXTRINSIC_CAM2,
    R_EXTRINSIC_CAM4,
    apply_valid_mask_to_image,
    build_equirect_lut,
    build_erp_valid_mask_from_lut_and_lens,
    build_fisheye_lens_mask,
    build_valid_mask_from_lut,
    convert_fisheye_to_equirect,
    encode_jpeg,
    make_horizontal_flow_valid_mask,
    select_camera_params,
)


JPEG_QUALITY = 95


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for fisheye-to-equirect export."""
    parser = argparse.ArgumentParser(description="Convert reviewed fisheye frames to equirect images.")
    parser.add_argument(
        "--reviewed",
        type=str,
        required=True,
        help="Path to reviewed.json.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output root directory, e.g. data_process/Output/20260311_riding.",
    )
    parser.add_argument(
        "--fisheye_dir",
        type=str,
        required=True,
        help="Fisheye image root directory, e.g. data_process/Output/20260320_riding/image/fisheye",
    )
    parser.add_argument(
        "--camera",
        type=str,
        choices=["old", "new"],
        default="old",
        help="Camera parameter group to use.",
    )
    parser.add_argument(
        "--apply_extrinsic",
        action="store_true",
        help="Apply camera extrinsic rotation before fisheye projection.",
    )
    parser.add_argument(
        "--lens_threshold",
        type=int,
        default=5,
        help="Threshold used to detect fisheye lens content mask.",
    )
    parser.add_argument(
        "--lens_close_kernel",
        type=int,
        default=51,
        help="Morphological close kernel size for fisheye lens mask.",
    )
    parser.add_argument(
        "--lens_erode_px",
        type=int,
        default=8,
        help="Erode fisheye lens mask before ERP remap to remove edge flare.",
    )
    parser.add_argument(
        "--export_flow_use",
        action="store_true",
        help="Also export flow-use equirect images with trimmed valid masks.",
    )
    parser.add_argument(
        "--flow_output_dir",
        type=str,
        default=None,
        help="Output directory for flow-use equirect images. If omitted, use sibling directory named equirect_rotate_for_flow_use.",
    )
    parser.add_argument(
        "--flow_trim_margin_px",
        type=int,
        default=40,
        help="Trim margin in pixels for flow-use valid mask.",
    )
    parser.add_argument(
        "--flow_trim_percentile",
        type=float,
        default=5.0,
        help="Percentile used to estimate horizontal flow valid boundary.",
    )
    return parser.parse_args()


def load_reviewed_frame_indices(reviewed_path: Path) -> list[int]:
    """Load reviewed.json and return deduplicated, sorted frame indices."""
    with reviewed_path.open("r", encoding="utf-8") as file:
        reviewed = json.load(file)

    if not isinstance(reviewed, list):
        raise ValueError("reviewed.json must contain a list")

    frame_indices: set[int] = set()
    for item in reviewed:
        if not isinstance(item, dict):
            raise ValueError("Each reviewed entry must be a dict")
        frame_indices.add(int(item["frame_idx"]))

    return sorted(frame_indices)


def find_first_existing_pair(frame_indices: list[int], fisheye_dir: Path) -> tuple[Path, Path]:
    """Find the first reviewed frame where both camera images exist."""
    for frame_idx in frame_indices:
        cam2_path = fisheye_dir / "cam2" / f"{frame_idx:05d}.jpg"
        cam4_path = fisheye_dir / "cam4" / f"{frame_idx:05d}.jpg"
        if cam2_path.exists() and cam4_path.exists():
            return cam2_path, cam4_path

    raise RuntimeError("No existing cam2/cam4 fisheye image pair found.")


def export_group(
    frame_indices: list[int],
    fisheye_dir: Path,
    cam2_dir: Path,
    cam4_dir: Path,
    cam2_lut: tuple,
    cam4_lut: tuple,
    progress: tqdm,
    flow_cam2_dir: Path | None = None,
    flow_cam4_dir: Path | None = None,
    flow_valid_cam2: np.ndarray | None = None,
    flow_valid_cam4: np.ndarray | None = None,
    export_flow_use: bool = False,
) -> None:
    """Export reviewed fisheye image pairs to equirect images for both cameras."""
    map_x_cam2, map_y_cam2 = cam2_lut
    map_x_cam4, map_y_cam4 = cam4_lut

    for frame_idx in frame_indices:
        main_frame = cv2.imread(str(fisheye_dir / "cam2" / f"{frame_idx:05d}.jpg"))
        partner_frame = cv2.imread(str(fisheye_dir / "cam4" / f"{frame_idx:05d}.jpg"))
        if main_frame is None or partner_frame is None:
            print(f"Warning: missing fisheye image pair, skipping frame {frame_idx}")
            progress.update(1)
            continue

        main_output = cam2_dir / f"{frame_idx:05d}.jpg"
        partner_output = cam4_dir / f"{frame_idx:05d}.jpg"
        cam2_frame = convert_fisheye_to_equirect(main_frame, map_x_cam2, map_y_cam2)
        cam4_frame = convert_fisheye_to_equirect(partner_frame, map_x_cam4, map_y_cam4)

        cam2_encoded = encode_jpeg(cam2_frame, quality=JPEG_QUALITY)
        cam4_encoded = encode_jpeg(cam4_frame, quality=JPEG_QUALITY)
        if cam2_encoded is None or cam4_encoded is None:
            print(f"Warning: failed to encode paired frames, skipping frame {frame_idx}")
            progress.update(1)
            continue

        try:
            main_output.write_bytes(cam2_encoded)
            partner_output.write_bytes(cam4_encoded)
        except OSError:
            print(f"Warning: failed to save paired frames, skipping frame {frame_idx}")
            progress.update(1)
            continue

        if export_flow_use:
            if (
                flow_cam2_dir is None
                or flow_cam4_dir is None
                or flow_valid_cam2 is None
                or flow_valid_cam4 is None
            ):
                raise ValueError("Flow-use export requested without output dirs or valid masks.")

            flow_cam2 = apply_valid_mask_to_image(cam2_frame, flow_valid_cam2, background_value=0)
            flow_cam4 = apply_valid_mask_to_image(cam4_frame, flow_valid_cam4, background_value=0)
            flow_cam2_output = flow_cam2_dir / f"{frame_idx:05d}.png"
            flow_cam4_output = flow_cam4_dir / f"{frame_idx:05d}.png"
            wrote_flow_cam2 = cv2.imwrite(str(flow_cam2_output), flow_cam2)
            wrote_flow_cam4 = cv2.imwrite(str(flow_cam4_output), flow_cam4)
            if not wrote_flow_cam2 or not wrote_flow_cam4:
                print(f"Warning: failed to save flow-use frames for frame {frame_idx}")

        progress.update(1)


def main() -> None:
    """Convert reviewed fisheye frames to equirect images for both cameras."""
    args = parse_args()
    reviewed_path = Path(args.reviewed)
    output_dir = Path(args.output_dir)
    fisheye_dir = Path(args.fisheye_dir)
    equirect_dir = output_dir
    cam2_dir = equirect_dir / "cam2"
    cam4_dir = equirect_dir / "cam4"
    valid_dir = equirect_dir / "valid"
    cam2_dir.mkdir(parents=True, exist_ok=True)
    cam4_dir.mkdir(parents=True, exist_ok=True)
    valid_dir.mkdir(parents=True, exist_ok=True)

    if args.flow_output_dir is not None:
        flow_dir = Path(args.flow_output_dir)
    else:
        flow_dir = equirect_dir.with_name(equirect_dir.name + "_for_flow_use")
    flow_cam2_dir = flow_dir / "cam2"
    flow_cam4_dir = flow_dir / "cam4"
    flow_valid_dir = flow_dir / "valid"
    if args.export_flow_use:
        flow_cam2_dir.mkdir(parents=True, exist_ok=True)
        flow_cam4_dir.mkdir(parents=True, exist_ok=True)
        flow_valid_dir.mkdir(parents=True, exist_ok=True)

    frame_indices = load_reviewed_frame_indices(reviewed_path)

    cam2_params, cam4_params = select_camera_params(args.camera)
    r_ext_cam2 = R_EXTRINSIC_CAM2 if args.apply_extrinsic else None
    r_ext_cam4 = R_EXTRINSIC_CAM4 if args.apply_extrinsic else None
    print(f"Building cam2 LUT with '{args.camera}' camera parameters...")
    cam2_lut = build_equirect_lut(cam2_params, R_extrinsic=r_ext_cam2, desc="Building cam2 LUT")
    print(f"Building cam4 LUT with '{args.camera}' camera parameters...")
    cam4_lut = build_equirect_lut(cam4_params, R_extrinsic=r_ext_cam4, desc="Building cam4 LUT")

    cam2_sample_path, cam4_sample_path = find_first_existing_pair(frame_indices, fisheye_dir)
    cam2_sample = cv2.imread(str(cam2_sample_path), cv2.IMREAD_COLOR)
    cam4_sample = cv2.imread(str(cam4_sample_path), cv2.IMREAD_COLOR)
    if cam2_sample is None or cam4_sample is None:
        raise RuntimeError(
            f"Failed to read fisheye sample pair: {cam2_sample_path}, {cam4_sample_path}"
        )

    cam2_lens_mask = build_fisheye_lens_mask(
        cam2_sample,
        threshold=args.lens_threshold,
        close_kernel=args.lens_close_kernel,
        erode_px=args.lens_erode_px,
    )
    cam4_lens_mask = build_fisheye_lens_mask(
        cam4_sample,
        threshold=args.lens_threshold,
        close_kernel=args.lens_close_kernel,
        erode_px=args.lens_erode_px,
    )

    valid_cam2 = build_erp_valid_mask_from_lut_and_lens(
        cam2_lut,
        cam2_lens_mask,
        output_size=(OUTPUT_WIDTH, OUTPUT_HEIGHT),
    )
    valid_cam4 = build_erp_valid_mask_from_lut_and_lens(
        cam4_lut,
        cam4_lens_mask,
        output_size=(OUTPUT_WIDTH, OUTPUT_HEIGHT),
    )
    cv2.imwrite(str(valid_dir / "cam2.png"), valid_cam2)
    cv2.imwrite(str(valid_dir / "cam4.png"), valid_cam4)

    flow_valid_cam2 = None
    flow_valid_cam4 = None
    if args.export_flow_use:
        flow_valid_cam2 = make_horizontal_flow_valid_mask(
            valid_cam2,
            camera_name="cam2",
            trim_edge_margin_px=args.flow_trim_margin_px,
            trim_edge_percentile=args.flow_trim_percentile,
        )
        flow_valid_cam4 = make_horizontal_flow_valid_mask(
            valid_cam4,
            camera_name="cam4",
            trim_edge_margin_px=args.flow_trim_margin_px,
            trim_edge_percentile=args.flow_trim_percentile,
        )
        cv2.imwrite(str(flow_valid_dir / "cam2.png"), flow_valid_cam2)
        cv2.imwrite(str(flow_valid_dir / "cam4.png"), flow_valid_cam4)

        flow_debug_valid_dir = flow_valid_dir / "debug"
        flow_debug_valid_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(flow_debug_valid_dir / "base_valid_cam2.png"), valid_cam2)
        cv2.imwrite(str(flow_debug_valid_dir / "base_valid_cam4.png"), valid_cam4)
        cv2.imwrite(str(flow_debug_valid_dir / "flow_valid_cam2.png"), flow_valid_cam2)
        cv2.imwrite(str(flow_debug_valid_dir / "flow_valid_cam4.png"), flow_valid_cam4)

    debug_valid_dir = valid_dir / "debug"
    debug_valid_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(debug_valid_dir / "fisheye_lens_cam2.png"), cam2_lens_mask)
    cv2.imwrite(str(debug_valid_dir / "fisheye_lens_cam4.png"), cam4_lens_mask)

    lut_valid_cam2 = build_valid_mask_from_lut(
        cam2_lut,
        output_size=(OUTPUT_WIDTH, OUTPUT_HEIGHT),
    )
    lut_valid_cam4 = build_valid_mask_from_lut(
        cam4_lut,
        output_size=(OUTPUT_WIDTH, OUTPUT_HEIGHT),
    )
    cv2.imwrite(str(debug_valid_dir / "lut_valid_cam2.png"), lut_valid_cam2)
    cv2.imwrite(str(debug_valid_dir / "lut_valid_cam4.png"), lut_valid_cam4)

    total_frames = len(frame_indices)
    with tqdm(total=total_frames, desc="Exporting equirect frames", unit="frame") as progress:
        export_group(
            frame_indices,
            fisheye_dir,
            cam2_dir,
            cam4_dir,
            cam2_lut,
            cam4_lut,
            progress,
            flow_cam2_dir=flow_cam2_dir,
            flow_cam4_dir=flow_cam4_dir,
            flow_valid_cam2=flow_valid_cam2,
            flow_valid_cam4=flow_valid_cam4,
            export_flow_use=args.export_flow_use,
        )


if __name__ == "__main__":
    main()

r'''
python src\script\test_function\debug021_run_fisheye_to_equirect.py `
  --reviewed data_process\Input\20260311_riding\review.json `
  --fisheye_dir data_process\Output\20260311_riding\image\fisheye `
  --output_dir data_process\Output\20260311_riding\image\equirect_rotate `
  --camera old `
  --apply_extrinsic `
  --export_flow_use
'''
