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

from tools.fisheye_extractor.config import DS_PARAMS, DS_PARAMS_CAM2, DS_PARAMS_CAM4
from tools.fisheye_extractor.ds_model import project


ERP_WIDTH = 5760
ERP_HEIGHT = 2880
OUTPUT_WIDTH = 5760
OUTPUT_HEIGHT = 2880
JPEG_QUALITY = 95
ROTATION_MATRIX = np.array(
    [1, 0, 0, 0, 0, 1, 0, -1, 0],
    dtype=np.float64,
).reshape(3, 3)
R_EXTRINSIC_CAM2 = np.eye(3, dtype=np.float64)
R_EXTRINSIC_CAM4 = np.array(
    [-1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 1.0],
    dtype=np.float64,
).reshape(3, 3)


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


def _select_camera_params(camera_group: str) -> tuple[dict[str, float], dict[str, float]]:
    """Return DS parameter sets for cam2 and cam4."""
    if camera_group == "new":
        return DS_PARAMS_CAM2, DS_PARAMS_CAM4
    return DS_PARAMS, DS_PARAMS


def build_equirect_lut(
    ds_params: dict[str, float],
    width: int = ERP_WIDTH,
    height: int = ERP_HEIGHT,
    chunk_rows: int = 128,
    R_extrinsic: np.ndarray | None = None,
    desc: str = "Building LUT",
) -> tuple[np.ndarray, np.ndarray]:
    """Build an equirect-to-fisheye remap LUT once for a given camera."""
    map_x = np.full((height, width), -1.0, dtype=np.float32)
    map_y = np.full((height, width), -1.0, dtype=np.float32)

    u_coords = np.arange(width, dtype=np.float64)
    lon = (u_coords / float(width)) * (2.0 * np.pi) - np.pi
    sin_lon = np.sin(lon)
    cos_lon = np.cos(lon)

    for row_start in tqdm(
        range(0, height, chunk_rows),
        desc=desc,
        unit="chunk",
        leave=False,
    ):
        row_end = min(row_start + chunk_rows, height)
        v_coords = np.arange(row_start, row_end, dtype=np.float64)
        lat = (v_coords / float(height)) * np.pi - (np.pi / 2.0)

        cos_lat = np.cos(lat)[:, None]
        sin_lat = np.sin(lat)[:, None]

        ray = np.stack(
            [
                cos_lat * sin_lon[None, :],
                np.broadcast_to(sin_lat, (row_end - row_start, width)),
                cos_lat * cos_lon[None, :],
            ],
            axis=-1,
        )
        if R_extrinsic is not None:
            ray_rot = ray.reshape(-1, 3) @ R_extrinsic.T @ ROTATION_MATRIX.T
        else:
            ray_rot = ray.reshape(-1, 3) @ ROTATION_MATRIX.T
        fisheye_pixels, valid_mask = project(ray_rot, ds_params)

        pixels_chunk = fisheye_pixels.reshape(row_end - row_start, width, 2)
        valid_chunk = valid_mask.reshape(row_end - row_start, width)
        map_x_chunk = map_x[row_start:row_end]
        map_y_chunk = map_y[row_start:row_end]
        map_x_chunk[valid_chunk] = pixels_chunk[..., 0][valid_chunk]
        map_y_chunk[valid_chunk] = pixels_chunk[..., 1][valid_chunk]

    return map_x, map_y


def _convert_fisheye_to_equirect(
    frame: np.ndarray,
    map_x: np.ndarray,
    map_y: np.ndarray,
) -> np.ndarray:
    """Remap one fisheye frame to equirect and resize for output."""
    equirect = cv2.remap(
        frame,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    return cv2.resize(equirect, (OUTPUT_WIDTH, OUTPUT_HEIGHT), interpolation=cv2.INTER_AREA)


def _encode_jpeg(frame: np.ndarray) -> bytes | None:
    """Encode one frame to JPEG bytes with quality 95."""
    ok, encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
    if not ok:
        return None
    return encoded.tobytes()


def export_group(
    frame_indices: list[int],
    fisheye_dir: Path,
    cam2_dir: Path,
    cam4_dir: Path,
    cam2_lut: tuple[np.ndarray, np.ndarray],
    cam4_lut: tuple[np.ndarray, np.ndarray],
    progress: tqdm,
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
        cam2_frame = _convert_fisheye_to_equirect(main_frame, map_x_cam2, map_y_cam2)
        cam4_frame = _convert_fisheye_to_equirect(partner_frame, map_x_cam4, map_y_cam4)

        cam2_encoded = _encode_jpeg(cam2_frame)
        cam4_encoded = _encode_jpeg(cam4_frame)
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


def main() -> None:
    """Convert reviewed fisheye frames to equirect images for both cameras."""
    args = parse_args()
    reviewed_path = Path(args.reviewed)
    output_dir = Path(args.output_dir)
    fisheye_dir = Path(args.fisheye_dir)
    equirect_dir = output_dir
    cam2_dir = equirect_dir / "cam2"
    cam4_dir = equirect_dir / "cam4"
    cam2_dir.mkdir(parents=True, exist_ok=True)
    cam4_dir.mkdir(parents=True, exist_ok=True)

    frame_indices = load_reviewed_frame_indices(reviewed_path)

    cam2_params, cam4_params = _select_camera_params(args.camera)
    r_ext_cam2 = R_EXTRINSIC_CAM2 if args.apply_extrinsic else None
    r_ext_cam4 = R_EXTRINSIC_CAM4 if args.apply_extrinsic else None
    print(f"Building cam2 LUT with '{args.camera}' camera parameters...")
    cam2_lut = build_equirect_lut(cam2_params, R_extrinsic=r_ext_cam2, desc="Building cam2 LUT")
    print(f"Building cam4 LUT with '{args.camera}' camera parameters...")
    cam4_lut = build_equirect_lut(cam4_params, R_extrinsic=r_ext_cam4, desc="Building cam4 LUT")

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
        )


if __name__ == "__main__":
    main()

r'''
python src\script\test_function\debug021_run_fisheye_to_equirect.py `
  --reviewed data_process\Input\20260311_riding\review.json `
  --fisheye_dir data_process\Output\20260311_riding\image\fisheye `
  --output_dir data_process\Output\20260311_riding\image\equirect `
  --camera old `
  --apply_extrinsic
'''
