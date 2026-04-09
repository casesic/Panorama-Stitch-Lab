import os
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')

import cv2
import numpy as np
from pathlib import Path
import config_script
import lib_pano.local_setting as ls
from lib_pano.calibration.rotate_equirect import rotate_equirectangular_image
from lib_pano.flow.raft_flow import RAFTFlow
from lib_pano.flow.warp_strategy import WarpStrategy, MeshConfig, DistanceConfig
from lib_pano.stitch.flow_stitcher import stitch_with_flow

INPUT_DIR = r"data_process/Output/20260311_riding/stitch_raw"
OUTPUT_DIR = r"data_process/Output/20260311_riding/raft_stitch"
INDEX_START = 513
NUM_FRAMES = 1
WARP_DIRECTION = "cam4_to_cam2"

RAFT_MODELS_PATH = os.path.join(ls.PROJECT_ROOT, 'tools/RAFT/models')

BIDIRECTIONAL = True
BLEND_THRESHOLD = 1
BLEND_ERODE_KERNEL_SIZE = 0
EXTEND_Y = 180
WARP_STRATEGY = WarpStrategy.MESH_UNIDIRECTIONAL
MESH_CONFIG = MeshConfig(
    grid_size=32,
    decay_radius=30.0,
    max_propagation=35.0,
    feather_width=20.0,
)
# WARP_STRATEGY = WarpStrategy.DISTANCE_UNIDIRECTIONAL
# DISTANCE_CONFIG = DistanceConfig(
#     decay_radius_px=200.0,
#     max_propagation_px=400.0,
#     downsample_factor=2,
# )
USE_SEAM_REPAIR = True

ROTATION_MATRIX = np.array([1, 0, 0, 0, 0, 1, 0, -1, 0], dtype=np.float64).reshape(3, 3)
ROTATION_INV = ROTATION_MATRIX.T


def process_frame(frame_idx, raft_flow):
    frame_dir = Path(INPUT_DIR) / f"frame_{frame_idx:05d}"
    source_path = frame_dir / f"rotated_cam4_frame_{frame_idx:05d}.png"
    target_path = frame_dir / f"rotated_cam2_frame_{frame_idx:05d}.png"
    source_valid_path = frame_dir / f"rotated_valid_cam4_frame_{frame_idx:05d}.png"
    target_valid_path = frame_dir / f"rotated_valid_cam2_frame_{frame_idx:05d}.png"

    if (
        not source_path.exists()
        or not target_path.exists()
        or not source_valid_path.exists()
        or not target_valid_path.exists()
    ):
        print(f"[warn] Missing input files for frame {frame_idx:05d}")
        return False

    source = cv2.imread(str(source_path), cv2.IMREAD_COLOR)
    target = cv2.imread(str(target_path), cv2.IMREAD_COLOR)
    source_valid = cv2.imread(str(source_valid_path), cv2.IMREAD_GRAYSCALE)
    target_valid = cv2.imread(str(target_valid_path), cv2.IMREAD_GRAYSCALE)

    if source is None or target is None or source_valid is None or target_valid is None:
        print(f"[warn] Failed to read input files for frame {frame_idx:05d}")
        return False

    output_dir = Path(OUTPUT_DIR) / f"frame_{frame_idx:05d}"
    output_dir.mkdir(parents=True, exist_ok=True)

    if WARP_DIRECTION == "cam4_to_cam2":
        img_target = target
        img_source = source
    elif WARP_DIRECTION == "cam2_to_cam4":
        img_target = source
        img_source = target
    else:
        raise ValueError(f"Unsupported WARP_DIRECTION: {WARP_DIRECTION}")

    result, _ = stitch_with_flow(
        img_source=img_source,
        img_target=img_target,
        raft_flow=raft_flow,
        source_valid_mask=source_valid,
        target_valid_mask=target_valid,
        strategy=WARP_STRATEGY,
        mesh_config=MESH_CONFIG,
        distance_config=DistanceConfig(),
        bidirectional=BIDIRECTIONAL,
        extend_y=EXTEND_Y,
        blend_threshold=BLEND_THRESHOLD,
        blend_erode_kernel_size=BLEND_ERODE_KERNEL_SIZE,
        use_seam_repair=USE_SEAM_REPAIR,
        debug_dir=str(output_dir),
    )

    output_path = output_dir / f"raft_{WARP_DIRECTION}_frame_{frame_idx:05d}.png"
    cv2.imwrite(str(output_path), result)

    result_unrotated = rotate_equirectangular_image(
        result,
        rotation_matrix=ROTATION_INV,
        interpolation=cv2.INTER_LINEAR,
        border_value=0,
    )
    unrotated_path = output_dir / f"raft_{WARP_DIRECTION}_unrotated_frame_{frame_idx:05d}.png"
    cv2.imwrite(str(unrotated_path), result_unrotated)
    return True


def main():
    print("Initializing RAFTFlow...")
    raft_flow = RAFTFlow(model_path=os.path.join(RAFT_MODELS_PATH, 'raft-things.pth'), consistency_thresh=40)

    success_count = 0
    for frame_idx in range(INDEX_START, INDEX_START + NUM_FRAMES):
        ok = process_frame(frame_idx, raft_flow)
        if ok:
            success_count += 1
            print(f"[frame {frame_idx:05d}] done")
        else:
            print(f"[frame {frame_idx:05d}] skipped")

    print(f"successful frames: {success_count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
