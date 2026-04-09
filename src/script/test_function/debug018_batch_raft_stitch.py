import os

os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')

import cv2
import numpy as np
import config_script
import lib_pano.local_setting as ls
from lib_pano.calibration.rotate_equirect import rotate_equirectangular_image
from lib_pano.flow.raft_align import RAFTAligner
from lib_pano.stitch.cumulative_stitch import stitch_cumulative
from pathlib import Path

INPUT_DIR = r"data_process/Output/20260311_riding/stitch_raw"
OUTPUT_DIR = r"data_process/Output/20260311_riding/raft_stitch"
INDEX_START = 513
NUM_FRAMES = 1
WARP_DIRECTION = "cam4_to_cam2"  # 可选 "cam4_to_cam2" 或 "cam2_to_cam4"

RAFT_MODELS_PATH = os.path.join(ls.PROJECT_ROOT, 'tools/RAFT/models')


STEP_IDX = 1
USE_MESH_WARP = True
DISTRIBUTE_WARP = True
USE_BIDIRECTIONAL = True
BLEND_THRESHOLD = 1
BLEND_ERODE_KERNEL_SIZE = 0
USE_SOFT_OVERLAP = True
EXTEND_Y = 180
MESH_DECAY_RADIUS = 6
MESH_MAX_PROPAGATION = 8
ROTATION_MATRIX = np.array([1, 0, 0, 0, 0, 1, 0, -1, 0], dtype=np.float64).reshape(3, 3)
ROTATION_INV = ROTATION_MATRIX.T


def process_frame(frame_idx, aligner):
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
        canvas = target
        new_shard = source
    elif WARP_DIRECTION == "cam2_to_cam4":
        canvas = source
        new_shard = target
    else:
        raise ValueError(f"Unsupported WARP_DIRECTION: {WARP_DIRECTION}")

    result = stitch_cumulative(
        canvas=canvas,
        new_shard=new_shard,
        aligner=aligner,
        source_valid_mask=source_valid,
        target_valid_mask=target_valid,
        debug_dir=str(output_dir),
        step_idx=STEP_IDX,
        use_mesh_warp=USE_MESH_WARP,
        distribute_warp=DISTRIBUTE_WARP,
        use_bidirectional=USE_BIDIRECTIONAL,
        blend_threshold=BLEND_THRESHOLD,
        blend_erode_kernel_size=BLEND_ERODE_KERNEL_SIZE,
        extend_y=EXTEND_Y if USE_SOFT_OVERLAP else 0,
        mesh_decay_radius=MESH_DECAY_RADIUS,
        mesh_max_propagation=MESH_MAX_PROPAGATION,
    )

    output_path = output_dir / f"raft_{WARP_DIRECTION}_frame_{frame_idx:05d}.png"
    cv2.imwrite(str(output_path), result)
    result_unrotated = rotate_equirectangular_image(
        result, rotation_matrix=ROTATION_INV, interpolation=cv2.INTER_LINEAR, border_value=0
    )
    unrotated_path = output_dir / f"raft_{WARP_DIRECTION}_unrotated_frame_{frame_idx:05d}.png"
    cv2.imwrite(str(unrotated_path), result_unrotated)
    return True


def main():
    print("Initializing RAFTAligner...")
    aligner = RAFTAligner(model_path=os.path.join(RAFT_MODELS_PATH, 'raft-things.pth'))

    success_count = 0
    for frame_idx in range(INDEX_START, INDEX_START + NUM_FRAMES):
        ok = process_frame(frame_idx, aligner)
        if ok:
            success_count += 1
            print(f"[frame {frame_idx:05d}] done")
        else:
            print(f"[frame {frame_idx:05d}] skipped")

    print(f"successful frames: {success_count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
