import os
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')

import json
import cv2
import numpy as np
from pathlib import Path
import config_script
import lib_pano.local_setting as ls
from lib_pano.calibration.rotate_equirect import rotate_equirectangular_image
from lib_pano.flow.debug_writer import DebugWriter
from lib_pano.flow.raft_flow import RAFTFlow
from lib_pano.flow.warp_strategy import WarpStrategy, MeshConfig, DistanceConfig
from lib_pano.stitch.flow_stitcher import stitch_with_flow

DATASET_ROOT = Path(r"data_process/Output/20260311_riding")
INPUT_DIR = DATASET_ROOT / "image" / "equirect_rotate_for_flow_use"
OUTPUT_DIR = DATASET_ROOT / "raft_stitch"
REVIEWED_PATH = Path(r"data_process/Input/20260311_riding/review.json")
DEBUG_LIMIT = 1
WARP_DIRECTION = "cam4_to_cam2"
USE_SEMANTIC_FLOW = True
SEMANTIC_FLOW_DIR = DATASET_ROOT / "semantic_flow" / "test_new_sil"
SEMANTIC_FLOW_WEIGHT_SCALE = 2.0
SEMANTIC_FLOW_CONF_THRESH = 0.05

RAFT_MODELS_PATH = os.path.join(ls.PROJECT_ROOT, 'tools/RAFT/models')

BIDIRECTIONAL = True
BLEND_THRESHOLD = 1
BLEND_ERODE_KERNEL_SIZE = 0
EXTEND_Y = 180
PHOTO_THRESHOLD = 80
RAFT_CONSISTENCY_THRESH = 40
FLOW_WEIGHT_MIN = 0.0
WARP_STRATEGY = WarpStrategy.MESH_UNIDIRECTIONAL
MESH_CONFIG = MeshConfig(
    grid_size=32,
    decay_radius=30.0,
    max_propagation=35.0,
    min_valid_ratio=0.03,
    feather_width=20.0,
    use_weighted_sampling=True,
    min_weight=0.02,
    min_weight_ratio=0.005,
)
# WARP_STRATEGY = WarpStrategy.DISTANCE_UNIDIRECTIONAL
# DISTANCE_CONFIG = DistanceConfig(
#     decay_radius_px=200.0,
#     max_propagation_px=400.0,
#     downsample_factor=2,
# )
USE_SEAM_REPAIR = True
SAVE_DEBUG = True
DEBUG_LEVEL = "minimal"  # options: "none", "minimal", "full"

# 1.交换x,y
ROTATION_MATRIX = np.array([
    1, 0, 0,
    0, 0, 1,
    0, 1, 0
], dtype=np.float64).reshape(3, 3)
ROTATION_INV = ROTATION_MATRIX.T


def load_frame_indices() -> list[int]:
    if REVIEWED_PATH.exists():
        with REVIEWED_PATH.open("r", encoding="utf-8") as f:
            reviewed = json.load(f)
        frame_indices = sorted({int(item["frame_idx"]) for item in reviewed})
        return frame_indices

    cam2_dir = INPUT_DIR / "cam2"
    frame_indices = []
    for pattern in ("*.png", "*.jpg"):
        for path in cam2_dir.glob(pattern):
            try:
                frame_indices.append(int(path.stem))
            except ValueError:
                continue
    return sorted(set(frame_indices))


def load_bbox_dense_semantic_flow(frame_idx: int, direction: str):
    if not USE_SEMANTIC_FLOW:
        return None

    frame_dir = SEMANTIC_FLOW_DIR / f"frame_{frame_idx:05d}"

    if direction == "cam2_to_cam4":
        flow_path = frame_dir / "semantic_flow_c2_to_c4.npy"
        conf_path = frame_dir / "semantic_flow_confidence_c2_to_c4.npy"
        bbox_path = frame_dir / "human_bbox.npy"
    elif direction == "cam4_to_cam2":
        flow_path = frame_dir / "semantic_flow_c4_to_c2.npy"
        conf_path = frame_dir / "semantic_flow_confidence_c4_to_c2.npy"
        bbox_path = frame_dir / "human_bbox.npy"
    else:
        raise ValueError(f"Unsupported direction: {direction}")

    flow = np.load(str(flow_path)).astype(np.float32)
    bbox = np.load(str(bbox_path)).astype(np.int32)
    conf = np.load(str(conf_path)).astype(np.float32)
    conf = np.clip(conf, 0.0, 1.0)

    x1, y1, x2, y2 = [int(v) for v in bbox.tolist()]
    expected_h = y2 - y1
    expected_w = x2 - x1

    if expected_h != flow.shape[0] or expected_w != flow.shape[1]:
        print(
            f"[warn] bbox size does not match semantic flow: "
            f"bbox=({expected_w},{expected_h}), flow={flow.shape[:2][::-1]}"
        )
        return None

    return {
        "flow": flow,
        "confidence": conf,
        "bbox": (x1, y1, x2, y2),
    }


def process_frame(frame_idx, raft_flow):
    cam2_path = INPUT_DIR / "cam2" / f"{frame_idx:05d}.png"
    cam4_path = INPUT_DIR / "cam4" / f"{frame_idx:05d}.png"
    cam2_valid_path = INPUT_DIR / "valid" / "cam2.png"
    cam4_valid_path = INPUT_DIR / "valid" / "cam4.png"


    cam2 = cv2.imread(str(cam2_path), cv2.IMREAD_COLOR)
    cam4 = cv2.imread(str(cam4_path), cv2.IMREAD_COLOR)

    cam2_valid = cv2.imread(str(cam2_valid_path), cv2.IMREAD_GRAYSCALE)
    cam4_valid = cv2.imread(str(cam4_valid_path), cv2.IMREAD_GRAYSCALE)

    output_dir = OUTPUT_DIR / f"frame_{frame_idx:05d}"
    output_dir.mkdir(parents=True, exist_ok=True)
    debug_writer = DebugWriter(
        debug_dir=str(output_dir),
        enabled=SAVE_DEBUG and DEBUG_LEVEL != "none",
        debug_level=DEBUG_LEVEL,
    )

    if WARP_DIRECTION == "cam4_to_cam2":
        img_source = cam4
        img_target = cam2
        raw_source_valid = cam4_valid
        raw_target_valid = cam2_valid
    elif WARP_DIRECTION == "cam2_to_cam4":
        img_source = cam2
        img_target = cam4
        raw_source_valid = cam2_valid
        raw_target_valid = cam4_valid
    else:
        raise ValueError(f"Unsupported WARP_DIRECTION: {WARP_DIRECTION}")

    source_valid = raw_source_valid
    target_valid = raw_target_valid

    if DEBUG_LEVEL == "full":
        debug_writer.write_mask("source_valid_mask.png", source_valid)
        debug_writer.write_mask("target_valid_mask.png", target_valid)

    semantic_prior = load_bbox_dense_semantic_flow(frame_idx, WARP_DIRECTION)

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
        photometric_threshold=PHOTO_THRESHOLD,
        photo_sigma=60.0,
        landed_invalid_multiplier=0.10,
        raft_invalid_multiplier=0.35,
        flow_weight_min=FLOW_WEIGHT_MIN,
        semantic_prior=semantic_prior,
        semantic_weight_scale=SEMANTIC_FLOW_WEIGHT_SCALE,
        semantic_conf_thresh=SEMANTIC_FLOW_CONF_THRESH,
        debug_dir=str(output_dir) if SAVE_DEBUG else None,
        debug_level=DEBUG_LEVEL,
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
    raft_flow = RAFTFlow(model_path=os.path.join(RAFT_MODELS_PATH, 'raft-things.pth'), consistency_thresh=RAFT_CONSISTENCY_THRESH)

    success_count = 0
    frame_indices = load_frame_indices()
    if DEBUG_LIMIT is not None:
        frame_indices = frame_indices[:DEBUG_LIMIT]

    for frame_idx in frame_indices:
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
