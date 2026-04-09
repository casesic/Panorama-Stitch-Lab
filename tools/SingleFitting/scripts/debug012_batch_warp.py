import os

os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')

import numpy as np
import cv2
from pathlib import Path
from scipy.ndimage import distance_transform_edt

BATCH_FLOW_DIR = r"data_process/Output/20260311_riding/batch_flow"
STITCH_DIR = r"data_process/Output/20260311_riding/stitch_batch"
INDEX_START = 513
NUM_FRAMES = 5
DECAY_RADIUS = 200.0
SAVE_VIS = True
WARP_DIRECTION = "cam2_to_cam4"  # 可选 "cam4_to_cam2" 或 "cam2_to_cam4"


def normalize_to_uint8(image: np.ndarray) -> np.ndarray:
    image = image.astype(np.float32)
    max_value = float(image.max())
    if max_value <= 0.0:
        return np.zeros(image.shape, dtype=np.uint8)
    return np.clip(image / max_value * 255.0, 0.0, 255.0).astype(np.uint8)


def process_frame(frame_idx, batch_flow_dir, stitch_dir, decay_radius, save_vis):
    flow_dir = Path(batch_flow_dir) / f"frame_{frame_idx:05d}"
    frame_dir = Path(stitch_dir) / f"frame_{frame_idx:05d}"
    flow_path = flow_dir / "semantic_flow.npy"
    conf_path = flow_dir / "semantic_flow_confidence.npy"
    bbox_path = flow_dir / "human_bbox.npy"
    cam2_path = frame_dir / f"rotated_cam2_frame_{frame_idx:05d}.png"
    cam4_path = frame_dir / f"rotated_cam4_frame_{frame_idx:05d}.png"

    if not flow_path.exists() or not conf_path.exists() or not bbox_path.exists():
        print(f"[warn] Missing flow data for frame {frame_idx:05d}")
        return {"frame_idx": frame_idx, "skipped": True, "reason": "no_flow"}

    if not cam2_path.exists() or not cam4_path.exists():
        print(f"[warn] Missing stitched image for frame {frame_idx:05d}")
        return {"frame_idx": frame_idx, "skipped": True, "reason": "no_image"}

    semantic_flow_roi = np.load(flow_path)
    semantic_confidence_roi = np.load(conf_path)
    human_bbox = np.load(bbox_path)
    if WARP_DIRECTION == "cam4_to_cam2":
        rotated_cam0 = cv2.imread(str(cam2_path))
        rotated_cam1 = cv2.imread(str(cam4_path))
    elif WARP_DIRECTION == "cam2_to_cam4":
        rotated_cam0 = cv2.imread(str(cam4_path))
        rotated_cam1 = cv2.imread(str(cam2_path))
    else:
        raise ValueError(f"Unsupported WARP_DIRECTION: {WARP_DIRECTION}")

    roi_x1, roi_y1, roi_x2, roi_y2 = [int(v) for v in human_bbox.tolist()]
    erp_h, erp_w = rotated_cam1.shape[:2]
    full_human_mask = np.zeros((erp_h, erp_w), dtype=bool)
    full_human_mask[roi_y1:roi_y2, roi_x1:roi_x2] = semantic_confidence_roi > 0
    full_flow = np.zeros((erp_h, erp_w, 2), dtype=np.float32)
    full_flow[roi_y1:roi_y2, roi_x1:roi_x2] = semantic_flow_roi.astype(np.float32)

    if not full_human_mask.any():
        print(f"[warn] Empty human mask for frame {frame_idx:05d}")
        return {"frame_idx": frame_idx, "skipped": True, "reason": "empty_mask"}

    dist_full, nearest_indices = distance_transform_edt(~full_human_mask, return_distances=True, return_indices=True)
    nearest_y = nearest_indices[0]
    nearest_x = nearest_indices[1]
    propagated_flow = full_flow[nearest_y, nearest_x]
    weight = np.clip(1.0 - dist_full / decay_radius, 0.0, 1.0).astype(np.float32)
    flow_with_decay = propagated_flow * weight[..., None]
    flow_with_decay[..., 0] = cv2.GaussianBlur(flow_with_decay[..., 0], (51, 51), 0)
    flow_with_decay[..., 1] = cv2.GaussianBlur(flow_with_decay[..., 1], (51, 51), 0)
    flow_with_decay[full_human_mask] = full_flow[full_human_mask]

    grid_x, grid_y = np.meshgrid(
        np.arange(erp_w, dtype=np.float32),
        np.arange(erp_h, dtype=np.float32),
    )
    map_x = grid_x + flow_with_decay[..., 0]
    map_y = grid_y + flow_with_decay[..., 1]
    source_preprocessed = cv2.remap(
        rotated_cam1,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )

    flow_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(flow_dir / "source_preprocessed.png"), source_preprocessed)

    if save_vis:
        source_preprocessed_overlay = cv2.addWeighted(
            rotated_cam0,
            0.5,
            source_preprocessed,
            0.5,
            0.0,
        )
        flow_magnitude = np.linalg.norm(flow_with_decay, axis=2)
        diag_flow_magnitude = normalize_to_uint8(flow_magnitude)
        diag_flow_weight = np.clip(weight * 255.0, 0.0, 255.0).astype(np.uint8)
        cv2.imwrite(str(flow_dir / "source_preprocessed_overlay.png"), source_preprocessed_overlay)
        cv2.imwrite(str(flow_dir / "flow_magnitude.png"), diag_flow_magnitude)
        # cv2.imwrite(str(flow_dir / "flow_weight.png"), diag_flow_weight)

    return {
        "frame_idx": frame_idx,
        "skipped": False,
        "valid_flow_pixels": int(full_human_mask.sum()),
        "valid_warped_pixels": int((source_preprocessed.sum(axis=2) > 0).sum()),
        "flow_magnitude_max": float(np.linalg.norm(flow_with_decay, axis=2).max()),
    }


def main():
    results = []
    for frame_idx in range(INDEX_START, INDEX_START + NUM_FRAMES):
        result = process_frame(
            frame_idx=frame_idx,
            batch_flow_dir=BATCH_FLOW_DIR,
            stitch_dir=STITCH_DIR,
            decay_radius=DECAY_RADIUS,
            save_vis=SAVE_VIS,
        )
        results.append(result)
        print(f"[frame {frame_idx:05d}] {result['reason'] if result.get('skipped') else 'ok'}")

    success_count = sum(1 for item in results if not item["skipped"])
    no_flow_count = sum(1 for item in results if item.get("reason") == "no_flow")
    no_image_count = sum(1 for item in results if item.get("reason") == "no_image")
    empty_mask_count = sum(1 for item in results if item.get("reason") == "empty_mask")

    print(f"successful frames: {success_count}")
    print(f"no_flow: {no_flow_count}")
    print(f"no_image: {no_image_count}")
    print(f"empty_mask: {empty_mask_count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

'''
首先大部分的重建结果左腿由于被遮挡,无法检测到关节点, 都是没有对齐的, 所以左腿的拟合效果大多都不好，手部也不没有精准重建的
512帧也可以去掉, cam2有完整的
513帧还可以, 但头部有些许不对齐, 主要原因cam4的身体不完整, 拟合效果不是很好
514帧还可以
515帧还可以
516帧不太行, cam2的重建效果不太好
517帧不太行, cam2的重建效果不太好, 还有sam2分割, 有的属于人体的部分, 没有割下来
518帧效果非常不好, cam2和cam4的重建效果都不好
519帧可以去掉, cam4有完整的
'''

'''
效果说明:
取了500-529的30帧
由于人的左腿被遮挡, sam2检测不到, 所以帧的左腿基本都是缺失的
500-512帧直接从cam2取, 因为cam2有完整的人体
519-529帧直接从cam4取, 因为cam4有完整的人体
513-518帧, 通过语义引导的光流传播, 从cam4传播到cam2, 进行补全, 具体每帧效果如下:
首先大部分的重建结果左腿由于被遮挡, 无法检测到关节点, 左腿基本都是没有对齐的, 
所以左腿的拟合效果大多都不好, 手部目前也不没有精准重建的
513帧还可以, 但头部有些许不对齐, 主要原因cam4的身体不太完整, 重建效果不是很好
514帧还可以
515帧还可以
516帧不太行, cam2的重建躯干大小拟合的不太好
517帧不太行, cam2的重建躯干大小拟合的不太好, 还有sam2分割, 有的属于人体的部分, 没有割下来
518帧效果非常不好, cam2的重建躯干大小拟合的不太好, cam4的重建左腿直接向外翻, 比较离谱(可以去掉)
总的来说, 还是要优化人体的重建效果, 这是最重要的
'''
