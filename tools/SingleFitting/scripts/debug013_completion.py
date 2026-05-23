import os

os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')

import numpy as np
import cv2
from pathlib import Path

BATCH_FLOW_DIR = r"data_process/Output/20260311_riding/batch_flow"
STITCH_DIR = r"data_process/Output/20260311_riding/stitch_batch"
OUTPUT_DIR = r"data_process/Output/20260311_riding/completion"
INDEX_START = 513
NUM_FRAMES = 5
SAVE_VIS = True
WARP_DIRECTION = "cam2_to_cam4"  # 可选 "cam2_to_cam4" 或 "cam4_to_cam2"


def process_frame(frame_idx):
    if WARP_DIRECTION == "cam2_to_cam4":
        ref_path = Path(STITCH_DIR) / f"frame_{frame_idx:05d}" / f"rotated_cam4_frame_{frame_idx:05d}.png"
        warp_path = Path(BATCH_FLOW_DIR) / f"frame_{frame_idx:05d}" / "source_preprocessed.png"
        mask_path = Path(STITCH_DIR) / f"frame_{frame_idx:05d}" / f"rotated_valid_cam4_frame_{frame_idx:05d}.png"
        out_name = "cam4_completed.png"
        labels = ("cam4 original", "cam2 warped", "completed")
    elif WARP_DIRECTION == "cam4_to_cam2":
        ref_path = Path(STITCH_DIR) / f"frame_{frame_idx:05d}" / f"rotated_cam2_frame_{frame_idx:05d}.png"
        warp_path = Path(BATCH_FLOW_DIR) / f"frame_{frame_idx:05d}" / "source_preprocessed.png"
        mask_path = Path(STITCH_DIR) / f"frame_{frame_idx:05d}" / f"rotated_valid_cam2_frame_{frame_idx:05d}.png"
        out_name = "cam2_completed.png"
        labels = ("cam2 original", "cam4 warped", "completed")
    else:
        raise ValueError(f"Unsupported WARP_DIRECTION: {WARP_DIRECTION}")

    if not ref_path.exists() or not warp_path.exists() or not mask_path.exists():
        print(f"[warn] Missing file for frame {frame_idx:05d}")
        return {"frame_idx": frame_idx, "skipped": True, "reason": "missing_file"}

    ref_img = cv2.imread(str(ref_path))
    warp_img = cv2.imread(str(warp_path))
    ref_mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

    ref_valid = ref_mask > 127
    warp_valid = warp_img.sum(axis=2) > 0
    top_boundary_per_col = np.full(ref_valid.shape[1], ref_valid.shape[0], dtype=np.int32)
    for col in range(ref_valid.shape[1]):
        valid_rows_in_col = np.where(ref_valid[:, col])[0]
        if len(valid_rows_in_col) > 0:
            top_boundary_per_col[col] = int(valid_rows_in_col.min())
    rows = np.arange(ref_valid.shape[0], dtype=np.int32)[:, None]
    boundary_mask = rows < top_boundary_per_col[None, :]
    completion_mask = warp_valid & (~ref_valid) & boundary_mask

    result = ref_img.copy()
    result[completion_mask] = warp_img[completion_mask]

    frame_output_dir = Path(OUTPUT_DIR) / f"frame_{frame_idx:05d}"
    frame_output_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(frame_output_dir / out_name), result)

    if SAVE_VIS:
        completion_mask_img = np.zeros(ref_mask.shape, dtype=np.uint8)
        completion_mask_img[completion_mask] = 255
        cv2.imwrite(str(frame_output_dir / "completion_mask.png"), completion_mask_img)
        boundary_vis = cv2.cvtColor(ref_mask, cv2.COLOR_GRAY2BGR)
        for col in range(0, ref_valid.shape[1], 4):  # 每4列画一个点，避免太密
            row = int(top_boundary_per_col[col])
            if row < ref_valid.shape[0]:
                cv2.circle(boundary_vis, (col, row), 1, (0, 255, 0), -1)
        cv2.imwrite(str(frame_output_dir / "boundary_debug.png"), boundary_vis)

        comparison = np.hstack([ref_img, warp_img, result])
        h, w = ref_img.shape[:2]
        font = cv2.FONT_HERSHEY_SIMPLEX
        for idx, label in enumerate(labels):
            x = idx * w + 20
            y = 40
            cv2.putText(comparison, label, (x, y), font, 1.0, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imwrite(str(frame_output_dir / "comparison.jpg"), comparison, [cv2.IMWRITE_JPEG_QUALITY, 95])

    return {
        "frame_idx": frame_idx,
        "skipped": False,
        "completion_pixels": int(completion_mask.sum()),
        "ref_valid_pixels": int(ref_valid.sum()),
        "warp_valid_pixels": int(warp_valid.sum()),
        "top_boundary_min": int(top_boundary_per_col.min()),
    }


def main():
    results = []
    for frame_idx in range(INDEX_START, INDEX_START + NUM_FRAMES):
        result = process_frame(frame_idx)
        results.append(result)
        print(f"[frame {frame_idx:05d}] {result['reason'] if result.get('skipped') else result}")

    success_count = sum(1 for item in results if not item["skipped"])
    skipped_count = sum(1 for item in results if item["skipped"])
    completion_values = [item["completion_pixels"] for item in results if not item["skipped"]]
    avg_completion_pixels = float(np.mean(completion_values)) if completion_values else 0.0

    print(f"successful frames: {success_count}")
    print(f"skipped frames: {skipped_count}")
    print(f"average completion pixels: {avg_completion_pixels}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
