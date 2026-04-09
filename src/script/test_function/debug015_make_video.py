import os
import shutil

import cv2

import config_script

STITCH_DIR = r"data_process/Output/20260311_riding/stitch_batch"
BATCH_FLOW_DIR = r"data_process/Output/20260311_riding/batch_flow"
COMPLETION_DIR = r"data_process/Output/20260311_riding/completion"
OUTPUT_PATH = r"data_process/Output/20260311_riding/preview_video.mp4"
OUTPUT_IMAGE_DIR = r"data_process/Output/20260311_riding/output"
FPS = 3

# 三段帧范围及对应图像来源
SEG1_START, SEG1_END = 500, 512
SEG2_START, SEG2_END = 513, 517
SEG3_START, SEG3_END = 518, 529


def get_frame_path(frame_idx):
    if SEG1_START <= frame_idx <= SEG1_END:
        return os.path.join(
            STITCH_DIR,
            f"frame_{frame_idx:05d}",
            f"rotated_cam2_frame_{frame_idx:05d}.png",
        )
    if SEG2_START <= frame_idx <= SEG2_END:
        return os.path.join(
            COMPLETION_DIR,
            f"frame_{frame_idx:05d}",
            "cam4_completed.png",
        )
    if SEG3_START <= frame_idx <= SEG3_END:
        return os.path.join(
            STITCH_DIR,
            f"frame_{frame_idx:05d}",
            f"rotated_cam4_frame_{frame_idx:05d}.png",
        )
    return None


def main():
    writer = None
    target_size = None
    written_frames = 0

    for frame_idx in range(490, 530):
        frame_path = get_frame_path(frame_idx)
        if frame_path is None:
            continue

        image = cv2.imread(frame_path, cv2.IMREAD_COLOR)
        if image is None:
            print(f"[warn] Failed to read frame: {frame_path}")
            continue

        os.makedirs(OUTPUT_IMAGE_DIR, exist_ok=True)
        copied_name = os.path.basename(frame_path)
        if copied_name == "source_preprocessed_overlay.png":
            stem, ext = os.path.splitext(copied_name)
            copied_name = f"{stem}_frame_{frame_idx:05d}{ext}"
        shutil.copy2(frame_path, os.path.join(OUTPUT_IMAGE_DIR, copied_name))

        if writer is None:
            h, w = image.shape[:2]
            target_size = (w, h)
            output_dir = os.path.dirname(OUTPUT_PATH)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
            writer = cv2.VideoWriter(
                OUTPUT_PATH,
                cv2.VideoWriter_fourcc(*"mp4v"),
                FPS,
                target_size,
            )

        if image.shape[1] != target_size[0] or image.shape[0] != target_size[1]:
            image = cv2.resize(image, target_size, interpolation=cv2.INTER_LINEAR)

        writer.write(image)
        written_frames += 1

    if writer is None:
        print("[warn] No valid frames were loaded.")
        return 1

    writer.release()
    print(f"Saved video to: {OUTPUT_PATH}")
    print(f"Total frames written: {written_frames}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
