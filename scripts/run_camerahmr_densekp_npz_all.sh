#!/bin/bash
set -euo pipefail

PROJECT_ROOT=/home/yuyue/Projects/pano-stitch
cd "$PROJECT_ROOT/tools/CameraHMR"

DATASETS=(
  # 20260311_riding
  20260320_riding_dataset1
  20260320_riding_dataset2
  20260320_riding_dataset3
)

CAMS=(cam2 cam4)

for dataset in "${DATASETS[@]}"; do
  for cam in "${CAMS[@]}"; do
    IMG_DIR="$PROJECT_ROOT/data_process/Output/$dataset/image/equirect_crop_rotate_upright/$cam"
    NPZ_DIR="$PROJECT_ROOT/data_process/Output/$dataset/camerahmr/densekp_npz/$cam"

    if [ ! -d "$IMG_DIR" ]; then
      echo "WARNING: input dir not found, skipping: $IMG_DIR"
      continue
    fi

    echo "Processing: dataset=$dataset cam=$cam"
    CUDA_VISIBLE_DEVICES=0 python demo_keypoints.py \
      --img_folder "$IMG_DIR" \
      --save_npz_folder "$NPZ_DIR" \
      --batch_size 4 \
      --num_workers 1 \
      --detector_threshold 0.25 \
      --no_vis
  done
done

echo "All done."
