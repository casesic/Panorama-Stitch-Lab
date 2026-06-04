#!/bin/bash
set -euo pipefail

PROJECT_ROOT="/home/yuyue/Projects/pano-stitch"
cd "$PROJECT_ROOT"

CONFIGS=(
  "20260311_riding|c2_to_c4||"
  #"20260320_riding_dataset1|c4_to_c2||"
  #"20260320_riding_dataset2|c4_to_c2||"
  #"20260320_riding_dataset3|c4_to_c2||"
)

for cfg in "${CONFIGS[@]}"; do
  IFS='|' read -r dataset directions start_frame end_frame <<< "$cfg"

  FITTING_DIR="$PROJECT_ROOT/data_process/Output/${dataset}/fitting/dual_cam_dense"
  IMAGE_DIR="$PROJECT_ROOT/data_process/Output/${dataset}/image/equirect_rotate"
  OUTPUT_DIR="$PROJECT_ROOT/data_process/Output/${dataset}/semantic_flow/dual_cam_dense"
  BBOX_DIR="$PROJECT_ROOT/data_process/Output/${dataset}/image/equirect_crop_rotate"
  INVALID_JSON="$PROJECT_ROOT/data_process/Output/${dataset}/vis/dual_erp_dense/invalid.json"

  echo "========================================"
  echo "dataset:      $dataset"
  echo "directions:   $directions"
  echo "start_frame:  ${start_frame:-(all)}"
  echo "end_frame:    ${end_frame:-(all)}"
  echo "fitting_dir:  $FITTING_DIR"
  echo "image_dir:    $IMAGE_DIR"
  echo "output_dir:   $OUTPUT_DIR"
  echo "bbox_dir:     $BBOX_DIR"
  echo "invalid_json: $INVALID_JSON"

  # check required dirs
  if [ ! -d "$FITTING_DIR" ]; then
    echo "WARNING: fitting_dir not found, skipping: $FITTING_DIR"
    continue
  fi
  if [ ! -d "$IMAGE_DIR" ]; then
    echo "WARNING: image_dir not found, skipping: $IMAGE_DIR"
    continue
  fi

  # build cmd array
  CMD=(
    python src/script/test_function/debug031_batch_semantic_flow.py
    --fitting_dir "$FITTING_DIR"
    --image_dir "$IMAGE_DIR"
    --output_dir "$OUTPUT_DIR"
    --directions "$directions"
    --image_width 5760
    --image_height 2880
  )

  if [ -n "$start_frame" ]; then
    CMD+=(--start_frame "$start_frame")
  fi
  if [ -n "$end_frame" ]; then
    CMD+=(--end_frame "$end_frame")
  fi

  if [ -f "$INVALID_JSON" ]; then
    CMD+=(--invalid_json "$INVALID_JSON")
  else
    echo "WARNING: invalid_json not found, skipping: $INVALID_JSON"
  fi

  if [ -d "$BBOX_DIR" ]; then
    CMD+=(--bbox_dir "$BBOX_DIR" --bbox_padding 100)
  else
    echo "WARNING: bbox_dir not found, skipping: $BBOX_DIR"
  fi

  echo "CMD: ${CMD[*]}"
  echo "========================================"
  echo ""

  "${CMD[@]}"
done

echo "All done."
