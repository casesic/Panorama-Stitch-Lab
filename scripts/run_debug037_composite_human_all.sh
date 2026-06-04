#!/bin/bash
set -euo pipefail

PROJECT_ROOT="/home/yuyue/Projects/pano-stitch"
cd "$PROJECT_ROOT"

CONFIGS=(
  "20260311_riding|data_process/Input/20260311_riding/review.json|c2_to_c4|top||"
  #"20260320_riding_dataset1|data_process/Input/20260320_riding/dataset1/review.json|c4_to_c2|bottom||"
  #"20260320_riding_dataset2|data_process/Input/20260320_riding/dataset2/review.json|c4_to_c2|bottom||"
  #"20260320_riding_dataset3|data_process/Input/20260320_riding/dataset3/review.json|c4_to_c2|bottom||"
)

for cfg in "${CONFIGS[@]}"; do
  IFS='|' read -r dataset reviewed directions blend_direction start end <<< "$cfg"

  SEG_DIR="$PROJECT_ROOT/data_process/Output/${dataset}/seg/equirect_crop_rotate_sam3"
  FLOW_DIR="$PROJECT_ROOT/data_process/Output/${dataset}/semantic_flow/dual_cam_dense"
  IMAGE_DIR="$PROJECT_ROOT/data_process/Output/${dataset}/image/equirect_rotate"
  BBOX_DIR="$PROJECT_ROOT/data_process/Output/${dataset}/image/equirect_crop_rotate"
  OUTPUT_DIR="$PROJECT_ROOT/data_process/Output/${dataset}/composite_human/dual_cam_dense"
  REVIEWED_PATH="$PROJECT_ROOT/${reviewed}"

  echo "========================================"
  echo "dataset:         $dataset"
  echo "directions:      $directions"
  echo "blend_direction: $blend_direction"
  echo "start:           ${start:-(all)}"
  echo "end:             ${end:-(all)}"
  echo "seg_dir:         $SEG_DIR"
  echo "flow_dir:        $FLOW_DIR"
  echo "image_dir:       $IMAGE_DIR"
  echo "bbox_dir:        $BBOX_DIR"
  echo "output_dir:      $OUTPUT_DIR"
  echo "reviewed:        $REVIEWED_PATH"

  # check required paths
  MISSING=0
  for path in "$SEG_DIR" "$FLOW_DIR" "$IMAGE_DIR" "$BBOX_DIR" "$REVIEWED_PATH"; do
    if [ ! -e "$path" ]; then
      echo "WARNING: not found, skipping: $path"
      MISSING=1
    fi
  done
  if [ "$MISSING" -eq 1 ]; then
    continue
  fi

  CMD=(
    python src/script/test_function/debug037_composite_human.py
    --seg_dir "$SEG_DIR"
    --flow_dir "$FLOW_DIR"
    --image_dir "$IMAGE_DIR"
    --bbox_dir "$BBOX_DIR"
    --output_dir "$OUTPUT_DIR"
    --reviewed "$REVIEWED_PATH"
    --blend_direction "$blend_direction"
    --directions "$directions"
  )

  if [ -n "$start" ]; then
    CMD+=(--start "$start")
  fi
  if [ -n "$end" ]; then
    CMD+=(--end "$end")
  fi

  echo "CMD: ${CMD[*]}"
  echo "========================================"
  echo ""

  "${CMD[@]}"
done

echo "All done."
