"""Command-line entry point for production staged-LM fitting."""

from __future__ import annotations

import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from script.human_fitting.staged_lm.app import run_production_pipeline
from script.human_fitting.staged_lm.cli import parse_args


def main() -> int:
    run_production_pipeline(parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

r'''
python src/script/human_fitting/staged_lm/run.py \
  --frames 513,514,515,516,517,518 \
  --hmr_dir data_process/Output/20260311_riding/hmr_cache \
  --sam3d_dense_dir data_process/Output/20260311_riding/sam3dbody/smplx_dense \
  --bbox_dir data_process/Output/20260311_riding/image/equirect_crop_rotate \
  --output_dir data_process/Vis/staged_lm_cpu \
  --stop_after_stage stage2 \
  --export_debug029_format
'''
