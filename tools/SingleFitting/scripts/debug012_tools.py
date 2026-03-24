#!/usr/bin/env python3
"""
Inspect pickle content from debug outputs.

Usage:
  python tools/SingleFitting/scripts/debug012_tools.py ^
    --pkl tools/SingleFitting/outputs/dual_cam/dual_s0_i0_29_frame15.pkl
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

SINGLE_FITTING_ROOT = Path(__file__).resolve().parents[1]
if str(SINGLE_FITTING_ROOT) not in sys.path:
    sys.path.insert(0, str(SINGLE_FITTING_ROOT))

from single_fitting.debug_inspect_shared import inspect_pickle_file


DEFAULT_PKL_PATH = Path("tools") / "SingleFitting" / "outputs" / "dual_cam" / "dual_s0_i0_0_frame0.pkl"


def main():
    parser = argparse.ArgumentParser(description="Inspect pickle content for debug outputs.")
    parser.add_argument("--pkl", type=Path, default=DEFAULT_PKL_PATH)
    args = parser.parse_args()
    inspect_pickle_file(args.pkl)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
