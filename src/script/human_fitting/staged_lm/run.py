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
