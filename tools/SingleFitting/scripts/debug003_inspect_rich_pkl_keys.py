#!/usr/bin/env python3
from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Any


DEFAULT_PKL = Path(
    "/mnt/hj_exosX18_data0/datasets/RICH/data/bodies/train/BBQ_001_guitar/00005/001.pkl"
)


def load_pickle(path: Path) -> Any:
    with path.open("rb") as f:
        return pickle.load(f)


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect top-level keys in a pickle dict.")
    parser.add_argument("--pkl", type=Path, default=DEFAULT_PKL, help="Path to .pkl file")
    args = parser.parse_args()

    if not args.pkl.exists():
        raise FileNotFoundError(f"PKL not found: {args.pkl}")

    obj = load_pickle(args.pkl)
    print(f"Loaded: {args.pkl}")
    print(f"Type: {type(obj)}")

    if isinstance(obj, dict):
        keys = list(obj.keys())
        print(f"Top-level dict keys ({len(keys)}):")
        for idx, key in enumerate(keys):
            print(f"  [{idx:03d}] {key}")
        print(obj['betas'])
    else:
        print("Object is not a dict; no dict keys to show.")
        if hasattr(obj, "__dict__"):
            attr_keys = list(vars(obj).keys())
            print(f"Object attributes ({len(attr_keys)}): {attr_keys}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
