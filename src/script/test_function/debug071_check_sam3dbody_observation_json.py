#!/usr/bin/env python3
"""Self-check: validate SAM 3D Body observation JSON format and content."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

SRC_ROOT = Path(__file__).resolve().parents[2]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


REQUIRED_FIELDS = [
    "source", "frame_idx", "cam", "image_path", "image_shape",
    "space", "bbox_crop", "bbox_full", "focal_length",
    "num_vertices", "num_valid_z", "num_inside",
    "vertices_2d", "valid_vertex_ids", "keypoints_2d",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Check SAM 3D Body observation JSON.")
    parser.add_argument("--json_path", type=str, required=True)
    args = parser.parse_args()

    with open(args.json_path, encoding="utf-8") as f:
        obs = json.load(f)

    # check required fields
    for field in REQUIRED_FIELDS:
        if field not in obs:
            print(f"[FAIL] missing field: {field}")
            return
    print("[PASS] all required fields present")

    # check space
    if obs["space"] not in ("crop_upright", "equirect_rotate"):
        print(f"[FAIL] unknown space: {obs['space']}")
        return
    print(f"[PASS] space={obs['space']}")

    # check vertices
    verts = np.asarray(obs["vertices_2d"], dtype=np.float32)
    vids = np.asarray(obs["valid_vertex_ids"], dtype=np.int64)
    if len(verts) != len(vids):
        print(f"[FAIL] vertices_2d count ({len(verts)}) != valid_vertex_ids count ({len(vids)})")
        return
    if not np.isfinite(verts).all():
        print("[FAIL] vertices_2d contains non-finite values")
        return
    print(f"[PASS] vertices_2d: {verts.shape}, all finite")

    # print summary
    kp = obs["keypoints_2d"]
    kp_shape = None if kp is None else (len(kp), len(kp[0]) if kp else 0)
    print(f"[INFO] frame_idx: {obs['frame_idx']}")
    print(f"[INFO] cam: {obs['cam']}")
    print(f"[INFO] space: {obs['space']}")
    print(f"[INFO] num_vertices exported: {len(verts)} / {obs['num_vertices']}")
    print(f"[INFO] bbox_crop: {obs['bbox_crop']}")
    print(f"[INFO] bbox_full: {obs['bbox_full']}")
    print(f"[INFO] keypoints_2d shape: {kp_shape}")
    print("[DONE] all checks passed")


if __name__ == "__main__":
    main()
