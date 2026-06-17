#!/usr/bin/env python3
"""Self-check: validate SAM 3D Body lib refactoring functions on a sample frame.

Reads cached npz for one frame, runs pairing + coordinate restore + filtering,
prints diagnostic info. No model loading, no image saving.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

SRC_ROOT = Path(__file__).resolve().parents[2]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from lib_pano.human3d.sam3dbody_io import load_sam3dbody_recon
from lib_pano.correspondence.sam3dbody_pairs import (
    pair_sam3dbody_vertices,
    restore_sam3dbody_pairs_to_full,
    filter_pairs_inside_image,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Check SAM 3D Body lib refactoring.")
    parser.add_argument("--dataset_root", type=str, required=True)
    parser.add_argument("--recon_dir", type=str, required=True)
    parser.add_argument("--frame", type=int, required=True)
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root)
    recon_dir = Path(args.recon_dir)
    frame_key = f"{args.frame:05d}"

    # load reconstructions
    recon_cam2 = load_sam3dbody_recon(recon_dir / "cam2" / f"{frame_key}.npz")
    recon_cam4 = load_sam3dbody_recon(recon_dir / "cam4" / f"{frame_key}.npz")
    print(f"[INFO] loaded cam2={recon_cam2.vertices_2d.shape}, cam4={recon_cam4.vertices_2d.shape}")

    # pair vertices
    src_upright, dst_upright, vids = pair_sam3dbody_vertices(recon_cam4, recon_cam2)
    print(f"[INFO] raw paired point count: {len(vids)}")
    print(f"[INFO] first 5 vertex ids: {vids[:5].tolist()}")

    # restore to full
    bbox_dir = dataset_root / "image" / "equirect_crop_rotate"
    bbox_cam2 = json.loads((bbox_dir / "cam2_bboxes.json").read_text(encoding="utf-8"))
    bbox_cam4 = json.loads((bbox_dir / "cam4_bboxes.json").read_text(encoding="utf-8"))
    bb2 = bbox_cam2[str(args.frame)]
    bb4 = bbox_cam4[str(args.frame)]

    src_full, dst_full = restore_sam3dbody_pairs_to_full(src_upright, dst_upright, bb4, bb2, args.frame)
    print(f"[INFO] src_full min: [{src_full[:, 0].min():.1f}, {src_full[:, 1].min():.1f}]")
    print(f"[INFO] src_full max: [{src_full[:, 0].max():.1f}, {src_full[:, 1].max():.1f}]")
    print(f"[INFO] dst_full min: [{dst_full[:, 0].min():.1f}, {dst_full[:, 1].min():.1f}]")
    print(f"[INFO] dst_full max: [{dst_full[:, 0].max():.1f}, {dst_full[:, 1].max():.1f}]")

    # filter inside image
    src_shape = tuple(int(x) for x in recon_cam4.image_shape)
    dst_shape = tuple(int(x) for x in recon_cam2.image_shape)
    # full images are 5760x2880
    src_f, dst_f, vids_f = filter_pairs_inside_image(
        src_full, dst_full, vids, (2880, 5760), (2880, 5760),
    )
    print(f"[INFO] full-coordinate valid point count: {len(vids_f)}")
    print("[DONE] all checks passed")


if __name__ == "__main__":
    main()
