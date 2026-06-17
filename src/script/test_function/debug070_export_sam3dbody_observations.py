#!/usr/bin/env python3
"""Export SAM 3D Body observations from cached reconstructions.

Reads debug066 npz cache, no model loading. Outputs JSON observation
dicts and pair summaries.
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
from lib_pano.human3d.sam3dbody_export import (
    sam3dbody_recon_to_observation_dict,
    build_sam3dbody_pair_summary_dict,
)
from lib_pano.correspondence.sam3dbody_pairs import (
    pair_sam3dbody_vertices,
    restore_sam3dbody_pairs_to_full,
    filter_pairs_inside_image,
)


def discover_all_frames(recon_dir: Path, cam: str = "cam2") -> list[int]:
    frames = set()
    for p in (recon_dir / cam).glob("*.npz"):
        try:
            frames.add(int(p.stem))
        except ValueError:
            continue
    return sorted(frames)


def parse_segment_arg(segment_str: str) -> list[int]:
    parts = segment_str.split("_")
    if len(parts) != 2:
        raise ValueError(f"segment must be 'start_end' or 'all', got: {segment_str}")
    start, end = int(parts[0]), int(parts[1])
    if start > end:
        raise ValueError(f"segment start must be <= end: {segment_str}")
    return list(range(start, end + 1))


def resolve_frame_indices(segment_args: list[str], recon_dir: Path) -> list[int]:
    all_frames: list[int] = []
    for seg in segment_args:
        if seg.lower() == "all":
            all_frames.extend(discover_all_frames(recon_dir, "cam2"))
        else:
            all_frames.extend(parse_segment_arg(seg))
    if not all_frames:
        raise ValueError("no frames to process (check --segment and recon_dir)")
    return sorted(set(all_frames))


def main() -> None:
    parser = argparse.ArgumentParser(description="Export SAM 3D Body observations to JSON.")
    parser.add_argument("--dataset_root", type=str, required=True)
    parser.add_argument("--recon_dir", type=str, required=True)
    parser.add_argument("--segment", type=str, nargs="+", default=["all"])
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--cams", type=str, nargs="+", default=["cam2", "cam4"])
    parser.add_argument("--restore_to_full", action="store_true", default=False)
    parser.add_argument("--max_vertices", type=int, default=None)
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root)
    recon_dir = Path(args.recon_dir)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    frame_indices = resolve_frame_indices(args.segment, recon_dir)
    print(f"[INFO] resolved {len(frame_indices)} frames: {frame_indices[0]:05d} ... {frame_indices[-1]:05d}")

    # load bboxes (needed for full restore)
    bbox_dir = dataset_root / "image" / "equirect_crop_rotate"
    bbox_cam2 = json.loads((bbox_dir / "cam2_bboxes.json").read_text(encoding="utf-8"))
    bbox_cam4 = json.loads((bbox_dir / "cam4_bboxes.json").read_text(encoding="utf-8"))

    ERP_W, ERP_H = 5760, 2880

    for space in (["crop_upright"] if not args.restore_to_full else ["crop_upright", "equirect_rotate"]):
        for frame_idx in frame_indices:
            frame_key = f"{frame_idx:05d}"
            for cam in args.cams:
                npz_path = recon_dir / cam / f"{frame_key}.npz"
                if not npz_path.exists():
                    print(f"[{frame_key}] [{cam}] SKIP: missing npz")
                    continue
                recon = load_sam3dbody_recon(npz_path)
                bbox = bbox_cam2[str(frame_idx)] if cam == "cam2" else bbox_cam4[str(frame_idx)]
                do_full = (space == "equirect_rotate")

                obs = sam3dbody_recon_to_observation_dict(
                    recon, bbox_full=bbox, restore_to_full=do_full, max_vertices=args.max_vertices,
                )

                json_dir = out / space / cam
                json_dir.mkdir(parents=True, exist_ok=True)
                json_path = json_dir / f"{frame_key}.json"
                with json_path.open("w", encoding="utf-8") as f:
                    json.dump(obs, f, ensure_ascii=False)
                print(f"[{frame_key}] [{cam}] [{space}] saved {len(obs['vertices_2d'])} vertices -> {json_path}")

        # pair summary (only for equirect_rotate space, cam4->cam2)
        if space == "equirect_rotate":
            pair_dir = out / "pair_summary"
            pair_dir.mkdir(parents=True, exist_ok=True)
            for frame_idx in frame_indices:
                frame_key = f"{frame_idx:05d}"
                npz4 = recon_dir / "cam4" / f"{frame_key}.npz"
                npz2 = recon_dir / "cam2" / f"{frame_key}.npz"
                if not npz4.exists() or not npz2.exists():
                    continue
                r4 = load_sam3dbody_recon(npz4)
                r2 = load_sam3dbody_recon(npz2)
                su, du, vids = pair_sam3dbody_vertices(r4, r2)
                bb4 = bbox_cam4[str(frame_idx)]
                bb2 = bbox_cam2[str(frame_idx)]
                sf, df = restore_sam3dbody_pairs_to_full(su, du, bb4, bb2, frame_idx)
                sf2, df2, vids2 = filter_pairs_inside_image(sf, df, vids, (ERP_H, ERP_W), (ERP_H, ERP_W))
                summary = build_sam3dbody_pair_summary_dict(frame_idx, "cam4", "cam2", sf2, df2, vids2)
                spath = pair_dir / f"{frame_key}_cam4_to_cam2.json"
                with spath.open("w", encoding="utf-8") as f:
                    json.dump(summary, f, ensure_ascii=False)
                print(f"[{frame_key}] [pair_summary] saved {summary['num_pairs']} pairs -> {spath}")

    print("[DONE]")


if __name__ == "__main__":
    main()
