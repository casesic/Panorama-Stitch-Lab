from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert debug045 sparse correspondence npz files to json.")
    parser.add_argument("--corr_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--start_frame", type=int, default=None)
    parser.add_argument("--end_frame", type=int, default=None)
    return parser.parse_args()


def convert_one_npz_to_json(npz_path: Path, json_path: Path) -> None:
    with np.load(npz_path, allow_pickle=False) as data:
        obj = {
            "frame_idx": int(data["frame_idx"]),
            "segment_name": str(data["segment_name"]),
            "image_width": int(data["image_width"]),
            "image_height": int(data["image_height"]),
            "matched_vertex_ids": data["matched_vertex_ids"].astype(int).tolist(),
            "pairs": data["pairs"].astype(float).tolist(),
        }

    json_path.parent.mkdir(parents=True, exist_ok=True)
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False)


def frame_idx_from_corr_name(path: Path, domain: str) -> int | None:
    match = re.match(rf"^(\d+)_{domain}_corr\.npz$", path.name)
    if match is None:
        return None
    return int(match.group(1))


def iter_corr_npz(corr_dir: Path, domain: str, start_frame: int | None, end_frame: int | None) -> list[Path]:
    paths: list[tuple[int, Path]] = []
    for npz_path in (corr_dir / f"{domain}_corr").glob(f"*_{domain}_corr.npz"):
        frame_idx = frame_idx_from_corr_name(npz_path, domain)
        if frame_idx is None:
            continue
        if start_frame is not None and frame_idx < start_frame:
            continue
        if end_frame is not None and frame_idx > end_frame:
            continue
        paths.append((frame_idx, npz_path))
    return [npz_path for _, npz_path in sorted(paths)]


def convert_domain(corr_dir: Path, output_dir: Path, domain: str, start_frame: int | None, end_frame: int | None) -> int:
    converted_count = 0
    json_dir = output_dir / f"{domain}_corr_json"
    for npz_path in iter_corr_npz(corr_dir, domain, start_frame, end_frame):
        json_path = json_dir / f"{npz_path.stem}.json"
        convert_one_npz_to_json(npz_path, json_path)
        converted_count += 1
    return converted_count


def main() -> None:
    args = parse_args()
    corr_dir = Path(args.corr_dir)
    output_dir = Path(args.output_dir)

    erp_count = convert_domain(corr_dir, output_dir, "erp", args.start_frame, args.end_frame)
    fisheye_count = convert_domain(corr_dir, output_dir, "fisheye", args.start_frame, args.end_frame)
    print(f"Converted erp={erp_count} fisheye={fisheye_count}")


if __name__ == "__main__":
    main()

r'''
python src/script/test_function/debug051_npz_corr_to_json.py `
  --corr_dir data_process/Output/20260311_riding/sparse_smpl_corr_debug045 `
  --output_dir data_process/Output/20260311_riding/sparse_smpl_corr_json 
'''
