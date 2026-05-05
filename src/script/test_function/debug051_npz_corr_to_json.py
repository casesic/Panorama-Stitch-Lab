from __future__ import annotations

import argparse
import sys
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[2]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from lib_pano.correspondence.sparse_corr_io import convert_corr_npz_to_json, list_corr_npz_files


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert debug045 sparse correspondence npz files to json.")
    parser.add_argument("--corr_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--start_frame", type=int, default=None)
    parser.add_argument("--end_frame", type=int, default=None)
    return parser.parse_args()


def convert_domain(corr_dir: Path, output_dir: Path, domain: str, start_frame: int | None, end_frame: int | None) -> int:
    converted_count = 0
    json_dir = output_dir / f"{domain}_corr_json"
    for npz_path in list_corr_npz_files(corr_dir, domain=domain, start_frame=start_frame, end_frame=end_frame):
        json_path = json_dir / f"{npz_path.stem}.json"
        convert_corr_npz_to_json(npz_path, json_path)
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
