from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge stage invalid.json files into one common invalid.json.")
    parser.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        help="Input invalid json paths. Accepts space-separated paths and comma-separated groups.",
    )
    parser.add_argument("--output", type=str, required=True, help="Merged invalid json output path.")
    parser.add_argument("--strict", action="store_true", help="Fail if any input json is missing.")
    parser.add_argument("--no_sources", action="store_true", help="Do not save source breakdown.")
    return parser.parse_args()


def expand_input_paths(raw_inputs: list[str]) -> list[Path]:
    paths: list[Path] = []
    for raw_item in raw_inputs:
        for item in raw_item.split(","):
            item = item.strip()
            if item:
                paths.append(Path(item))
    return paths


def _extract_all_ints(obj) -> set[int]:
    frames: set[int] = set()

    if isinstance(obj, bool):
        return frames

    if isinstance(obj, int):
        frames.add(int(obj))
        return frames

    if isinstance(obj, list):
        for item in obj:
            frames.update(_extract_all_ints(item))
        return frames

    if isinstance(obj, dict):
        if "frame_idx" in obj:
            frames.update(_extract_all_ints(obj["frame_idx"]))
        if "frame_idxs" in obj:
            frames.update(_extract_all_ints(obj["frame_idxs"]))
        if "frames" in obj:
            frames.update(_extract_all_ints(obj["frames"]))
        for value in obj.values():
            frames.update(_extract_all_ints(value))
        return frames

    return frames


def extract_invalid_frames(data) -> set[int]:
    if isinstance(data, dict):
        if "invalid_frames" in data:
            return _extract_all_ints(data["invalid_frames"])
        if "frames" in data:
            return _extract_all_ints(data["frames"])
        return _extract_all_ints(data)

    return _extract_all_ints(data)


def load_invalid_frames(json_path: Path) -> set[int]:
    with json_path.open("r", encoding="utf-8-sig") as file:
        data = json.load(file)
    return extract_invalid_frames(data)


def main() -> int:
    args = parse_args()

    input_paths = expand_input_paths(args.inputs)
    output_path = Path(args.output)

    merged_frames: set[int] = set()
    source_frames: dict[str, list[int]] = {}
    num_existing_sources = 0

    for input_path in input_paths:
        if not input_path.exists():
            message = f"missing invalid json: {input_path}"
            if args.strict:
                raise FileNotFoundError(message)
            print(f"[WARN] missing invalid json, skipped: {input_path}")
            continue

        frames = load_invalid_frames(input_path)
        sorted_frames = sorted(frames)
        merged_frames.update(frames)
        source_frames[str(input_path)] = sorted_frames
        num_existing_sources += 1
        print(f"[OK] {input_path}: {len(sorted_frames)} invalid frames")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_data = {
        "invalid_frames": sorted(merged_frames),
        "meta": {
            "total_invalid": len(merged_frames),
            "num_sources": num_existing_sources,
        },
    }
    if not args.no_sources:
        output_data["meta"]["sources"] = source_frames

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(output_data, file, ensure_ascii=False, indent=2)

    print(f"[SUMMARY] merged invalid frames: {len(merged_frames)}")
    print(f"[OUTPUT] saved to: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


r'''
python src\script\test_function\debug022_merge_invalid_jsons.py `
  --inputs `
    data_process\Output\20260320_riding_dataset1\bbox_grounding_dino\equirect_rotate_smooth\invalid.json `
    data_process\Output\20260320_riding_dataset1\image\equirect_crop_rotate\invalid.json `
  --output data_process\Output\20260320_riding_dataset1\invalid.json

python src\script\test_function\debug022_merge_invalid_jsons.py `
  --inputs `
    data_process\Output\20260311_riding\bbox_grounding_dino\equirect_rotate_smooth\invalid.json `
    data_process\Output\20260311_riding\image\equirect_crop_rotate\invalid.json `
  --output data_process\Output\20260311_riding\invalid.json
  
python src\script\test_function\debug022_merge_invalid_jsons.py `
  --inputs data_process\Output\xxx\stage1\invalid.json,data_process\Output\xxx\stage2\invalid.json `
  --output data_process\Output\xxx\invalid.json
'''
