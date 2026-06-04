import argparse
import json
from pathlib import Path

import cv2

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}

ROTATE_MAP = {
    "cw90": cv2.ROTATE_90_CLOCKWISE,
    "ccw90": cv2.ROTATE_90_COUNTERCLOCKWISE,
    "180": cv2.ROTATE_180,
    "none": None,
}


def _resolve_rotate_code(frame_id: int, rules: list[dict]):
    for rule in rules:
        if rule["start"] <= frame_id <= rule["end"]:
            rotate_str = rule.get("rotate", "none")
            return ROTATE_MAP.get(rotate_str)
    return None


def process_cam(input_dir: Path, output_dir: Path, rules: list[dict], unmatched: str) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(
        p for p in input_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )

    stats = {"total": len(image_paths), "processed": 0, "skipped": 0, "copied": 0, "rotated": 0}

    for img_path in image_paths:
        try:
            frame_id = int(img_path.stem)
        except ValueError:
            print(f"  [WARN] skip non-numeric filename: {img_path.name}")
            stats["skipped"] += 1
            continue

        rotate_code = _resolve_rotate_code(frame_id, rules)

        if rotate_code is None and rules and unmatched == "skip":
            stats["skipped"] += 1
            continue

        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img is None:
            print(f"  [WARN] failed to read: {img_path}")
            stats["skipped"] += 1
            continue

        if rotate_code is None:
            rotated = img
            stats["copied"] += 1
        else:
            rotated = cv2.rotate(img, rotate_code)
            stats["rotated"] += 1

        out_path = output_dir / img_path.name
        cv2.imwrite(str(out_path), rotated)
        stats["processed"] += 1

    return stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    config_path = Path(args.config)
    datasets = json.loads(config_path.read_text())

    for ds in datasets:
        name = ds.get("name", "unnamed")
        input_root = Path(ds["input_root"])
        output_root = Path(ds["output_root"])
        cams = ds.get("cams", ["cam2", "cam4"])
        unmatched = ds.get("unmatched", "copy")
        rules = ds.get("rules", [])

        print(f"\n{'='*60}")
        print(f"[{name}]")
        print(f"  input:  {input_root}")
        print(f"  output: {output_root}")
        print(f"  cams:   {cams}")
        print(f"  unmatched: {unmatched}")
        print(f"  rules:  {len(rules)}")

        for cam in cams:
            in_dir = input_root / cam
            out_dir = output_root / cam
            if not in_dir.exists():
                print(f"  [WARN] missing folder: {in_dir}")
                continue

            print(f"  [{cam}] {in_dir}")
            stats = process_cam(in_dir, out_dir, rules, unmatched)
            print(
                f"    total={stats['total']}  processed={stats['processed']}"
                f"  rotated={stats['rotated']}  copied={stats['copied']}"
                f"  skipped={stats['skipped']}"
            )


if __name__ == "__main__":
    main()


r'''
python src/script/test_function/debug001_rotate_cam_by_frame_rule.py \
  --config configs/rotate_upright_rules.json
'''
