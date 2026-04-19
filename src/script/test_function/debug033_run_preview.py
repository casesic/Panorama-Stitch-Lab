from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


TITLE_HEIGHT = 30
TITLE_BG = (255, 255, 255)
TITLE_FG = (0, 0, 0)

r'''
python src\script\test_function\debug033_run_preview.py `
    --reviewed data_process\Input\20260311_riding\review.json `
    --image_dir data_process\Output\20260311_riding\image\fisheye `
    --output data_process\Output\preview\20260311_riding_preview.jpg `
'''


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a vertical preview sheet for reviewed frame segments.")
    parser.add_argument("--reviewed", type=str, required=True, help="Path to reviewed.json")
    parser.add_argument("--image_dir", type=str, required=True, help="Fisheye image root containing cam2/ and cam4/")
    parser.add_argument("--output", type=str, required=True, help="Output preview image path")
    parser.add_argument("--gap_threshold", type=int, default=2, help="Gap threshold for splitting frame segments")
    parser.add_argument("--thumb_size", type=int, default=256, help="Thumbnail side length")
    parser.add_argument("--cols", type=int, default=6, help="Number of segments per row.")
    return parser.parse_args()


def load_reviewed_frame_indices(reviewed_path: Path) -> list[int]:
    with reviewed_path.open("r", encoding="utf-8") as file:
        reviewed = json.load(file)

    if not isinstance(reviewed, list):
        raise ValueError("reviewed.json must contain a list")

    frame_indices: set[int] = set()
    for item in reviewed:
        if not isinstance(item, dict):
            raise ValueError("Each reviewed entry must be a dict")
        if "frame_idx" not in item:
            raise ValueError("Each reviewed entry must contain frame_idx")
        frame_indices.add(int(item["frame_idx"]))

    return sorted(frame_indices)


def split_frame_segments(frame_indices: list[int], gap_threshold: int = 2) -> list[list[int]]:
    if not frame_indices:
        return []

    segments: list[list[int]] = [[frame_indices[0]]]
    for frame_idx in frame_indices[1:]:
        if frame_idx - segments[-1][-1] > gap_threshold:
            segments.append([frame_idx])
        else:
            segments[-1].append(frame_idx)
    return segments


def _load_thumbnail(image_path: Path, thumb_size: int) -> np.ndarray:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        return np.zeros((thumb_size, thumb_size, 3), dtype=np.uint8)
    return cv2.resize(image, (thumb_size, thumb_size), interpolation=cv2.INTER_AREA)


def _build_segment_block(segment: list[int], image_dir: Path, thumb_size: int) -> np.ndarray:
    start_frame = segment[0]
    end_frame = segment[-1]
    total = len(segment)

    cam2_first = _load_thumbnail(image_dir / "cam2" / f"{start_frame:05d}.jpg", thumb_size)
    cam4_first = _load_thumbnail(image_dir / "cam4" / f"{start_frame:05d}.jpg", thumb_size)
    cam2_last = _load_thumbnail(image_dir / "cam2" / f"{end_frame:05d}.jpg", thumb_size)
    cam4_last = _load_thumbnail(image_dir / "cam4" / f"{end_frame:05d}.jpg", thumb_size)

    row_top = np.hstack([cam2_first, cam4_first])
    row_bottom = np.hstack([cam2_last, cam4_last])
    image_grid = np.vstack([row_top, row_bottom])

    title = np.full((TITLE_HEIGHT, thumb_size * 2, 3), TITLE_BG, dtype=np.uint8)
    text = f"seg {start_frame}-{end_frame}  ({total}f)  start={start_frame}  end={end_frame}"
    cv2.putText(
        title,
        text,
        (8, 21),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        TITLE_FG,
        1,
        cv2.LINE_AA,
    )
    block = np.vstack([title, image_grid])
    block = cv2.copyMakeBorder(
        block,
        4,
        4,
        4,
        4,
        cv2.BORDER_CONSTANT,
        value=(255, 255, 255),
    )
    return block


def main() -> None:
    args = parse_args()

    reviewed_path = Path(args.reviewed)
    image_dir = Path(args.image_dir)
    output_path = Path(args.output)

    frame_indices = load_reviewed_frame_indices(reviewed_path)
    segments = split_frame_segments(frame_indices, gap_threshold=args.gap_threshold)
    if not segments:
        raise ValueError("No frame segments found from reviewed.json")

    blocks: list[np.ndarray] = []
    for segment in tqdm(segments, desc="Building preview", unit="segment"):
        blocks.append(_build_segment_block(segment, image_dir, args.thumb_size))

    rows: list[np.ndarray] = []
    gap_col = np.full((blocks[0].shape[0], 4, 3), 255, dtype=np.uint8)
    for i in range(0, len(blocks), args.cols):
        row_blocks = blocks[i:i + args.cols]
        while len(row_blocks) < args.cols:
            row_blocks.append(np.zeros_like(blocks[0]))
        row_parts: list[np.ndarray] = []
        for block_idx, block in enumerate(row_blocks):
            if block_idx > 0:
                row_parts.append(gap_col)
            row_parts.append(block)
        rows.append(np.hstack(row_parts))

    gap_row = np.full((4, rows[0].shape[1], 3), 255, dtype=np.uint8)
    preview_parts: list[np.ndarray] = []
    for row_idx, row in enumerate(rows):
        if row_idx > 0:
            preview_parts.append(gap_row)
        preview_parts.append(row)

    preview = np.vstack(preview_parts)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(output_path), preview, [cv2.IMWRITE_JPEG_QUALITY, 95])
    if not ok:
        raise RuntimeError(f"Failed to save preview image: {output_path}")


if __name__ == "__main__":
    main()
