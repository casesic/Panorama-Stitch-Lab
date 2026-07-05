#!/usr/bin/env python3
"""Standalone smpl_corr delivery visualization — no project-internal deps.

Reads minimal pair JSON, draws points_bg and pair_lines for ERP & fisheye.
Pure numpy + cv2 + json + pathlib + argparse.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# constants
# ---------------------------------------------------------------------------
EXPECTED_KEYS = frozenset({"frame_idx", "image_width", "image_height", "pairs"})
FORBIDDEN_KEYS = frozenset(
    {"point_ids", "valid", "values", "confidence", "coord_type", "segment_name"}
)

COLOR_CAM2 = (0, 255, 0)       # green BGR
COLOR_CAM4 = (0, 255, 255)     # yellow BGR
COLOR_LINE = (190, 190, 190)   # light grey BGR

FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.7
FONT_THICK = 2
FONT_COLOR = (255, 255, 255)

ERP_W, ERP_H = 5760, 2880
FISH_W, FISH_H = 2880, 2880

VALID_DATASETS = frozenset(
    {"20260311_riding", "20260320_riding_dataset1", "20260320_riding_dataset2"}
)


# ---------------------------------------------------------------------------
# segment parsing
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class SegmentSpec:
    name: str
    start_frame: int
    end_frame: int
    frame_indices: list[int]


def parse_segment_specs(segment_args: list[str]) -> list[SegmentSpec]:
    """Parse start_end strings → SegmentSpec list, with overlap detection."""
    specs: list[SegmentSpec] = []
    seen: dict[int, str] = {}
    for seg in segment_args:
        parts = seg.split("_")
        if len(parts) != 2:
            raise ValueError(f"Invalid segment: {seg!r}")
        start, end = int(parts[0]), int(parts[1])
        if start > end:
            raise ValueError(f"Segment start > end: {seg!r}")
        frames = list(range(start, end + 1))
        for f in frames:
            if f in seen:
                raise ValueError(
                    f"Overlapping frame {f} in segment '{seg}' "
                    f"(already present in segment '{seen[f]}')"
                )
            seen[f] = seg
        specs.append(SegmentSpec(name=seg, start_frame=start, end_frame=end, frame_indices=frames))
    return specs


# ---------------------------------------------------------------------------
# JSON I/O
# ---------------------------------------------------------------------------
def load_and_validate_json(path: Path, expected_frame: int) -> tuple[int, int, int, np.ndarray]:
    """Read minimal JSON, validate format, return (frame_idx, w, h, pairs [N,4])."""
    with open(str(path), encoding="utf-8") as f:
        data = json.load(f)

    keys = set(data.keys())
    if keys != EXPECTED_KEYS:
        raise ValueError(f"{path}: unexpected keys {keys}, expected {EXPECTED_KEYS}")
    for fk in FORBIDDEN_KEYS:
        if fk in data:
            raise ValueError(f"{path}: forbidden key '{fk}' present")

    frame_idx = int(data["frame_idx"])
    if frame_idx != expected_frame:
        raise ValueError(
            f"{path}: frame_idx={frame_idx}, expected {expected_frame}"
        )

    w = int(data["image_width"])
    h = int(data["image_height"])
    pairs = np.array(data["pairs"], dtype=np.float64)
    if pairs.ndim != 2 or (pairs.shape[0] > 0 and pairs.shape[1] != 4):
        raise ValueError(f"{path}: pairs shape {pairs.shape}, expected [N, 4]")
    if not np.isfinite(pairs).all():
        raise ValueError(f"{path}: pairs contain non-finite values")

    return frame_idx, w, h, pairs


# ---------------------------------------------------------------------------
# image loading
# ---------------------------------------------------------------------------
def load_image_checked(path: Path, expected_w: int, expected_h: int) -> np.ndarray:
    """Load BGR image, validate dimensions."""
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    h, w = img.shape[:2]
    if w != expected_w or h != expected_h:
        raise ValueError(
            f"{path}: size {w}x{h}, expected {expected_w}x{expected_h}"
        )
    return img


# ---------------------------------------------------------------------------
# drawing
# ---------------------------------------------------------------------------
def _point_in_bounds(x: float, y: float, w: int, h: int) -> bool:
    return 0 <= x < w and 0 <= y < h


def draw_points_bg(
    bg: np.ndarray,
    pts: np.ndarray,   # [N, 2]
    color: tuple[int, int, int],
    radius: int,
    label_pairs: int,
    label_drawn: int,
) -> np.ndarray:
    """Draw points on background image."""
    out = bg.copy()
    h, w = out.shape[:2]
    drawn = 0
    for i in range(len(pts)):
        x, y = float(pts[i, 0]), float(pts[i, 1])
        if _point_in_bounds(x, y, w, h):
            cv2.circle(out, (int(round(x)), int(round(y))), radius, color, -1,
                       lineType=cv2.LINE_AA)
            drawn += 1

    text_lines = [f"pairs: {label_pairs}", f"drawn: {drawn}"]
    y0 = 25
    for line in text_lines:
        cv2.putText(out, line, (8, y0), FONT, FONT_SCALE, FONT_COLOR,
                    FONT_THICK, cv2.LINE_AA)
        y0 += 25
    return out


def _subsample_pairs(pairs: np.ndarray, max_lines: int) -> np.ndarray:
    """Deterministic equal-interval subsample of up to max_lines pairs."""
    n = len(pairs)
    if n == 0 or n <= max_lines:
        return pairs
    step = n / max_lines
    idx = np.arange(max_lines) * step
    idx = np.unique(np.round(idx).astype(int))
    idx = idx[idx < n]
    return pairs[idx]


def draw_erp_pair_lines_black(
    pairs: np.ndarray,      # [N, 4]
    max_lines: int,
    point_radius: int,
) -> tuple[np.ndarray, int]:
    """Black ERP canvas — cam2/cam4 points and grey lines in same 5760x2880 space."""
    canvas = np.zeros((ERP_H, ERP_W, 3), dtype=np.uint8)
    pairs_sub = _subsample_pairs(pairs, max_lines)
    lines_drawn = 0
    for i in range(len(pairs_sub)):
        c2x, c2y = float(pairs_sub[i, 0]), float(pairs_sub[i, 1])
        c4x, c4y = float(pairs_sub[i, 2]), float(pairs_sub[i, 3])
        in_c2 = _point_in_bounds(c2x, c2y, ERP_W, ERP_H)
        in_c4 = _point_in_bounds(c4x, c4y, ERP_W, ERP_H)
        if in_c2 and in_c4:
            pt_c2 = (int(round(c2x)), int(round(c2y)))
            pt_c4 = (int(round(c4x)), int(round(c4y)))
            cv2.line(canvas, pt_c2, pt_c4, COLOR_LINE, 1, cv2.LINE_AA)
            cv2.circle(canvas, pt_c2, point_radius, COLOR_CAM2, -1, cv2.LINE_AA)
            cv2.circle(canvas, pt_c4, point_radius, COLOR_CAM4, -1, cv2.LINE_AA)
            lines_drawn += 1
        else:
            print(
                f"  [WARN] ERP pair OOB: "
                f"cam2=({c2x:.1f},{c2y:.1f}) cam4=({c4x:.1f},{c4y:.1f})"
            )
    if lines_drawn < len(pairs_sub):
        print(
            f"  [WARN] erp pair_lines: {len(pairs_sub) - lines_drawn} lines "
            f"skipped (out of bounds)"
        )
    return canvas, lines_drawn


def draw_fisheye_pair_lines_black(
    pairs: np.ndarray,      # [N, 4]
    max_lines: int,
    point_radius: int,
) -> tuple[np.ndarray, int]:
    """Black side-by-side canvas — cam2 left [0,2880), cam4 right [2880,5760)."""
    canvas = np.zeros((FISH_H, FISH_W * 2, 3), dtype=np.uint8)
    pairs_sub = _subsample_pairs(pairs, max_lines)
    lines_drawn = 0
    for i in range(len(pairs_sub)):
        c2x, c2y = float(pairs_sub[i, 0]), float(pairs_sub[i, 1])
        c4x, c4y = float(pairs_sub[i, 2]), float(pairs_sub[i, 3])
        in_c2 = _point_in_bounds(c2x, c2y, FISH_W, FISH_H)
        in_c4 = _point_in_bounds(c4x, c4y, FISH_W, FISH_H)
        if in_c2 and in_c4:
            pt_c2 = (int(round(c2x)), int(round(c2y)))
            pt_c4 = (int(round(c4x + FISH_W)), int(round(c4y)))
            cv2.line(canvas, pt_c2, pt_c4, COLOR_LINE, 1, cv2.LINE_AA)
            cv2.circle(canvas, pt_c2, point_radius, COLOR_CAM2, -1, cv2.LINE_AA)
            cv2.circle(canvas, pt_c4, point_radius, COLOR_CAM4, -1, cv2.LINE_AA)
            lines_drawn += 1
    return canvas, lines_drawn


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize smpl_corr delivery JSON pairs."
    )
    parser.add_argument("--dataset", type=str, required=True,
                         help=f"Dataset name. One of: {', '.join(sorted(VALID_DATASETS))}")
    parser.add_argument("--corr_root", type=str, required=True,
                         help="smpl_corr root containing {method}/erp_corr_json/...")
    parser.add_argument("--image_root", type=str, required=True,
                         help="image root with equirect_rotate/ and fisheye/")
    parser.add_argument("--output_root", type=str, required=True)
    parser.add_argument("--methods", type=str, nargs="+",
                         default=["densekp_raw", "densekp_fitting",
                                   "sam3dbody_raw", "sam3dbody_fitting"])
    parser.add_argument("--segment", type=str, nargs="+", required=True,
                         help="One or more start_end ranges (e.g. 513_518 704_711)")
    parser.add_argument("--max_lines", type=int, default=300)
    parser.add_argument("--point_radius", type=int, default=4)

    args = parser.parse_args()

    # validate dataset
    if args.dataset not in VALID_DATASETS:
        parser.error(
            f"Unknown dataset: {args.dataset}. "
            f"Choose from: {', '.join(sorted(VALID_DATASETS))}"
        )

    segment_specs = parse_segment_specs(args.segment)
    frame_indices: list[int] = []
    for spec in segment_specs:
        frame_indices.extend(spec.frame_indices)

    seg_labels = ", ".join(f"{spec.start_frame}_{spec.end_frame}" for spec in segment_specs)
    print(f"[VIS]")
    print(f"  dataset: {args.dataset}")
    print(f"  methods: {', '.join(args.methods)}")
    print(f"  segments: {seg_labels}")
    print(f"  total_frames: {len(frame_indices)}")

    corr_root = Path(args.corr_root)
    image_root = Path(args.image_root)
    output_root = Path(args.output_root)

    erp_dir = image_root / "equirect_rotate"
    fish_dir = image_root / "fisheye"

    for method in args.methods:
        method_corr = corr_root / method
        if not method_corr.is_dir():
            print(f"[SKIP] {method}: corr dir not found at {method_corr}")
            continue

        method_out = output_root / method
        print(f"\n[VIS] {method}")

        for fid in frame_indices:
            fkey = f"{fid:05d}"
            erp_json = method_corr / "erp_corr_json" / f"{fkey}.json"
            fish_json = method_corr / "fisheye_corr_json" / f"{fkey}.json"

            if not erp_json.exists():
                print(f"  [SKIP] missing ERP JSON: {erp_json}")
                continue
            if not fish_json.exists():
                print(f"  [SKIP] missing fisheye JSON: {fish_json}")
                continue

            # load + validate
            _, erp_w, erp_h, erp_pairs = load_and_validate_json(erp_json, fid)
            _, fish_w, fish_h, fish_pairs = load_and_validate_json(fish_json, fid)

            if erp_w != ERP_W or erp_h != ERP_H:
                raise ValueError(
                    f"ERP {erp_json}: expected {ERP_W}x{ERP_H}, got {erp_w}x{erp_h}"
                )
            if fish_w != FISH_W or fish_h != FISH_H:
                raise ValueError(
                    f"fisheye {fish_json}: expected {FISH_W}x{FISH_H}, got {fish_w}x{fish_h}"
                )
            if len(erp_pairs) != len(fish_pairs):
                raise ValueError(
                    f"frame {fid}: ERP pairs={len(erp_pairs)}, fisheye={len(fish_pairs)}"
                )

            n_pairs = len(erp_pairs)
            print(f"\n[{method}][{fkey}]")

            # ERP bg images
            erp_c2 = load_image_checked(erp_dir / "cam2" / f"{fkey}.jpg", ERP_W, ERP_H)
            erp_c4 = load_image_checked(erp_dir / "cam4" / f"{fkey}.jpg", ERP_W, ERP_H)
            fish_c2 = load_image_checked(fish_dir / "cam2" / f"{fkey}.jpg", FISH_W, FISH_H)
            fish_c4 = load_image_checked(fish_dir / "cam4" / f"{fkey}.jpg", FISH_W, FISH_H)

            # output dirs
            erp_pts_c2_dir = method_out / "erp" / "erp_points_bg" / "cam2"
            erp_pts_c4_dir = method_out / "erp" / "erp_points_bg" / "cam4"
            erp_lines_dir = method_out / "erp" / "erp_pair_lines"
            fish_pts_c2_dir = method_out / "fisheye" / "fisheye_points_bg" / "cam2"
            fish_pts_c4_dir = method_out / "fisheye" / "fisheye_points_bg" / "cam4"
            fish_lines_dir = method_out / "fisheye" / "fisheye_pair_lines"

            for d in (erp_pts_c2_dir, erp_pts_c4_dir, erp_lines_dir,
                      fish_pts_c2_dir, fish_pts_c4_dir, fish_lines_dir):
                d.mkdir(parents=True, exist_ok=True)

            # --- ERP points_bg ---
            erp_c2_vis = draw_points_bg(
                erp_c2, erp_pairs[:, :2], COLOR_CAM2, args.point_radius,
                n_pairs, n_pairs,
            )
            cv2.imwrite(str(erp_pts_c2_dir / f"{fkey}.jpg"), erp_c2_vis,
                        [cv2.IMWRITE_JPEG_QUALITY, 95])

            erp_c4_vis = draw_points_bg(
                erp_c4, erp_pairs[:, 2:4], COLOR_CAM4, args.point_radius,
                n_pairs, n_pairs,
            )
            cv2.imwrite(str(erp_pts_c4_dir / f"{fkey}.jpg"), erp_c4_vis,
                        [cv2.IMWRITE_JPEG_QUALITY, 95])

            # --- ERP pair_lines ---
            erp_lines, erp_lines_drawn = draw_erp_pair_lines_black(
                erp_pairs, args.max_lines, args.point_radius,
            )
            cv2.imwrite(str(erp_lines_dir / f"{fkey}.jpg"), erp_lines,
                        [cv2.IMWRITE_JPEG_QUALITY, 95])

            # --- fisheye points_bg ---
            fish_c2_drawn = 0
            fish_c2_out = fish_c2.copy()
            for i in range(len(fish_pairs)):
                x, y = float(fish_pairs[i, 0]), float(fish_pairs[i, 1])
                if _point_in_bounds(x, y, FISH_W, FISH_H):
                    cv2.circle(fish_c2_out,
                               (int(round(x)), int(round(y))),
                               args.point_radius, COLOR_CAM2, -1,
                               lineType=cv2.LINE_AA)
                    fish_c2_drawn += 1
            cv2.putText(fish_c2_out, f"pairs: {n_pairs}", (8, 25),
                        FONT, FONT_SCALE, FONT_COLOR, FONT_THICK, cv2.LINE_AA)
            cv2.putText(fish_c2_out, f"drawn: {fish_c2_drawn}", (8, 50),
                        FONT, FONT_SCALE, FONT_COLOR, FONT_THICK, cv2.LINE_AA)
            cv2.imwrite(str(fish_pts_c2_dir / f"{fkey}.jpg"), fish_c2_out,
                        [cv2.IMWRITE_JPEG_QUALITY, 95])

            fish_c4_drawn = 0
            fish_c4_out = fish_c4.copy()
            for i in range(len(fish_pairs)):
                x, y = float(fish_pairs[i, 2]), float(fish_pairs[i, 3])
                if _point_in_bounds(x, y, FISH_W, FISH_H):
                    cv2.circle(fish_c4_out,
                               (int(round(x)), int(round(y))),
                               args.point_radius, COLOR_CAM4, -1,
                               lineType=cv2.LINE_AA)
                    fish_c4_drawn += 1
            cv2.putText(fish_c4_out, f"pairs: {n_pairs}", (8, 25),
                        FONT, FONT_SCALE, FONT_COLOR, FONT_THICK, cv2.LINE_AA)
            cv2.putText(fish_c4_out, f"drawn: {fish_c4_drawn}", (8, 50),
                        FONT, FONT_SCALE, FONT_COLOR, FONT_THICK, cv2.LINE_AA)
            cv2.imwrite(str(fish_pts_c4_dir / f"{fkey}.jpg"), fish_c4_out,
                        [cv2.IMWRITE_JPEG_QUALITY, 95])

            # --- fisheye pair_lines ---
            fish_lines, fish_lines_drawn = draw_fisheye_pair_lines_black(
                fish_pairs, args.max_lines, args.point_radius,
            )
            cv2.imwrite(str(fish_lines_dir / f"{fkey}.jpg"), fish_lines,
                        [cv2.IMWRITE_JPEG_QUALITY, 95])

            fish_skipped = n_pairs - fish_lines_drawn
            print(f"erp_pairs={n_pairs}")
            print(f"erp_lines_drawn={erp_lines_drawn}")
            print(f"fisheye_pairs={n_pairs}")
            print(f"fisheye_lines_drawn={fish_lines_drawn}")
            print(f"fisheye_skipped_out_of_bounds={fish_skipped}")

    print("\n[DONE] visualization complete")


if __name__ == "__main__":
    main()


r'''
# A. 多 method、多段：
python src/script/test_function/debug080_vis_smpl_corr_delivery.py \
  --dataset 20260320_riding_dataset2 \
  --corr_root Circular_Video/01_send/20260625_smpl_corr/data/20260320_riding_dataset2/smpl_corr/ \
  --image_root data_process/Output/20260320_riding_dataset2/image \
  --output_root Circular_Video/01_send/20260625_smpl_corr/data/20260320_riding_dataset2/smpl_corr_vis \
  --methods densekp_raw densekp_fitting sam3dbody_raw sam3dbody_fitting \
  --segment 519_528 529_531

# B. 单 method、多段：
python src/script/test_function/debug080_vis_smpl_corr_delivery.py \
  --dataset 20260311_riding \
  --corr_root Circular_Video/01_send/20260625_smpl_corr/data/20260311_riding/smpl_corr/ \
  --image_root data_process/Output/20260311_riding/image \
  --output_root Circular_Video/01_send/20260625_smpl_corr/data/20260311_riding/smpl_corr_vis \
  --methods sam3dbody_fitting \
  --segment 513_518 704_711
TODO:
20260311_riding: 513_518 704_711 1217_1220
20260320_riding_dataset1: 886_895
20260320_riding_dataset2: 519_528 529_531
'''