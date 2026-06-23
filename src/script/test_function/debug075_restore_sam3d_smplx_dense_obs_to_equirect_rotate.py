#!/usr/bin/env python3
"""
Restore SAM 3D Body -> SMPL-X dense 2D observations from the
`equirect_crop_rotate_upright` crop space back to the full `equirect_rotate`
image space, then render an overlay for sanity check.

Per input obs npz (per frame / per camera):
  1. Load points_2d (crop-upright coords) + valid masks
  2. Undo the upright rotation and add the crop bbox offset -> full ERP coords
  3. Recompute geometric in-bounds mask against the full image; valid_full
     stays purely geometric (NO black-pixel / RGB / human-mask filtering)
  4. Save a lightweight observation npz in `equirect_rotate` space
  5. Render an overlay (cam2 yellow, cam4 green), drawing every draw_step point

No SAM 3D Body, no CPW, no fitting, no debug074 changes.

Output:
  output_dir/
    cam2/   cam4/                 # restored observation npz
    vis/cam2/   vis/cam4/         # overlay jpg
    summary.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

SMPLX_NUM_VERTS = 10475

CAM_COLORS = {
    "cam2": (0, 255, 255),   # yellow (BGR)
    "cam4": (0, 255, 0),     # green
}
DEFAULT_COLOR = (0, 255, 255)


# ---------------------------------------------------------------------------
# rotation rule + coordinate restore (20260311_riding specific)
# ---------------------------------------------------------------------------
def get_rotate_rule(frame_id: int) -> str:
    if 0 <= frame_id <= 3494:
        return "ccw90"
    if 3495 <= frame_id <= 5381:
        return "cw90"
    if 5951 <= frame_id <= 5954:
        return "ccw90"
    return "none"


def upright_points_to_equirect_rotate(points_upright, bbox, frame_id):
    x1, y1, x2, y2 = bbox
    crop_w = x2 - x1
    crop_h = y2 - y1

    pts = points_upright.astype(np.float32)
    out = np.empty_like(pts)

    rotate = get_rotate_rule(frame_id)

    if rotate == "ccw90":
        out[:, 0] = crop_w - 1 - pts[:, 1]
        out[:, 1] = pts[:, 0]
    elif rotate == "cw90":
        out[:, 0] = pts[:, 1]
        out[:, 1] = crop_h - 1 - pts[:, 0]
    elif rotate == "180":
        out[:, 0] = crop_w - 1 - pts[:, 0]
        out[:, 1] = crop_h - 1 - pts[:, 1]
    elif rotate == "none":
        out[:, 0] = pts[:, 0]
        out[:, 1] = pts[:, 1]
    else:
        raise ValueError(f"unknown rotate rule: {rotate}")

    out[:, 0] += x1
    out[:, 1] += y1
    return out


# ---------------------------------------------------------------------------
# bbox json
# ---------------------------------------------------------------------------
def load_bboxes(dataset_root: Path, cam: str) -> dict[str, list]:
    bbox_path = dataset_root / "image" / "equirect_crop_rotate" / f"{cam}_bboxes.json"
    with open(str(bbox_path)) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# vis
# ---------------------------------------------------------------------------
def draw_points(img: np.ndarray, points_2d: np.ndarray, valid: np.ndarray,
                draw_step: int, color) -> np.ndarray:
    vis = img.copy()
    ids = np.where(valid)[0]
    if draw_step > 1:
        ids = ids[::draw_step]
    for vid in ids:
        x, y = points_2d[vid]
        cv2.circle(vis, (int(round(x)), int(round(y))), 2, color, -1)
    return vis


# ---------------------------------------------------------------------------
# frame discovery / segment parsing
# ---------------------------------------------------------------------------
def discover_all_frames(obs_dir: Path, cam: str) -> list[int]:
    frames = set()
    for p in (obs_dir / cam).glob("*.npz"):
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


def resolve_frame_indices(segment_args: list[str], obs_dir: Path, cam: str) -> list[int]:
    all_frames: list[int] = []
    saw_all = False
    for seg in segment_args:
        if seg.lower() == "all":
            saw_all = True
        else:
            all_frames.extend(parse_segment_arg(seg))
    if saw_all:
        all_frames.extend(discover_all_frames(obs_dir, cam))
    if not all_frames:
        raise ValueError("no frames to process (check --segment and --obs_dir)")
    return sorted(set(all_frames))


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Restore SMPL-X dense obs from crop-upright to equirect_rotate."
    )
    parser.add_argument("--dataset_root", type=str, required=True)
    parser.add_argument("--obs_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument(
        "--segment", type=str, nargs="+", default=["all"],
        help="Segment range(s) e.g. 513_518 704_711, or 'all' for all frames.",
    )
    parser.add_argument("--cams", type=str, nargs="+", default=["cam2", "cam4"])
    parser.add_argument("--draw_step", type=int, default=20)
    parser.add_argument("--overwrite", action="store_true", default=False)
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root)
    obs_dir = Path(args.obs_dir)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    vis_dir = out / "vis"
    full_dir = dataset_root / "image" / "equirect_rotate"

    bboxes_by_cam = {cam: load_bboxes(dataset_root, cam) for cam in args.cams}

    frame_indices = resolve_frame_indices(args.segment, obs_dir, args.cams[0])
    print(f"[INFO] resolved {len(frame_indices)} frames: "
          f"{frame_indices[0]:05d} ... {frame_indices[-1]:05d}")

    csv_path = out / "summary.csv"
    csv_file = open(str(csv_path), "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow([
        "frame", "cam", "status", "reason",
        "total_points", "input_valid_points", "full_inside_points", "valid_points",
        "output_npz", "output_vis",
    ])

    n_ok = 0
    n_skip = 0

    for frame_idx in frame_indices:
        frame_key = f"{frame_idx:05d}"
        for cam in args.cams:
            tag = f"[{frame_key}] [{cam}]"

            npz_dir = out / cam
            npz_dir.mkdir(parents=True, exist_ok=True)
            out_npz = npz_dir / f"{frame_key}.npz"
            vis_cam_dir = vis_dir / cam
            vis_cam_dir.mkdir(parents=True, exist_ok=True)
            out_vis = vis_cam_dir / f"{frame_key}_overlay.jpg"

            if out_npz.exists() and not args.overwrite:
                print(f"{tag} SKIP: npz exists (use --overwrite to redo)")
                csv_writer.writerow([frame_idx, cam, "skip", "npz exists", *([""] * 6)])
                n_skip += 1
                continue

            obs_npz = obs_dir / cam / f"{frame_key}.npz"
            if not obs_npz.exists():
                print(f"{tag} SKIP: missing obs npz {obs_npz}")
                csv_writer.writerow([frame_idx, cam, "skip", "missing obs npz", *([""] * 6)])
                n_skip += 1
                continue

            bbox_key = str(int(frame_idx))
            if bbox_key not in bboxes_by_cam[cam]:
                print(f"{tag} SKIP: missing bbox for frame {bbox_key}")
                csv_writer.writerow([frame_idx, cam, "skip", "missing bbox", *([""] * 6)])
                n_skip += 1
                continue
            bbox = bboxes_by_cam[cam][bbox_key]

            full_image_path = full_dir / cam / f"{frame_key}.jpg"
            full_img = cv2.imread(str(full_image_path))
            if full_img is None:
                print(f"{tag} SKIP: missing full image {full_image_path}")
                csv_writer.writerow([frame_idx, cam, "skip", "missing full image", *([""] * 6)])
                n_skip += 1
                continue
            full_h, full_w = full_img.shape[:2]

            d = np.load(str(obs_npz), allow_pickle=True)
            vertex_ids = d["vertex_ids"]
            points_2d = d["points_2d"]
            valid_z = d["valid_z"].astype(bool)
            inside = d["inside"].astype(bool)
            valid = d["valid"].astype(bool)

            points_2d_full = upright_points_to_equirect_rotate(points_2d, bbox, frame_idx)

            full_inside = (
                np.isfinite(points_2d_full).all(axis=1)
                & (points_2d_full[:, 0] >= 0)
                & (points_2d_full[:, 0] < full_w)
                & (points_2d_full[:, 1] >= 0)
                & (points_2d_full[:, 1] < full_h)
            )
            valid_full = valid & full_inside

            n_total = int(SMPLX_NUM_VERTS)
            n_in_valid = int(valid.sum())
            n_full_inside = int(full_inside.sum())
            n_valid_full = int(valid_full.sum())
            print(f"{tag} input_valid={n_in_valid} full_inside={n_full_inside} "
                  f"valid_full={n_valid_full} / {n_total}")

            np.savez_compressed(
                out_npz,
                frame_idx=np.array(frame_idx, dtype=np.int32),
                cam=np.array(cam),
                source=np.array("sam3dbody_mhr2smplx"),
                space=np.array("equirect_rotate"),
                image_path=np.array(str(full_image_path)),
                image_shape=np.array(full_img.shape, dtype=np.int32),
                vertex_ids=vertex_ids.astype(np.int32),
                points_2d=points_2d_full.astype(np.float32),
                valid_z=valid_z.astype(bool),
                input_inside=inside.astype(bool),
                full_inside=full_inside.astype(bool),
                valid=valid_full.astype(bool),
            )

            color = CAM_COLORS.get(cam, DEFAULT_COLOR)
            overlay = draw_points(full_img, points_2d_full, valid_full, args.draw_step, color)
            cv2.imwrite(str(out_vis), overlay, [cv2.IMWRITE_JPEG_QUALITY, 95])

            csv_writer.writerow([
                frame_idx, cam, "ok", "",
                n_total, n_in_valid, n_full_inside, n_valid_full,
                str(out_npz), str(out_vis),
            ])
            n_ok += 1

    csv_file.close()
    total = len(frame_indices) * len(args.cams)
    print(f"\n[DONE] {n_ok} ok, {n_skip} skipped (total {total} frame/cam)")
    print(f"summary: {csv_path}")


if __name__ == "__main__":
    main()


r'''
# Single segment, cam2 + cam4:
python src/script/test_function/debug075_restore_sam3d_smplx_dense_obs_to_equirect_rotate.py \
  --dataset_root data_process/Output/20260311_riding \
  --obs_dir data_process/Output/20260311_riding/sam3dbody/smplx_dense \
  --output_dir data_process/Output/20260311_riding/sam3dbody_smplx_dense_obs_equirect_rotate \
  --segment 513_518 \
  --cams cam2 cam4 \
  --draw_step 20 \
  --overwrite

# All frames discovered from obs_dir:
python src/script/test_function/debug075_restore_sam3d_smplx_dense_obs_to_equirect_rotate.py \
  --dataset_root data_process/Output/20260311_riding \
  --obs_dir data_process/Output/20260311_riding/sam3dbody/smplx_dense \
  --output_dir data_process/Output/20260311_riding/sam3dbody_smplx_dense_obs_equirect_rotate \
  --segment all \
  --cams cam2 cam4 \
  --draw_step 20
'''
