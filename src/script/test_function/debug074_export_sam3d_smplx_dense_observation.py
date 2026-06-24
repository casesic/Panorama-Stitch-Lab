#!/usr/bin/env python3
"""
Export SMPL-X (10475) dense 2D observations from SAM 3D Body recon npz.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

SMPLX_NUM_VERTS = 10475


# ---------------------------------------------------------------------------
# vis: draw subsampled valid points on the original image
# ---------------------------------------------------------------------------
def draw_points(img: np.ndarray, points_2d: np.ndarray, valid: np.ndarray,
                draw_step: int) -> np.ndarray:
    vis = img.copy()
    ids = np.where(valid)[0]
    if draw_step > 1:
        ids = ids[::draw_step]
    for vid in ids:
        x, y = points_2d[vid]
        cv2.circle(vis, (int(round(x)), int(round(y))), 2, (0, 255, 255), -1)
    return vis


# ---------------------------------------------------------------------------
# frame discovery / segment parsing
# ---------------------------------------------------------------------------
def discover_all_frames(recon_dir: Path, cam: str) -> list[int]:
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


def resolve_frame_indices(segment_args: list[str], recon_dir: Path, cam: str) -> list[int]:
    all_frames: list[int] = []
    saw_all = False
    for seg in segment_args:
        if seg.lower() == "all":
            saw_all = True
        else:
            all_frames.extend(parse_segment_arg(seg))
    if saw_all:
        all_frames.extend(discover_all_frames(recon_dir, cam))
    if not all_frames:
        raise ValueError("no frames to process (check --segment and --recon_dir)")
    return sorted(set(all_frames))


# ---------------------------------------------------------------------------
# mapping + barycentric surface -> 2D projection
# ---------------------------------------------------------------------------
def load_mapping(mapping_path: Path) -> tuple[np.ndarray, np.ndarray]:
    m = np.load(str(mapping_path))
    triangle_ids = m["triangle_ids"].astype(np.int64)
    baryc_coords = m["baryc_coords"].astype(np.float64)
    if triangle_ids.shape[0] != SMPLX_NUM_VERTS:
        raise ValueError(
            f"triangle_ids count {triangle_ids.shape[0]} != {SMPLX_NUM_VERTS}"
        )
    return triangle_ids, baryc_coords


def map_and_project(
    pred_vertices: np.ndarray,
    pred_cam_t: np.ndarray,
    faces: np.ndarray,
    focal_length: float,
    image_shape,
    triangle_ids: np.ndarray,
    baryc_coords: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    H, W = int(image_shape[0]), int(image_shape[1])

    verts_cam = pred_vertices.astype(np.float64) + pred_cam_t.astype(np.float64)[None, :]

    mhr_face_vids = faces[triangle_ids]                      # [10475, 3]
    mhr_tri_verts = verts_cam[mhr_face_vids]                 # [10475, 3, 3]
    smplx_surface_3d = (baryc_coords[..., None] * mhr_tri_verts).sum(axis=1)  # [10475, 3]

    X = smplx_surface_3d[:, 0]
    Y = smplx_surface_3d[:, 1]
    Z = smplx_surface_3d[:, 2]

    with np.errstate(divide="ignore", invalid="ignore"):
        u = focal_length * X / Z + W / 2.0
        v = focal_length * Y / Z + H / 2.0
    points_2d = np.stack([u, v], axis=1).astype(np.float32)

    valid_z = Z > 1e-6
    inside = (
        (points_2d[:, 0] >= 0) & (points_2d[:, 0] < W)
        & (points_2d[:, 1] >= 0) & (points_2d[:, 1] < H)
    )
    valid = valid_z & inside
    return points_2d, valid_z, inside, valid


def resolve_image_path(image_path: str) -> Path:
    p = Path(image_path)
    if p.is_absolute() and p.exists():
        return p
    candidate = PROJECT_ROOT / image_path
    return candidate if candidate.exists() else p


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export SMPL-X dense 2D observations from SAM 3D Body recon npz."
    )
    parser.add_argument("--recon_dir", type=str, required=True)
    parser.add_argument("--mapping_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument(
        "--segment", type=str, nargs="+", default=["all"],
        help="Segment range(s) e.g. 513_518 704_711, or 'all' for all frames.",
    )
    parser.add_argument("--cams", type=str, nargs="+", default=["cam2", "cam4"])
    parser.add_argument("--draw_step", type=int, default=20)
    parser.add_argument("--overwrite", action="store_true", default=False)
    args = parser.parse_args()

    recon_dir = Path(args.recon_dir)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    vis_dir = out / "vis"

    triangle_ids, baryc_coords = load_mapping(Path(args.mapping_path))
    print(f"[INFO] mapping loaded: {triangle_ids.shape[0]} smplx verts")

    frame_indices = resolve_frame_indices(args.segment, recon_dir, args.cams[0])
    print(f"[INFO] resolved {len(frame_indices)} frames: "
          f"{frame_indices[0]:05d} ... {frame_indices[-1]:05d}")

    csv_path = out / "summary.csv"
    csv_file = open(str(csv_path), "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow([
        "frame", "cam", "status", "reason",
        "total_points", "valid_points", "inside_points",
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
                csv_writer.writerow([frame_idx, cam, "skip", "npz exists", "", "", "", "", ""])
                n_skip += 1
                continue

            recon_npz = recon_dir / cam / f"{frame_key}.npz"
            if not recon_npz.exists():
                print(f"{tag} SKIP: missing recon npz {recon_npz}")
                csv_writer.writerow([frame_idx, cam, "skip", "missing recon npz", "", "", "", "", ""])
                n_skip += 1
                continue

            d = np.load(str(recon_npz), allow_pickle=True)
            pred_vertices = d["pred_vertices"]
            pred_cam_t = d["pred_cam_t"]
            faces = d["faces"]
            focal_length = float(d["focal_length"])
            image_shape = np.asarray(d["image_shape"], dtype=np.int32)
            image_path = str(d["image_path"])

            points_2d, valid_z, inside, valid = map_and_project(
                pred_vertices, pred_cam_t, faces, focal_length,
                image_shape, triangle_ids, baryc_coords,
            )
            vertex_ids = np.arange(SMPLX_NUM_VERTS, dtype=np.int32)

            n_total = int(SMPLX_NUM_VERTS)
            n_valid = int(valid.sum())
            n_inside = int(inside.sum())
            print(f"{tag} valid={n_valid} inside={n_inside} / {n_total}")

            np.savez_compressed(
                out_npz,
                frame_idx=np.array(frame_idx, dtype=np.int32),
                cam=np.array(cam),
                source=np.array("sam3dbody_mhr2smplx"),
                space=np.array("equirect_crop_rotate_upright"),
                image_path=np.array(image_path),
                image_shape=image_shape,
                vertex_ids=vertex_ids,
                points_2d=points_2d,
                valid_z=valid_z.astype(bool),
                inside=inside.astype(bool),
                valid=valid.astype(bool),
            )

            img = cv2.imread(str(resolve_image_path(image_path)))
            if img is None:
                print(f"{tag} WARN: failed to read image for vis: {image_path}")
                csv_writer.writerow([
                    frame_idx, cam, "ok", "vis skipped (image unreadable)",
                    n_total, n_valid, n_inside, str(out_npz), "",
                ])
                n_ok += 1
                continue

            overlay = draw_points(img, points_2d, valid, args.draw_step)
            cv2.imwrite(str(out_vis), overlay, [cv2.IMWRITE_JPEG_QUALITY, 95])

            csv_writer.writerow([
                frame_idx, cam, "ok", "",
                n_total, n_valid, n_inside, str(out_npz), str(out_vis),
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
python src/script/test_function/debug074_export_sam3d_smplx_dense_observation.py \
  --recon_dir data_process/Output/20260311_riding/sam3dbody/recon \
  --mapping_path tools/MHR/tools/mhr_smpl_conversion/assets/mhr2smplx_mapping.npz \
  --output_dir data_process/Output/20260311_riding/sam3dbody/smplx_dense \
  --segment 513_518 \
  --cams cam2 cam4 \
  --draw_step 20 \
  --overwrite

# All frames discovered from recon_dir:
python src/script/test_function/debug074_export_sam3d_smplx_dense_observation.py \
  --recon_dir data_process/Output/20260320_riding_dataset2/sam3dbody/recon \
  --mapping_path tools/MHR/tools/mhr_smpl_conversion/assets/mhr2smplx_mapping.npz \
  --output_dir data_process/Output/20260320_riding_dataset2/sam3dbody/smplx_dense \
  --segment all \
  --cams cam2 cam4 \
  --draw_step 20
'''
