#!/usr/bin/env python3
"""
Batch export SAM 3D Body reconstruction per camera (crop_upright images only).

Per frame / per camera:
  1. Run SAM 3D Body on equirect_crop_rotate_upright image
  2. Project mesh vertices to 2D
  3. Save npz + visualization images

Output:
  output_dir/
    cam2/   cam4/
    vis/cam2/   vis/cam4/
    summary.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np

SRC_ROOT = Path(__file__).resolve().parents[2]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

SAM3D_ROOT = SRC_ROOT.parent / "tools" / "sam-3d-body"
if str(SAM3D_ROOT) not in sys.path:
    sys.path.insert(0, str(SAM3D_ROOT))

import torch
from sam_3d_body import load_sam_3d_body, SAM3DBodyEstimator
from tools.build_detector import HumanDetector
from tools.build_fov_estimator import FOVEstimator
from tools.vis_utils import visualize_sample_together

from lib_pano.human3d.sam3dbody_project import project_vertices_opencv_style


# ---------------------------------------------------------------------------
# draw projection points
# ---------------------------------------------------------------------------
def draw_points(img: np.ndarray, pts2d: np.ndarray, valid: np.ndarray,
                bbox=None, max_pts: int = 1500) -> np.ndarray:
    vis = img.copy()
    h, w = img.shape[:2]
    inside = (
        valid
        & np.isfinite(pts2d[:, 0]) & np.isfinite(pts2d[:, 1])
        & (pts2d[:, 0] >= 0) & (pts2d[:, 0] < w)
        & (pts2d[:, 1] >= 0) & (pts2d[:, 1] < h)
    )
    ids = np.where(inside)[0]
    if len(ids) > max_pts:
        ids = np.linspace(0, len(ids) - 1, max_pts).astype(int)
        ids = np.where(inside)[0][ids]
    for vid in ids:
        x, y = pts2d[vid]
        cv2.circle(vis, (int(round(x)), int(round(y))), 2, (0, 255, 255), -1)
    if bbox is not None:
        x1, y1, x2, y2 = bbox
        cv2.rectangle(vis, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
    return vis


# ---------------------------------------------------------------------------
# model init (once)
# ---------------------------------------------------------------------------
def init_model(checkpoint_path: str, mhr_path: str, device: torch.device):
    model, model_cfg = load_sam_3d_body(checkpoint_path, device=device, mhr_path=mhr_path)
    human_detector = HumanDetector(name="vitdet", device=device, path="")
    fov_estimator = FOVEstimator(name="moge2", device=device, path="")
    estimator = SAM3DBodyEstimator(
        sam_3d_body_model=model,
        model_cfg=model_cfg,
        human_detector=human_detector,
        human_segmentor=None,
        fov_estimator=fov_estimator,
    )
    return estimator


# ---------------------------------------------------------------------------
# process single image
# ---------------------------------------------------------------------------
def process_one_image(
    estimator: SAM3DBodyEstimator,
    image_path: Path,
    bbox_thr: float,
) -> dict | None:
    if not image_path.exists():
        return None
    outputs = estimator.process_one_image(str(image_path), bbox_thr=bbox_thr, use_mask=False)
    if len(outputs) == 0:
        return None
    # select largest bbox
    best = None
    best_area = -1
    for out in outputs:
        x1, y1, x2, y2 = out["bbox"]
        area = max(0, x2 - x1) * max(0, y2 - y1)
        if area > best_area:
            best_area = area
            best = out
    return {"selected": best, "num_persons": len(outputs), "pid": int(np.argmax(
        [max(0, o["bbox"][2] - o["bbox"][0]) * max(0, o["bbox"][3] - o["bbox"][1]) for o in outputs]
    ))}


# ---------------------------------------------------------------------------
# parse segment(s)
# ---------------------------------------------------------------------------
def discover_all_frames(upright_dir: Path, cam: str = "cam2") -> list[int]:
    """Discover available frame indices from upright image filenames."""
    glob_pat = "*.jpg"
    frames = set()
    for p in (upright_dir / cam).glob(glob_pat):
        try:
            frames.add(int(p.stem))
        except ValueError:
            continue
    return sorted(frames)


def parse_segment_arg(segment_str: str) -> list[int]:
    """Parse a single segment string, e.g. '513_518'."""
    parts = segment_str.split("_")
    if len(parts) != 2:
        raise ValueError(f"segment must be 'start_end' or 'all', got: {segment_str}")
    start, end = int(parts[0]), int(parts[1])
    if start > end:
        raise ValueError(f"segment start must be <= end: {segment_str}")
    return list(range(start, end + 1))


def resolve_frame_indices(
    segment_args: list[str],
    upright_dir: Path,
    cam: str = "cam2",
) -> list[int]:
    """Resolve --segment args to a flat, deduplicated, sorted list of frame indices."""
    all_frames: list[int] = []
    saw_all = False
    for seg in segment_args:
        if seg.lower() == "all":
            saw_all = True
        else:
            all_frames.extend(parse_segment_arg(seg))
    if saw_all:
        discovered = discover_all_frames(upright_dir, cam)
        all_frames.extend(discovered)
    if not all_frames:
        raise ValueError("no frames to process (check --segment and input directory)")
    # deduplicate and sort
    return sorted(set(all_frames))


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Batch export SAM 3D Body reconstruction per camera."
    )
    parser.add_argument("--dataset_root", type=str, required=True)
    parser.add_argument(
        "--segment", type=str, nargs="+", default=["all"],
        help="Segment range(s) e.g. 513_518 704_711, or 'all' for all frames.",
    )
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--mhr_path", type=str, required=True)
    parser.add_argument("--cams", type=str, nargs="+", default=["cam2", "cam4"])
    parser.add_argument("--bbox_thresh", type=float, default=0.8)
    parser.add_argument("--overwrite", action="store_true", default=False)
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # init
    # ------------------------------------------------------------------
    dataset_root = Path(args.dataset_root)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    upright_dir = dataset_root / "image" / "equirect_crop_rotate_upright"
    vis_dir = out / "vis"

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"[INFO] device: {device}")

    # resolve frame list
    frame_indices = resolve_frame_indices(args.segment, upright_dir, cam=args.cams[0])
    print(f"[INFO] resolved {len(frame_indices)} frames: {frame_indices[0]:05d} ... {frame_indices[-1]:05d}")

    print("[INFO] loading SAM 3D Body ...")
    estimator = init_model(args.checkpoint_path, args.mhr_path, device)
    print("[INFO] model loaded.")

    # ------------------------------------------------------------------
    # summary csv
    # ------------------------------------------------------------------
    csv_path = out / "summary.csv"
    csv_file = open(str(csv_path), "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow([
        "frame", "cam", "status", "reason",
        "num_persons", "selected_pid",
        "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2",
        "inside_points", "valid_z_points", "npz_path",
    ])

    n_ok = 0
    n_skip = 0

    for frame_idx in frame_indices:
        frame_key = f"{frame_idx:05d}"
        for cam in args.cams:
            tag = f"[{frame_key}] [{cam}]"
            print(f"{tag} processing ...")

            # output dirs
            npz_dir = out / cam
            npz_dir.mkdir(parents=True, exist_ok=True)
            npz_path = npz_dir / f"{frame_key}.npz"
            vis_cam_dir = vis_dir / cam
            vis_cam_dir.mkdir(parents=True, exist_ok=True)

            # skip if exists
            if npz_path.exists() and not args.overwrite:
                print(f"{tag} SKIP: npz exists (use --overwrite to redo)")
                csv_writer.writerow([frame_idx, cam, "skip", "npz exists", *([""] * 9)])
                n_skip += 1
                continue

            # read image
            img_path = upright_dir / cam / f"{frame_key}.jpg"
            if not img_path.exists():
                print(f"{tag} SKIP: missing {img_path}")
                csv_writer.writerow([frame_idx, cam, "skip", "missing image", *([""] * 9)])
                n_skip += 1
                continue

            # run SAM 3D Body
            result = process_one_image(estimator, img_path, bbox_thr=args.bbox_thresh)
            if result is None:
                print(f"{tag} SKIP: no person detected")
                csv_writer.writerow([frame_idx, cam, "skip", "no person detected", 0, *([""] * 8)])
                n_skip += 1
                continue

            sel = result["selected"]
            num_persons = result["num_persons"]
            pid = result["pid"]

            img = cv2.imread(str(img_path))
            if img is None:
                print(f"{tag} SKIP: failed to read image")
                csv_writer.writerow([frame_idx, cam, "skip", "cv2.imread failed", *([""] * 9)])
                n_skip += 1
                continue

            # project vertices
            vertices = sel["pred_vertices"]
            cam_t = sel["pred_cam_t"]
            focal = float(sel["focal_length"])
            bbox = sel["bbox"]
            faces = estimator.faces

            pts2d, verts_cam, valid_z = project_vertices_opencv_style(
                vertices, cam_t, focal, img.shape,
            )

            inside = (
                np.isfinite(pts2d).all(axis=1)
                & valid_z
                & (pts2d[:, 0] >= 0) & (pts2d[:, 0] < img.shape[1])
                & (pts2d[:, 1] >= 0) & (pts2d[:, 1] < img.shape[0])
            )

            n_inside = int(inside.sum())
            n_valid_z = int(valid_z.sum())
            print(f"{tag} inside={n_inside} valid_z={n_valid_z}")

            # save npz
            np.savez_compressed(
                npz_path,
                image_path=str(img_path),
                cam=cam,
                frame_idx=frame_idx,
                image_shape=np.array(img.shape),
                bbox=bbox,
                focal_length=np.array(focal, dtype=np.float32),
                pred_vertices=vertices,
                pred_cam_t=cam_t,
                vertices_cam=verts_cam,
                vertices_2d=pts2d,
                valid_z=valid_z,
                inside=inside,
                faces=faces,
                pred_keypoints_2d=sel["pred_keypoints_2d"],
            )

            # visualization
            overlay = draw_points(img, pts2d, valid_z, bbox=bbox)
            cv2.imwrite(str(vis_cam_dir / f"{frame_key}_vertices_overlay.jpg"), overlay, [cv2.IMWRITE_JPEG_QUALITY, 95])

            official_mesh = visualize_sample_together(img, [sel], faces)
            cv2.imwrite(str(vis_cam_dir / f"{frame_key}_official_mesh.jpg"), official_mesh.astype(np.uint8), [cv2.IMWRITE_JPEG_QUALITY, 95])

            csv_writer.writerow([
                frame_idx, cam, "ok", "",
                num_persons, pid,
                f"{bbox[0]:.1f}", f"{bbox[1]:.1f}", f"{bbox[2]:.1f}", f"{bbox[3]:.1f}",
                n_inside, n_valid_z, str(npz_path),
            ])
            n_ok += 1

    csv_file.close()
    print(f"\n[DONE] {n_ok} ok, {n_skip} skipped (total {len(frame_indices) * len(args.cams)} frames)")
    print(f"summary: {csv_path}")


if __name__ == "__main__":
    main()


r'''
# Multiple segments:
python src/script/test_function/debug066_batch_export_sam3dbody_recon.py \
  --dataset_root data_process/Output/20260311_riding \
  --segment 513_518 704_711 1217_1225 \
  --output_dir data_process/Output/20260311_riding/sam3dbody/recon \
  --checkpoint_path ~/.cache/modelscope/hub/models/facebook/sam-3d-body-dinov3/model.ckpt \
  --mhr_path ~/.cache/modelscope/hub/models/facebook/sam-3d-body-dinov3/assets/mhr_model.pt \
  --cams cam2 cam4 \
  --bbox_thresh 0.8

# All frames:
python src/script/test_function/debug066_batch_export_sam3dbody_recon.py \
  --dataset_root data_process/Output/20260320_riding_dataset2 \
  --segment all \
  --output_dir data_process/Output/20260320_riding_dataset2/sam3dbody/recon \
  --checkpoint_path ~/.cache/modelscope/hub/models/facebook/sam-3d-body-dinov3/model.ckpt \
  --mhr_path ~/.cache/modelscope/hub/models/facebook/sam-3d-body-dinov3/assets/mhr_model.pt \
  --cams cam2 cam4 \
  --bbox_thresh 0.8

# With overwrite:
python src/script/test_function/debug066_batch_export_sam3dbody_recon.py \
  --dataset_root data_process/Output/20260311_riding \
  --segment 513_518 \
  --output_dir data_process/Output/20260311_riding/sam3dbody/recon \
  --checkpoint_path ~/.cache/modelscope/hub/models/facebook/sam-3d-body-dinov3/model.ckpt \
  --mhr_path ~/.cache/modelscope/hub/models/facebook/sam-3d-body-dinov3/assets/mhr_model.pt \
  --cams cam2 cam4 \
  --bbox_thresh 0.8 \
  --overwrite
'''
