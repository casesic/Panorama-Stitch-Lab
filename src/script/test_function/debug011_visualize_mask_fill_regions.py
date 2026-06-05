from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import config_script  # noqa: F401

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import cv2
import numpy as np

from lib_pano.human_vis.smplx_forward_vis import (
    SMPLXForwarder,
    _load_pickle,
    unpack_smplx_result,
)
from lib_pano.human_vis.dual_erp_vis import project_cam2, project_cam4
from lib_pano.pipeline.bbox_io import load_bbox_dict

# Rotation constant matching lib_pano.human_vis.dual_erp_vis._R_CAM2_NP
_R_CAM2_NP = np.array([1, 0, 0, 0, 0, 1, 0, -1, 0], dtype=np.float32).reshape(3, 3)

BLACK_THRESH = 3
UNDERFILL_MIN_AREA = 30

# ---------------------------------------------------------------------------
# argparse
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize underfill/overfill regions between SAM3 mask and SMPL-X silhouette."
    )
    parser.add_argument("--image_root", type=str, required=True)
    parser.add_argument("--result_dir", type=str, required=True)
    parser.add_argument("--segment", type=str, required=True)
    parser.add_argument("--seg_dir", type=str, required=True)
    parser.add_argument("--bbox_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--image_width", type=int, default=5760)
    parser.add_argument("--image_height", type=int, default=2880)
    parser.add_argument("--dilate_smpl", type=int, default=5)
    parser.add_argument("--dilate_mask", type=int, default=5)
    parser.add_argument("--sample_points", type=int, default=200)
    parser.add_argument("--panel_width", type=int, default=480)
    parser.add_argument("--pad", type=int, default=50, help="Padding around bbox for crop panels.")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# helpers: mask rotation (upright -> crop orientation)
# ---------------------------------------------------------------------------
def _get_rotate_code(frame_id: int):
    if frame_id < 3495:
        return cv2.ROTATE_90_COUNTERCLOCKWISE
    if 3495 <= frame_id <= 5381:
        return cv2.ROTATE_90_CLOCKWISE
    if 5951 <= frame_id <= 5954:
        return cv2.ROTATE_90_COUNTERCLOCKWISE
    return None


def upright_mask_to_crop(mask_upright: np.ndarray, frame_id: int) -> np.ndarray:
    """Rotate an upright-crop mask back to the original crop orientation."""
    code = _get_rotate_code(frame_id)
    if code is None:
        return mask_upright
    if code == cv2.ROTATE_90_COUNTERCLOCKWISE:
        return cv2.rotate(mask_upright, cv2.ROTATE_90_CLOCKWISE)
    if code == cv2.ROTATE_90_CLOCKWISE:
        return cv2.rotate(mask_upright, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return mask_upright


# ---------------------------------------------------------------------------
# helpers: load and place SAM3 mask into full ERP
# ---------------------------------------------------------------------------
def load_mask_full_erp(
    frame_idx: int,
    seg_dir: Path,
    cam: str,
    bbox: list[int],
    erp_h: int,
    erp_w: int,
    frame_id: int,
) -> np.ndarray | None:
    """Load SAM3 mask and place it into a full-ERP binary mask."""
    mask_path = seg_dir / cam / "mask" / f"{frame_idx:05d}.png"
    if not mask_path.exists():
        print(f"  [warn] mask not found: {mask_path}")
        return None

    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return None

    # If mask is already full ERP size, threshold and return
    if mask.shape[0] == erp_h and mask.shape[1] == erp_w:
        return (mask > 127).astype(np.uint8) * 255

    x1, y1, x2, y2 = [int(v) for v in bbox]
    crop_h = y2 - y1
    crop_w = x2 - x1

    # Detect orientation: mask may be upright (H,W swapped) or already crop-local
    if mask.shape[0] == crop_w and mask.shape[1] == crop_h:
        # Upright orientation — rotate back to crop
        mask = upright_mask_to_crop(mask, frame_id)

    if mask.shape[0] != crop_h or mask.shape[1] != crop_w:
        mask = cv2.resize(mask, (crop_w, crop_h), interpolation=cv2.INTER_NEAREST)

    full = np.zeros((erp_h, erp_w), dtype=np.uint8)
    full[y1:y2, x1:x2] = (mask > 127).astype(np.uint8) * 255
    return full


# ---------------------------------------------------------------------------
# helpers: rasterize SMPL-X mesh to binary silhouette mask
# ---------------------------------------------------------------------------
def _erp_project_np(points_3d: np.ndarray, width: int, height: int) -> np.ndarray:
    x = points_3d[..., 0]
    y = points_3d[..., 1]
    z = points_3d[..., 2]
    r = np.linalg.norm(points_3d, axis=-1).clip(1e-6, None)
    lon = np.arctan2(x, z)
    lat = np.arcsin((-y / r).clip(-1 + 1e-6, 1 - 1e-6))
    u = (lon / (2.0 * np.pi) + 0.5) * width
    v = (0.5 - lat / np.pi) * height
    return np.stack([u, v], axis=-1)


def rasterize_smplx_mask(
    verts: np.ndarray,
    faces: np.ndarray,
    erp_w: int,
    erp_h: int,
    cam: str,
) -> np.ndarray:
    """Rasterize SMPL-X mesh faces into a full-ERP binary mask."""
    # Apply camera rotation to verts
    verts_rot = verts @ _R_CAM2_NP.T
    v2d = _erp_project_np(verts_rot, erp_w, erp_h).astype(np.int32)

    mask = np.zeros((erp_h, erp_w), dtype=np.uint8)
    for face in faces:
        pts = v2d[face]
        # Skip faces that wrap around the ERP seam
        if pts[:, 0].max() - pts[:, 0].min() > erp_w // 2:
            continue
        if pts[:, 0].max() < 0 or pts[:, 0].min() >= erp_w:
            continue
        if pts[:, 1].max() < 0 or pts[:, 1].min() >= erp_h:
            continue
        cv2.fillPoly(mask, [pts], 255)

    return mask


# ---------------------------------------------------------------------------
# helpers: region analysis
# ---------------------------------------------------------------------------
def compute_regions(
    mask_m: np.ndarray,
    mask_s: np.ndarray,
    dilate_smpl: int,
    dilate_mask: int,
) -> dict:
    M_bool = mask_m > 0
    S_bool = mask_s > 0

    kernel_s = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_smpl * 2 + 1, dilate_smpl * 2 + 1))
    kernel_m = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_mask * 2 + 1, dilate_mask * 2 + 1))

    S_dilated = cv2.dilate(S_bool.astype(np.uint8), kernel_s).astype(bool)
    M_dilated = cv2.dilate(M_bool.astype(np.uint8), kernel_m).astype(bool)

    underfill = M_bool & (~S_dilated)
    overfill = S_bool & (~M_dilated)
    covered = M_bool & S_bool

    mask_area = int(M_bool.sum())
    smpl_area = int(S_bool.sum())

    return {
        "M_bool": M_bool,
        "S_bool": S_bool,
        "underfill": underfill,
        "overfill": overfill,
        "covered": covered,
        "mask_area": mask_area,
        "smpl_area": smpl_area,
        "underfill_area": int(underfill.sum()),
        "overfill_area": int(overfill.sum()),
        "covered_area": int(covered.sum()),
    }


# ---------------------------------------------------------------------------
# helpers: underfill component analysis
# ---------------------------------------------------------------------------
def analyze_underfill_components(
    underfill: np.ndarray, min_area: int
) -> tuple[np.ndarray, list[dict]]:
    """Analyze connected components in underfill mask, filtering small fragments."""
    u8 = underfill.astype(np.uint8) * 255
    n_labels, labels, stats_arr, centroids = cv2.connectedComponentsWithStats(u8, connectivity=8)
    components = []
    filtered = np.zeros(underfill.shape, dtype=bool)
    for label_id in range(1, n_labels):
        area = int(stats_arr[label_id, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        x = int(stats_arr[label_id, cv2.CC_STAT_LEFT])
        y = int(stats_arr[label_id, cv2.CC_STAT_TOP])
        w = int(stats_arr[label_id, cv2.CC_STAT_WIDTH])
        h = int(stats_arr[label_id, cv2.CC_STAT_HEIGHT])
        cx = float(centroids[label_id, 0])
        cy = float(centroids[label_id, 1])
        components.append({
            "id": len(components) + 1,
            "area": area,
            "bbox": [x, y, w, h],
            "centroid": [cx, cy],
        })
        filtered |= (labels == label_id)
    return filtered, components


# ---------------------------------------------------------------------------
# helpers: sample underfill points (per-component allocation)
# ---------------------------------------------------------------------------
def sample_underfill_points(
    underfill: np.ndarray,
    components: list[dict],
    n_samples: int,
    rng: np.random.RandomState,
) -> np.ndarray:
    """Sample underfill points, allocating at least a few per component."""
    if not components or underfill.sum() == 0:
        return np.empty((0, 2), dtype=np.int32)

    ys, xs = np.where(underfill)
    all_pts = np.stack([xs, ys], axis=1).astype(np.int32)

    min_per_comp = min(10, n_samples // max(len(components), 1))
    n_headroom = max(0, n_samples - min_per_comp * len(components))

    # Allocate per component: at least min_per_comp, rest proportional to area
    total_area = sum(c["area"] for c in components)
    sampled = []
    remaining = n_headroom
    for ci, comp in enumerate(components):
        cx, cy = int(comp["centroid"][0]), int(comp["centroid"][1])
        w, h = comp["bbox"][2], comp["bbox"][3]
        bx, by = comp["bbox"][0], comp["bbox"][1]
        comp_mask = underfill[by:by + h, bx:bx + w]
        comp_ys, comp_xs = np.where(comp_mask)
        comp_pts = np.stack([comp_xs + bx, comp_ys + by], axis=1).astype(np.int32)
        n_comp = len(comp_pts)
        if n_comp == 0:
            continue

        n_alloc = min_per_comp
        if total_area > 0 and remaining > 0 and ci < len(components) - 1:
            extra = int(round(n_headroom * comp["area"] / total_area))
            n_alloc += extra
            remaining -= extra
        elif ci == len(components) - 1:
            n_alloc += remaining  # last component gets the rest
        n_alloc = min(n_alloc, n_comp)

        if n_alloc >= n_comp:
            sampled.append(comp_pts)
        else:
            idxs = rng.choice(n_comp, size=n_alloc, replace=False)
            sampled.append(comp_pts[idxs])

    if not sampled:
        return np.empty((0, 2), dtype=np.int32)
    result = np.concatenate(sampled, axis=0)
    if len(result) > n_samples:
        idxs = rng.choice(len(result), size=n_samples, replace=False)
        result = result[idxs]
    return result.astype(np.int32)


# ---------------------------------------------------------------------------
# helpers: build debug panel
# ---------------------------------------------------------------------------
def _resize_to_width(img: np.ndarray, target_w: int) -> np.ndarray:
    h, w = img.shape[:2]
    if w == target_w:
        return img
    scale = target_w / w
    new_h = int(round(h * scale))
    return cv2.resize(img, (target_w, new_h), interpolation=cv2.INTER_LINEAR)


def _to_3ch(img_1ch: np.ndarray, color: tuple[int, int, int]) -> np.ndarray:
    """Convert a 1-channel uint8 mask to a 3-channel color image (0/255 -> bg/color)."""
    out = np.zeros((*img_1ch.shape, 3), dtype=np.uint8)
    fg = img_1ch > 0
    out[fg] = color
    return out


def _add_label(img: np.ndarray, text: str) -> np.ndarray:
    out = img.copy()
    cv2.putText(out, text, (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(out, text, (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def build_cam_panel(
    img_crop: np.ndarray,       # [H, W, 3] original image crop
    M_bool: np.ndarray,         # [H, W] bool, SAM3 mask
    S_bool: np.ndarray,         # [H, W] bool, SMPL-X silhouette
    underfill: np.ndarray,      # [H, W] bool
    overfill: np.ndarray,       # [H, W] bool
    covered: np.ndarray,        # [H, W] bool
    components: list[dict],     # underfill component list
    sample_pts: np.ndarray,     # [K, 2] (x, y) in crop coords
    panel_w: int,
    stats: dict,
    cam_label: str,
) -> np.ndarray:
    # 1. image
    p1 = img_crop.copy()

    # 2. M/S overlay: M=green, S=blue
    p2 = img_crop.copy().astype(np.float32)
    p2[M_bool] = p2[M_bool] * 0.5 + np.array([0, 230, 0], dtype=np.float32) * 0.5
    p2[S_bool] = p2[S_bool] * 0.5 + np.array([200, 100, 0], dtype=np.float32) * 0.5
    p2 = p2.clip(0, 255).astype(np.uint8)

    # 3. fill region overlay
    p3 = img_crop.copy().astype(np.float32)
    p3[underfill] = p3[underfill] * 0.4 + np.array([0, 220, 220], dtype=np.float32) * 0.6
    p3[overfill] = p3[overfill] * 0.4 + np.array([0, 0, 255], dtype=np.float32) * 0.6
    p3[covered] = p3[covered] * 0.4 + np.array([0, 180, 180], dtype=np.float32) * 0.6
    p3 = p3.clip(0, 255).astype(np.uint8)

    # Draw underfill component contours + labels
    underfill_u8 = underfill.astype(np.uint8) * 255
    contours, _ = cv2.findContours(underfill_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(p3, contours, -1, (255, 255, 255), 1)
    for comp in components:
        cx = int(round(comp["centroid"][0]))
        cy = int(round(comp["centroid"][1]))
        cv2.putText(p3, f"C{comp['id']}", (cx + 2, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(p3, f"C{comp['id']}", (cx + 2, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1, cv2.LINE_AA)

    # 4. underfill sampled points on image
    p4 = img_crop.copy()
    for pt in sample_pts:
        cv2.circle(p4, (int(pt[0]), int(pt[1])), 3, (255, 200, 50), -1)

    panels = [_resize_to_width(p, panel_w) for p in [p1, p2, p3, p4]]
    labels = ["image", "M/S overlay", "underfill/overfill", "fill samples"]

    # Stats text overlay on the 3rd panel
    stat_lines = [
        f"{cam_label}",
        f"M={stats['mask_area']} S={stats['smpl_area']}",
        f"under={stats['underfill_area']} ({stats['underfill_ratio']:.2%})",
        f"over={stats['overfill_area']} ({stats['overfill_ratio']:.2%})",
        f"covered={stats['covered_area']}",
    ]
    for li, line in enumerate(stat_lines):
        y = 20 + li * 16
        cv2.putText(panels[2], line, (5, y), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1, cv2.LINE_AA)

    labeled = [_add_label(p, l) for p, l in zip(panels, labels)]

    # Stack: 2x2 grid
    row1 = np.hstack(labeled[:2])
    row2 = np.hstack(labeled[2:])
    max_w = max(row1.shape[1], row2.shape[1])
    r1_pad = np.zeros((row1.shape[0], max_w, 3), dtype=np.uint8)
    r1_pad[:, :row1.shape[1]] = row1
    r2_pad = np.zeros((row2.shape[0], max_w, 3), dtype=np.uint8)
    r2_pad[:, :row2.shape[1]] = row2
    return np.vstack([r1_pad, r2_pad])


# ---------------------------------------------------------------------------
# per-camera processing
# ---------------------------------------------------------------------------
def process_cam(
    frame_idx: int,
    image_root: Path,
    seg_dir: Path,
    bbox_dict: dict,
    output_dir: Path,
    erp_w: int,
    erp_h: int,
    dilate_smpl: int,
    dilate_mask: int,
    n_sample_points: int,
    panel_w: int,
    pad: int,
    params: list,
    param_idx: int,
    forwarder: SMPLXForwarder,
    faces: np.ndarray,
    frame_id: int,
    cam: str,
    rng: np.random.RandomState,
) -> dict | None:
    bbox_key = str(frame_idx)
    bbox = bbox_dict.get(bbox_key)
    if bbox is None:
        print(f"  [{cam}] missing bbox")
        return None

    # Load image
    img_path = image_root / cam / f"{frame_idx:05d}.jpg"
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"  [{cam}] failed to read image")
        return None

    # Load SAM3 mask (full ERP)
    mask_m = load_mask_full_erp(frame_idx, seg_dir, cam, bbox, erp_h, erp_w, frame_id)
    if mask_m is None:
        print(f"  [{cam}] failed to load mask")
        return None

    # Forward SMPL-X params -> verts
    verts, _, _ = forwarder.forward_param_to_numpy(params[param_idx])
    verts = np.asarray(verts, dtype=np.float32)

    # Rasterize SMPL-X mesh to binary mask (full ERP)
    mask_s = rasterize_smplx_mask(verts, faces, erp_w, erp_h, cam)

    # Exclude black (invalid) image regions from both masks
    valid_image = np.any(img > BLACK_THRESH, axis=2)
    mask_m = mask_m.copy()
    mask_s = mask_s.copy()
    mask_m[~valid_image] = 0
    mask_s[~valid_image] = 0

    # Compute regions
    regions = compute_regions(mask_m, mask_s, dilate_smpl, dilate_mask)

    # Analyze underfill connected components
    filtered_underfill, underfill_components = analyze_underfill_components(
        regions["underfill"], UNDERFILL_MIN_AREA
    )
    regions["underfill"] = filtered_underfill
    underfill_area = int(filtered_underfill.sum())
    overfill_area = regions["overfill_area"]
    mask_area = regions["mask_area"]
    smpl_area = regions["smpl_area"]

    stats = {
        "mask_area": mask_area,
        "smpl_area": smpl_area,
        "underfill_area": underfill_area,
        "overfill_area": overfill_area,
        "covered_area": regions["covered_area"],
        "underfill_ratio": underfill_area / max(mask_area, 1),
        "overfill_ratio": overfill_area / max(smpl_area, 1),
        "underfill_components": underfill_components,
    }

    # Sample underfill points (per-component allocation)
    sample_pts = sample_underfill_points(filtered_underfill, underfill_components, n_sample_points, rng)

    # Crop region for visualization (using bbox + pad)
    x1 = max(0, int(bbox[0]) - pad)
    y1 = max(0, int(bbox[1]) - pad)
    x2 = min(erp_w, int(bbox[2]) + pad)
    y2 = min(erp_h, int(bbox[3]) + pad)

    img_crop = img[y1:y2, x1:x2].copy()

    # Crop all masks to the same region
    M_crop = regions["M_bool"][y1:y2, x1:x2]
    S_crop = regions["S_bool"][y1:y2, x1:x2]
    underfill_crop = regions["underfill"][y1:y2, x1:x2]
    overfill_crop = regions["overfill"][y1:y2, x1:x2]
    covered_crop = regions["covered"][y1:y2, x1:x2]

    # Offset underfill component coordinates to crop coords
    components_crop = []
    for comp in underfill_components:
        c = dict(comp)
        c["bbox"] = [c["bbox"][0] - x1, c["bbox"][1] - y1, c["bbox"][2], c["bbox"][3]]
        c["centroid"] = [c["centroid"][0] - x1, c["centroid"][1] - y1]
        components_crop.append(c)

    # Offset sample points to crop coords
    sample_pts_crop = sample_pts.copy()
    if len(sample_pts_crop) > 0:
        sample_pts_crop[:, 0] -= x1
        sample_pts_crop[:, 1] -= y1

    # Build panel
    panel = build_cam_panel(
        img_crop=img_crop,
        M_bool=M_crop,
        S_bool=S_crop,
        underfill=underfill_crop,
        overfill=overfill_crop,
        covered=covered_crop,
        components=components_crop,
        sample_pts=sample_pts_crop,
        panel_w=panel_w,
        stats=stats,
        cam_label=cam,
    )

    # Save
    out_path = output_dir / f"{frame_idx:05d}_{cam}.jpg"
    cv2.imwrite(str(out_path), panel, [cv2.IMWRITE_JPEG_QUALITY, 92])
    print(f"  [{cam}] saved {out_path}  under={underfill_area} over={overfill_area}")

    return stats


# ---------------------------------------------------------------------------
# segment helpers
# ---------------------------------------------------------------------------
def parse_segment_range(segment: str) -> tuple[int, int]:
    parts = segment.split("_")
    if len(parts) != 2:
        raise ValueError(f"segment must match start_end, got: {segment}")
    start, end = int(parts[0]), int(parts[1])
    if start > end:
        raise ValueError(f"segment start must be <= end: {segment}")
    return start, end


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main() -> None:
    args = parse_args()

    image_root = Path(args.image_root)
    result_dir = Path(args.result_dir)
    seg_dir = Path(args.seg_dir)
    bbox_dir = Path(args.bbox_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    erp_w = args.image_width
    erp_h = args.image_height

    start_frame, end_frame = parse_segment_range(args.segment)
    frame_indices = list(range(start_frame, end_frame + 1))

    # Load bbox dicts
    bbox_c2 = load_bbox_dict(bbox_dir / "cam2_bboxes.json")
    bbox_c4 = load_bbox_dict(bbox_dir / "cam4_bboxes.json")

    # Load SMPL-X fitting results
    print("Loading fitting results ...")
    result_c2 = _load_pickle(result_dir / "cam2" / f"seg_{args.segment}_seq.pkl")
    result_c4 = _load_pickle(result_dir / "cam4" / f"seg_{args.segment}_seq.pkl")
    params_c2 = unpack_smplx_result(result_c2)
    params_c4 = unpack_smplx_result(result_c4)

    total_frames = min(len(frame_indices), len(params_c2), len(params_c4))
    print(f"Segment: {args.segment}, frames: {start_frame}-{end_frame}, usable: {total_frames}")
    print()

    # Initialize SMPLX forwarder
    print("Loading SMPLX forwarder ...")
    forwarder = SMPLXForwarder(gender="neutral")
    faces = forwarder.get_model_faces()
    faces = np.asarray(faces, dtype=np.int32)
    print(f"  faces: {faces.shape}")
    print()

    rng = np.random.RandomState(0)
    all_stats: dict = {"frames": {}}

    for idx in range(total_frames):
        frame_idx = frame_indices[idx]
        frame_key = f"{frame_idx:05d}"
        print(f"[{frame_key}] processing ...")

        frame_stats = {}

        for cam, params, bbox_dict in [
            ("cam2", params_c2, bbox_c2),
            ("cam4", params_c4, bbox_c4),
        ]:
            stats = process_cam(
                frame_idx=frame_idx,
                image_root=image_root,
                seg_dir=seg_dir,
                bbox_dict=bbox_dict,
                output_dir=output_dir,
                erp_w=erp_w,
                erp_h=erp_h,
                dilate_smpl=args.dilate_smpl,
                dilate_mask=args.dilate_mask,
                n_sample_points=args.sample_points,
                panel_w=args.panel_width,
                pad=args.pad,
                params=params,
                param_idx=idx,
                forwarder=forwarder,
                faces=faces,
                frame_id=frame_idx,
                cam=cam,
                rng=rng,
            )
            if stats is not None:
                frame_stats[cam] = stats

        if frame_stats:
            all_stats["frames"][frame_key] = frame_stats

    # Save summary JSON
    summary_path = output_dir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(all_stats, f, indent=2, ensure_ascii=False)
    print(f"\nSummary saved to {summary_path}")
    print(f"Done. {len(all_stats['frames'])} frames processed.")


if __name__ == "__main__":
    main()


r'''
python src/script/test_function/debug011_visualize_mask_fill_regions.py \
  --image_root data_process/Output/20260311_riding/image/equirect_rotate \
  --result_dir data_process/Output/20260311_riding/fitting/kp_densekp_bbox_filter_maskfill_stage3 \
  --segment 704_711 \
  --seg_dir data_process/Output/20260311_riding/seg/equirect_crop_rotate_sam3 \
  --bbox_dir data_process/Output/20260311_riding/image/equirect_crop_rotate \
  --output_dir data_process/Output/20260311_riding/vis/mask_fill_regions/kp_densekp_bbox_filter_maskfill_stage3 \
  --image_width 5760 \
  --image_height 2880
'''
