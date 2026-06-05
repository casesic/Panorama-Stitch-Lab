from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch

# ---------------------------------------------------------------------------
# constants
# ---------------------------------------------------------------------------
BLACK_THRESH = 3
UNDERFILL_MIN_AREA = 30
DILATE_SMPL = 5
DILATE_MASK = 5
DEFAULT_SAMPLE_POINTS = 200

# ---------------------------------------------------------------------------
# mask loading
# ---------------------------------------------------------------------------
def _get_rotate_code(frame_id: int):
    if frame_id < 3495:
        return cv2.ROTATE_90_COUNTERCLOCKWISE
    if 3495 <= frame_id <= 5381:
        return cv2.ROTATE_90_CLOCKWISE
    if 5951 <= frame_id <= 5954:
        return cv2.ROTATE_90_COUNTERCLOCKWISE
    return None


def _upright_mask_to_crop(mask_upright: np.ndarray, frame_id: int) -> np.ndarray:
    code = _get_rotate_code(frame_id)
    if code is None:
        return mask_upright
    if code == cv2.ROTATE_90_COUNTERCLOCKWISE:
        return cv2.rotate(mask_upright, cv2.ROTATE_90_CLOCKWISE)
    if code == cv2.ROTATE_90_CLOCKWISE:
        return cv2.rotate(mask_upright, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return mask_upright


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
        mask = _upright_mask_to_crop(mask, frame_id)

    if mask.shape[0] != crop_h or mask.shape[1] != crop_w:
        mask = cv2.resize(mask, (crop_w, crop_h), interpolation=cv2.INTER_NEAREST)

    full = np.zeros((erp_h, erp_w), dtype=np.uint8)
    full[y1:y2, x1:x2] = (mask > 127).astype(np.uint8) * 255
    return full


# ---------------------------------------------------------------------------
# SMPL-X silhouette rasterization
# ---------------------------------------------------------------------------
_R_CAM2_NP = np.array([1, 0, 0, 0, 0, 1, 0, -1, 0], dtype=np.float32).reshape(3, 3)


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


def rasterize_smplx_mask_np(
    verts: np.ndarray,
    faces: np.ndarray,
    erp_w: int,
    erp_h: int,
) -> np.ndarray:
    """Rasterize SMPL-X mesh faces into a full-ERP binary mask."""
    verts_rot = verts @ _R_CAM2_NP.T
    v2d = _erp_project_np(verts_rot, erp_w, erp_h).astype(np.int32)

    mask = np.zeros((erp_h, erp_w), dtype=np.uint8)
    for face in faces:
        pts = v2d[face]
        if pts[:, 0].max() - pts[:, 0].min() > erp_w // 2:
            continue
        if pts[:, 0].max() < 0 or pts[:, 0].min() >= erp_w:
            continue
        if pts[:, 1].max() < 0 or pts[:, 1].min() >= erp_h:
            continue
        cv2.fillPoly(mask, [pts], 255)

    return mask


# ---------------------------------------------------------------------------
# region computation
# ---------------------------------------------------------------------------
def compute_mask_fill_regions(
    mask_m: np.ndarray,
    mask_s: np.ndarray,
    image: np.ndarray | None = None,
) -> dict:
    """Compute underfill/overfill/covered regions and connected components."""
    M = mask_m.copy()
    S = mask_s.copy()

    if image is not None:
        valid_image = np.any(image > BLACK_THRESH, axis=2)
        M[~valid_image] = 0
        S[~valid_image] = 0

    M_bool = M > 0
    S_bool = S > 0

    kernel_s = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (DILATE_SMPL * 2 + 1, DILATE_SMPL * 2 + 1))
    kernel_m = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (DILATE_MASK * 2 + 1, DILATE_MASK * 2 + 1))

    S_dilated = cv2.dilate(S_bool.astype(np.uint8), kernel_s).astype(bool)
    M_dilated = cv2.dilate(M_bool.astype(np.uint8), kernel_m).astype(bool)

    underfill = M_bool & (~S_dilated)
    overfill = S_bool & (~M_dilated)
    covered = M_bool & S_bool

    # Connected component analysis on underfill
    filtered_underfill, components = _analyze_underfill_components(underfill, UNDERFILL_MIN_AREA)

    mask_area = int(M_bool.sum())
    smpl_area = int(S_bool.sum())

    return {
        "M_bool": M_bool,
        "S_bool": S_bool,
        "underfill": filtered_underfill,
        "overfill": overfill,
        "covered": covered,
        "components": components,
        "mask_area": mask_area,
        "smpl_area": smpl_area,
        "underfill_area": int(filtered_underfill.sum()),
        "overfill_area": int(overfill.sum()),
        "covered_area": int(covered.sum()),
    }


def _analyze_underfill_components(
    underfill: np.ndarray, min_area: int
) -> tuple[np.ndarray, list[dict]]:
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
# underfill point sampling (per-component allocation)
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

    min_per_comp = min(10, n_samples // max(len(components), 1))
    n_headroom = max(0, n_samples - min_per_comp * len(components))
    total_area = sum(c["area"] for c in components)

    sampled = []
    remaining = n_headroom
    for ci, comp in enumerate(components):
        bx, by, bw, bh = comp["bbox"]
        comp_mask = underfill[by:by + bh, bx:bx + bw]
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
            n_alloc += remaining
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
# prepare samples for a whole sequence
# ---------------------------------------------------------------------------
def prepare_mask_fill_samples_for_sequence(
    frame_indices: list[int],
    verts_list: list[np.ndarray],
    image_root: Path,
    seg_dir: Path,
    bbox_dict: dict,
    faces: np.ndarray,
    erp_w: int,
    erp_h: int,
    cam: str,
    n_sample_points: int = DEFAULT_SAMPLE_POINTS,
    rng: np.random.RandomState | None = None,
) -> tuple[list[np.ndarray], list[dict]]:
    """Generate underfill sample points for a sequence of frames.

    Returns:
        samples_list: list of [K_i, 2] arrays in full ERP coords
        stats_list: per-frame region stats
    """
    if rng is None:
        rng = np.random.RandomState(0)

    samples_list: list[np.ndarray] = []
    stats_list: list[dict] = []

    for i, frame_idx in enumerate(frame_indices):
        bbox_key = str(frame_idx)
        bbox = bbox_dict.get(bbox_key)
        if bbox is None:
            samples_list.append(np.empty((0, 2), dtype=np.int32))
            stats_list.append({"error": "missing bbox"})
            continue

        # Load SAM3 mask
        mask_m = load_mask_full_erp(frame_idx, seg_dir, cam, bbox, erp_h, erp_w, frame_idx)
        if mask_m is None:
            samples_list.append(np.empty((0, 2), dtype=np.int32))
            stats_list.append({"error": "missing mask"})
            continue

        # Load image for black-region filtering
        img_path = image_root / cam / f"{frame_idx:05d}.jpg"
        img = cv2.imread(str(img_path))
        if img is None:
            samples_list.append(np.empty((0, 2), dtype=np.int32))
            stats_list.append({"error": "missing image"})
            continue

        # Rasterize SMPL-X silhouette
        verts = verts_list[i]
        mask_s = rasterize_smplx_mask_np(verts, faces, erp_w, erp_h)

        # Compute regions
        regions = compute_mask_fill_regions(mask_m, mask_s, image=img)

        # Sample underfill points
        sample_pts = sample_underfill_points(
            regions["underfill"], regions["components"], n_sample_points, rng
        )
        samples_list.append(sample_pts.astype(np.float32))

        stats = {
            "mask_area": regions["mask_area"],
            "smpl_area": regions["smpl_area"],
            "underfill_area": regions["underfill_area"],
            "overfill_area": regions["overfill_area"],
            "covered_area": regions["covered_area"],
            "underfill_ratio": regions["underfill_area"] / max(regions["mask_area"], 1),
            "overfill_ratio": regions["overfill_area"] / max(regions["smpl_area"], 1),
            "underfill_components": regions["components"],
            "sample_count": len(sample_pts),
        }
        stats_list.append(stats)

    return samples_list, stats_list


# ---------------------------------------------------------------------------
# MaskFillLoss (differentiable)
# ---------------------------------------------------------------------------
class MaskFillLoss:
    def __init__(self, image_width: int, image_height: int, robust_scale: float = 100.0):
        self.W = image_width
        self.H = image_height
        self.robust_scale = robust_scale

    def compute(
        self,
        verts_seq: torch.Tensor,
        samples_list: list[np.ndarray | None],
        project_fn,
    ) -> torch.Tensor:
        """Compute mask fill loss for a sequence.

        Args:
            verts_seq: [T, V, 3] vertices
            samples_list: list of [K_i, 2] underfill sample points (numpy, full ERP)
            project_fn: ERP projection function (Tensor, W, H) -> Tensor [N, 2]
        """
        device = verts_seq.device
        dtype = verts_seq.dtype
        losses = []

        for t, samples in enumerate(samples_list):
            if samples is None or len(samples) == 0:
                continue
            verts = verts_seq[t]  # [V, 3]
            verts_2d = project_fn(verts, self.W, self.H)  # [V, 2]

            samples_t = torch.from_numpy(np.asarray(samples, dtype=np.float32)).to(device=device, dtype=dtype)

            # Compute min distance from each sample point to any vertex
            # chunked cdist to avoid OOM for large V
            chunk_size = 4096
            n_v = verts_2d.shape[0]
            min_dists = torch.full((samples_t.shape[0],), float("inf"), device=device, dtype=dtype)
            for start in range(0, n_v, chunk_size):
                end = min(start + chunk_size, n_v)
                chunk = verts_2d[start:end]  # [chunk, 2]
                dists = torch.cdist(samples_t, chunk)  # [K, chunk]
                min_dists = torch.minimum(min_dists, dists.min(dim=1).values)

            robust = torch.sqrt((min_dists / self.robust_scale) ** 2 + 1e-8)
            losses.append(robust.mean())

        if not losses:
            return verts_seq.new_zeros(())
        return torch.stack(losses).mean()
