from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch


def prepare_silhouette_data(
    mask_path: Path,
    n_samples: int = 300,
    device: str = "cpu",
) -> torch.Tensor | None:
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return None

    valid_mask = (mask > 127).astype(np.uint8)
    h, w = valid_mask.shape[:2]
    kernel_size = max(3, int(min(h, w) * 0.02))
    if kernel_size % 2 == 0:
        kernel_size += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    eroded_mask = cv2.erode(valid_mask, kernel, iterations=1)

    contours, _ = cv2.findContours(eroded_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None

    min_area = float(h * w) * 0.05
    valid_contours = [cnt for cnt in contours if cv2.contourArea(cnt) >= min_area]
    if not valid_contours:
        return None

    boundary_pts = np.concatenate([cnt.reshape(-1, 2) for cnt in valid_contours], axis=0)
    if boundary_pts.shape[0] == 0:
        return None

    sample_count = min(int(n_samples), boundary_pts.shape[0])
    sample_idx = np.linspace(0, boundary_pts.shape[0] - 1, num=sample_count, dtype=np.int64)
    sampled_pts = boundary_pts[sample_idx]
    return torch.tensor(sampled_pts, dtype=torch.float32, device=device)


def compute_silhouette_loss(
    verts_world: torch.Tensor,
    boundary_pts: torch.Tensor,
    bbox: list[int],
    project_fn,
    erp_w: int,
    erp_h: int,
) -> torch.Tensor:
    proj = project_fn(verts_world, erp_w, erp_h)
    x1, y1, x2, y2 = bbox
    offset = proj.new_tensor([x1, y1])
    proj_local = proj - offset

    crop_w = x2 - x1
    crop_h = y2 - y1
    valid_mask = (
        (proj_local[:, 0] >= 0)
        & (proj_local[:, 0] <= crop_w)
        & (proj_local[:, 1] >= 0)
        & (proj_local[:, 1] <= crop_h)
    )
    proj_local_valid = proj_local[valid_mask]
    if proj_local_valid.shape[0] == 0:
        return verts_world.new_zeros(())

    boundary_pts = boundary_pts.to(proj_local_valid.device)
    dists = torch.cdist(boundary_pts, proj_local_valid)
    return dists.min(dim=1).values.mean()
