"""IRLS (Iteratively Reweighted Least Squares) linearisation and validation.

When ``--reproj_objective=charbonnier_irls`` is selected, the robust
Charbonnier loss is minimised through IRLS: at each linearisation point the
per-point weights are recomputed so that a weighted least-squares surrogate has
the same first-order gradient as the true robust objective.

This module provides:

* :func:`build_primary_aggregation_spec` — the per-(frame, camera) weighting
  spec that normalises each group by its valid-point count and frame count.
* :func:`compute_primary_charbonnier_tensor` — the true robust objective.
* :func:`compute_irls_point_weights` — per-point IRLS weights.
* :func:`make_irls_linearization` — builds the weighted surrogate residual.
* :func:`validate_first_order` — numerically checks that the surrogate's
  gradient matches the true objective's gradient.
"""

from __future__ import annotations

from typing import Any, Callable

import torch
from torch.func import jacfwd

from .residuals import flatten_reprojection_groups
from .types import ReprojectionGroup


def build_primary_aggregation_spec(groups: list[ReprojectionGroup]) -> dict[str, Any]:
    """Build the per-group aggregation spec used by the Charbonnier objective.

    Each (frame, camera) group receives ``alpha = 1 / (n_frames * n_valid)``
    so that the total objective is a normalised average over all frames and
    points of a single camera.
    """
    frames_by_camera: dict[str, set[int]] = {"cam2": set(), "cam4": set()}
    for group in groups:
        frames_by_camera[group.camera].add(group.frame_index)

    terms: dict[str, Any] = {}
    for group in groups:
        camera = group.camera
        n_frames = len(frames_by_camera[camera])
        n_valid = group.valid_point_count
        alpha = 1.0 / (n_frames * n_valid) if n_valid > 0 and n_frames > 0 else 0.0
        key = f"{group.frame_index}|{group.camera}"
        terms[key] = {
            "n_valid": n_valid,
            "valid_frame_count": n_frames,
            "camera_weight": 1.0,
            "alpha_per_point": alpha,
        }
    return {"terms": terms, "source": "sam3d_dense.py:272,135 + erp_dual.py:393"}


def build_primary_aggregation_spec_from_observations(
    frame_indices: list[int],
    cam2_vertex_ids_by_frame: list[torch.Tensor],
    cam4_vertex_ids_by_frame: list[torch.Tensor],
) -> dict[str, Any]:
    """Build the production aggregation spec without evaluating residuals.

    The group order and frame-count semantics exactly match
    :func:`build_primary_aggregation_spec` for the groups emitted by
    ``make_raw_reproj_groups``: every frame contributes cam2 followed by cam4,
    including zero-length groups.
    """
    frame_count = len(frame_indices)
    if not (
        len(cam2_vertex_ids_by_frame) == frame_count
        and len(cam4_vertex_ids_by_frame) == frame_count
    ):
        raise ValueError("OBSERVATION_AGGREGATION_SPEC_LENGTH_MISMATCH")
    valid_frame_count = len(set(frame_indices))
    terms: dict[str, Any] = {}
    for frame_index, cam2_ids, cam4_ids in zip(
        frame_indices, cam2_vertex_ids_by_frame, cam4_vertex_ids_by_frame,
    ):
        for camera, vertex_ids in (("cam2", cam2_ids), ("cam4", cam4_ids)):
            n_valid = int(vertex_ids.numel())
            alpha = (
                1.0 / (valid_frame_count * n_valid)
                if n_valid > 0 and valid_frame_count > 0 else 0.0
            )
            terms[f"{frame_index}|{camera}"] = {
                "n_valid": n_valid,
                "valid_frame_count": valid_frame_count,
                "camera_weight": 1.0,
                "alpha_per_point": alpha,
            }
    return {"terms": terms, "source": "sam3d_dense.py:272,135 + erp_dual.py:393"}


def compute_primary_charbonnier_tensor(
    groups: list[ReprojectionGroup],
    eps: float,
    spec: dict[str, Any],
) -> torch.Tensor:
    """Evaluate the robust Charbonnier objective from reprojection groups.

    ``rho(r) = sqrt(r^2 + eps^2) - eps`` is the smoothed absolute value; the
    per-group contribution is ``alpha * sum(rho(||r||))``.
    """
    total = torch.tensor(0.0, device=groups[0].residual_2d.device, dtype=groups[0].residual_2d.dtype)
    for group in groups:
        if group.valid_point_count == 0:
            continue
        key = f"{group.frame_index}|{group.camera}"
        term = spec["terms"][key]
        alpha = term["alpha_per_point"]
        if alpha <= 0:
            continue
        norms = torch.norm(group.residual_2d, dim=1)
        rho = torch.sqrt(norms ** 2 + eps ** 2) - eps
        total = total + alpha * torch.sum(rho)
    return total


def compute_irls_point_weights(
    groups: list[ReprojectionGroup],
    eps: float,
    spec: dict[str, Any],
) -> list[torch.Tensor]:
    """Compute IRLS weights ``w_i = alpha / sqrt(||r_i||^2 + eps^2)``.

    These weights turn the robust Charbonnier objective into a weighted
    least-squares surrogate whose gradient matches the true objective at the
    linearisation point.
    """
    weights: list[torch.Tensor] = []
    for group in groups:
        if group.valid_point_count == 0:
            weights.append(group.residual_2d.new_empty(0))
            continue
        key = f"{group.frame_index}|{group.camera}"
        term = spec["terms"][key]
        alpha = term["alpha_per_point"]
        if alpha <= 0:
            weights.append(group.residual_2d.new_empty(0))
            continue
        norms = torch.norm(group.residual_2d, dim=1)
        weights.append((alpha / torch.sqrt(norms ** 2 + eps ** 2)).detach())
    return weights


def make_irls_linearization(
    raw_groups_fn: Callable[[torch.Tensor], list[ReprojectionGroup]],
    anchor_residual_fn: Callable[[torch.Tensor], torch.Tensor] | None,
    linearisation_point: torch.Tensor,
    eps: float,
    spec: dict[str, Any],
) -> dict[str, Any]:
    """Build an IRLS weighted-least-squares surrogate at ``linearisation_point``.

    Returns a dict with:

    * ``surrogate_fn`` — callable returning the weighted residual vector
      (plus optional anchor residual) for any ``x``.
    * ``weight_stats`` — summary statistics of the IRLS weights.
    * ``spec`` — the aggregation spec, passed through for later reference.
    """
    groups_lin = raw_groups_fn(linearisation_point)
    weights = compute_irls_point_weights(groups_lin, eps, spec)
    flat_weights = torch.cat([w for w in weights if w.numel() > 0])
    weight_stats = {
        "min": float(flat_weights.min()) if flat_weights.numel() else 0,
        "mean": float(flat_weights.mean()) if flat_weights.numel() else 0,
        "median": float(flat_weights.median()) if flat_weights.numel() else 0,
        "max": float(flat_weights.max()) if flat_weights.numel() else 0,
        "n_points": int(flat_weights.numel()),
    }

    def surrogate_fn(x: torch.Tensor) -> torch.Tensor:
        groups = raw_groups_fn(x)
        parts: list[torch.Tensor] = []
        for index, group in enumerate(groups):
            if group.valid_point_count == 0 or index >= len(weights) or weights[index].numel() == 0:
                parts.append(group.residual_2d.reshape(-1))
            else:
                sqrt_w = torch.sqrt(weights[index].to(group.residual_2d.device))
                parts.append((group.residual_2d * sqrt_w.unsqueeze(1)).reshape(-1))
        if anchor_residual_fn is not None:
            parts.append(anchor_residual_fn(x))
        return torch.cat(parts)

    return {"surrogate_fn": surrogate_fn, "weight_stats": weight_stats, "spec": spec}


def validate_first_order(
    surrogate_fn: Callable[[torch.Tensor], torch.Tensor],
    true_objective_fn: Callable[[torch.Tensor], torch.Tensor],
    x: torch.Tensor,
    label: str,
    irls_gradient_rel_tol: float,
) -> dict[str, Any]:
    """Check that the IRLS surrogate gradient matches the true objective.

    A correct IRLS majoriser must satisfy ``grad(surrogate) == grad(true_obj)``
    at the linearisation point.  This function computes both gradients in
    float64 and reports the relative L2 error.
    """
    x_grad = x.detach().clone().requires_grad_(True)
    objective_true = true_objective_fn(x_grad)
    if not isinstance(objective_true, torch.Tensor):
        raise RuntimeError(f"Type: {type(objective_true)}")
    gradient_true = torch.autograd.grad(objective_true, x_grad, create_graph=False)[0].detach().to(torch.float64)

    jacobian = jacfwd(surrogate_fn)(x.detach().clone().requires_grad_(True)).detach().to(torch.float64)
    gradient_surrogate = jacobian.T @ surrogate_fn(x).detach().to(torch.float64)

    diff = gradient_true - gradient_surrogate
    abs_max = float(diff.abs().max())
    rel_l2 = float(diff.norm() / max(gradient_true.norm().item(), 1.0))
    ok = rel_l2 <= irls_gradient_rel_tol
    print(f"[IRLS {label}] abs_max={abs_max:.2e} rel_l2={rel_l2:.2e} {'PASS' if ok else 'FAIL'}")
    return {"abs_max": abs_max, "rel_l2": rel_l2, "pass": ok}
