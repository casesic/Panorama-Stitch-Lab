"""Arrowhead normal equations and damped Schur-complement solve.

The parameter order is ``[frame_0, ..., frame_T-1, shared_beta]``.  This
module is deliberately stage agnostic: it only knows the arrowhead block
layout and never imports SMPL-X or stage orchestration code.
"""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any

import torch


@dataclass
class ArrowheadNormalEquationSystem:
    """Block representation of a symmetric arrowhead Gauss-Newton system."""

    frame_hessians: torch.Tensor       # [T, P, P]
    frame_beta_cross: torch.Tensor     # [T, P, B]
    beta_hessian: torch.Tensor         # [B, B]
    frame_gradients: torch.Tensor      # [T, P]
    beta_gradient: torch.Tensor        # [B]
    timing_sink: Any | None = None
    to_dense_calls: int = 0
    schur_success_count: int = 0
    schur_failure_count: int = 0

    def __post_init__(self) -> None:
        tensors = (
            self.frame_hessians, self.frame_beta_cross, self.beta_hessian,
            self.frame_gradients, self.beta_gradient,
        )
        if any(t.dtype != torch.float64 for t in tensors):
            raise TypeError("ARROWHEAD_REQUIRES_FLOAT64")
        if any(t.device != tensors[0].device for t in tensors):
            raise RuntimeError("ARROWHEAD_DEVICE_MISMATCH")
        t, p, p2 = self.frame_hessians.shape
        b = self.beta_hessian.shape[0]
        if (p != p2 or self.frame_beta_cross.shape != (t, p, b)
                or self.beta_hessian.shape != (b, b)
                or self.frame_gradients.shape != (t, p)
                or self.beta_gradient.shape != (b,)):
            raise ValueError("INVALID_ARROWHEAD_SHAPES")
        if not all(torch.isfinite(x).all() for x in tensors):
            raise ValueError("NONFINITE_ARROWHEAD_SYSTEM")

    @property
    def frame_count(self) -> int:
        return int(self.frame_hessians.shape[0])

    @property
    def frame_dimension(self) -> int:
        return int(self.frame_hessians.shape[1])

    @property
    def beta_dimension(self) -> int:
        return int(self.beta_hessian.shape[0])

    @property
    def parameter_dimension(self) -> int:
        return self.frame_count * self.frame_dimension + self.beta_dimension

    def flat_gradient(self) -> torch.Tensor:
        return torch.cat((self.frame_gradients.reshape(-1), self.beta_gradient))

    def max_diagonal(self) -> torch.Tensor:
        frame_diagonal = torch.diagonal(self.frame_hessians, dim1=-2, dim2=-1)
        return torch.maximum(frame_diagonal.max(), torch.diagonal(self.beta_hessian).max())

    def hessian_frobenius_norm(self) -> torch.Tensor:
        squared = (
            self.frame_hessians.square().sum()
            + 2.0 * self.frame_beta_cross.square().sum()
            + self.beta_hessian.square().sum()
        )
        return torch.sqrt(squared)

    def split_delta(self, delta: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if delta.ndim != 1 or delta.numel() != self.parameter_dimension:
            raise ValueError("ARROWHEAD_DELTA_DIMENSION_MISMATCH")
        pose_size = self.frame_count * self.frame_dimension
        return delta[:pose_size].reshape(self.frame_count, self.frame_dimension), delta[pose_size:]

    def quadratic_form(self, delta: torch.Tensor) -> torch.Tensor:
        frame_delta, beta_delta = self.split_delta(delta)
        frame_term = torch.einsum("tp,tpq,tq->", frame_delta, self.frame_hessians, frame_delta)
        cross_term = 2.0 * torch.einsum(
            "tp,tpb,b->", frame_delta, self.frame_beta_cross, beta_delta,
        )
        beta_term = torch.einsum("b,bc,c->", beta_delta, self.beta_hessian, beta_delta)
        return frame_term + cross_term + beta_term

    def to_dense(self, *, count_call: bool = True) -> torch.Tensor:
        if count_call:
            self.to_dense_calls += 1
            if self.timing_sink is not None:
                self.timing_sink.structured_to_dense_calls += 1
        total = self.parameter_dimension
        dense = torch.zeros(total, total, dtype=torch.float64, device=self.frame_hessians.device)
        p = self.frame_dimension
        beta_start = self.frame_count * p
        for frame in range(self.frame_count):
            frame_slice = slice(frame * p, (frame + 1) * p)
            dense[frame_slice, frame_slice] = self.frame_hessians[frame]
            dense[frame_slice, beta_start:] = self.frame_beta_cross[frame]
            dense[beta_start:, frame_slice] = self.frame_beta_cross[frame].T
        dense[beta_start:, beta_start:] = self.beta_hessian
        return dense

    def solve_damped(self, damping: float) -> torch.Tensor | None:
        """Solve the uniformly damped system without materialising dense H."""
        sink = self.timing_sink
        do_sync = bool(sink is not None and sink.profile_timing and self.frame_hessians.is_cuda)
        if do_sync:
            torch.cuda.synchronize(self.frame_hessians.device)
        solve_start = time.perf_counter()
        try:
            p, b = self.frame_dimension, self.beta_dimension
            eye_p = torch.eye(p, dtype=torch.float64, device=self.frame_hessians.device)
            local_damped = self.frame_hessians + float(damping) * eye_p.unsqueeze(0)
            local_rhs = torch.cat(
                (self.frame_gradients.unsqueeze(-1), self.frame_beta_cross), dim=-1,
            )

            part_start = time.perf_counter()
            local_cholesky, local_info = torch.linalg.cholesky_ex(local_damped)
            if torch.any(local_info != 0):
                raise RuntimeError("SCHUR_LOCAL_CHOLESKY_FAILED")
            solved = torch.cholesky_solve(local_rhs, local_cholesky)
            xg, xb = solved[..., 0], solved[..., 1:]
            if do_sync:
                torch.cuda.synchronize(self.frame_hessians.device)
            if sink is not None:
                sink.schur_local_factor_ms += (time.perf_counter() - part_start) * 1000.0

            part_start = time.perf_counter()
            eye_b = torch.eye(b, dtype=torch.float64, device=self.frame_hessians.device)
            schur = self.beta_hessian + float(damping) * eye_b
            schur = schur - torch.einsum("tpb,tpc->bc", self.frame_beta_cross, xb)
            rhs_beta = -self.beta_gradient + torch.einsum(
                "tpb,tp->b", self.frame_beta_cross, xg,
            )
            if do_sync:
                torch.cuda.synchronize(self.frame_hessians.device)
            if sink is not None:
                sink.schur_reduced_build_ms += (time.perf_counter() - part_start) * 1000.0

            part_start = time.perf_counter()
            reduced_cholesky, reduced_info = torch.linalg.cholesky_ex(schur)
            if reduced_info.item() != 0:
                raise RuntimeError("SCHUR_REDUCED_CHOLESKY_FAILED")
            delta_beta = torch.cholesky_solve(
                rhs_beta.unsqueeze(1), reduced_cholesky,
            ).squeeze(1)
            if do_sync:
                torch.cuda.synchronize(self.frame_hessians.device)
            if sink is not None:
                sink.schur_reduced_solve_ms += (time.perf_counter() - part_start) * 1000.0

            part_start = time.perf_counter()
            delta_frames = -xg - torch.einsum("tpb,b->tp", xb, delta_beta)
            delta = torch.cat((delta_frames.reshape(-1), delta_beta))
            if not torch.isfinite(delta).all():
                raise RuntimeError("SCHUR_NONFINITE_DELTA")
            if do_sync:
                torch.cuda.synchronize(self.frame_hessians.device)
            if sink is not None:
                sink.schur_back_substitution_ms += (time.perf_counter() - part_start) * 1000.0
            self.schur_success_count += 1
            return delta
        except Exception:
            self.schur_failure_count += 1
            return None
        finally:
            if do_sync:
                torch.cuda.synchronize(self.frame_hessians.device)
            if sink is not None:
                sink.schur_solve_ms += (time.perf_counter() - solve_start) * 1000.0
                sink.schur_solve_calls += 1
