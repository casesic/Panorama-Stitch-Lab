"""Staged Levenberg–Marquardt fitting for dual-camera ERP SMPL-X.

The package contains the production implementation for:

- Stage 1 rigid alignment;
- Stage 2 pose warmup;
- Stage 2 joint pose/shape refinement;
- analytic Jacobians;
- arrowhead normal equations and Schur solving;
- CPU/CUDA device handling.

Validation and benchmark entry points live under ``src/script/validation`` and
``src/script/benchmark``.
"""

from __future__ import annotations

__all__: list[str] = []
