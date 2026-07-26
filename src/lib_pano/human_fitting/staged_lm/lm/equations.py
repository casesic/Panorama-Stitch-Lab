"""Jacobian evaluation, normal equations, and dense/structured solve dispatch.

This module owns tensor-level least-squares algebra only. It does not decide
trial acceptance, update damping, check convergence, or run the LM state
machine. Read :mod:`.trial`, then :mod:`.convergence`, then :mod:`.runner`.
"""

from __future__ import annotations

import torch
from torch.func import jacfwd

from ..linearization import Linearization
from ..types import LinearSolveResult, NormalEquations


# 1. Jacobian evaluation
def compute_jacobian(
    linearization: Linearization,
    x: torch.Tensor,
) -> torch.Tensor:
    """Use an optional analytic Jacobian, otherwise forward-mode AD."""
    if linearization.jacobian is not None:
        return linearization.jacobian(x).detach()
    return jacfwd(linearization.surrogate)(
        x.detach().clone().requires_grad_(True)
    ).detach()


# 2. Dense solve
def solve_damped_normal_equations(
    hessian: torch.Tensor,
    gradient: torch.Tensor,
    damping: float,
) -> LinearSolveResult:
    """Solve one dense damped system with Cholesky then ``linalg.solve``."""
    parameter_dimension = int(hessian.shape[0])
    identity = torch.eye(
        parameter_dimension, dtype=hessian.dtype, device=hessian.device,
    )
    matrix = hessian + damping * identity
    try:
        lower, info = torch.linalg.cholesky_ex(matrix)
        if info.item():
            raise RuntimeError
        delta = torch.cholesky_solve(-gradient.unsqueeze(1), lower).squeeze(1)
        return LinearSolveResult(delta, "cholesky")
    except Exception:
        try:
            delta = torch.linalg.solve(matrix, -gradient.unsqueeze(1)).squeeze(1)
            return LinearSolveResult(delta, "dense_solve")
        except Exception:
            return LinearSolveResult(None, "dense_solve")


def compute_scaled_gradient_from_normal_equations(
    hessian: torch.Tensor, gradient: torch.Tensor,
) -> float:
    """Scaled gradient ``||g|| / max(1, ||H||)``."""
    return float(gradient.norm() / max(1, hessian.norm()))


# 3. Normal-equation validation
def _validate_normal_equations(
    equations: NormalEquations,
    parameters: torch.Tensor,
) -> None:
    """Validate representation, device and dtype once after construction."""
    dense = equations.hessian is not None
    structured = equations.structured_system is not None
    if dense == structured:
        raise RuntimeError("INVALID_NORMAL_EQUATION_REPRESENTATION")
    if structured:
        flat = equations.structured_system.flat_gradient()
        if equations.gradient.shape != flat.shape or not torch.equal(equations.gradient, flat):
            raise RuntimeError("INVALID_NORMAL_EQUATION_REPRESENTATION")

    expected_device = parameters.device
    tensors = [("residual", equations.residual), ("gradient", equations.gradient)]
    if dense:
        tensors.append(("hessian", equations.hessian))
    else:
        system = equations.structured_system
        tensors.extend((
            ("frame_hessians", system.frame_hessians),
            ("frame_beta_cross", system.frame_beta_cross),
            ("beta_hessian", system.beta_hessian),
            ("frame_gradients", system.frame_gradients),
            ("beta_gradient", system.beta_gradient),
        ))
    for name, tensor in tensors:
        if tensor.device != expected_device:
            error = (
                "UNEXPECTED_CPU_FALLBACK_IN_CUDA_PIPELINE"
                if parameters.is_cuda else "DEVICE_CONTRACT_VIOLATION"
            )
            raise RuntimeError(
                f"{error}: {name} expected={expected_device} actual={tensor.device}"
            )

    double_tensors = [("gradient", equations.gradient)]
    if dense:
        double_tensors.append(("hessian", equations.hessian))
    else:
        double_tensors.extend(
            (f"structured.{name}", tensor) for name, tensor in tensors[2:]
        )
    for name, tensor in double_tensors:
        if tensor.dtype != torch.float64:
            raise RuntimeError(
                "DEVICE_CONTRACT_VIOLATION: "
                f"{name} expected_dtype=torch.float64 actual_dtype={tensor.dtype}"
            )


# 4. Dense and structured matrix properties
def normal_equations_max_diagonal(equations: NormalEquations) -> float:
    value = (torch.diagonal(equations.hessian).max() if equations.hessian is not None
             else equations.structured_system.max_diagonal())
    return float(value)


def normal_equations_hessian_norm(equations: NormalEquations) -> torch.Tensor:
    return (equations.hessian.norm() if equations.hessian is not None
            else equations.structured_system.hessian_frobenius_norm())


def normal_equations_quadratic_form(
    equations: NormalEquations, delta: torch.Tensor,
) -> torch.Tensor:
    if equations.hessian is not None:
        return delta @ equations.hessian @ delta
    return equations.structured_system.quadratic_form(delta)


def compute_scaled_gradient(equations: NormalEquations) -> float:
    return float(equations.gradient.norm() / max(1, normal_equations_hessian_norm(equations)))


# 5. Arrowhead solve dispatch
def _solve_structured_normal_equations(
    equations: NormalEquations, damping: float,
) -> LinearSolveResult:
    """Solve Arrowhead equations, materialising dense H only after failure."""
    system = equations.structured_system
    delta = system.solve_damped(damping)
    if delta is not None:
        return LinearSolveResult(delta, "schur")
    dense = system.to_dense()
    if system.timing_sink is not None:
        system.timing_sink.schur_dense_fallback_calls += 1
    dense_result = solve_damped_normal_equations(
        dense, equations.gradient, damping,
    )
    return LinearSolveResult(
        dense_result.delta, dense_result.backend, schur_failed=True,
    )


def solve_normal_equations(
    equations: NormalEquations, damping: float,
) -> LinearSolveResult:
    """Dispatch a dense or Arrowhead normal-equation solve."""
    if equations.hessian is not None:
        return solve_damped_normal_equations(
            equations.hessian, equations.gradient, damping,
        )
    return _solve_structured_normal_equations(equations, damping)


# 6. Linearization evaluation
def _evaluate_linearization(
    linearization: Linearization,
    x: torch.Tensor,
) -> tuple[NormalEquations, bool]:
    if linearization.normal_equations is not None:
        equations = linearization.normal_equations(x)
        if not isinstance(equations, NormalEquations):
            raise TypeError(f"normal_equations_fn returned {type(equations)}")
        _validate_normal_equations(equations, x)
        return equations, True
    residual = linearization.surrogate(x).detach()
    jacobian = compute_jacobian(linearization, x)
    jacobian_64 = jacobian.to(torch.float64)
    residual_64 = residual.to(torch.float64)
    equations = NormalEquations(
        residual=residual,
        hessian=jacobian_64.T @ jacobian_64,
        gradient=jacobian_64.T @ residual_64,
    )
    _validate_normal_equations(equations, x)
    return equations, False

