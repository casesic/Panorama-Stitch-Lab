"""Full-forward L-BFGS oracle used only by validation scripts."""

from __future__ import annotations

import time
from typing import Callable

import torch

from .types import LBFGSResult, LBFGS_LR, LBFGS_MAX_ITER


def run_lbfgs_oracle(
    label: str,
    true_objective_fn: Callable[[torch.Tensor], torch.Tensor],
    raw_metrics_fn: Callable[[torch.Tensor], dict[str, float]],
    initial_parameters: torch.Tensor,
    outer_steps: int,
) -> LBFGSResult:
    """Run L-BFGS on the full differentiable objective as a reference oracle."""
    x = initial_parameters.detach().clone().requires_grad_(True)
    optimizer = torch.optim.LBFGS(
        [x], line_search_fn="strong_wolfe", lr=LBFGS_LR,
        max_iter=LBFGS_MAX_ITER,
    )
    closure_calls = [0]
    if x.is_cuda:
        torch.cuda.synchronize(x.device)
    start = time.perf_counter()

    def closure():
        closure_calls[0] += 1
        optimizer.zero_grad()
        objective = true_objective_fn(x)
        if not isinstance(objective, torch.Tensor):
            raise TypeError(f"Type: {type(objective)}")
        if objective.ndim != 0:
            raise RuntimeError(f"Ndims: {objective.shape}")
        objective.backward()
        return objective

    for _ in range(outer_steps):
        optimizer.step(closure)

    final_parameters = x.detach()
    if final_parameters.is_cuda:
        torch.cuda.synchronize(final_parameters.device)
    wall_ms = (time.perf_counter() - start) * 1000.0
    objective_final = float(true_objective_fn(final_parameters).detach().item())
    print(
        f"[{label}] cls={closure_calls[0]} obj={objective_final:.6e} "
        f"rms={raw_metrics_fn(final_parameters)['rms']:.1f}"
    )
    return LBFGSResult(
        label=label,
        parameters=final_parameters,
        objective=objective_final,
        closure_calls=closure_calls[0],
        wall_time_ms=wall_ms,
    )
