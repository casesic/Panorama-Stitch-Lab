"""Full-forward L-BFGS oracle used only by validation scripts."""

from __future__ import annotations

import time
from typing import Callable

import torch

from lib_pano.human_fitting.staged_lm.config import SolverConfig
from lib_pano.human_fitting.staged_lm.types import LBFGSResult


def run_lbfgs_oracle(
    label: str,
    true_objective_fn: Callable[[torch.Tensor], torch.Tensor],
    raw_metrics_fn: Callable[[torch.Tensor], dict[str, float]],
    initial_parameters: torch.Tensor,
    outer_steps: int,
    solver_config: SolverConfig,
) -> LBFGSResult:
    """Run L-BFGS on the full differentiable objective as a reference oracle."""
    x = initial_parameters.detach().clone().requires_grad_(True)
    optimizer = torch.optim.LBFGS(
        [x], line_search_fn="strong_wolfe", lr=solver_config.lbfgs_lr,
        max_iter=solver_config.lbfgs_max_iter,
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
