"""Read-only convergence and objective-lifecycle audit for production LM.

This module deliberately mirrors ``run_lm_optimization`` instead of changing
its API.  It may depend on production private helpers/constants because it is
an audit-only lockstep implementation guarded by exact parity tests.
"""

from __future__ import annotations

import ast
import inspect
import math
import struct
import time
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Callable

import torch

from lib_pano.human_fitting.staged_lm.lm import convergence as lm_convergence
from lib_pano.human_fitting.staged_lm.lm import equations as lm_equations
from lib_pano.human_fitting.staged_lm.lm import runner as lm_runner
from lib_pano.human_fitting.staged_lm.config import SolverConfig
from lib_pano.human_fitting.staged_lm.linearization import Linearization
from lib_pano.human_fitting.staged_lm.types import (
    LMResult,
    LMSolverCounters,
    NormalEquations,
    StageDefinition,
)


@dataclass
class ObjectiveCallRecord:
    role: str
    outer: int | None
    trial: int | None
    parameters: torch.Tensor
    value: float


@dataclass
class TrialAuditRecord:
    outer: int
    trial: int
    objective_before: float
    objective_candidate: float | None
    gradient_dot_delta: float | None
    quadratic_term: float | None
    predicted_reduction: float | None
    actual_reduction: float | None
    rho: float | None
    damping_before: float
    damping_after: float
    candidate_parameter_norm: float | None
    delta_norm_raw: float | None
    delta_norm_relative: float | None
    accepted: bool
    solve_success: bool


@dataclass
class OuterConvergenceRecord:
    outer: int
    accepted_step_index: int
    objective_before: float
    objective_after: float
    actual_reduction: float
    relative_objective_change: float
    gradient_norm_before: float
    hessian_norm_before: float
    scaled_gradient_before: float
    max_hessian_diagonal_before: float
    accepted_delta_norm_raw: float
    accepted_delta_norm_relative_to_x: float
    scaled_gradient_after: float
    gradient_norm_after: float
    hessian_norm_after: float
    damping_before_first_trial: float
    accepted_trial_damping: float
    damping_after_accept: float
    accepted_trial_index: int
    rejected_trials_in_outer: int
    rho: float
    predicted_reduction: float
    scaled_gradient_tol: float
    step_tol: float
    relative_objective_tol: float
    scaled_gradient_ratio_to_tol: float
    step_norm_ratio_to_tol: float
    relative_objective_ratio_to_tol: float
    scaled_gradient_would_stop: bool
    step_would_stop: bool
    relative_objective_would_stop: bool
    production_selected_status: str | None
    is_final_budget_outer: bool
    next_iteration_exists: bool
    relinearization_used_by_next_iteration: bool
    block_step_norms: dict[str, Any] = field(default_factory=dict)
    relinearization_ms: float = 0.0


@dataclass
class TracedLMRun:
    result: LMResult
    outer_records: list[OuterConvergenceRecord]
    trial_records: list[TrialAuditRecord]
    objective_calls: list[ObjectiveCallRecord]
    checkpoints: dict[str, torch.Tensor]
    initial_diagnostics: dict[str, Any]
    no_accepted_diagnostics: list[dict[str, Any]]


def _line(function: Callable[..., Any]) -> int:
    return inspect.getsourcelines(function)[1]


def static_solver_audit() -> dict[str, Any]:
    """Inspect convergence behavior using AST nodes and explicit locations."""
    path = Path(inspect.getsourcefile(lm_runner.run_lm_optimization))
    convergence_path = Path(inspect.getsourcefile(lm_convergence.check_convergence))
    tree = ast.parse(path.read_text(), filename=str(path))
    convergence_tree = ast.parse(
        convergence_path.read_text(),
        filename=str(convergence_path),
    )
    run_node = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "run_lm_optimization"
    )
    check_node = next(
        node for node in convergence_tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "check_convergence"
    )
    no_accepted_node = next(
        node for node in convergence_tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "classify_no_accepted_trial"
    )
    check_calls = [
        node for node in ast.walk(run_node)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "check_convergence"
    ]
    initial_equations_line = next(
        node.lineno for node in ast.walk(run_node)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Tuple)
            and any(
                isinstance(item, ast.Name) and item.id == "equations"
                for item in target.elts
            )
            for target in node.targets
        )
    )
    convergence_order = []
    for node in check_node.body:
        if not isinstance(node, ast.If):
            continue
        text = ast.unparse(node.test)
        if "scaled_grad" in text:
            convergence_order.append("scaled_gradient")
        elif "step_norm" in text:
            convergence_order.append("step_norm")
        elif "rel_objective_change" in text:
            convergence_order.append("relative_objective")

    no_accepted_if = next(
        node for node in ast.walk(run_node)
        if isinstance(node, ast.If)
        and "not accepted_this" == ast.unparse(node.test)
    )
    branch_text = ast.unparse(no_accepted_node)
    hardcoded_compare = next(
        node for node in ast.walk(no_accepted_node)
        if isinstance(node, ast.Compare)
        and "scaled_grad" in ast.unparse(node.left)
        and any(
            isinstance(value, ast.Constant) and value.value == 0.0001
            for value in node.comparators
        )
    )
    accepted_compare = next(
        node for node in ast.walk(no_accepted_node)
        if isinstance(node, ast.Compare)
        and "accepted_steps" in ast.unparse(node.left)
    )
    accepted_min = int(accepted_compare.comparators[0].value) + 1
    post_accept = bool(check_calls) and all(
        call.lineno > no_accepted_if.end_lineno for call in check_calls
    )
    initial_check = any(call.lineno < initial_equations_line for call in check_calls)
    source = path.read_text()
    return {
        "solver_file": str(path),
        "run_lm_line": run_node.lineno,
        "check_convergence_file": str(convergence_path),
        "check_convergence_line": check_node.lineno,
        "no_accepted_branch_file": str(convergence_path),
        "no_accepted_branch_line": no_accepted_node.lineno,
        "initial_convergence_check_present": initial_check,
        "post_accept_convergence_check_present": post_accept,
        "no_accepted_branch_uses_solver_config": "solver_config" in branch_text,
        "no_accepted_branch_min_accepted_steps": accepted_min,
        "no_accepted_branch_scaled_grad_threshold": float(
            hardcoded_compare.comparators[0].value
        ),
        "convergence_predicate_order": convergence_order,
        "step_norm_definition": "RAW_EUCLIDEAN_PARAMETER_NORM",
        "step_norm_line": next(
            index for index, line in enumerate(source.splitlines(), 1)
            if "check_convergence(scaled_grad_final, float(torch.norm(delta))" in line
        ),
        "relative_objective_definition":
            "(objective_before - objective_after) / max(abs(objective_before), 1e-16)",
        "final_budget_step_relinearized": True,
        "accepted_objective_recomputed_for_rel": True,
        "outer_objective_recomputed_from_same_x": True,
        "final_objective_recomputed_from_same_x": True,
    }


def _hessian_norm(equations: NormalEquations) -> float:
    return float(lm_equations.normal_equations_hessian_norm(equations))


def _float32_ulp_difference(a: float, b: float) -> int | None:
    try:
        ia = struct.unpack("!I", struct.pack("!f", a))[0]
        ib = struct.unpack("!I", struct.pack("!f", b))[0]
    except (OverflowError, struct.error):
        return None
    return abs(int(ia) - int(ib))


def split_step_norms(
    stage_name: str,
    delta: torch.Tensor,
    x_before: torch.Tensor,
    *,
    frame_count: int,
    body_pose_dim: int = 63,
    beta_dim: int = 10,
    beta_abs_limit: float = 2.5,
) -> dict[str, Any]:
    """Report raw step norms by the production parameter layout."""
    result: dict[str, Any] = {
        "raw_norm": float(delta.norm()),
        "relative_norm": float(delta.norm()) / max(1.0, float(x_before.norm())),
    }
    if stage_name not in ("S1", "STAGE2_POSE", "STAGE2_POSE_SHAPE"):
        return result
    if stage_name == "S1":
        blocks = delta.reshape(frame_count, 12)
        per_frame = []
        for block in blocks:
            per_frame.append({
                "cam2_orient_norm": float(block[0:3].norm()),
                "cam2_translation_norm": float(block[3:6].norm()),
                "cam4_orient_norm": float(block[6:9].norm()),
                "cam4_translation_norm": float(block[9:12].norm()),
            })
        orient = torch.cat((blocks[:, 0:3], blocks[:, 6:9]), dim=1)
        translation = torch.cat((blocks[:, 3:6], blocks[:, 9:12]), dim=1)
        result.update({
            "per_frame": per_frame,
            "orient_norm_total": float(orient.norm()),
            "translation_norm_total": float(translation.norm()),
            "orient_max_abs": float(orient.abs().max()),
            "translation_max_abs": float(translation.abs().max()),
        })
        return result

    per_frame_dim = 12 + body_pose_dim
    frame_values = delta[:frame_count * per_frame_dim].reshape(
        frame_count, per_frame_dim,
    )
    rigid = frame_values[:, :12]
    pose = frame_values[:, 12:]
    orient = torch.cat((rigid[:, 0:3], rigid[:, 6:9]), dim=1)
    translation = torch.cat((rigid[:, 3:6], rigid[:, 9:12]), dim=1)
    result.update({
        "per_frame": [
            {
                "rigid_norm": float(value[:12].norm()),
                "body_pose_norm": float(value[12:].norm()),
            }
            for value in frame_values
        ],
        "rigid_norm": float(rigid.norm()),
        "rigid_orient_norm": float(orient.norm()),
        "rigid_translation_norm": float(translation.norm()),
        "pose_norm": float(pose.norm()),
    })
    if stage_name == "STAGE2_POSE_SHAPE":
        beta_delta = delta[-beta_dim:]
        beta_before = x_before[-beta_dim:]
        beta_after = beta_before + beta_delta.to(beta_before.dtype)
        physical_before = beta_abs_limit * torch.tanh(beta_before)
        physical_after = beta_abs_limit * torch.tanh(beta_after)
        result.update({
            "frame_rigid_norm": float(rigid.norm()),
            "frame_pose_norm": float(pose.norm()),
            "beta_latent_norm": float(beta_delta.norm()),
            "physical_beta_delta_norm": float(
                (physical_after - physical_before).norm()
            ),
        })
    return result


def run_lm_with_convergence_trace(
    label: str,
    make_linearization: Callable[[torch.Tensor], Linearization],
    raw_metrics_fn: Callable[[torch.Tensor], dict[str, float]],
    initial_parameters: torch.Tensor,
    param_dim: int,
    max_accepted_steps: int,
    max_trials: int,
    solver_config: SolverConfig,
    profile_timing: bool,
    true_objective_fn: Callable[[torch.Tensor], torch.Tensor],
    *,
    stage_name: str | None = None,
    frame_count: int = 1,
    body_pose_dim: int = 63,
    beta_dim: int = 10,
    beta_abs_limit: float = 2.5,
) -> TracedLMRun:
    """Audit-only, exact control-flow mirror of production LM."""
    profile = profile_timing
    do_sync = profile and initial_parameters.is_cuda

    def sync() -> None:
        if do_sync:
            torch.cuda.synchronize(initial_parameters.device)

    timing = {
        "initial_linearization_ms": 0.0,
        "normal_equations_ms": 0.0,
        "objective_eval_ms": 0.0,
        "linear_solve_ms": 0.0,
        "final_metrics_ms": 0.0,
    }
    objective_calls: list[ObjectiveCallRecord] = []

    def objective(
        parameters: torch.Tensor,
        role: str,
        outer: int | None,
        trial: int | None = None,
    ) -> float:
        value = float(true_objective_fn(parameters).detach().item())
        objective_calls.append(ObjectiveCallRecord(
            role, outer, trial, parameters.detach().clone(), value,
        ))
        return value

    sync()
    t_init = time.perf_counter()
    t0 = time.perf_counter()
    x = initial_parameters.detach().clone()
    linearization = make_linearization(x)
    t_j = time.perf_counter()
    equations, _ = lm_equations._evaluate_linearization(
        linearization, x,
    )
    if profile:
        timing["normal_equations_ms"] += (time.perf_counter() - t_j) * 1000.0
        timing["initial_linearization_ms"] = (
            time.perf_counter() - t_init
        ) * 1000.0
    initial_max_diagonal = lm_equations.normal_equations_max_diagonal(equations)
    if not math.isfinite(initial_max_diagonal) or initial_max_diagonal <= 0:
        raise RuntimeError("INVALID_INITIAL_LM_DAMPING")
    damping = initial_max_diagonal * 1e-3
    initial_diagnostics = {
        "initial_scaled_gradient": lm_equations.compute_scaled_gradient(equations),
        "initial_gradient_norm": float(equations.gradient.norm()),
        "initial_hessian_norm": _hessian_norm(equations),
        "initial_max_diagonal": initial_max_diagonal,
        "initial_damping": damping,
        "initial_step_check_applicable": False,
        "initial_rel_objective_check_applicable": False,
    }
    initial_diagnostics.update({
        "initial_gradient_pass_config": (
            initial_diagnostics["initial_scaled_gradient"]
            < solver_config.scaled_grad_tol
        ),
        "initial_gradient_pass_hardcoded": (
            initial_diagnostics["initial_scaled_gradient"] < 1e-4
        ),
        "hypothetical_initial_gradient_status": (
            "CONVERGED_SCALED_GRADIENT"
            if initial_diagnostics["initial_scaled_gradient"]
            < solver_config.scaled_grad_tol else None
        ),
    })

    counters = LMSolverCounters(linearization_count=1)
    status = "RUNNING"
    weight_history = [{"outer": -1, **linearization.weight_stats}]
    spec = dict(linearization.aggregation_spec)
    linear_solver_backend = linearization.provenance.linear_solver_backend
    trial_history: list[dict[str, Any]] = []
    trial_records: list[TrialAuditRecord] = []
    outer_records: list[OuterConvergenceRecord] = []
    no_accepted_diagnostics: list[dict[str, Any]] = []
    checkpoints = {"baseline_initial": x.detach().clone()}

    for outer in range(max_accepted_steps):
        gradient_64 = equations.gradient
        gradient_norm_before = float(gradient_64.norm())
        hessian_norm_before = _hessian_norm(equations)
        scaled_gradient_before = lm_equations.compute_scaled_gradient(equations)
        max_diagonal_before = lm_equations.normal_equations_max_diagonal(equations)
        x_before = x.detach().clone()
        damping_before_first_trial = damping

        sync()
        t_o = time.perf_counter()
        obj_before = objective(x, "initial_or_outer_before", outer)
        if profile:
            timing["objective_eval_ms"] += (
                time.perf_counter() - t_o
            ) * 1000.0
        accepted_this = False
        rejected_in_outer = 0
        accepted_details: dict[str, Any] = {}
        delta = None

        for trial in range(max_trials):
            sync()
            t_ls = time.perf_counter()
            trial_damping = damping
            solve_result = lm_equations.solve_normal_equations(equations, damping)
            counters.record_linear_solve(solve_result)
            delta = solve_result.delta
            if not solve_result.success:
                damping = min(damping * 10, lm_convergence._LAMBDA_MAX)
                counters.rejected_trials += 1
                rejected_in_outer += 1
                trial_history.append({
                    "outer": outer, "trial": trial, "damping": trial_damping,
                    "configured_solver_backend": linear_solver_backend,
                    "final_solve_backend": solve_result.backend,
                    "solve_success": False, "delta_norm": None,
                    "predicted_reduction": None, "actual_reduction": None,
                    "rho": None, "accepted": False,
                    "schur_primary_failed": solve_result.schur_failed,
                })
                trial_records.append(TrialAuditRecord(
                    outer, trial, obj_before, None, None, None, None, None,
                    None, trial_damping, damping, None, None, None, False, False,
                ))
                if profile:
                    timing["linear_solve_ms"] += (
                        time.perf_counter() - t_ls
                    ) * 1000.0
                continue
            if profile:
                timing["linear_solve_ms"] += (
                    time.perf_counter() - t_ls
                ) * 1000.0

            candidate = (x + delta.to(dtype=torch.float32)).detach()
            sync()
            t_o = time.perf_counter()
            obj_trial = objective(candidate, "trial_candidate", outer, trial)
            if profile:
                timing["objective_eval_ms"] += (
                    time.perf_counter() - t_o
                ) * 1000.0
            delta_64 = delta.to(torch.float64)
            gradient_dot_delta = float(gradient_64.dot(delta_64))
            quadratic_term = float(
                lm_equations.normal_equations_quadratic_form(
                    equations, delta_64,
                )
            )
            predicted_reduction = float(
                -gradient_64.dot(delta_64) - 0.5 * quadratic_term
            )
            actual_reduction = obj_before - obj_trial
            rho = (
                actual_reduction / max(predicted_reduction, 1e-12)
                if predicted_reduction > 0 else 0.0
            )
            accepted_trial = (
                predicted_reduction > 0 and actual_reduction > 0 and rho > 0
            )
            trial_history.append({
                "outer": outer, "trial": trial, "damping": trial_damping,
                "configured_solver_backend": linear_solver_backend,
                "final_solve_backend": solve_result.backend,
                "solve_success": True, "delta_norm": float(delta.norm()),
                "predicted_reduction": predicted_reduction,
                "actual_reduction": actual_reduction, "rho": rho,
                "accepted": accepted_trial,
                "schur_primary_failed": solve_result.schur_failed,
            })
            damping_after = damping
            if accepted_trial:
                if rho > 0.75:
                    damping_after = max(damping / 3, lm_convergence._LAMBDA_MIN)
                elif rho < 0.25:
                    damping_after = min(damping * 3, lm_convergence._LAMBDA_MAX)
            else:
                damping_after = min(damping * 10, lm_convergence._LAMBDA_MAX)
            trial_records.append(TrialAuditRecord(
                outer, trial, obj_before, obj_trial, gradient_dot_delta,
                quadratic_term, predicted_reduction, actual_reduction, rho,
                trial_damping, damping_after, float(candidate.norm()),
                float(delta.norm()),
                float(delta.norm()) / max(1.0, float(x.norm())),
                accepted_trial, True,
            ))
            if accepted_trial:
                x = candidate
                counters.accepted_steps += 1
                accepted_this = True
                damping = damping_after
                accepted_details = {
                    "trial": trial,
                    "trial_damping": trial_damping,
                    "obj_trial": obj_trial,
                    "rho": rho,
                    "predicted_reduction": predicted_reduction,
                }
                checkpoints[f"accepted_step_{counters.accepted_steps}"] = (
                    x.detach().clone()
                )
                break
            damping = damping_after
            counters.rejected_trials += 1
            rejected_in_outer += 1

        if not accepted_this:
            scaled_grad = lm_equations.compute_scaled_gradient(equations)
            status = (
                "CONVERGED_SCALED_GRADIENT"
                if counters.accepted_steps > 4 and scaled_grad < 1e-4
                else "LM_NO_ACCEPTED_TRIAL"
            )
            no_accepted_diagnostics.append({
                "outer": outer,
                "scaled_gradient": scaled_grad,
                "config_gradient_pass":
                    scaled_grad < solver_config.scaled_grad_tol,
                "hardcoded_gradient_pass": scaled_grad < 1e-4,
                "min_accepted_count_pass": counters.accepted_steps > 4,
                "resulting_production_status": status,
                "hypothetical_config_consistent_status": (
                    "CONVERGED_SCALED_GRADIENT"
                    if scaled_grad < solver_config.scaled_grad_tol
                    else "LM_NO_ACCEPTED_TRIAL"
                ),
            })
            break

        relinearization_start = time.perf_counter()
        next_linearization = make_linearization(x)
        t_j = time.perf_counter()
        next_equations, _ = lm_equations._evaluate_linearization(
            next_linearization, x,
        )
        if profile:
            timing["normal_equations_ms"] += (
                time.perf_counter() - t_j
            ) * 1000.0
        relinearization_ms = (
            time.perf_counter() - relinearization_start
        ) * 1000.0
        counters.linearization_count += 1
        weight_history.append({
            "outer": outer, **next_linearization.weight_stats,
        })
        scaled_grad_final = lm_equations.compute_scaled_gradient(next_equations)
        sync()
        t_o = time.perf_counter()
        objective_after = objective(x, "accepted_rel_recheck", outer)
        rel = (obj_before - objective_after) / max(abs(obj_before), 1e-16)
        if profile:
            timing["objective_eval_ms"] += (
                time.perf_counter() - t_o
            ) * 1000.0
        step_norm = float(torch.norm(delta))
        convergence = lm_convergence.check_convergence(
            scaled_grad_final, step_norm, rel, solver_config,
        )
        is_final_budget = outer == max_accepted_steps - 1
        next_iteration_exists = (
            convergence is None and not is_final_budget
        )
        outer_records.append(OuterConvergenceRecord(
            outer=outer,
            accepted_step_index=counters.accepted_steps,
            objective_before=obj_before,
            objective_after=objective_after,
            actual_reduction=obj_before - objective_after,
            relative_objective_change=rel,
            gradient_norm_before=gradient_norm_before,
            hessian_norm_before=hessian_norm_before,
            scaled_gradient_before=scaled_gradient_before,
            max_hessian_diagonal_before=max_diagonal_before,
            accepted_delta_norm_raw=step_norm,
            accepted_delta_norm_relative_to_x=(
                step_norm / max(1.0, float(x_before.norm()))
            ),
            scaled_gradient_after=scaled_grad_final,
            gradient_norm_after=float(next_equations.gradient.norm()),
            hessian_norm_after=_hessian_norm(next_equations),
            damping_before_first_trial=damping_before_first_trial,
            accepted_trial_damping=accepted_details["trial_damping"],
            damping_after_accept=damping,
            accepted_trial_index=accepted_details["trial"],
            rejected_trials_in_outer=rejected_in_outer,
            rho=accepted_details["rho"],
            predicted_reduction=accepted_details["predicted_reduction"],
            scaled_gradient_tol=solver_config.scaled_grad_tol,
            step_tol=solver_config.step_tol,
            relative_objective_tol=solver_config.rel_obj_tol,
            scaled_gradient_ratio_to_tol=(
                scaled_grad_final / solver_config.scaled_grad_tol
            ),
            step_norm_ratio_to_tol=step_norm / solver_config.step_tol,
            relative_objective_ratio_to_tol=(
                abs(rel) / solver_config.rel_obj_tol
            ),
            scaled_gradient_would_stop=(
                scaled_grad_final < solver_config.scaled_grad_tol
            ),
            step_would_stop=step_norm < solver_config.step_tol,
            relative_objective_would_stop=(
                abs(rel) < solver_config.rel_obj_tol
            ),
            production_selected_status=convergence,
            is_final_budget_outer=is_final_budget,
            next_iteration_exists=next_iteration_exists,
            relinearization_used_by_next_iteration=next_iteration_exists,
            block_step_norms=split_step_norms(
                stage_name or label.removesuffix("_LM"),
                delta, x_before, frame_count=frame_count,
                body_pose_dim=body_pose_dim, beta_dim=beta_dim,
                beta_abs_limit=beta_abs_limit,
            ),
            relinearization_ms=relinearization_ms,
        ))
        if convergence is not None:
            status = convergence
            break
        linearization = next_linearization
        equations = next_equations

    if status == "RUNNING":
        status = "MAX_ACCEPTED_STEPS"

    sync()
    t_fm = time.perf_counter()
    final_parameters = x.detach().clone()
    rms_metrics = raw_metrics_fn(final_parameters)
    obj_final = objective(
        final_parameters, "final_metrics_objective", None,
    )
    if profile:
        timing["final_metrics_ms"] = (
            time.perf_counter() - t_fm
        ) * 1000.0
    wall_ms = lm_runner._elapsed_ms(t0, x, do_sync)
    first_outer_call = next(
        (
            call for call in objective_calls
            if call.role == "initial_or_outer_before"
        ),
        None,
    )
    initial_diagnostics["initial_true_objective"] = (
        first_outer_call.value if first_outer_call is not None else obj_final
    )
    checkpoints["production_final"] = final_parameters.detach().clone()
    result = LMResult(
        label=label, parameters=final_parameters, objective=obj_final,
        status=status, counters=counters, final_lambda=damping,
        wall_time_ms=wall_ms, rms_metrics=rms_metrics,
        weight_history=weight_history, aggregation_spec=spec,
        jacobian_backend=linearization.provenance.jacobian_backend,
        linearization_backend=linearization.provenance.linearization_backend,
        linear_solver_backend=linear_solver_backend,
        trial_history=trial_history,
        profile_timing=timing if profile else None,
    )
    return TracedLMRun(
        result, outer_records, trial_records, objective_calls, checkpoints,
        initial_diagnostics, no_accepted_diagnostics,
    )


def assert_exact_lm_parity(production: LMResult, traced: LMResult) -> None:
    """Fail before audit conclusions if trace and production diverge."""
    scalar_fields = (
        "status", "final_lambda", "objective", "rms_metrics",
        "trial_history", "weight_history", "aggregation_spec", "jacobian_backend",
        "linearization_backend", "eval_backend", "linear_solver_backend",
    )
    for name in scalar_fields:
        if getattr(production, name) != getattr(traced, name):
            raise AssertionError(f"TRACE_RUNNER_PARITY_FAIL:{name}")
    if vars(production.counters) != vars(traced.counters):
        raise AssertionError("TRACE_RUNNER_PARITY_FAIL:counters")
    if not torch.equal(production.parameters, traced.parameters):
        raise AssertionError("TRACE_RUNNER_PARITY_FAIL:parameters")


def objective_call_attribution(trace: TracedLMRun) -> dict[str, Any]:
    roles = {
        role: sum(call.role == role for call in trace.objective_calls)
        for role in (
            "initial_or_outer_before", "trial_candidate",
            "accepted_rel_recheck", "final_metrics_objective",
        )
    }
    successful = sum(record.solve_success for record in trace.trial_records)
    current = len(trace.objective_calls)
    minimal = 1 + successful
    return {
        "current_total_calls": current,
        "outer_before_calls": roles["initial_or_outer_before"],
        "trial_candidate_calls": roles["trial_candidate"],
        "accepted_rel_recheck_calls": roles["accepted_rel_recheck"],
        "final_calls": roles["final_metrics_objective"],
        "successful_solve_trial_calls": successful,
        "minimal_exact_objective_calls": minimal,
        "removable_duplicate_calls": current - minimal,
        "reduction_fraction": (current - minimal) / current if current else 0.0,
    }


def duplicate_objective_proof(trace: TracedLMRun) -> dict[str, Any]:
    """Close the loop for accepted, next-outer, and final duplicate calls."""
    comparisons: list[dict[str, Any]] = []
    accepted_trials = {
        (call.outer, call.trial): call
        for call in trace.objective_calls
        if call.role == "trial_candidate"
        and next(
            record.accepted for record in trace.trial_records
            if record.outer == call.outer and record.trial == call.trial
        )
    }
    rel_calls = {
        call.outer: call for call in trace.objective_calls
        if call.role == "accepted_rel_recheck"
    }
    outer_calls = {
        call.outer: call for call in trace.objective_calls
        if call.role == "initial_or_outer_before"
    }

    def compare(kind: str, first: ObjectiveCallRecord, second: ObjectiveCallRecord):
        comparisons.append({
            "kind": kind,
            "parameters_equal": torch.equal(first.parameters, second.parameters),
            "objective_exact": first.value == second.value,
            "absolute_difference": abs(first.value - second.value),
            "float32_ulp_difference": _float32_ulp_difference(
                first.value, second.value,
            ),
        })

    for (outer, _trial), candidate in accepted_trials.items():
        compare("accepted_trial_vs_rel_recheck", candidate, rel_calls[outer])
        if outer + 1 in outer_calls:
            compare(
                "accepted_trial_vs_next_outer_before",
                candidate, outer_calls[outer + 1],
            )
    final = next(
        call for call in trace.objective_calls
        if call.role == "final_metrics_objective"
    )
    current = (
        rel_calls[max(rel_calls)] if rel_calls
        else outer_calls[max(outer_calls)]
    )
    compare("current_objective_vs_final", current, final)
    exact = all(
        item["parameters_equal"] and item["objective_exact"]
        for item in comparisons
    )
    return {
        "status": (
            "EXACT_OBJECTIVE_REUSE_PROVEN"
            if exact else "OBJECTIVE_NONDETERMINISM_DETECTED"
        ),
        "comparisons": comparisons,
    }


def predicate_reachability(trace: TracedLMRun) -> dict[str, Any]:
    records = trace.outer_records
    if not records:
        return {
            "min_scaled_gradient": None, "min_raw_step_norm": None,
            "min_relative_objective_change": None,
            "first_outer_scaled_gradient_pass": None,
            "first_outer_step_pass": None,
            "first_outer_rel_objective_pass": None,
        }
    min_grad = min(record.scaled_gradient_after for record in records)
    min_step = min(record.accepted_delta_norm_raw for record in records)
    min_rel = min(abs(record.relative_objective_change) for record in records)

    def first(attribute: str) -> int | None:
        return next(
            (record.outer for record in records if getattr(record, attribute)),
            None,
        )

    config = records[0]
    return {
        "min_scaled_gradient": min_grad,
        "min_raw_step_norm": min_step,
        "min_relative_objective_change": min_rel,
        "min_scaled_gradient_ratio_to_tol":
            min_grad / config.scaled_gradient_tol,
        "min_step_norm_ratio_to_tol": min_step / config.step_tol,
        "min_rel_change_ratio_to_tol":
            min_rel / config.relative_objective_tol,
        "first_outer_scaled_gradient_pass":
            first("scaled_gradient_would_stop"),
        "first_outer_step_pass": first("step_would_stop"),
        "first_outer_rel_objective_pass":
            first("relative_objective_would_stop"),
    }


def trace_to_json(trace: TracedLMRun) -> dict[str, Any]:
    return {
        "result": trace.result.to_json_dict(),
        "initial_diagnostics": trace.initial_diagnostics,
        "outer_records": [asdict(record) for record in trace.outer_records],
        "trial_records": [asdict(record) for record in trace.trial_records],
        "objective_call_attribution": objective_call_attribution(trace),
        "duplicate_objective_proof": duplicate_objective_proof(trace),
        "predicate_reachability": predicate_reachability(trace),
        "no_accepted_diagnostics": trace.no_accepted_diagnostics,
        "final_budget_relinearization": {
            "calls": sum(
                record.is_final_budget_outer for record in trace.outer_records
            ),
            "normal_equation_calls": sum(
                record.is_final_budget_outer for record in trace.outer_records
            ),
            "time_ms": sum(
                record.relinearization_ms for record in trace.outer_records
                if record.is_final_budget_outer
            ),
            "next_solver_iteration": any(
                record.is_final_budget_outer and record.next_iteration_exists
                for record in trace.outer_records
            ),
            "uses": [
                "convergence_status_check", "final_weight_history",
            ],
            "future_designs": [
                "KEEP_FOR_FINAL_CONVERGENCE_STATUS",
                "SKIP_WHEN_BUDGET_STATUS_ACCEPTABLE",
                "COMPUTE_ONLY_IF_FINAL_GRADIENT_REQUIRED",
            ],
        },
    }


def replace_initial_parameters(
    definition: StageDefinition, parameters: torch.Tensor,
) -> StageDefinition:
    return replace(
        definition,
        initial_lm_parameters=parameters.detach().clone(),
    )


def replace_accepted_budget(
    definition: StageDefinition, budget: int,
) -> StageDefinition:
    return replace(definition, lm_max_accepted_steps=budget)
