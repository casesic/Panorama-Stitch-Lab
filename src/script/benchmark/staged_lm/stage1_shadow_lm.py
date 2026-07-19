"""Serial-driven and independent Stage-1 FULL_BATCH LM shadows."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch

from lib_pano.human_fitting.staged_lm.lm_solver import (
    check_convergence, compute_scaled_gradient, normal_equations_max_diagonal,
    normal_equations_quadratic_form, run_lm_optimization, solve_normal_equations,
)
from lib_pano.human_fitting.staged_lm.types import LMResult, StageDefinition
from script.benchmark.staged_lm.stage1_batch_candidates import Stage1BatchCache
from script.benchmark.staged_lm.stage1_shadow_common import (
    Stage1NEMode, build_stage1_shadow_linearization_common, equation_errors,
    evaluate_stage1_equations,
)


@dataclass
class Stage1ShadowTrialRecord:
    outer: int; trial: int; damping: float
    serial_solve_success: bool; candidate_solve_success: bool
    serial_cholesky_success: bool; candidate_cholesky_success: bool
    serial_dense_fallback: bool; candidate_dense_fallback: bool
    delta_max_abs: float | None; delta_relative: float | None
    predicted_serial: float | None; predicted_candidate: float | None
    predicted_relative: float | None
    actual_serial: float | None; actual_candidate: float | None
    rho_serial: float | None; rho_candidate: float | None; rho_abs_diff: float | None
    accepted_serial: bool; accepted_candidate: bool; acceptance_match: bool
    residual_relative: float; jacobian_relative: float
    H_relative: float; g_relative: float
    serial_equation_ms: float; candidate_equation_ms: float


@dataclass
class Stage1LockstepResult:
    status: str; final_parameters: torch.Tensor; final_objective: float
    trial_records: list[Stage1ShadowTrialRecord]
    acceptance_mismatch_count: int
    first_acceptance_mismatch: dict[str, Any] | None
    solve_path_mismatch_count: int
    first_solve_path_mismatch: dict[str, Any] | None
    accepted_steps: int; rejected_trials: int; final_lambda: float
    def to_json_dict(self):
        return {"status":self.status,"final_objective":self.final_objective,
                "acceptance_mismatch_count":self.acceptance_mismatch_count,
                "first_acceptance_mismatch":self.first_acceptance_mismatch,
                "solve_path_mismatch_count":self.solve_path_mismatch_count,
                "first_solve_path_mismatch":self.first_solve_path_mismatch,
                "accepted_steps":self.accepted_steps,"rejected_trials":self.rejected_trials,
                "final_lambda":self.final_lambda,
                "trial_records":[asdict(x) for x in self.trial_records]}


def make_stage1_batch_linearization(stage, batch, context, point, mode):
    common = build_stage1_shadow_linearization_common(
        stage.analytic_cache, batch, point, is_irls=context.is_irls,
        charbonnier_eps=context.charbonnier_eps,
        aggregation_spec=stage.aggregation_spec, rep_scale=context.scaling.rep_scale,
    )
    return {"surrogate_fn":common.surrogate_fn,"jacobian_fn":common.jacobian_fn,
            "normal_equations_fn":lambda x:evaluate_stage1_equations(
                mode,stage.analytic_cache,batch,x,common.padded_row_scales,
                common.flat_row_scales).equations,
            "weight_stats":common.weight_stats,"spec":common.spec,
            "jacobian_backend":"analytic_stage1_batched_shadow",
            "linearization_backend":"direct_normal_equations_batched_shadow",
            "linear_solver_backend":"dense"}


def run_stage1_lm_independent_batch(
    stage: StageDefinition, batch: Stage1BatchCache, context: Any, *,
    mode: Stage1NEMode = Stage1NEMode.FULL_BATCH_GROUP_GEMM,
    initial_parameters: torch.Tensor | None = None, profile_timing: bool = True,
) -> LMResult:
    initial = stage.initial_lm_parameters if initial_parameters is None else initial_parameters
    return run_lm_optimization(
        "STAGE1_BATCH_SHADOW",lambda x:make_stage1_batch_linearization(stage,batch,context,x,mode),
        stage.raw_metrics_fn,initial,stage.parameter_dimension,stage.lm_max_accepted_steps,
        context.solver_config.lm_max_trials,context.solver_config,profile_timing,0,
        stage.lm_true_objective_fn,validate_irls=False,
    )


def run_stage1_serial_reference(
    stage: StageDefinition, context: Any, initial_parameters: torch.Tensor,
    *, profile_timing: bool = True,
) -> LMResult:
    """Run the unmodified production analytic Stage-1 linearisation."""
    initial = initial_parameters.detach().clone()
    return run_lm_optimization(
        "STAGE1_SERIAL_REFERENCE",
        lambda x: stage.linearization_builder(x, stage.aggregation_spec),
        stage.raw_metrics_fn, initial, stage.parameter_dimension,
        stage.lm_max_accepted_steps, context.solver_config.lm_max_trials,
        context.solver_config, profile_timing, 0, stage.lm_true_objective_fn,
        validate_irls=False,
    )


def run_stage1_full_batch_candidate(
    stage: StageDefinition, batch: Stage1BatchCache, context: Any,
    initial_parameters: torch.Tensor, *, profile_timing: bool = True,
) -> LMResult:
    """Run the shadow whose only replacement is FULL_BATCH normal equations."""
    return run_stage1_lm_independent_batch(
        stage, batch, context, mode=Stage1NEMode.FULL_BATCH_GROUP_GEMM,
        initial_parameters=initial_parameters.detach().clone(),
        profile_timing=profile_timing,
    )


def run_stage1_lm_lockstep(
    stage: StageDefinition, batch: Stage1BatchCache, context: Any, *,
    candidate_mode: Stage1NEMode, initial_parameters: torch.Tensor | None = None,
) -> Stage1LockstepResult:
    x = (stage.initial_lm_parameters if initial_parameters is None else initial_parameters).detach().clone()
    records=[]; damping=None; accepted=rejected=0; status="RUNNING"
    for outer in range(stage.lm_max_accepted_steps):
        common=build_stage1_shadow_linearization_common(
            stage.analytic_cache,batch,x,is_irls=context.is_irls,
            charbonnier_eps=context.charbonnier_eps,aggregation_spec=stage.aggregation_spec,
            rep_scale=context.scaling.rep_scale)
        serial=evaluate_stage1_equations(Stage1NEMode.SERIAL_REFERENCE,stage.analytic_cache,batch,x,
                                         common.padded_row_scales,common.flat_row_scales)
        candidate=evaluate_stage1_equations(candidate_mode,stage.analytic_cache,batch,x,
                                            common.padded_row_scales,common.flat_row_scales)
        errors=equation_errors(candidate,serial)
        if damping is None:damping=normal_equations_max_diagonal(serial.equations)*1e-3
        before=float(stage.lm_true_objective_fn(x).detach()); accepted_this=False
        for trial in range(context.solver_config.lm_max_trials):
            trial_damping=damping
            sd,sc,sf,_,_,sm=solve_normal_equations(serial.equations,damping,stage.parameter_dimension)
            cd,cc,cf,_,_,cm=solve_normal_equations(candidate.equations,damping,stage.parameter_dimension)
            if sd is None:
                damping=min(damping*10,1e12);rejected+=1;continue
            sx=(x+sd.to(torch.float32)).detach(); so=float(stage.lm_true_objective_fn(sx).detach())
            sp=float(-serial.equations.gradient.dot(sd)-.5*normal_equations_quadratic_form(serial.equations,sd))
            sa=before-so;sr=sa/max(sp,1e-12) if sp>0 else 0.;sacc=sp>0 and sa>0 and sr>0
            if cd is None: cp=ca=cr=None;cacc=False
            else:
                cx=(x+cd.to(torch.float32)).detach();co=float(stage.lm_true_objective_fn(cx).detach())
                cp=float(-candidate.equations.gradient.dot(cd)-.5*normal_equations_quadratic_form(candidate.equations,cd))
                ca=before-co;cr=ca/max(cp,1e-12) if cp>0 else 0.;cacc=cp>0 and ca>0 and cr>0
            dd=None if cd is None else cd-sd
            records.append(Stage1ShadowTrialRecord(
                outer,trial,trial_damping,sd is not None,cd is not None,sc,cc,bool(sf or sm),bool(cf or cm),
                None if dd is None else float(dd.abs().max()),
                None if dd is None else float(dd.norm())/max(float(sd.norm()),1e-30),
                sp,cp,None if cp is None else abs(cp-sp)/max(abs(sp),1e-30),sa,ca,sr,cr,
                None if cr is None else abs(cr-sr),sacc,cacc,sacc==cacc,
                errors["scaled_residual"]["relative_frobenius"],0.0,
                errors["H"]["relative_frobenius"],errors["g"]["relative_frobenius"],
                serial.equation_ms,candidate.equation_ms))
            if sacc:
                x=sx;accepted+=1;accepted_this=True
                if sr>.75:damping=max(damping/3,1e-10)
                elif sr<.25:damping=min(damping*3,1e12)
                break
            damping=min(damping*10,1e12);rejected+=1
        if not accepted_this:status="LM_NO_ACCEPTED_TRIAL";break
        nxt=make_stage1_batch_linearization(stage,batch,context,x,Stage1NEMode.SERIAL_REFERENCE)
        eq=nxt["normal_equations_fn"](x)
        rel=(before-float(stage.lm_true_objective_fn(x).detach()))/max(abs(before),1e-16)
        convergence=check_convergence(compute_scaled_gradient(eq),float(sd.norm()),rel,context.solver_config)
        if convergence is not None:status=convergence;break
    if status=="RUNNING":status="MAX_ACCEPTED_STEPS"
    am=[r for r in records if not r.acceptance_match]
    sm=[r for r in records if (r.serial_solve_success,r.serial_cholesky_success,r.serial_dense_fallback)!=(r.candidate_solve_success,r.candidate_cholesky_success,r.candidate_dense_fallback)]
    return Stage1LockstepResult(status,x,float(stage.lm_true_objective_fn(x).detach()),records,len(am),
        None if not am else {"outer":am[0].outer,"trial":am[0].trial},len(sm),
        None if not sm else {"outer":sm[0].outer,"trial":sm[0].trial},accepted,rejected,float(damping))


def compare_stage1_lm_results(serial: LMResult,candidate: LMResult) -> dict[str,Any]:
    diff=candidate.parameters-serial.parameters;first=None
    for index,(left,right) in enumerate(zip(serial.trial_history,candidate.trial_history)):
        for key in ("outer","trial","solve_success","accepted","used_dense_fallback"):
            if left.get(key)!=right.get(key):first={"index":index,"field":key,"serial":left.get(key),"candidate":right.get(key)};break
        if first:break
    structure=(len(serial.weight_history)==len(candidate.weight_history) and all(
        a.get("outer")==b.get("outer") and a.get("n_points")==b.get("n_points")
        for a,b in zip(serial.weight_history,candidate.weight_history)))
    return {"first_divergent_trial":first,"final_parameter_max_abs":float(diff.abs().max()),
            "final_parameter_relative":float(diff.norm())/max(float(serial.parameters.norm()),1e-30),
            "objective_relative":abs(candidate.objective-serial.objective)/max(abs(serial.objective),1e-30),
            "metrics_max_abs":max(abs(candidate.rms_metrics[k]-serial.rms_metrics[k]) for k in ("rms","median","p95")),
            "status_match":serial.status==candidate.status,
            "accepted_match":serial.counters.accepted_steps==candidate.counters.accepted_steps,
            "rejected_match":serial.counters.rejected_trials==candidate.counters.rejected_trials,
            "trial_count_match":len(serial.trial_history)==len(candidate.trial_history),
            "solve_path_match":(serial.counters.cholesky_success_count,serial.counters.solve_fallback_count)==(candidate.counters.cholesky_success_count,candidate.counters.solve_fallback_count),
            "final_lambda_relative":abs(candidate.final_lambda-serial.final_lambda)/max(abs(serial.final_lambda),1e-30),
            "weight_history_structure_match":structure}
