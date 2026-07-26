"""Independent and serial-driven lockstep Joint FULL_BATCH LM shadows."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch

from lib_pano.human_fitting.staged_lm.lm.equations import (
    normal_equations_max_diagonal, normal_equations_quadratic_form,
    solve_normal_equations,
)
from lib_pano.human_fitting.staged_lm.lm.runner import run_lm_optimization
from lib_pano.human_fitting.staged_lm.linearization import (
    Linearization,
    LinearizationProvenance,
)
from lib_pano.human_fitting.staged_lm.joint import physical_beta_from_latent
from lib_pano.human_fitting.staged_lm.types import LMResult, StageDefinition
from script.benchmark.staged_lm.joint_batch_candidates import JointBatchCache, JointNEMode
from script.benchmark.staged_lm.joint_shadow_common import (
    arrowhead_errors, build_joint_shadow_linearization_common,
    evaluate_joint_equations, schur_errors,
)


@dataclass
class JointShadowTrialRecord:
    outer: int; trial: int; damping: float
    serial_solve_success: bool; batch_solve_success: bool
    serial_schur_success: bool; batch_schur_success: bool
    serial_used_dense_fallback: bool; batch_used_dense_fallback: bool
    delta_frame_max_abs: float | None; delta_beta_max_abs: float | None
    delta_relative: float | None
    predicted_serial: float | None; predicted_batch: float | None
    predicted_relative: float | None
    actual_serial: float | None; actual_batch: float | None
    rho_serial: float | None; rho_batch: float | None; rho_abs_diff: float | None
    accepted_serial: bool; accepted_batch: bool; acceptance_match: bool
    A_relative: float; B_relative: float; C_relative: float
    g_frame_relative: float; g_beta_relative: float
    reduced_schur_relative: float | None; reduced_rhs_relative: float | None
    serial_equation_ms: float; batch_equation_ms: float
    worst_frame_index: int | None; worst_A_index: tuple[int,...] | None
    worst_B_index: tuple[int,...] | None; worst_C_index: tuple[int,...] | None
    worst_beta_gradient_index: tuple[int,...] | None


@dataclass
class JointLockstepResult:
    status: str; final_parameters: torch.Tensor; final_objective: float
    trial_records: list[JointShadowTrialRecord]
    acceptance_mismatch_count: int; first_acceptance_mismatch: dict[str,Any]|None
    accepted_steps: int; rejected_trials: int; final_lambda: float
    def to_json_dict(self):
        return {"status":self.status,"final_objective":self.final_objective,
                "acceptance_mismatch_count":self.acceptance_mismatch_count,
                "first_acceptance_mismatch":self.first_acceptance_mismatch,
                "accepted_steps":self.accepted_steps,"rejected_trials":self.rejected_trials,
                "final_lambda":self.final_lambda,"trial_records":[asdict(x) for x in self.trial_records]}


def make_batched_joint_linearization(
    stage: StageDefinition, batch: JointBatchCache, context: Any,
    point: torch.Tensor, mode: JointNEMode,
) -> Linearization:
    common=build_joint_shadow_linearization_common(stage,batch,context,point)
    return Linearization(
        surrogate=lambda x: x.new_empty(0),
        normal_equations=lambda x: evaluate_joint_equations(
            mode, stage, batch, x, common["row_scales_by_group"], context,
        ),
        weight_stats=common["weight_stats"],
        aggregation_spec=common["spec"],
        provenance=LinearizationProvenance(
            jacobian_backend="analytic_stage2_joint_batched_shadow",
            linearization_backend="direct_arrowhead_normal_equations_batched_shadow",
            linear_solver_backend="arrowhead_schur",
        ),
    )


def run_joint_lm_independent_batch(
    stage: StageDefinition,batch: JointBatchCache,context: Any,*,mode: JointNEMode,
    initial_parameters: torch.Tensor|None=None,profile_timing: bool=True,
) -> LMResult:
    initial=stage.initial_lm_parameters if initial_parameters is None else initial_parameters
    return run_lm_optimization(
        "STAGE2_JOINT_BATCH_SHADOW",lambda x:make_batched_joint_linearization(stage,batch,context,x,mode),
        stage.raw_metrics_fn,initial,stage.parameter_dimension,stage.lm_max_accepted_steps,
        context.solver_config.lm_max_trials,context.solver_config,profile_timing,
        stage.lm_true_objective_fn,
    )


def run_joint_lm_lockstep(
    stage: StageDefinition,batch: JointBatchCache,context: Any,*,mode: JointNEMode,
    initial_parameters: torch.Tensor|None=None,
) -> JointLockstepResult:
    x=(stage.initial_lm_parameters if initial_parameters is None else initial_parameters).detach().clone()
    records=[];accepted=rejected=0;damping=None;status="RUNNING";beta_dim=batch.layout.beta_dim
    for outer in range(stage.lm_max_accepted_steps):
        common=build_joint_shadow_linearization_common(stage,batch,context,x)
        scales=common["row_scales_by_group"]
        if x.is_cuda: torch.cuda.synchronize(x.device)
        import time
        begin=time.perf_counter(); serial=evaluate_joint_equations(JointNEMode.SERIAL_REFERENCE,stage,batch,x,scales,context)
        if x.is_cuda:torch.cuda.synchronize(x.device)
        serial_ms=(time.perf_counter()-begin)*1000
        begin=time.perf_counter(); candidate=evaluate_joint_equations(mode,stage,batch,x,scales,context)
        if x.is_cuda:torch.cuda.synchronize(x.device)
        batch_ms=(time.perf_counter()-begin)*1000
        errors=arrowhead_errors(candidate,serial)
        if damping is None:damping=normal_equations_max_diagonal(serial)*1e-3
        before=float(stage.lm_true_objective_fn(x).detach());accepted_this=False
        for trial in range(context.solver_config.lm_max_trials):
            serial_solve=solve_normal_equations(serial,damping)
            batch_solve=solve_normal_equations(candidate,damping)
            sd,bd=serial_solve.delta,batch_solve.delta
            if sd is None:
                damping=min(damping*10,1e12);rejected+=1;continue
            sx=(x+sd.to(x.dtype)).detach();sobj=float(stage.lm_true_objective_fn(sx).detach())
            sp=float(-serial.gradient.dot(sd)-.5*normal_equations_quadratic_form(serial,sd));sa=before-sobj
            sr=sa/max(sp,1e-12) if sp>0 else 0.;sacc=sp>0 and sa>0 and sr>0
            if bd is None: bp=ba=br=None;bacc=False;delta=None;schur={}
            else:
                bx=(x+bd.to(x.dtype)).detach();bobj=float(stage.lm_true_objective_fn(bx).detach())
                bp=float(-candidate.gradient.dot(bd)-.5*normal_equations_quadratic_form(candidate,bd));ba=before-bobj
                br=ba/max(bp,1e-12) if bp>0 else 0.;bacc=bp>0 and ba>0 and br>0;delta=bd-sd
                schur=schur_errors(candidate,serial,damping)
            df=None if delta is None else delta[:-beta_dim].reshape(batch.layout.frame_count,75)
            db=None if delta is None else delta[-beta_dim:]
            records.append(JointShadowTrialRecord(
                outer,trial,float(damping),sd is not None,bd is not None,
                serial_solve.schur_success,batch_solve.schur_success,
                serial_solve.schur_failed,batch_solve.schur_failed,
                None if df is None else float(df.abs().max()),
                None if db is None else float(db.abs().max()),None if delta is None else float(delta.norm())/max(float(sd.norm()),1e-30),
                sp,bp,None if bp is None else abs(bp-sp)/max(abs(sp),1e-30),sa,ba,sr,br,
                None if br is None else abs(br-sr),sacc,bacc,sacc==bacc,
                errors["A"]["relative"],errors["B"]["relative"],errors["C"]["relative"],
                errors["g_frame"]["relative"],errors["g_beta"]["relative"],
                schur.get("reduced_schur",{}).get("relative"),schur.get("reduced_rhs",{}).get("relative"),
                serial_ms,batch_ms,errors["A"]["worst_index"][0] if errors["A"]["worst_index"] else None,
                errors["A"]["worst_index"],errors["B"]["worst_index"],errors["C"]["worst_index"],errors["g_beta"]["worst_index"],
            ))
            if sacc:
                x=sx;accepted+=1;accepted_this=True
                damping=max(damping/3,1e-10) if sr>.75 else min(damping*3,1e12) if sr<.25 else damping
                break
            damping=min(damping*10,1e12);rejected+=1
        if not accepted_this:status="LM_NO_ACCEPTED_TRIAL";break
    if status=="RUNNING":status="MAX_ACCEPTED_STEPS"
    mismatch=[r for r in records if not r.acceptance_match]
    return JointLockstepResult(status,x,float(stage.lm_true_objective_fn(x).detach()),records,len(mismatch),
        None if not mismatch else {"outer":mismatch[0].outer,"trial":mismatch[0].trial},accepted,rejected,float(damping))


def compare_joint_lm_results(serial: LMResult,candidate: LMResult,beta_abs_limit: float,beta_dim: int)->dict[str,Any]:
    diff=candidate.parameters-serial.parameters
    sb,_=physical_beta_from_latent(serial.parameters[-beta_dim:],beta_abs_limit)
    cb,_=physical_beta_from_latent(candidate.parameters[-beta_dim:],beta_abs_limit)
    first=None
    for index,(left,right) in enumerate(zip(serial.trial_history,candidate.trial_history)):
        for key in ("outer","trial","solve_success","accepted","used_dense_fallback"):
            if left.get(key)!=right.get(key):first={"index":index,"difference_type":key,"serial":left.get(key),"batch":right.get(key)};break
        if first:break
    weight_structure_match = (
        len(serial.weight_history) == len(candidate.weight_history)
        and all(
            left.get("outer") == right.get("outer")
            and left.get("n_points") == right.get("n_points")
            for left, right in zip(serial.weight_history, candidate.weight_history)
        )
    )
    weight_numeric_max_abs = max(
        (abs(float(left[key]) - float(right[key]))
         for left, right in zip(serial.weight_history, candidate.weight_history)
         for key in ("min", "mean", "median", "max")
         if key in left and key in right),
        default=0.0,
    )
    return {"first_divergent_trial":first,"final_parameter_max_abs":float(diff.abs().max()),
            "final_parameter_relative":float(diff.norm())/max(float(serial.parameters.norm()),1e-30),
            "physical_beta_max_abs":float((cb-sb).abs().max()),
            "objective_relative":abs(candidate.objective-serial.objective)/max(abs(serial.objective),1e-30),
            "metrics_max_abs":max(abs(candidate.rms_metrics[k]-serial.rms_metrics[k]) for k in ("rms","median","p95")),
            "status_match":serial.status==candidate.status,"accepted_match":serial.counters.accepted_steps==candidate.counters.accepted_steps,
            "rejected_match":serial.counters.rejected_trials==candidate.counters.rejected_trials,
            "trial_count_match":len(serial.trial_history)==len(candidate.trial_history),
            "schur_match":(serial.counters.schur_success_count,counters_value(serial,"schur_dense_fallback_count"),serial.counters.schur_failure_count)==(candidate.counters.schur_success_count,counters_value(candidate,"schur_dense_fallback_count"),candidate.counters.schur_failure_count),
            "weight_history_match":serial.weight_history==candidate.weight_history,
            "weight_history_structure_match":weight_structure_match,
            "weight_history_numeric_max_abs":weight_numeric_max_abs}


def counters_value(result:LMResult,name:str)->int:return int(getattr(result.counters,name,0))
