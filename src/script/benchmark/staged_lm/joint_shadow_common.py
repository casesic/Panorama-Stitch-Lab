"""Shared comparison and serial-weight utilities for Joint LM shadows."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any

import torch

from lib_pano.human_fitting.staged_lm.irls import compute_irls_point_weights
from lib_pano.human_fitting.staged_lm.stage2_joint_fast import joint_fast_raw_groups,evaluate_joint_shape_state
from lib_pano.human_fitting.staged_lm.stage2_joint_jacobian import (
    build_stage2_joint_analytic_linearization,joint_group_raw_residual_and_jacobians,
)
from lib_pano.human_fitting.staged_lm.types import NormalEquations, StageDefinition
from script.benchmark.staged_lm.joint_batch_candidates import (
    JointBatchCache, JointNEMode, evaluate_joint_normal_equations,
    joint_batched_group_analytic,
    pad_joint_row_scales,
)


@dataclass(frozen=True)
class TensorError:
    max_abs: float
    relative: float
    reference_norm: float
    candidate_norm: float
    cosine_similarity: float
    worst_index: tuple[int, ...] | None


def tensor_error(candidate: torch.Tensor, reference: torch.Tensor) -> TensorError:
    delta = (candidate-reference).to(torch.float64)
    ref = reference.to(torch.float64); cand = candidate.to(torch.float64)
    flat = delta.abs().reshape(-1)
    worst = None
    if flat.numel():
        linear = int(flat.argmax())
        indices=[]
        for size in reversed(delta.shape): indices.append(linear % size); linear //= size
        worst=tuple(reversed(indices))
    rn=float(torch.linalg.vector_norm(ref)); cn=float(torch.linalg.vector_norm(cand))
    cosine=float(torch.nn.functional.cosine_similarity(ref.reshape(1,-1),cand.reshape(1,-1),eps=1e-30)) if ref.numel() else 1.0
    return TensorError(float(flat.max()) if flat.numel() else 0.0,
                       float(torch.linalg.vector_norm(delta))/max(rn,1e-30),rn,cn,cosine,worst)


def serial_joint_row_scales(
    stage: StageDefinition, parameters: torch.Tensor, context: Any,
) -> tuple[list[torch.Tensor], dict[str, Any]]:
    cache=stage.analytic_cache
    groups=joint_fast_raw_groups(cache,parameters,context.objective_config.beta_abs_limit)
    if context.is_irls:
        weights=compute_irls_point_weights(groups,context.charbonnier_eps,stage.aggregation_spec)
        scales=[torch.sqrt(weight).repeat_interleave(2).detach() for weight in weights]
        flat=torch.cat(weights)
        stats={"min":float(flat.min()),"mean":float(flat.mean()),"median":float(flat.median()),
               "max":float(flat.max()),"n_points":flat.numel()}
    else:
        scales=[torch.full((group.residual_2d.numel(),),context.scaling.rep_scale,
                           dtype=parameters.dtype,device=parameters.device) for group in groups]
        stats={}
    return scales,stats


def build_joint_shadow_linearization_common(
    stage: StageDefinition, batch: JointBatchCache, context: Any,
    linearization_point: torch.Tensor,
) -> dict[str,Any]:
    scales,stats=serial_joint_row_scales(stage,linearization_point,context)
    # The production builder is called only as a reference oracle in tests;
    # shadow execution binds its already-computed scales directly.
    return {"row_scales_by_group":scales,"padded_row_scales":pad_joint_row_scales(batch,scales),
            "weight_stats":stats,"spec":stage.aggregation_spec or {}}


def evaluate_joint_equations(
    mode: JointNEMode, stage: StageDefinition, batch: JointBatchCache,
    parameters: torch.Tensor, scales: list[torch.Tensor], context: Any,
) -> NormalEquations:
    return evaluate_joint_normal_equations(
        mode,stage.analytic_cache,batch,parameters,context.objective_config.beta_abs_limit,scales,
        context.body_pose_anchors,context.beta_anchor,context.scaling.pose_anchor_scale,
        context.scaling.shape_anchor_scale,
    )


def arrowhead_errors(candidate: NormalEquations, reference: NormalEquations) -> dict[str,Any]:
    c=candidate.structured_system;r=reference.structured_system
    if c is None or r is None: raise RuntimeError("MISSING_ARROWHEAD_SYSTEM")
    output={
        "A":asdict(tensor_error(c.frame_hessians,r.frame_hessians)),
        "B":asdict(tensor_error(c.frame_beta_cross,r.frame_beta_cross)),
        "C":asdict(tensor_error(c.beta_hessian,r.beta_hessian)),
        "g_frame":asdict(tensor_error(c.frame_gradients,r.frame_gradients)),
        "g_beta":asdict(tensor_error(c.beta_gradient,r.beta_gradient)),
        "flat_gradient":asdict(tensor_error(c.flat_gradient(),r.flat_gradient())),
        "residual":asdict(tensor_error(candidate.residual,reference.residual)),
        "max_diagonal_abs":abs(float(c.max_diagonal())-float(r.max_diagonal())),
        "frobenius_abs":abs(float(c.hessian_frobenius_norm())-float(r.hessian_frobenius_norm())),
    }
    output["g_frame_sign_mismatch"] = int(((torch.sign(c.frame_gradients)!=torch.sign(r.frame_gradients)) & (r.frame_gradients.abs()>1e-6)).sum())
    output["g_beta_sign_mismatch"] = int(((torch.sign(c.beta_gradient)!=torch.sign(r.beta_gradient)) & (r.beta_gradient.abs()>1e-6)).sum())
    return output


def joint_microkernel_errors(
    stage: StageDefinition,batch: JointBatchCache,parameters: torch.Tensor,
    beta_abs_limit: float,
) -> dict[str,Any]:
    """Compare valid padded batch residual/J blocks with production groups."""
    cache=stage.analytic_cache
    shape=evaluate_joint_shape_state(cache,parameters[-batch.layout.beta_dim:],beta_abs_limit)
    candidate=joint_batched_group_analytic(cache,batch,parameters,beta_abs_limit)
    residual=[];pose=[];beta=[]
    for index,(group,count) in enumerate(zip(cache.groups,batch.layout.point_counts_host)):
        value=joint_group_raw_residual_and_jacobians(cache,group,parameters,beta_abs_limit,shape)
        indices=batch.active_indices_cam2 if batch.layout.camera_indices_host[index]==0 else batch.active_indices_cam4
        residual.append((candidate.residual[index,:count].reshape(-1),value.raw_residual))
        pose.append((candidate.active_pose_jacobian[index,:count].reshape(-1,69),value.local_pose_jacobian.index_select(1,indices)))
        beta.append((candidate.beta_jacobian[index,:count].reshape(-1,batch.layout.beta_dim),value.shared_beta_jacobian))
    return {"residual":asdict(tensor_error(torch.cat([x[0] for x in residual]),torch.cat([x[1] for x in residual]))),
            "pose_jacobian":asdict(tensor_error(torch.cat([x[0] for x in pose]),torch.cat([x[1] for x in pose]))),
            "beta_jacobian":asdict(tensor_error(torch.cat([x[0] for x in beta]),torch.cat([x[1] for x in beta])))}


def build_schur_diagnostics(system, damping: float) -> dict[str,torch.Tensor|bool]:
    p,b=system.frame_dimension,system.beta_dimension
    local=system.frame_hessians+float(damping)*torch.eye(p,dtype=torch.float64,device=system.frame_hessians.device)[None]
    chol,info=torch.linalg.cholesky_ex(local)
    rhs=torch.cat((system.frame_gradients[...,None],system.frame_beta_cross),-1)
    solved=torch.cholesky_solve(rhs,chol);xg,xb=solved[...,0],solved[...,1:]
    schur=system.beta_hessian+float(damping)*torch.eye(b,dtype=torch.float64,device=local.device)-torch.einsum("tpb,tpc->bc",system.frame_beta_cross,xb)
    reduced_rhs=-system.beta_gradient+torch.einsum("tpb,tp->b",system.frame_beta_cross,xg)
    rchol,rinfo=torch.linalg.cholesky_ex(schur)
    db=torch.cholesky_solve(reduced_rhs[:,None],rchol).squeeze(1)
    df=-xg-torch.einsum("tpb,b->tp",xb,db)
    return {"local_damped":local,"local_cholesky_info":info,"xg":xg,"xb":xb,
            "reduced_schur":schur,"reduced_rhs":reduced_rhs,
            "reduced_cholesky_info":rinfo,"delta_beta":db,"delta_frames":df,
            "flat_delta":torch.cat((df.reshape(-1),db)),
            "success":bool(torch.all(info==0) and int(rinfo)==0)}


def schur_errors(candidate: NormalEquations, reference: NormalEquations, damping: float) -> dict[str,Any]:
    c=build_schur_diagnostics(candidate.structured_system,damping)
    r=build_schur_diagnostics(reference.structured_system,damping)
    return {name:asdict(tensor_error(c[name],r[name])) for name in
            ("local_damped","reduced_schur","reduced_rhs","delta_beta","delta_frames","flat_delta")}


def diagnostic_gate(errors: dict[str,Any], schur: dict[str,Any]) -> bool:
    return (errors["residual"]["relative"]<=1e-6 and errors["A"]["relative"]<=2e-5
            and errors["B"]["relative"]<=2e-5 and errors["C"]["relative"]<=2e-5
            and errors["g_frame"]["relative"]<=2e-5 and errors["g_beta"]["relative"]<=2e-5
            and schur["flat_delta"]["relative"]<=1e-4)
