"""Serial Pose/Shape arrowhead H/g construction and Linearization binding.

For each frame ``t`` this module accumulates
``A[t] += J_frame.T J_frame``, ``B[t] += J_frame.T J_beta``,
``C += J_beta.T J_beta``, and the corresponding gradient blocks.
Pose anchors enter frame blocks; the shape anchor enters the shared beta
block. It does not solve the system: Schur math remains in
:mod:`..arrowhead_schur`. Read :mod:`.batched` for the production equivalent.
"""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any, Callable

import torch

from ..arrowhead_schur import ArrowheadNormalEquationSystem
from ..config import STAGE1_PER_FRAME_DIM
from ..irls import compute_irls_point_weights
from ..linearization import Linearization, LinearizationProvenance
from ..types import NormalEquations
from .beta_derivatives import pose_shape_anchor_residual_and_jacobian
from .cache import Stage2PoseShapeFastCache, validate_stage2_pose_shape_layout
from .derivatives import (
    pose_shape_group_raw_residual_and_jacobians,
    pose_shape_pose_anchor_residual_and_jacobian,
)
from .forward import pose_shape_fast_raw_groups
from .shape import evaluate_pose_shape_state, physical_beta_from_latent


@dataclass(frozen=True)
class Stage2PoseShapeLinearizationCommon:
    row_scales_by_group: list[torch.Tensor]
    weight_stats: dict[str, Any]
    surrogate_fn: Callable[[torch.Tensor], torch.Tensor]
    jacobian_fn: Callable[[torch.Tensor], torch.Tensor]
    spec: dict[str, Any]


def accumulate_stage2_pose_shape_arrowhead_system(
    cache: Stage2PoseShapeFastCache, parameters: torch.Tensor, beta_abs_limit: float,
    row_scales_by_group: list[torch.Tensor], body_pose_anchors: list[torch.Tensor],
    beta_anchor: torch.Tensor, pose_anchor_scale: float, shape_anchor_scale: float,
) -> tuple[torch.Tensor, ArrowheadNormalEquationSystem]:
    """Accumulate ``A[T,75,75]``, ``B[T,75,B]``, ``C[B,B]`` and gradients."""
    per_frame_dim,total_dim=validate_stage2_pose_shape_layout(cache,parameters)
    if len(row_scales_by_group)!=len(cache.groups):
        raise ValueError("pose/shape row scale count")
    if cache.profile_timing and parameters.is_cuda: torch.cuda.synchronize(parameters.device)
    start=time.perf_counter();shape=evaluate_pose_shape_state(cache,parameters[-cache.model.beta_dim:],beta_abs_limit)
    frame_hessians=torch.zeros(cache.frame_count,per_frame_dim,per_frame_dim,dtype=torch.float64,device=parameters.device)
    frame_beta_cross=torch.zeros(cache.frame_count,per_frame_dim,cache.model.beta_dim,dtype=torch.float64,device=parameters.device)
    beta_hessian=torch.zeros(cache.model.beta_dim,cache.model.beta_dim,dtype=torch.float64,device=parameters.device)
    frame_gradients=torch.zeros(cache.frame_count,per_frame_dim,dtype=torch.float64,device=parameters.device)
    beta_gradient=torch.zeros(cache.model.beta_dim,dtype=torch.float64,device=parameters.device);residual_parts=[]
    for group,scale in zip(cache.groups,row_scales_by_group):
        result=pose_shape_group_raw_residual_and_jacobians(cache,group,parameters,beta_abs_limit,shape)
        scale=scale.to(parameters);r=result.raw_residual*scale
        jf=(result.local_pose_jacobian*scale.unsqueeze(1)).to(torch.float64)
        jb=(result.shared_beta_jacobian*scale.unsqueeze(1)).to(torch.float64);r64=r.to(torch.float64)
        frame=group.frame_position
        frame_hessians[frame]+=jf.T@jf
        frame_beta_cross[frame]+=jf.T@jb
        beta_hessian+=jb.T@jb
        frame_gradients[frame]+=jf.T@r64;beta_gradient+=jb.T@r64;residual_parts.append(r)
    pose_residuals=[];pose_scale_squared=float(pose_anchor_scale)**2
    for frame in range(cache.frame_count):
        parameter_body_slice=slice(frame*per_frame_dim+STAGE1_PER_FRAME_DIM,(frame+1)*per_frame_dim)
        local_body_slice=slice(STAGE1_PER_FRAME_DIM,per_frame_dim)
        delta=parameters[parameter_body_slice]-body_pose_anchors[frame].to(parameters)
        pose_residuals.append(delta*pose_anchor_scale)
        frame_hessians[frame,local_body_slice,local_body_slice]+=torch.eye(cache.model.body_pose_dim,dtype=torch.float64,device=parameters.device)*pose_scale_squared
        frame_gradients[frame,local_body_slice]+=delta.to(torch.float64)*pose_scale_squared
    beta,dbdz=physical_beta_from_latent(parameters[-cache.model.beta_dim:],beta_abs_limit)
    shape_residual=(beta-beta_anchor.to(parameters))*shape_anchor_scale
    diagonal=(dbdz*shape_anchor_scale).to(torch.float64)
    beta_hessian+=torch.diag(diagonal.square())
    beta_gradient+=diagonal*shape_residual.to(torch.float64)
    residual=torch.cat((*residual_parts,*pose_residuals,shape_residual))
    system=ArrowheadNormalEquationSystem(
        frame_hessians,frame_beta_cross,beta_hessian,frame_gradients,beta_gradient,
        timing_sink=cache,
    )
    if cache.profile_timing and parameters.is_cuda: torch.cuda.synchronize(parameters.device)
    cache.arrowhead_accumulation_ms+=(time.perf_counter()-start)*1000;cache.arrowhead_accumulation_calls+=1
    return residual,system


def stage2_pose_shape_arrowhead_normal_equations(
    cache: Stage2PoseShapeFastCache, parameters: torch.Tensor, beta_abs_limit: float,
    row_scales_by_group: list[torch.Tensor], body_pose_anchors: list[torch.Tensor],
    beta_anchor: torch.Tensor, pose_anchor_scale: float, shape_anchor_scale: float,
) -> NormalEquations:
    start=time.perf_counter()
    residual,system=accumulate_stage2_pose_shape_arrowhead_system(
        cache,parameters,beta_abs_limit,row_scales_by_group,body_pose_anchors,
        beta_anchor,pose_anchor_scale,shape_anchor_scale,
    )
    result=NormalEquations(residual,None,system.flat_gradient(),structured_system=system)
    cache.normal_equations_ms+=(time.perf_counter()-start)*1000;cache.normal_equations_calls+=1
    return result


def build_stage2_pose_shape_linearization_common(
    cache: Stage2PoseShapeFastCache, linearisation_point: torch.Tensor, beta_abs_limit: float,
    beta_anchor: torch.Tensor, body_pose_anchors: list[torch.Tensor],
    pose_anchor_scale: float, shape_anchor_scale: float, *, is_irls: bool,
    charbonnier_eps: float, aggregation_spec: dict[str,Any]|None, rep_scale: float,
) -> Stage2PoseShapeLinearizationCommon:
    groups=pose_shape_fast_raw_groups(cache,linearisation_point,beta_abs_limit)
    if is_irls:
        if aggregation_spec is None: raise ValueError("aggregation spec required")
        weights=compute_irls_point_weights(groups,charbonnier_eps,aggregation_spec)
        scales=[torch.sqrt(w).repeat_interleave(2).detach() for w in weights]
        flat=torch.cat(weights);stats={"min":float(flat.min()),"mean":float(flat.mean()),"median":float(flat.median()),"max":float(flat.max()),"n_points":flat.numel()}
    else:
        scales=[torch.full((g.residual_2d.numel(),),rep_scale,dtype=linearisation_point.dtype,device=linearisation_point.device) for g in groups];stats={}
    per_frame_dim,total_dim=validate_stage2_pose_shape_layout(cache,linearisation_point)
    def anchors(parameters):
        pose=torch.cat([(parameters[f*per_frame_dim+STAGE1_PER_FRAME_DIM:(f+1)*per_frame_dim]-body_pose_anchors[f].to(parameters))*pose_anchor_scale for f in range(cache.frame_count)])
        beta,_=physical_beta_from_latent(parameters[-cache.model.beta_dim:],beta_abs_limit)
        shape=(beta-beta_anchor.to(parameters))*shape_anchor_scale
        return pose,shape
    def surrogate_fn(parameters):
        current=pose_shape_fast_raw_groups(cache,parameters,beta_abs_limit)
        reproj=[g.residual_2d.reshape(-1)*s.to(parameters) for g,s in zip(current,scales)]
        pose,shape=anchors(parameters);return torch.cat((*reproj,pose,shape))
    def jacobian_fn(parameters):
        if cache.profile_timing and parameters.is_cuda:torch.cuda.synchronize(parameters.device)
        start=time.perf_counter();shape_state=evaluate_pose_shape_state(cache,parameters[-cache.model.beta_dim:],beta_abs_limit)
        blocks=[]
        for group,scale in zip(cache.groups,scales):
            result=pose_shape_group_raw_residual_and_jacobians(cache,group,parameters,beta_abs_limit,shape_state)
            block=torch.zeros(result.raw_residual.numel(),total_dim,dtype=parameters.dtype,device=parameters.device)
            fs=slice(group.frame_position*per_frame_dim,(group.frame_position+1)*per_frame_dim)
            block[:,fs]=result.local_pose_jacobian*scale.to(parameters).unsqueeze(1)
            block[:,-cache.model.beta_dim:]=result.shared_beta_jacobian*scale.to(parameters).unsqueeze(1);blocks.append(block)
        _,pose_j=pose_shape_pose_anchor_residual_and_jacobian(cache,parameters,body_pose_anchors,pose_anchor_scale)
        _,shape_j=pose_shape_anchor_residual_and_jacobian(parameters[-cache.model.beta_dim:],beta_anchor,beta_abs_limit,shape_anchor_scale,parameter_dimension=total_dim)
        value=torch.cat((*blocks,pose_j,shape_j))
        if cache.profile_timing and parameters.is_cuda:torch.cuda.synchronize(parameters.device)
        cache.explicit_jacobian_ms+=(time.perf_counter()-start)*1000;cache.explicit_jacobian_calls+=1
        return value
    return Stage2PoseShapeLinearizationCommon(
        scales, stats, surrogate_fn, jacobian_fn, aggregation_spec or {},
    )


def build_stage2_pose_shape_analytic_linearization(
    cache: Stage2PoseShapeFastCache, linearisation_point: torch.Tensor, beta_abs_limit: float,
    beta_anchor: torch.Tensor, body_pose_anchors: list[torch.Tensor],
    pose_anchor_scale: float, shape_anchor_scale: float, *, is_irls: bool,
    charbonnier_eps: float, aggregation_spec: dict[str,Any]|None, rep_scale: float,
) -> Linearization:
    common = build_stage2_pose_shape_linearization_common(
        cache, linearisation_point, beta_abs_limit, beta_anchor,
        body_pose_anchors, pose_anchor_scale, shape_anchor_scale,
        is_irls=is_irls, charbonnier_eps=charbonnier_eps,
        aggregation_spec=aggregation_spec, rep_scale=rep_scale,
    )
    def normal_equations_fn(parameters):
        return stage2_pose_shape_arrowhead_normal_equations(
            cache, parameters, beta_abs_limit, common.row_scales_by_group,
            body_pose_anchors, beta_anchor, pose_anchor_scale, shape_anchor_scale,
        )
    return Linearization(
        surrogate=common.surrogate_fn,
        jacobian=common.jacobian_fn,
        normal_equations=normal_equations_fn,
        weight_stats=common.weight_stats,
        aggregation_spec=common.spec,
        provenance=LinearizationProvenance(
            jacobian_backend="analytic_stage2_pose_shape",
            linearization_backend="direct_normal_equations",
            linear_solver_backend="arrowhead_schur",
        ),
    )
