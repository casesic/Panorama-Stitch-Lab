"""Complete dynamic Stage 2 joint pose+beta analytic Jacobian.

Gauss-Newton needs the two first-order blocks at the current parameter point,
``J_pose(x, beta)`` and ``J_beta(x, beta)``. It does not require the mixed
second derivative ``d(J_pose)/d beta``. Consequently the pose block here is
evaluated with the current dynamic ``v_shaped`` and joints, never with the
fixed-beta pose-warmup cache.
"""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any

import torch

from .analytic_stage1 import project_cam2_erp_with_jacobian
from .arrowhead_schur import ArrowheadNormalEquationSystem
from .config import STAGE1_PER_FRAME_DIM
from .irls import build_primary_aggregation_spec, compute_irls_point_weights
from .parameter_layout import unpack_frame_parameters
from .stage2_joint_beta_jacobian import joint_shape_anchor_residual_and_jacobian
from .stage2_joint_fast import (
    Stage2JointFastCache, Stage2JointFrameCameraCache, Stage2JointShapeState,
    evaluate_joint_shape_state, validate_stage2_joint_layout, joint_fast_raw_groups,
    physical_beta_from_latent,
)
from .stage2_pose_jacobian import (
    assemble_full_pose_rotation_jacobians,
    compute_selected_lbs_with_jacobian,
    compute_selected_pose_offsets_with_jacobian,
)
from .types import NormalEquations, ReprojectionGroup


@dataclass
class CombinedFKResult:
    posed_joints: torch.Tensor
    global_transforms: torch.Tensor
    relative_transforms: torch.Tensor
    d_global_transforms_d_pose: torch.Tensor
    d_relative_transforms_d_pose: torch.Tensor
    d_global_transforms_d_beta: torch.Tensor
    d_relative_transforms_d_beta: torch.Tensor


@dataclass
class JointGroupAnalyticState:
    vertices: torch.Tensor
    d_vertices_d_pose_rotation: torch.Tensor
    d_vertices_d_beta_latent: torch.Tensor
    intermediates: dict[str, torch.Tensor]


@dataclass
class JointGroupJacobianResult:
    raw_residual: torch.Tensor
    local_pose_jacobian: torch.Tensor
    shared_beta_jacobian: torch.Tensor
    intermediates: dict[str, torch.Tensor] | None = None


@dataclass
class JointSurrogateResult:
    residual: torch.Tensor
    jacobian: torch.Tensor


def compute_joint_fk_with_combined_jacobian(
    rotations: torch.Tensor,
    d_rotations_d_pose: torch.Tensor,
    joints: torch.Tensor,
    d_joints_d_beta_latent: torch.Tensor,
    parents: torch.Tensor,
) -> CombinedFKResult:
    """Propagate ``[pose | beta]`` FK derivative columns in one tree pass."""
    joint_count = joints.shape[0]
    pose_dim = d_rotations_d_pose.shape[-1]
    beta_dim = d_joints_d_beta_latent.shape[-1]
    combined_dim = pose_dim + beta_dim
    relative_joints = joints.clone()
    relative_joints[1:] -= joints.index_select(0, parents[1:])
    d_relative_beta = d_joints_d_beta_latent.clone()
    d_relative_beta[1:] -= d_joints_d_beta_latent.index_select(0, parents[1:])

    local = torch.zeros(joint_count, 4, 4, dtype=joints.dtype, device=joints.device)
    local[:, :3, :3] = rotations
    local[:, :3, 3] = relative_joints
    local[:, 3, 3] = 1
    d_local = torch.zeros(
        joint_count, 4, 4, combined_dim, dtype=joints.dtype, device=joints.device,
    )
    d_local[:, :3, :3, :pose_dim] = d_rotations_d_pose
    d_local[:, :3, 3, pose_dim:] = d_relative_beta

    globals_ = [local[0]]
    derivatives = [d_local[0]]
    for joint in range(1, joint_count):
        parent = int(parents[joint])
        parent_global = globals_[parent]
        globals_.append(parent_global @ local[joint])
        derivatives.append(
            torch.einsum("abq,bc->acq", derivatives[parent], local[joint])
            + torch.einsum("ab,bcq->acq", parent_global, d_local[joint])
        )
    global_transforms = torch.stack(globals_)
    d_global = torch.stack(derivatives)

    rest = torch.eye(4, dtype=joints.dtype, device=joints.device).repeat(joint_count, 1, 1)
    rest[:, :3, 3] = -joints
    d_rest = torch.zeros(
        joint_count, 4, 4, combined_dim, dtype=joints.dtype, device=joints.device,
    )
    d_rest[:, :3, 3, pose_dim:] = -d_joints_d_beta_latent
    relative = global_transforms @ rest
    d_relative = (
        torch.einsum("jabq,jbc->jacq", d_global, rest)
        + torch.einsum("jab,jbcq->jacq", global_transforms, d_rest)
    )
    return CombinedFKResult(
        global_transforms[:, :3, 3], global_transforms, relative,
        d_global[..., :pose_dim], d_relative[..., :pose_dim],
        d_global[..., pose_dim:], d_relative[..., pose_dim:],
    )


def evaluate_joint_group_analytic_state(
    cache: Stage2JointFastCache, camera_cache: Stage2JointFrameCameraCache,
    shape_state: Stage2JointShapeState, global_orient: torch.Tensor,
    translation: torch.Tensor, body_pose: torch.Tensor,
) -> JointGroupAnalyticState:
    rotations, d_rotations = assemble_full_pose_rotation_jacobians(
        global_orient, body_pose, camera_cache.left_hand_pose.to(global_orient),
        camera_cache.right_hand_pose.to(global_orient), cache.model,
    )
    pose_offsets, d_pose_offsets, pose_feature, d_pose_feature = (
        compute_selected_pose_offsets_with_jacobian(
            rotations, d_rotations, camera_cache.selected_posedirs.to(global_orient),
        )
    )
    beta_dirs = camera_cache.selected_beta_shapedirs.to(global_orient)
    selected_v_shaped = camera_cache.selected_base_v_shaped.to(global_orient) + torch.einsum(
        "ncb,b->nc", beta_dirs, shape_state.beta,
    )
    d_v_beta = beta_dirs * shape_state.d_beta_d_latent.reshape(1, 1, -1)
    selected_v_posed = selected_v_shaped + pose_offsets
    cache.fk_combined_calls += 1
    fk = compute_joint_fk_with_combined_jacobian(
        rotations, d_rotations, shape_state.joints, shape_state.d_joints_d_latent,
        cache.model.parents.to(global_orient.device),
    )
    combined_v = torch.cat((d_pose_offsets, d_v_beta), -1)
    combined_a = torch.cat((fk.d_relative_transforms_d_pose, fk.d_relative_transforms_d_beta), -1)
    cache.selected_lbs_forward_calls += 1
    skinned, d_skinned, transforms, d_transforms = compute_selected_lbs_with_jacobian(
        selected_v_posed, combined_v, camera_cache.selected_lbs_weights.to(global_orient),
        fk.relative_transforms, combined_a,
    )
    pose_rotation_dim = 3 + cache.model.body_pose_dim
    scale = camera_cache.scale.to(global_orient).reshape(1, 1)
    vertices = skinned * scale + translation.reshape(1, 3)
    d_pose = d_skinned[..., :pose_rotation_dim] * scale.unsqueeze(-1)
    d_beta = d_skinned[..., pose_rotation_dim:] * scale.unsqueeze(-1)
    return JointGroupAnalyticState(vertices, d_pose, d_beta, {
        "rotations": rotations, "d_rotations_d_pose": d_rotations,
        "pose_feature": pose_feature, "d_pose_feature_d_pose": d_pose_feature,
        "selected_pose_offsets": pose_offsets, "d_pose_offsets_d_pose": d_pose_offsets,
        "selected_v_shaped": selected_v_shaped, "d_selected_v_shaped_d_beta_latent": d_v_beta,
        "selected_v_posed": selected_v_posed, "d_selected_v_posed_d_pose": d_pose_offsets,
        "d_selected_v_posed_d_beta_latent": d_v_beta, "joints": shape_state.joints,
        "d_joints_d_beta_latent": shape_state.d_joints_d_latent,
        "global_transforms": fk.global_transforms, "relative_transforms": fk.relative_transforms,
        "d_relative_transforms_d_pose": fk.d_relative_transforms_d_pose,
        "d_relative_transforms_d_beta_latent": fk.d_relative_transforms_d_beta,
        "selected_transforms": transforms,
        "d_selected_transforms_d_pose": d_transforms[..., :pose_rotation_dim],
        "d_selected_transforms_d_beta_latent": d_transforms[..., pose_rotation_dim:],
        "selected_skinned_vertices": skinned,
    })


def joint_group_raw_residual_and_jacobians(
    cache: Stage2JointFastCache, camera_cache: Stage2JointFrameCameraCache,
    parameters: torch.Tensor, beta_abs_limit: float,
    shape_state: Stage2JointShapeState | None = None, *, return_intermediates: bool = False,
) -> JointGroupJacobianResult:
    per_frame_dim, _ = validate_stage2_joint_layout(cache, parameters)
    if shape_state is None:
        shape_state = evaluate_joint_shape_state(cache, parameters[-cache.model.beta_dim:], beta_abs_limit)
    frame = unpack_frame_parameters(parameters, camera_cache.frame_position, per_frame_dim, True)
    if camera_cache.camera == "cam2":
        orient, translation, camera_column = frame.cam2_orient, frame.cam2_transl, 0
    else:
        orient, translation, camera_column = frame.cam4_orient, frame.cam4_transl, 6
    cache.group_analytic_calls += 1
    state = evaluate_joint_group_analytic_state(
        cache, camera_cache, shape_state, orient.reshape(3), translation.reshape(3),
        frame.body_pose.reshape(-1),
    )
    uv, projection_j = project_cam2_erp_with_jacobian(state.vertices, cache.image_width, cache.image_height)
    uv_pose = torch.einsum("nab,nbq->naq", projection_j, state.d_vertices_d_pose_rotation)
    uv_beta = torch.einsum("nab,nbq->naq", projection_j, state.d_vertices_d_beta_latent)
    point_count = state.vertices.shape[0]
    local = torch.zeros(point_count, 2, per_frame_dim, dtype=parameters.dtype, device=parameters.device)
    local[:, :, camera_column:camera_column+3] = uv_pose[:, :, :3]
    local[:, :, camera_column+3:camera_column+6] = projection_j
    local[:, :, STAGE1_PER_FRAME_DIM:] = uv_pose[:, :, 3:]
    return JointGroupJacobianResult(
        (uv-camera_cache.targets.to(parameters)).reshape(-1),
        local.reshape(-1, per_frame_dim), uv_beta.reshape(-1, cache.model.beta_dim),
        state.intermediates if return_intermediates else None,
    )


def joint_group_explicit_local_jacobian(*args, **kwargs) -> torch.Tensor:
    result = joint_group_raw_residual_and_jacobians(*args, **kwargs)
    return torch.cat((result.local_pose_jacobian, result.shared_beta_jacobian), 1)


def joint_explicit_raw_global_jacobian(
    cache: Stage2JointFastCache, parameters: torch.Tensor, beta_abs_limit: float,
) -> torch.Tensor:
    """DEBUG_ONLY_GLOBAL_JOINT_JACOBIAN."""
    per_frame_dim, total_dim = validate_stage2_joint_layout(cache, parameters)
    shape = evaluate_joint_shape_state(cache, parameters[-cache.model.beta_dim:], beta_abs_limit)
    rows = sum(group.targets.numel() for group in cache.groups)
    jacobian = torch.zeros(rows, total_dim, dtype=parameters.dtype, device=parameters.device)
    row = 0
    for group in cache.groups:
        result = joint_group_raw_residual_and_jacobians(cache, group, parameters, beta_abs_limit, shape)
        block = group.frame_position * per_frame_dim
        count = result.raw_residual.numel()
        jacobian[row:row+count, block:block+per_frame_dim] = result.local_pose_jacobian
        jacobian[row:row+count, -cache.model.beta_dim:] = result.shared_beta_jacobian
        row += count
    return jacobian


def joint_pose_anchor_residual_and_jacobian(
    cache: Stage2JointFastCache, parameters: torch.Tensor,
    body_pose_anchors: list[torch.Tensor], pose_anchor_scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    per_frame_dim, total_dim = validate_stage2_joint_layout(cache, parameters)
    residuals=[]; jacobian=torch.zeros(
        cache.frame_count*cache.model.body_pose_dim, total_dim,
        dtype=parameters.dtype, device=parameters.device,
    )
    identity=torch.eye(cache.model.body_pose_dim,dtype=parameters.dtype,device=parameters.device)*pose_anchor_scale
    for frame in range(cache.frame_count):
        column=frame*per_frame_dim+STAGE1_PER_FRAME_DIM; row=frame*cache.model.body_pose_dim
        residuals.append((parameters[column:column+cache.model.body_pose_dim]-body_pose_anchors[frame].to(parameters))*pose_anchor_scale)
        jacobian[row:row+cache.model.body_pose_dim,column:column+cache.model.body_pose_dim]=identity
    return torch.cat(residuals),jacobian


def joint_surrogate_residual_and_jacobian(
    cache: Stage2JointFastCache, parameters: torch.Tensor, beta_abs_limit: float,
    beta_anchor: torch.Tensor, body_pose_anchors: list[torch.Tensor],
    pose_anchor_scale: float, shape_anchor_scale: float, *,
    objective_mode: str, charbonnier_eps: float, rep_scale: float,
) -> JointSurrogateResult:
    _, total_dim=validate_stage2_joint_layout(cache,parameters)
    shape=evaluate_joint_shape_state(cache,parameters[-cache.model.beta_dim:],beta_abs_limit)
    results=[joint_group_raw_residual_and_jacobians(cache,g,parameters,beta_abs_limit,shape) for g in cache.groups]
    groups=[ReprojectionGroup(g.frame_index,g.camera,r.raw_residual.reshape(-1,2)) for g,r in zip(cache.groups,results)]
    if objective_mode=="charbonnier_irls":
        spec=build_primary_aggregation_spec(groups); weights=compute_irls_point_weights(groups,charbonnier_eps,spec)
        scales=[torch.sqrt(w).repeat_interleave(2) for w in weights]
    elif objective_mode=="raw_l2":
        scales=[torch.full((r.raw_residual.numel(),),rep_scale,dtype=parameters.dtype,device=parameters.device) for r in results]
    else: raise ValueError(objective_mode)
    residual_parts=[]; jacobian_parts=[]
    per_frame_dim=STAGE1_PER_FRAME_DIM+cache.model.body_pose_dim
    for group,result,scale in zip(cache.groups,results,scales):
        residual_parts.append(result.raw_residual*scale)
        block=torch.zeros(result.raw_residual.numel(),total_dim,dtype=parameters.dtype,device=parameters.device)
        column=group.frame_position*per_frame_dim
        block[:,column:column+per_frame_dim]=result.local_pose_jacobian*scale.unsqueeze(1)
        block[:,-cache.model.beta_dim:]=result.shared_beta_jacobian*scale.unsqueeze(1)
        jacobian_parts.append(block)
    pose_r,pose_j=joint_pose_anchor_residual_and_jacobian(cache,parameters,body_pose_anchors,pose_anchor_scale)
    shape_r,shape_j=joint_shape_anchor_residual_and_jacobian(
        parameters[-cache.model.beta_dim:],beta_anchor,beta_abs_limit,shape_anchor_scale,
        parameter_dimension=total_dim,
    )
    return JointSurrogateResult(torch.cat((*residual_parts,pose_r,shape_r)),torch.cat((*jacobian_parts,pose_j,shape_j)))


def accumulate_stage2_joint_arrowhead_system(
    cache: Stage2JointFastCache, parameters: torch.Tensor, beta_abs_limit: float,
    row_scales_by_group: list[torch.Tensor], body_pose_anchors: list[torch.Tensor],
    beta_anchor: torch.Tensor, pose_anchor_scale: float, shape_anchor_scale: float,
) -> tuple[torch.Tensor, ArrowheadNormalEquationSystem]:
    per_frame_dim,total_dim=validate_stage2_joint_layout(cache,parameters)
    if len(row_scales_by_group)!=len(cache.groups): raise ValueError("joint row scale count")
    if cache.profile_timing and parameters.is_cuda: torch.cuda.synchronize(parameters.device)
    start=time.perf_counter();shape=evaluate_joint_shape_state(cache,parameters[-cache.model.beta_dim:],beta_abs_limit)
    frame_hessians=torch.zeros(cache.frame_count,per_frame_dim,per_frame_dim,dtype=torch.float64,device=parameters.device)
    frame_beta_cross=torch.zeros(cache.frame_count,per_frame_dim,cache.model.beta_dim,dtype=torch.float64,device=parameters.device)
    beta_hessian=torch.zeros(cache.model.beta_dim,cache.model.beta_dim,dtype=torch.float64,device=parameters.device)
    frame_gradients=torch.zeros(cache.frame_count,per_frame_dim,dtype=torch.float64,device=parameters.device)
    beta_gradient=torch.zeros(cache.model.beta_dim,dtype=torch.float64,device=parameters.device);residual_parts=[]
    for group,scale in zip(cache.groups,row_scales_by_group):
        result=joint_group_raw_residual_and_jacobians(cache,group,parameters,beta_abs_limit,shape)
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


def stage2_joint_arrowhead_normal_equations(
    cache: Stage2JointFastCache, parameters: torch.Tensor, beta_abs_limit: float,
    row_scales_by_group: list[torch.Tensor], body_pose_anchors: list[torch.Tensor],
    beta_anchor: torch.Tensor, pose_anchor_scale: float, shape_anchor_scale: float,
) -> NormalEquations:
    start=time.perf_counter()
    residual,system=accumulate_stage2_joint_arrowhead_system(
        cache,parameters,beta_abs_limit,row_scales_by_group,body_pose_anchors,
        beta_anchor,pose_anchor_scale,shape_anchor_scale,
    )
    result=NormalEquations(residual,None,system.flat_gradient(),structured_system=system)
    cache.normal_equations_ms+=(time.perf_counter()-start)*1000;cache.normal_equations_calls+=1
    return result


def build_stage2_joint_analytic_linearization(
    cache: Stage2JointFastCache, linearisation_point: torch.Tensor, beta_abs_limit: float,
    beta_anchor: torch.Tensor, body_pose_anchors: list[torch.Tensor],
    pose_anchor_scale: float, shape_anchor_scale: float, *, is_irls: bool,
    charbonnier_eps: float, aggregation_spec: dict[str,Any]|None, rep_scale: float,
) -> dict[str,Any]:
    groups=joint_fast_raw_groups(cache,linearisation_point,beta_abs_limit)
    if is_irls:
        if aggregation_spec is None: raise ValueError("aggregation spec required")
        weights=compute_irls_point_weights(groups,charbonnier_eps,aggregation_spec)
        scales=[torch.sqrt(w).repeat_interleave(2).detach() for w in weights]
        flat=torch.cat(weights);stats={"min":float(flat.min()),"mean":float(flat.mean()),"median":float(flat.median()),"max":float(flat.max()),"n_points":flat.numel()}
    else:
        scales=[torch.full((g.residual_2d.numel(),),rep_scale,dtype=linearisation_point.dtype,device=linearisation_point.device) for g in groups];stats={}
    per_frame_dim,total_dim=validate_stage2_joint_layout(cache,linearisation_point)
    def anchors(parameters):
        pose=torch.cat([(parameters[f*per_frame_dim+STAGE1_PER_FRAME_DIM:(f+1)*per_frame_dim]-body_pose_anchors[f].to(parameters))*pose_anchor_scale for f in range(cache.frame_count)])
        beta,_=physical_beta_from_latent(parameters[-cache.model.beta_dim:],beta_abs_limit)
        shape=(beta-beta_anchor.to(parameters))*shape_anchor_scale
        return pose,shape
    def surrogate_fn(parameters):
        current=joint_fast_raw_groups(cache,parameters,beta_abs_limit)
        reproj=[g.residual_2d.reshape(-1)*s.to(parameters) for g,s in zip(current,scales)]
        pose,shape=anchors(parameters);return torch.cat((*reproj,pose,shape))
    def jacobian_fn(parameters):
        if cache.profile_timing and parameters.is_cuda:torch.cuda.synchronize(parameters.device)
        start=time.perf_counter();shape_state=evaluate_joint_shape_state(cache,parameters[-cache.model.beta_dim:],beta_abs_limit)
        blocks=[]
        for group,scale in zip(cache.groups,scales):
            result=joint_group_raw_residual_and_jacobians(cache,group,parameters,beta_abs_limit,shape_state)
            block=torch.zeros(result.raw_residual.numel(),total_dim,dtype=parameters.dtype,device=parameters.device)
            fs=slice(group.frame_position*per_frame_dim,(group.frame_position+1)*per_frame_dim)
            block[:,fs]=result.local_pose_jacobian*scale.to(parameters).unsqueeze(1)
            block[:,-cache.model.beta_dim:]=result.shared_beta_jacobian*scale.to(parameters).unsqueeze(1);blocks.append(block)
        _,pose_j=joint_pose_anchor_residual_and_jacobian(cache,parameters,body_pose_anchors,pose_anchor_scale)
        _,shape_j=joint_shape_anchor_residual_and_jacobian(parameters[-cache.model.beta_dim:],beta_anchor,beta_abs_limit,shape_anchor_scale,parameter_dimension=total_dim)
        value=torch.cat((*blocks,pose_j,shape_j))
        if cache.profile_timing and parameters.is_cuda:torch.cuda.synchronize(parameters.device)
        cache.explicit_jacobian_ms+=(time.perf_counter()-start)*1000;cache.explicit_jacobian_calls+=1
        return value
    def normal_equations_fn(parameters):return stage2_joint_arrowhead_normal_equations(cache,parameters,beta_abs_limit,scales,body_pose_anchors,beta_anchor,pose_anchor_scale,shape_anchor_scale)
    return {"surrogate_fn":surrogate_fn,"jacobian_fn":jacobian_fn,"normal_equations_fn":normal_equations_fn,
            "weight_stats":stats,"spec":aggregation_spec or {},"jacobian_backend":"analytic_stage2_joint","linearization_backend":"direct_normal_equations",
            "linear_solver_backend":"arrowhead_schur"}
