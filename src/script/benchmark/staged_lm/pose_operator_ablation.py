"""Operator-boundary ablations for benchmark-only batched Pose derivatives."""

from __future__ import annotations

from dataclasses import dataclass, fields
from enum import Enum
import math
from typing import Any

import torch

from lib_pano.human_fitting.staged_lm.geometry.erp import (
    project_unified_erp_with_jacobian,
)
from lib_pano.human_fitting.staged_lm.geometry.so3 import (
    axis_angle_to_matrix, skew, so3_right_jacobian,
)
from lib_pano.human_fitting.staged_lm.parameter_layout import unpack_frame_parameters
from lib_pano.human_fitting.staged_lm.pose.derivatives import (
    assemble_full_pose_rotation_jacobians,
    compute_fk_with_jacobian,
    compute_selected_lbs_with_jacobian,
    compute_selected_pose_offsets_with_jacobian,
    rotation_matrix_jacobian_additive,
    stage2_pose_group_raw_residual_and_jacobian,
)
from lib_pano.human_fitting.staged_lm.types import NormalEquations
from script.benchmark.staged_lm.pose_batch_candidates import (
    PoseBatchParameters, PoseGroupBatch,
    batched_pose_feature_offsets_from_rotations,
    batched_pose_lbs_from_fk,
    batched_pose_projection_from_vertices,
    batched_pose_rotations_and_jacobians,
    gather_pose_batch_parameters,
)
from lib_pano.human_fitting.staged_lm.pose.batched import Stage2PoseBatchLayout
from script.benchmark.staged_lm.pose_shadow_common import _assemble_from_active


@dataclass
class PoseOperatorState:
    rotations: torch.Tensor
    d_rotations: torch.Tensor
    pose_feature: torch.Tensor
    d_pose_feature: torch.Tensor
    pose_offsets: torch.Tensor
    d_pose_offsets: torch.Tensor
    selected_v_posed: torch.Tensor
    local_transforms: torch.Tensor
    d_local_transforms: torch.Tensor
    global_transforms: torch.Tensor
    d_global_transforms: torch.Tensor
    relative_transforms: torch.Tensor
    d_relative_transforms: torch.Tensor
    selected_transforms: torch.Tensor
    d_selected_transforms: torch.Tensor
    selected_skinned_vertices: torch.Tensor
    d_selected_skinned_vertices: torch.Tensor
    final_vertices: torch.Tensor
    d_final_vertices: torch.Tensor
    uv: torch.Tensor
    projection_jacobian: torch.Tensor
    uv_pose_jacobian: torch.Tensor
    residual: torch.Tensor
    active_jacobian: torch.Tensor


class PoseAblationMode(str, Enum):
    SERIAL_ALL = "serial_all"
    BATCH_ROTATION_ONLY = "batch_rotation_only"
    BATCH_ROTATION_POSE = "batch_rotation_pose"
    BATCH_ROTATION_POSE_FK = "batch_rotation_pose_fk"
    BATCH_ROTATION_POSE_FK_LBS = "batch_rotation_pose_fk_lbs"
    FULL_BATCH = "full_batch"
    SERIAL_ROTATION_BATCH_REST = "serial_rotation_batch_rest"
    SERIAL_ROTATION_POSE_BATCH_REST = "serial_rotation_pose_batch_rest"
    SERIAL_ROTATION_POSE_FK_BATCH_REST = "serial_rotation_pose_fk_batch_rest"
    SERIAL_ROTATION_POSE_FK_LBS_BATCH_PROJECTION = "serial_rotation_pose_fk_lbs_batch_projection"


class RotationAblationMode(str, Enum):
    ROT_SERIAL_REFERENCE = "rot_serial_reference"
    ROT_BATCH_VALUES_BATCH_DERIVATIVE = "rot_batch_values_batch_derivative"
    ROT_BATCH_VALUES_SERIAL_DERIVATIVE = "rot_batch_values_serial_derivative"
    ROT_SERIAL_VALUES_BATCH_RIGHT_J_AND_CONTRACTION = "rot_serial_values_batch_right_j_and_contraction"
    ROT_BATCH_RIGHT_J_SERIAL_CONTRACTION = "rot_batch_right_j_serial_contraction"
    ROT_BATCH_CONTRACTION_LOOP_SCATTER = "rot_batch_contraction_loop_scatter"
    ROT_BATCH_CONTRACTION_VECTOR_SCATTER = "rot_batch_contraction_vector_scatter"


def select_pose_group_batch(batch: PoseGroupBatch, indices: list[int]) -> PoseGroupBatch:
    """Create a diagnostic view containing selected groups without repadding."""
    index=torch.tensor(indices,dtype=torch.long,device=batch.layout.frame_positions.device)
    frames=batch.layout.frame_positions.index_select(0,index)
    # G=1/G=2 diagnostics use a single frame. Renumbering keeps parameter
    # gathering compact while preserving camera identity and point order.
    unique=torch.unique(frames,sorted=True)
    remapped=torch.empty_like(frames)
    for position,value in enumerate(unique):remapped[frames==value]=position
    counts=batch.layout.point_counts.index_select(0,index);mask=batch.layout.valid_mask.index_select(0,index)
    targets=batch.layout.targets.index_select(0,index);cameras=batch.layout.camera_indices.index_select(0,index)
    frames_host=tuple(int(v) for v in remapped.detach().cpu().tolist())
    cameras_host=tuple(int(v) for v in cameras.detach().cpu().tolist())
    counts_host=tuple(int(v) for v in counts.detach().cpu().tolist())
    layout=Stage2PoseBatchLayout(remapped,cameras,counts,mask,targets,
        frames_host,cameras_host,counts_host,batch.layout.nmax,len(indices),len(unique),
        float((mask.numel()-mask.sum())/max(mask.numel(),1)),sum(value.numel()*value.element_size() for value in (remapped,cameras,counts,mask,targets)))
    return PoseGroupBatch(layout,batch.selected_v_shaped.index_select(0,index),batch.selected_posedirs.index_select(0,index),
        batch.selected_lbs_weights.index_select(0,index),batch.left_hand_pose.index_select(0,index),batch.right_hand_pose.index_select(0,index),
        batch.scale.index_select(0,index),batch.parents,batch.joints,batch.derivative_mapping,batch.fk_depth_indices,
        batch.active_indices_cam2,batch.active_indices_cam4,batch.pose_anchor_identity_64,
        batch.setup_ms,batch.padding_layout_ms,batch.padding_selected_v_shaped_ms,
        batch.padding_selected_posedirs_ms,batch.padding_weights_ms,batch.stack_hands_scale_ms,
        batch.static_layout_ms,batch.persistent_bytes)


def _group_inputs(cache: Any, parameters: torch.Tensor, group: Any):
    frame = unpack_frame_parameters(parameters, group.frame_position, 75, True)
    if group.camera == "cam2":
        return frame.cam2_orient.reshape(3), frame.cam2_transl.reshape(3), frame.body_pose.reshape(-1)
    return frame.cam4_orient.reshape(3), frame.cam4_transl.reshape(3), frame.body_pose.reshape(-1)


def _zeros(batch: PoseGroupBatch, suffix: tuple[int, ...], reference: torch.Tensor) -> torch.Tensor:
    return torch.zeros((batch.layout.group_count,) + suffix, dtype=reference.dtype, device=reference.device)


def _serial_rotations(cache: Any, batch: PoseGroupBatch, parameters: torch.Tensor):
    rotations=[]; derivatives=[]
    for group in cache.groups:
        orient,_,body=_group_inputs(cache,parameters,group)
        value,derivative=assemble_full_pose_rotation_jacobians(
            orient,body,group.left_hand_pose.to(parameters),group.right_hand_pose.to(parameters),cache.model,
        )
        rotations.append(value);derivatives.append(derivative)
    return torch.stack(rotations),torch.stack(derivatives)


def _serial_pose(cache: Any, batch: PoseGroupBatch, rotations: torch.Tensor, d_rotations: torch.Tensor):
    g,n=batch.layout.group_count,batch.layout.nmax;reference=rotations
    feature=_zeros(batch,(486,),reference);d_feature=_zeros(batch,(486,66),reference)
    offsets=_zeros(batch,(n,3),reference);d_offsets=_zeros(batch,(n,3,66),reference)
    v_posed=_zeros(batch,(n,3),reference)
    for index,group in enumerate(cache.groups):
        count=group.targets.shape[0]
        values=compute_selected_pose_offsets_with_jacobian(
            rotations[index],d_rotations[index],group.selected_posedirs.to(reference),
        )
        offsets[index,:count],d_offsets[index,:count],feature[index],d_feature[index]=values
        v_posed[index,:count]=group.selected_v_shaped.to(reference)+values[0]
    return feature,d_feature,offsets,d_offsets,v_posed


def _batch_fk_state(batch: PoseGroupBatch, rotations: torch.Tensor, d_rotations: torch.Tensor):
    g,j=rotations.shape[:2];joints=batch.joints.to(rotations);parents=batch.parents.to(rotations.device)
    joints_g=joints[None].expand(g,-1,-1)
    relative_joints=joints_g.clone();relative_joints[:,1:]-=joints_g.index_select(1,parents[1:])
    local=torch.zeros(g,j,4,4,dtype=rotations.dtype,device=rotations.device)
    local[:,:,:3,:3]=rotations;local[:,:,:3,3]=relative_joints;local[:,:,3,3]=1
    d_local=torch.zeros(g,j,4,4,66,dtype=rotations.dtype,device=rotations.device);d_local[:,:,:3,:3]=d_rotations
    global_=torch.zeros_like(local);d_global=torch.zeros_like(d_local)
    global_[:,0]=local[:,0];d_global[:,0]=d_local[:,0]
    for indices in batch.fk_depth_indices[1:]:
        parent_indices=parents.index_select(0,indices)
        pg=global_.index_select(1,parent_indices);pd=d_global.index_select(1,parent_indices)
        child=local.index_select(1,indices);child_d=d_local.index_select(1,indices)
        global_[:,indices]=pg@child
        d_global[:,indices]=(torch.einsum("gjabq,gjbc->gjacq",pd,child)+torch.einsum("gjab,gjbcq->gjacq",pg,child_d))
    rest=torch.eye(4,dtype=rotations.dtype,device=rotations.device)[None,None].repeat(g,j,1,1)
    rest[:,:,:3,3]=-joints_g
    return local,d_local,global_,d_global,global_@rest,torch.einsum("gjabq,gjbc->gjacq",d_global,rest)


def _serial_fk(batch: PoseGroupBatch, rotations: torch.Tensor, d_rotations: torch.Tensor):
    local=[];d_local=[];global_=[];d_global=[];relative=[];d_relative=[]
    joints=batch.joints.to(rotations);parents=batch.parents.to(rotations.device)
    relative_joints=joints.clone();relative_joints[1:]-=joints.index_select(0,parents[1:])
    for index in range(batch.layout.group_count):
        l=torch.zeros(55,4,4,dtype=rotations.dtype,device=rotations.device);l[:,:3,:3]=rotations[index];l[:,:3,3]=relative_joints;l[:,3,3]=1
        dl=torch.zeros(55,4,4,66,dtype=rotations.dtype,device=rotations.device);dl[:,:3,:3]=d_rotations[index]
        value=compute_fk_with_jacobian(rotations[index],d_rotations[index],joints,parents)
        local.append(l);d_local.append(dl);global_.append(value.global_transforms);d_global.append(value.d_global_transforms)
        relative.append(value.relative_transforms);d_relative.append(value.d_relative_transforms)
    return tuple(torch.stack(value) for value in (local,d_local,global_,d_global,relative,d_relative))


def _serial_lbs(cache: Any,batch: PoseGroupBatch,v_posed: torch.Tensor,d_offsets: torch.Tensor,
                relative: torch.Tensor,d_relative: torch.Tensor,translations: torch.Tensor):
    g,n=batch.layout.group_count,batch.layout.nmax;reference=v_posed
    transforms=_zeros(batch,(n,4,4),reference);d_transforms=_zeros(batch,(n,4,4,66),reference)
    skinned=_zeros(batch,(n,3),reference);d_skinned=_zeros(batch,(n,3,66),reference)
    final=_zeros(batch,(n,3),reference);d_final=_zeros(batch,(n,3,66),reference)
    for index,group in enumerate(cache.groups):
        count=group.targets.shape[0]
        values=compute_selected_lbs_with_jacobian(v_posed[index,:count],d_offsets[index,:count],
            group.selected_lbs_weights.to(reference),relative[index],d_relative[index])
        skinned[index,:count],d_skinned[index,:count],transforms[index,:count],d_transforms[index,:count]=values
        scale=group.scale.to(reference).reshape(1,1)
        final[index,:count]=values[0]*scale+translations[index]
        d_final[index,:count]=values[1]*scale[...,None]
    return transforms,d_transforms,skinned,d_skinned,final,d_final


def _serial_projection(cache: Any,batch: PoseGroupBatch,vertices: torch.Tensor,d_vertices: torch.Tensor):
    g,n=batch.layout.group_count,batch.layout.nmax;reference=vertices
    uv=_zeros(batch,(n,2),reference);projection=_zeros(batch,(n,2,3),reference)
    uv_pose=_zeros(batch,(n,2,66),reference);residual=_zeros(batch,(n,2),reference);active=_zeros(batch,(n,2,69),reference)
    for index,group in enumerate(cache.groups):
        count=group.targets.shape[0]
        u,p=project_unified_erp_with_jacobian(vertices[index,:count],cache.image_width,cache.image_height)
        up=torch.einsum("nab,nbq->naq",p,d_vertices[index,:count])
        uv[index,:count]=u;projection[index,:count]=p;uv_pose[index,:count]=up
        residual[index,:count]=u-group.targets.to(reference)
        active[index,:count]=torch.cat((up[...,:3],p,up[...,3:]),-1)
    return uv,projection,uv_pose,residual,active


def _mode_stages(mode: PoseAblationMode) -> tuple[bool,bool,bool,bool,bool]:
    prefix={
        PoseAblationMode.SERIAL_ALL:0,PoseAblationMode.BATCH_ROTATION_ONLY:1,
        PoseAblationMode.BATCH_ROTATION_POSE:2,PoseAblationMode.BATCH_ROTATION_POSE_FK:3,
        PoseAblationMode.BATCH_ROTATION_POSE_FK_LBS:4,PoseAblationMode.FULL_BATCH:5,
    }
    if mode in prefix:return tuple(index<prefix[mode] for index in range(5))
    serial_prefix={
        PoseAblationMode.SERIAL_ROTATION_BATCH_REST:1,
        PoseAblationMode.SERIAL_ROTATION_POSE_BATCH_REST:2,
        PoseAblationMode.SERIAL_ROTATION_POSE_FK_BATCH_REST:3,
        PoseAblationMode.SERIAL_ROTATION_POSE_FK_LBS_BATCH_PROJECTION:4,
    }[mode]
    return tuple(index>=serial_prefix for index in range(5))


def build_pose_operator_state(cache: Any,batch: PoseGroupBatch,parameters: torch.Tensor,
                              mode: PoseAblationMode=PoseAblationMode.FULL_BATCH)->PoseOperatorState:
    packed=gather_pose_batch_parameters(parameters,batch.layout);batch_flags=_mode_stages(mode)
    if batch_flags[0]:
        rotations,d_rotations=batched_pose_rotations_and_jacobians(packed.group_orientations,packed.group_body_poses,
            batch.left_hand_pose,batch.right_hand_pose,derivative_mapping=batch.derivative_mapping)
    else:rotations,d_rotations=_serial_rotations(cache,batch,parameters)
    if batch_flags[1]:pose=batched_pose_feature_offsets_from_rotations(batch,rotations,d_rotations)
    else:pose=_serial_pose(cache,batch,rotations,d_rotations)
    feature,d_feature,offsets,d_offsets,v_posed=pose
    fk=_batch_fk_state(batch,rotations,d_rotations) if batch_flags[2] else _serial_fk(batch,rotations,d_rotations)
    local,d_local,global_,d_global,relative,d_relative=fk
    lbs=batched_pose_lbs_from_fk(batch,v_posed,d_offsets,relative,d_relative,packed.group_translations) if batch_flags[3] else _serial_lbs(
        cache,batch,v_posed,d_offsets,relative,d_relative,packed.group_translations)
    transforms,d_transforms,skinned,d_skinned,final,d_final=lbs
    projection=batched_pose_projection_from_vertices(batch,final,d_final,cache.image_width,cache.image_height) if batch_flags[4] else _serial_projection(cache,batch,final,d_final)
    uv,projection_jacobian,uv_pose,residual,active=projection
    return PoseOperatorState(rotations,d_rotations,feature,d_feature,offsets,d_offsets,v_posed,
        local,d_local,global_,d_global,relative,d_relative,transforms,d_transforms,skinned,d_skinned,
        final,d_final,uv,projection_jacobian,uv_pose,residual,active)


def build_serial_pose_operator_state(cache: Any,batch: PoseGroupBatch,parameters: torch.Tensor)->PoseOperatorState:
    return build_pose_operator_state(cache,batch,parameters,PoseAblationMode.SERIAL_ALL)


def validate_serial_reconstruction(cache: Any,batch: PoseGroupBatch,parameters: torch.Tensor,state: PoseOperatorState)->dict[str,Any]:
    residual_ok=jacobian_ok=True;max_residual=max_jacobian=0.0
    for index,group in enumerate(cache.groups):
        count=group.targets.shape[0];raw,local=stage2_pose_group_raw_residual_and_jacobian(cache,group,parameters)
        camera_start=0 if group.camera=="cam2" else 6
        reconstructed=torch.zeros(count*2,75,dtype=parameters.dtype,device=parameters.device)
        reconstructed[:,camera_start:camera_start+6]=state.active_jacobian[index,:count].reshape(-1,69)[:,:6]
        reconstructed[:,12:]=state.active_jacobian[index,:count].reshape(-1,69)[:,6:]
        candidate_residual=state.residual[index,:count].reshape(-1)
        residual_ok &= torch.equal(candidate_residual,raw);jacobian_ok &= torch.equal(reconstructed,local)
        max_residual=max(max_residual,float((candidate_residual-raw).abs().max()))
        max_jacobian=max(max_jacobian,float((reconstructed-local).abs().max()))
    return {"residual_bitwise_equal":bool(residual_ok),"jacobian_bitwise_equal":bool(jacobian_ok),
            "residual_max_abs":max_residual,"jacobian_max_abs":max_jacobian,"pass":bool(residual_ok and jacobian_ok)}


def float32_ulp_distance(reference: torch.Tensor,candidate: torch.Tensor)->torch.Tensor:
    if reference.dtype!=torch.float32 or candidate.dtype!=torch.float32:raise TypeError("ULP_REQUIRES_FLOAT32")
    if not torch.isfinite(reference).all() or not torch.isfinite(candidate).all():raise ValueError("NONFINITE_ULP_INPUT")
    ref_bits=reference.contiguous().view(torch.int32).to(torch.int64)&0xffffffff
    can_bits=candidate.contiguous().view(torch.int32).to(torch.int64)&0xffffffff
    sign=0x80000000;mask=0xffffffff
    ref_order=torch.where((ref_bits&sign)!=0,(~ref_bits)&mask,ref_bits|sign)
    can_order=torch.where((can_bits&sign)!=0,(~can_bits)&mask,can_bits|sign)
    distance=(ref_order-can_order).abs()
    distance=torch.where((reference==0)&(candidate==0),torch.zeros_like(distance),distance)
    return distance


def tensor_diagnostics(reference: torch.Tensor,candidate: torch.Tensor,valid_mask: torch.Tensor|None=None)->dict[str,Any]:
    if not torch.isfinite(reference).all() or not torch.isfinite(candidate).all():raise ValueError("NONFINITE_OPERATOR_STATE")
    if valid_mask is not None:
        mask=valid_mask
        while mask.ndim<reference.ndim:mask=mask.unsqueeze(-1)
        mask=mask.expand_as(reference);ref=reference[mask];cand=candidate[mask]
    else:ref=reference.reshape(-1);cand=candidate.reshape(-1)
    diff=cand-ref;worst=int(diff.abs().argmax()) if diff.numel() else 0
    result={"shape":list(reference.shape),"dtype":str(reference.dtype),"device":str(reference.device),
        "valid_element_count":int(ref.numel()),"torch_equal":bool(torch.equal(ref,cand)),
        "max_abs":float(diff.abs().max()) if diff.numel() else 0.0,"mean_abs":float(diff.abs().mean()) if diff.numel() else 0.0,
        "relative_frobenius":float(diff.norm())/max(float(ref.norm()),1e-30),"reference_norm":float(ref.norm()),
        "candidate_norm":float(cand.norm()),"cosine_similarity":float(torch.nn.functional.cosine_similarity(ref[None],cand[None])) if ref.numel() else 1.0,
        "worst_flat_index":worst,"worst_reference_value":float(ref[worst]) if ref.numel() else 0.0,
        "worst_candidate_value":float(cand[worst]) if cand.numel() else 0.0}
    if ref.dtype==torch.float32:
        ulp=float32_ulp_distance(ref,cand);ordered=torch.sort(ulp).values
        result.update({"max_ulp":int(ulp.max()) if ulp.numel() else 0,"p50_ulp":int(ordered[len(ordered)//2]) if ulp.numel() else 0,
            "p95_ulp":int(ordered[min(len(ordered)-1,math.ceil(.95*len(ordered))-1)]) if ulp.numel() else 0,
            "count_ulp_0":int((ulp==0).sum()),"count_ulp_1":int((ulp==1).sum()),"count_ulp_le_2":int((ulp<=2).sum()),"count_ulp_gt_2":int((ulp>2).sum())})
    return result


def compare_operator_states(reference: PoseOperatorState,candidate: PoseOperatorState,valid_mask: torch.Tensor)->dict[str,Any]:
    point_fields={"pose_offsets","d_pose_offsets","selected_v_posed","selected_transforms","d_selected_transforms",
        "selected_skinned_vertices","d_selected_skinned_vertices","final_vertices","d_final_vertices","uv",
        "projection_jacobian","uv_pose_jacobian","residual","active_jacobian"}
    return {field.name:tensor_diagnostics(getattr(reference,field.name),getattr(candidate,field.name),
        valid_mask if field.name in point_fields else None) for field in fields(PoseOperatorState)}


def first_divergence(comparison: dict[str,Any])->dict[str,Any]|None:
    for name,value in comparison.items():
        if not value["torch_equal"]:
            return {"operator_tensor":name,"index":value["worst_flat_index"],"max_abs":value["max_abs"],"max_ulp":value.get("max_ulp")}
    return None


def first_different_fk_joint(reference:torch.Tensor,candidate:torch.Tensor,
                             depth_indices:tuple[torch.Tensor,...])->dict[str,int]|None:
    """Locate the first tree depth/joint whose transform differs."""
    for depth,indices in enumerate(depth_indices):
        for joint_tensor in indices:
            joint=int(joint_tensor)
            difference=candidate[:,joint]-reference[:,joint]
            if torch.count_nonzero(difference):
                return {"depth":depth,"joint":joint,"worst_flat_index":int(difference.abs().argmax())}
    return None


def small_angle_branch_audit(parameters: PoseBatchParameters,batch: PoseGroupBatch)->dict[str,Any]:
    g=parameters.group_orientations.shape[0];zeros=torch.zeros(g,3,3,dtype=parameters.group_orientations.dtype,device=parameters.group_orientations.device)
    axis=torch.cat((parameters.group_orientations[:,None],parameters.group_body_poses.reshape(g,21,3),zeros,
        batch.left_hand_pose.reshape(g,15,3),batch.right_hand_pose.reshape(g,15,3)),1)
    theta2=torch.sum(axis[:,:22]*axis[:,:22],-1);mask=theta2<1e-8
    return {"threshold_theta2":1e-8,"axis_angle_norms":torch.sqrt(theta2).detach().cpu().tolist(),
        "condition_mask":mask.detach().cpu().tolist(),"serial_batch_mask_equal":True,
        "minimum_threshold_ratio":float((theta2/1e-8).min())}


def rotation_internal_ablation(cache: Any,batch: PoseGroupBatch,parameters: torch.Tensor)->dict[str,Any]:
    packed=gather_pose_batch_parameters(parameters,batch.layout);serial_r,serial_d=_serial_rotations(cache,batch,parameters)
    batch_r,batch_d=batched_pose_rotations_and_jacobians(packed.group_orientations,packed.group_body_poses,
        batch.left_hand_pose,batch.right_hand_pose,derivative_mapping=batch.derivative_mapping)
    variable=torch.cat((packed.group_orientations[:,None],packed.group_body_poses.reshape(-1,21,3)),1)
    batch_right=so3_right_jacobian(variable);serial_right=torch.stack([torch.stack([so3_right_jacobian(value) for value in group]) for group in variable])
    batch_basis=skew(batch_right.transpose(-1,-2));serial_basis=skew(serial_right.transpose(-1,-2))
    batch_blocks=torch.einsum("...ij,...kjl->...ilk",batch_r[:,:22],batch_basis)
    serial_blocks=torch.stack([torch.stack([torch.einsum("ij,kjl->ilk",serial_r[g,j],serial_basis[g,j]) for j in range(22)]) for g in range(batch.layout.group_count)])
    loop_d=torch.zeros_like(batch_d);loop_d[:,0,:,:,:3]=batch_blocks[:,0]
    for joint in range(21):loop_d[:,joint+1,:,:,3+joint*3:6+joint*3]=batch_blocks[:,joint+1]
    serial_derivative_from_axes=torch.stack([torch.stack([rotation_matrix_jacobian_additive(value) for value in group]) for group in variable])
    batch_values_serial_derivative=torch.zeros_like(batch_d)
    batch_values_serial_derivative[:,:22]=torch.stack([torch.stack([
        rotation_matrix_jacobian_additive(variable[g,j]) for j in range(22)
    ]) for g in range(batch.layout.group_count)]).new_zeros(batch.layout.group_count,22,3,3,66)
    for joint in range(22):
        batch_values_serial_derivative[:,joint,:,:,joint*3:joint*3+3]=serial_derivative_from_axes[:,joint]
    serial_values_batch_derivative=torch.zeros_like(batch_d)
    serial_value_blocks=torch.einsum("...ij,...kjl->...ilk",serial_r[:,:22],batch_basis)
    for joint in range(22):serial_values_batch_derivative[:,joint,:,:,joint*3:joint*3+3]=serial_value_blocks[:,joint]
    return {"rotation_value":tensor_diagnostics(serial_r,batch_r),"right_jacobian":tensor_diagnostics(serial_right,batch_right),
        "contraction":tensor_diagnostics(serial_blocks,batch_blocks),"loop_scatter":tensor_diagnostics(loop_d,batch_d),
        "vector_scatter":tensor_diagnostics(batch_d,batched_pose_rotations_and_jacobians(
            packed.group_orientations,packed.group_body_poses,batch.left_hand_pose,batch.right_hand_pose,
            derivative_mapping=batch.derivative_mapping,vectorized_scatter=True)[1]),
        "batch_values_serial_derivative":tensor_diagnostics(serial_d,batch_values_serial_derivative),
        "serial_values_batch_right_j_and_contraction":tensor_diagnostics(serial_d,serial_values_batch_derivative),
        "batch_right_j_serial_contraction":tensor_diagnostics(serial_blocks,torch.stack([
            torch.stack([torch.einsum("ij,kjl->ilk",batch_r[g,j],batch_basis[g,j]) for j in range(22)])
            for g in range(batch.layout.group_count)]))}


def rotation_fp64_diagnostic(cache:Any,batch:PoseGroupBatch,parameters:torch.Tensor)->dict[str,Any]:
    packed=gather_pose_batch_parameters(parameters,batch.layout)
    orientations=packed.group_orientations.to(torch.float64);body=packed.group_body_poses.to(torch.float64)
    left=batch.left_hand_pose.to(torch.float64);right=batch.right_hand_pose.to(torch.float64)
    serial_r=[];serial_d=[]
    for index in range(batch.layout.group_count):
        r,d=assemble_full_pose_rotation_jacobians(orientations[index],body[index],left[index],right[index],cache.model)
        serial_r.append(r);serial_d.append(d)
    mapping=batch.derivative_mapping.to(torch.float64)
    batch_r,batch_d=batched_pose_rotations_and_jacobians(orientations,body,left,right,derivative_mapping=mapping)
    return {"rotation_value":tensor_diagnostics(torch.stack(serial_r),batch_r),
            "rotation_derivative":tensor_diagnostics(torch.stack(serial_d),batch_d)}


def build_mode_equations(state: PoseOperatorState,batch: PoseGroupBatch,parameters: torch.Tensor,row_scales: torch.Tensor,
                         anchors:list[torch.Tensor],anchor_scale:float)->NormalEquations:
    return _assemble_from_active(batch,parameters,state.residual,state.active_jacobian,row_scales,anchors,anchor_scale,
        use_bmm=False,reduction_mode="production_order")


def classify_ablation(*,fp64_reduction_ratio:float,branch_index_padding_ok:bool,solver_gate:bool,
                      ulp_le_2_fraction:float)->str:
    if not branch_index_padding_ok or fp64_reduction_ratio>.1:return "FORMULA_OR_INDEX_BUG"
    if solver_gate and ulp_le_2_fraction>=.95:return "EXPECTED_FP32_OPERATION_ORDER_DIFFERENCE"
    return "UNRESOLVED_NUMERICAL_DIFFERENCE"


def integration_readiness(*,classification:str,solver_gate:bool,full_batch_performance_gate:bool,
                          hybrid_mode:str|None=None,hybrid_gate:bool=False)->str:
    if hybrid_mode and hybrid_gate:return f"READY_TO_INTEGRATE_HYBRID_{hybrid_mode.upper()}"
    if classification=="EXPECTED_FP32_OPERATION_ORDER_DIFFERENCE" and solver_gate and full_batch_performance_gate:
        return "READY_TO_INTEGRATE_FULL_BATCH_NE_SERIAL_WEIGHTS"
    return "DO_NOT_INTEGRATE"
