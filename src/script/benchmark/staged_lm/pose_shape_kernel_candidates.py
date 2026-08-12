"""Diagnostic-only Stage 2 joint analytic H/g kernel candidates.

Nothing in this module is imported by the production stage builders.  The
functions intentionally keep timing, padding experiments, and alternative
contractions isolated from the validated solver path.
"""

from __future__ import annotations

from dataclasses import dataclass
import statistics
import time
from typing import Any, Callable

import torch

from lib_pano.human_fitting.staged_lm.geometry.erp import (
    project_unified_erp_with_jacobian,
)
from lib_pano.human_fitting.staged_lm.geometry.so3 import (
    axis_angle_to_matrix, skew, so3_right_jacobian,
)
from lib_pano.human_fitting.staged_lm.config import STAGE1_PER_FRAME_DIM
from lib_pano.human_fitting.staged_lm.irls import compute_primary_charbonnier_tensor
from lib_pano.human_fitting.staged_lm.parameter_layout import unpack_frame_parameters
from lib_pano.human_fitting.staged_lm.pose_shape import (
    Stage2PoseShapeFastCache, Stage2PoseShapeFrameCameraCache, Stage2PoseShapeState,
    evaluate_pose_shape_state, physical_beta_from_latent,
)
from lib_pano.human_fitting.staged_lm.pose_shape import (
    CombinedFKResult, PoseShapeGroupJacobianResult,
    compute_pose_shape_fk_with_combined_jacobian,
    pose_shape_group_raw_residual_and_jacobians,
)
from script.validation.staged_lm.references.pose_shape_backends import (
    compute_pose_shape_fk_with_pose_and_beta_jacobians_separate_reference,
)
from lib_pano.human_fitting.staged_lm.arrowhead_schur import ArrowheadNormalEquationSystem
from lib_pano.human_fitting.staged_lm.types import NormalEquations
from lib_pano.human_fitting.staged_lm.types import ReprojectionGroup
from lib_pano.human_fitting.staged_lm.pose.derivatives import (
    assemble_full_pose_rotation_jacobians,
    compute_selected_lbs_with_jacobian,
    compute_selected_pose_offsets_with_jacobian,
)


def _sync(reference: torch.Tensor) -> None:
    if reference.is_cuda:
        torch.cuda.synchronize(reference.device)


def _timed(reference: torch.Tensor, fn: Callable[[], Any]) -> tuple[Any, float]:
    _sync(reference)
    start = time.perf_counter()
    value = fn()
    _sync(reference)
    return value, (time.perf_counter() - start) * 1000.0


def tensor_bytes(tensor: torch.Tensor) -> int:
    return tensor.numel() * tensor.element_size()


@dataclass
class GroupProfile:
    result: PoseShapeGroupJacobianResult
    section_ms: dict[str, float]
    tensors: dict[str, torch.Tensor]


@dataclass
class HGContribution:
    frame_hessian: torch.Tensor
    frame_beta_cross: torch.Tensor
    beta_hessian: torch.Tensor
    frame_gradient: torch.Tensor
    beta_gradient: torch.Tensor
    residual: torch.Tensor


@dataclass
class JointGroupBatch:
    valid_mask: torch.Tensor
    point_counts: torch.Tensor
    selected_base_v: torch.Tensor
    selected_beta_dirs: torch.Tensor
    selected_posedirs: torch.Tensor
    selected_weights: torch.Tensor
    targets: torch.Tensor
    left_hand_pose: torch.Tensor
    right_hand_pose: torch.Tensor
    scale: torch.Tensor
    frame_positions: torch.Tensor
    camera_ids: torch.Tensor

    @property
    def padded_bytes(self) -> int:
        return sum(tensor_bytes(value) for value in self.__dict__.values()
                   if isinstance(value, torch.Tensor))


def _group_parameters(
    cache: Stage2PoseShapeFastCache, group: Stage2PoseShapeFrameCameraCache,
    parameters: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, int]:
    per_frame_dim = STAGE1_PER_FRAME_DIM + cache.model.body_pose_dim
    frame = unpack_frame_parameters(parameters, group.frame_position, per_frame_dim, True)
    if group.camera == "cam2":
        return frame.cam2_orient.reshape(3), frame.cam2_transl.reshape(3), frame.body_pose.reshape(-1), 0, per_frame_dim
    return frame.cam4_orient.reshape(3), frame.cam4_transl.reshape(3), frame.body_pose.reshape(-1), 6, per_frame_dim


def profile_pose_shape_group_baseline(
    cache: Stage2PoseShapeFastCache, group: Stage2PoseShapeFrameCameraCache,
    shape_state: Stage2PoseShapeState, parameters: torch.Tensor,
    row_scale: torch.Tensor | None = None,
) -> GroupProfile:
    """Reproduce the production group kernel with section-level timings."""
    sections: dict[str, float] = {}
    tensors: dict[str, torch.Tensor] = {}
    (orient, translation, body_pose, camera_column, per_frame_dim), sections["A_unpack_prepare"] = _timed(
        parameters, lambda: _group_parameters(cache, group, parameters),
    )
    (rotations, d_rotations), sections["B_rotations_jacobian"] = _timed(
        parameters, lambda: assemble_full_pose_rotation_jacobians(
            orient, body_pose, group.left_hand_pose.to(parameters),
            group.right_hand_pose.to(parameters), cache.model,
        ),
    )
    tensors.update(rotations=rotations, d_rotations=d_rotations)
    (pose_offsets, d_pose_offsets, pose_feature, d_pose_feature), sections["C_pose_feature_offsets"] = _timed(
        parameters, lambda: compute_selected_pose_offsets_with_jacobian(
            rotations, d_rotations, group.selected_posedirs.to(parameters),
        ),
    )
    tensors.update(
        pose_feature=pose_feature, d_pose_feature=d_pose_feature,
        selected_pose_offsets=pose_offsets, d_pose_offsets=d_pose_offsets,
    )

    def dynamic_shape():
        beta_dirs = group.selected_beta_shapedirs.to(parameters)
        selected_v_shaped = group.selected_base_v_shaped.to(parameters) + torch.einsum(
            "ncb,b->nc", beta_dirs, shape_state.beta,
        )
        d_v_beta = beta_dirs * shape_state.d_beta_d_latent.reshape(1, 1, -1)
        return selected_v_shaped, d_v_beta, selected_v_shaped + pose_offsets
    (selected_v_shaped, d_v_beta, selected_v_posed), sections["D_dynamic_shape"] = _timed(parameters, dynamic_shape)
    tensors.update(selected_v_shaped=selected_v_shaped, d_v_beta=d_v_beta, selected_v_posed=selected_v_posed)
    fk, sections["E_combined_production_fk"] = _timed(
        parameters, lambda: compute_pose_shape_fk_with_combined_jacobian(
            rotations, d_rotations, shape_state.joints,
            shape_state.d_joints_d_latent, cache.model.parents.to(parameters.device),
        ),
    )
    tensors.update(
        global_transforms=fk.global_transforms,
        d_global_pose=fk.d_global_transforms_d_pose,
        d_global_beta=fk.d_global_transforms_d_beta,
        relative_transforms=fk.relative_transforms,
        d_relative_pose=fk.d_relative_transforms_d_pose,
        d_relative_beta=fk.d_relative_transforms_d_beta,
    )
    (combined_v, combined_a), sections["F_derivative_concat"] = _timed(
        parameters, lambda: (
            torch.cat((d_pose_offsets, d_v_beta), -1),
            torch.cat((fk.d_relative_transforms_d_pose, fk.d_relative_transforms_d_beta), -1),
        ),
    )
    tensors.update(combined_v=combined_v, combined_a=combined_a)
    (skinned, d_skinned, transforms, d_transforms), sections["G_selected_lbs"] = _timed(
        parameters, lambda: compute_selected_lbs_with_jacobian(
            selected_v_posed, combined_v, group.selected_lbs_weights.to(parameters),
            fk.relative_transforms, combined_a,
        ),
    )
    tensors.update(
        selected_transforms=transforms, d_selected_transforms=d_transforms,
        selected_skinned_vertices=skinned, d_skinned=d_skinned,
    )
    pose_dim = 3 + cache.model.body_pose_dim

    def finish_vertices():
        scale = group.scale.to(parameters).reshape(1, 1)
        return (skinned * scale + translation.reshape(1, 3),
                d_skinned[..., :pose_dim] * scale.unsqueeze(-1),
                d_skinned[..., pose_dim:] * scale.unsqueeze(-1))
    (vertices, d_pose, d_beta), sections["H_scale_translation"] = _timed(parameters, finish_vertices)
    (uv, projection_j), sections["I_erp_projection"] = _timed(
        parameters, lambda: project_unified_erp_with_jacobian(vertices, cache.image_width, cache.image_height),
    )
    tensors["projection_jacobian"] = projection_j
    (uv_pose, uv_beta), sections["J_projection_chain"] = _timed(
        parameters, lambda: (
            torch.einsum("nab,nbq->naq", projection_j, d_pose),
            torch.einsum("nab,nbq->naq", projection_j, d_beta),
        ),
    )
    tensors.update(uv_pose=uv_pose, uv_beta=uv_beta)

    def materialize_local():
        local = torch.zeros(vertices.shape[0], 2, per_frame_dim,
                            dtype=parameters.dtype, device=parameters.device)
        local[:, :, camera_column:camera_column+3] = uv_pose[:, :, :3]
        local[:, :, camera_column+3:camera_column+6] = projection_j
        local[:, :, STAGE1_PER_FRAME_DIM:] = uv_pose[:, :, 3:]
        return local.reshape(-1, per_frame_dim), uv_beta.reshape(-1, cache.model.beta_dim)
    (local_j, beta_j), sections["K_local_j_materialize"] = _timed(parameters, materialize_local)
    tensors.update(local_pose_jacobian=local_j, shared_beta_jacobian=beta_j)
    residual = (uv - group.targets.to(parameters)).reshape(-1)
    scale_rows = (torch.ones_like(residual) if row_scale is None else row_scale.to(parameters))
    (scaled_r, scaled_jf, scaled_jb), sections["L_row_scaling"] = _timed(
        parameters, lambda: (
            residual * scale_rows,
            local_j * scale_rows.unsqueeze(1),
            beta_j * scale_rows.unsqueeze(1),
        ),
    )
    tensors.update(scaled_J_frame_FP32=scaled_jf, scaled_J_beta_FP32=scaled_jb)
    (r64, jf64, jb64), sections["M_fp64_conversion"] = _timed(
        parameters, lambda: (scaled_r.to(torch.float64), scaled_jf.to(torch.float64), scaled_jb.to(torch.float64)),
    )
    tensors.update(scaled_J_frame_FP64=jf64, scaled_J_beta_FP64=jb64)
    (hff, hfb, hbb), sections["N_hessian_gemm"] = _timed(
        parameters, lambda: (jf64.T @ jf64, jf64.T @ jb64, jb64.T @ jb64),
    )
    (gf, gb), sections["O_gradient_gemv"] = _timed(
        parameters, lambda: (jf64.T @ r64, jb64.T @ r64),
    )

    def block_update():
        a = torch.zeros_like(hff); b = torch.zeros_like(hfb); c = torch.zeros_like(hbb)
        ga = torch.zeros_like(gf); gc = torch.zeros_like(gb)
        a += hff; b += hfb; c += hbb; ga += gf; gc += gb
        return a, b, c, ga, gc
    _, sections["P_arrowhead_update"] = _timed(parameters, block_update)
    result = PoseShapeGroupJacobianResult(residual, local_j, beta_j)
    return GroupProfile(result, sections, tensors)


def profile_pose_shape_group_production_combined(*args, **kwargs) -> GroupProfile:
    """Explicitly named production baseline used by current benchmark reports."""
    return profile_pose_shape_group_baseline(*args, **kwargs)


def profile_joint_separate_fk_reference(
    rotations: torch.Tensor, d_rotations: torch.Tensor,
    shape_state: Stage2PoseShapeState, parents: torch.Tensor,
) -> tuple[CombinedFKResult, float]:
    """REFERENCE_ONLY: time the historical separate pose/beta recursion."""
    return _timed(rotations, lambda: compute_pose_shape_fk_with_pose_and_beta_jacobians_separate_reference(
        rotations, d_rotations, shape_state.joints,
        shape_state.d_joints_d_latent, parents,
    ))


def compute_pose_shape_fk_with_combined_jacobian_bench(
    rotations: torch.Tensor, d_rotations_pose: torch.Tensor,
    joints: torch.Tensor, d_joints_beta: torch.Tensor, parents: torch.Tensor,
) -> tuple[CombinedFKResult, torch.Tensor, torch.Tensor]:
    """Benchmark-compatible wrapper around the production combined FK."""
    result = compute_pose_shape_fk_with_combined_jacobian(
        rotations, d_rotations_pose, joints, d_joints_beta, parents,
    )
    d_global = torch.cat(
        (result.d_global_transforms_d_pose, result.d_global_transforms_d_beta), -1,
    )
    d_relative = torch.cat(
        (result.d_relative_transforms_d_pose, result.d_relative_transforms_d_beta), -1,
    )
    return result, d_global, d_relative


def selected_lbs_low_memory_b1(
    selected_v_posed: torch.Tensor, d_selected_v_posed: torch.Tensor,
    weights: torch.Tensor, relative_transforms: torch.Tensor,
    d_relative_transforms: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Candidate B1: materialize only the first three transform rows."""
    transforms_xyz = (weights @ relative_transforms[:, :3].reshape(relative_transforms.shape[0], 12)).reshape(-1, 3, 4)
    d_transforms_xyz = torch.einsum("nj,jabq->nabq", weights, d_relative_transforms[:, :3])
    homogeneous = torch.cat((selected_v_posed, torch.ones_like(selected_v_posed[:, :1])), 1)
    d_homogeneous = torch.cat((d_selected_v_posed, torch.zeros_like(d_selected_v_posed[:, :1])), 1)
    vertices = torch.einsum("nab,nb->na", transforms_xyz, homogeneous)
    derivative = (
        torch.einsum("nabq,nb->naq", d_transforms_xyz, homogeneous)
        + torch.einsum("nab,nbq->naq", transforms_xyz, d_homogeneous)
    )
    return vertices, derivative


def selected_lbs_direct_b2(
    selected_v_posed: torch.Tensor, d_selected_v_posed: torch.Tensor,
    weights: torch.Tensor, relative_transforms: torch.Tensor,
    d_relative_transforms: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Candidate B2: directly contract joint-transform derivatives with v."""
    homogeneous = torch.cat((selected_v_posed, torch.ones_like(selected_v_posed[:, :1])), 1)
    d_homogeneous = torch.cat((d_selected_v_posed, torch.zeros_like(d_selected_v_posed[:, :1])), 1)
    transforms_xyz = torch.einsum("nj,jab->nab", weights, relative_transforms[:, :3])
    vertices = torch.einsum("nab,nb->na", transforms_xyz, homogeneous)
    derivative_transform = torch.einsum(
        "nj,jabq,nb->naq", weights, d_relative_transforms[:, :3], homogeneous,
    )
    derivative_vertex = torch.einsum("nab,nbq->naq", transforms_xyz, d_homogeneous)
    return vertices, derivative_transform + derivative_vertex


def pack_exact_lbs_weights(weights: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Pack every exactly nonzero influence; no threshold or top-k approximation."""
    counts = (weights != 0).sum(1)
    width = int(counts.max())
    # Sorting a boolean key only moves exact zeros behind nonzeros. Selecting
    # ``width`` therefore retains every exact nonzero and may include harmless
    # zero padding for vertices with fewer influences.
    indices = torch.argsort(weights == 0, dim=1, stable=True)[:, :width]
    return indices, weights.gather(1, indices)


def selected_lbs_exact_sparse_b3(
    selected_v_posed: torch.Tensor, d_selected_v_posed: torch.Tensor,
    exact_indices: torch.Tensor, exact_weights: torch.Tensor,
    relative_transforms: torch.Tensor, d_relative_transforms: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Candidate B3 using all exactly nonzero packed LBS influences."""
    selected_a = relative_transforms.index_select(0, exact_indices.reshape(-1)).reshape(
        exact_indices.shape + (4,4),
    )
    selected_da = d_relative_transforms.index_select(0, exact_indices.reshape(-1)).reshape(
        exact_indices.shape + (4,4,d_relative_transforms.shape[-1]),
    )
    transforms_xyz = torch.einsum("nk,nkab->nab", exact_weights, selected_a[..., :3, :])
    homogeneous = torch.cat((selected_v_posed, torch.ones_like(selected_v_posed[:, :1])), 1)
    d_homogeneous = torch.cat((d_selected_v_posed, torch.zeros_like(d_selected_v_posed[:, :1])), 1)
    vertices = torch.einsum("nab,nb->na", transforms_xyz, homogeneous)
    derivative = (
        torch.einsum("nk,nkabq,nb->naq", exact_weights, selected_da[..., :3, :, :], homogeneous)
        + torch.einsum("nab,nbq->naq", transforms_xyz, d_homogeneous)
    )
    return vertices, derivative


def active_group_jacobians(
    result: PoseShapeGroupJacobianResult, camera: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    camera_slice = slice(0, 6) if camera == "cam2" else slice(6, 12)
    active = torch.cat((result.local_pose_jacobian[:, camera_slice],
                        result.local_pose_jacobian[:, STAGE1_PER_FRAME_DIM:]), 1)
    indices = torch.cat((
        torch.arange(camera_slice.start, camera_slice.stop, device=active.device),
        torch.arange(STAGE1_PER_FRAME_DIM, result.local_pose_jacobian.shape[1], device=active.device),
    ))
    return active, result.shared_beta_jacobian, indices


def accumulate_active_group_hg_bench(
    result: PoseShapeGroupJacobianResult, camera: str, row_scale: torch.Tensor,
    *, accumulation_dtype: torch.dtype = torch.float64,
) -> HGContribution:
    """Candidate C: avoid the inactive six zero-filled camera columns."""
    active, beta_j, indices = active_group_jacobians(result, camera)
    scale = row_scale.to(result.raw_residual)
    residual = result.raw_residual * scale
    ja = (active * scale.unsqueeze(1)).to(accumulation_dtype)
    jb = (beta_j * scale.unsqueeze(1)).to(accumulation_dtype)
    r = residual.to(accumulation_dtype)
    p = result.local_pose_jacobian.shape[1]
    b = beta_j.shape[1]
    frame_h = torch.zeros(p, p, dtype=accumulation_dtype, device=ja.device)
    cross = torch.zeros(p, b, dtype=accumulation_dtype, device=ja.device)
    gradient = torch.zeros(p, dtype=accumulation_dtype, device=ja.device)
    frame_h[indices[:, None], indices[None, :]] = ja.T @ ja
    cross[indices] = ja.T @ jb
    gradient[indices] = ja.T @ r
    return HGContribution(frame_h, cross, jb.T @ jb, gradient, jb.T @ r, residual)


def accumulate_full_group_hg_bench(
    result: PoseShapeGroupJacobianResult, row_scale: torch.Tensor,
    *, gemm_dtype: torch.dtype = torch.float64,
) -> HGContribution:
    """Candidate D0/D1 depending on ``gemm_dtype``."""
    scale = row_scale.to(result.raw_residual)
    residual = result.raw_residual * scale
    jf = (result.local_pose_jacobian * scale.unsqueeze(1)).to(gemm_dtype)
    jb = (result.shared_beta_jacobian * scale.unsqueeze(1)).to(gemm_dtype)
    r = residual.to(gemm_dtype)
    return HGContribution(
        (jf.T @ jf).to(torch.float64), (jf.T @ jb).to(torch.float64),
        (jb.T @ jb).to(torch.float64), (jf.T @ r).to(torch.float64),
        (jb.T @ r).to(torch.float64), residual,
    )


def hg_error(candidate: HGContribution, reference: HGContribution) -> dict[str, float]:
    def relative(a, b):
        return float((a-b).norm() / max(float(b.norm()), 1e-30))
    return {
        "H_rel": relative(candidate.frame_hessian, reference.frame_hessian),
        "H_max": float((candidate.frame_hessian-reference.frame_hessian).abs().max()),
        "cross_rel": relative(candidate.frame_beta_cross, reference.frame_beta_cross),
        "beta_H_rel": relative(candidate.beta_hessian, reference.beta_hessian),
        "g_rel": relative(candidate.frame_gradient, reference.frame_gradient),
        "g_max": float((candidate.frame_gradient-reference.frame_gradient).abs().max()),
        "beta_g_rel": relative(candidate.beta_gradient, reference.beta_gradient),
    }


def accumulate_stage2_pose_shape_arrowhead_fp32_bench(
    cache: Stage2PoseShapeFastCache, parameters: torch.Tensor, beta_abs_limit: float,
    row_scales_by_group: list[torch.Tensor], body_pose_anchors: list[torch.Tensor],
    beta_anchor: torch.Tensor, pose_anchor_scale: float, shape_anchor_scale: float,
) -> NormalEquations:
    """Candidate D1 full system: FP32 group GEMMs, FP64 small-block accumulation."""
    p=STAGE1_PER_FRAME_DIM+cache.model.body_pose_dim;b=cache.model.beta_dim
    a=torch.zeros(cache.frame_count,p,p,dtype=torch.float64,device=parameters.device)
    cross=torch.zeros(cache.frame_count,p,b,dtype=torch.float64,device=parameters.device)
    c=torch.zeros(b,b,dtype=torch.float64,device=parameters.device)
    gf=torch.zeros(cache.frame_count,p,dtype=torch.float64,device=parameters.device)
    gb=torch.zeros(b,dtype=torch.float64,device=parameters.device);residuals=[]
    shape=evaluate_pose_shape_state(cache,parameters[-b:],beta_abs_limit)
    for group,scale in zip(cache.groups,row_scales_by_group):
        result=pose_shape_group_raw_residual_and_jacobians(cache,group,parameters,beta_abs_limit,shape)
        contribution=accumulate_full_group_hg_bench(result,scale,gemm_dtype=torch.float32)
        frame=group.frame_position;a[frame]+=contribution.frame_hessian
        cross[frame]+=contribution.frame_beta_cross;c+=contribution.beta_hessian
        gf[frame]+=contribution.frame_gradient;gb+=contribution.beta_gradient
        residuals.append(contribution.residual)
    body_slice=slice(STAGE1_PER_FRAME_DIM,p);pose_scale2=float(pose_anchor_scale)**2
    pose_residuals=[]
    for frame in range(cache.frame_count):
        delta=parameters[frame*p+STAGE1_PER_FRAME_DIM:(frame+1)*p]-body_pose_anchors[frame].to(parameters)
        pose_residuals.append(delta*pose_anchor_scale)
        a[frame,body_slice,body_slice]+=torch.eye(cache.model.body_pose_dim,dtype=torch.float64,device=parameters.device)*pose_scale2
        gf[frame,body_slice]+=delta.to(torch.float64)*pose_scale2
    beta,dbdz=physical_beta_from_latent(parameters[-b:],beta_abs_limit)
    shape_residual=(beta-beta_anchor.to(parameters))*shape_anchor_scale
    diagonal=(dbdz*shape_anchor_scale).to(torch.float64)
    c+=torch.diag(diagonal.square());gb+=diagonal*shape_residual.to(torch.float64)
    system=ArrowheadNormalEquationSystem(a,cross,c,gf,gb)
    return NormalEquations(torch.cat((*residuals,*pose_residuals,shape_residual)),None,system.flat_gradient(),system)


def build_pose_shape_group_batch(
    groups: list[Stage2PoseShapeFrameCameraCache], *, device: torch.device | None = None,
) -> JointGroupBatch:
    """Candidate E layout: pad selected-point tensors without reordering."""
    if not groups:
        raise ValueError("EMPTY_JOINT_GROUP_BATCH")
    reference = groups[0].selected_base_v_shaped
    device = reference.device if device is None else device
    dtype = reference.dtype
    g, nmax = len(groups), max(int(item.vertex_ids.numel()) for item in groups)
    beta_dim = groups[0].selected_beta_shapedirs.shape[-1]
    pose_feature_dim = groups[0].selected_posedirs.shape[0]
    joint_count = groups[0].selected_lbs_weights.shape[-1]
    valid = torch.zeros(g, nmax, dtype=torch.bool, device=device)
    base = torch.zeros(g, nmax, 3, dtype=dtype, device=device)
    beta_dirs = torch.zeros(g, nmax, 3, beta_dim, dtype=dtype, device=device)
    posedirs = torch.zeros(g, pose_feature_dim, nmax, 3, dtype=dtype, device=device)
    weights = torch.zeros(g, nmax, joint_count, dtype=dtype, device=device)
    targets = torch.zeros(g, nmax, 2, dtype=dtype, device=device)
    for index, item in enumerate(groups):
        count = int(item.vertex_ids.numel()); valid[index, :count] = True
        base[index, :count] = item.selected_base_v_shaped.to(device)
        beta_dirs[index, :count] = item.selected_beta_shapedirs.to(device)
        posedirs[index, :, :count] = item.selected_posedirs.to(device)
        weights[index, :count] = item.selected_lbs_weights.to(device)
        targets[index, :count] = item.targets.to(device)
    return JointGroupBatch(
        valid, torch.tensor([item.vertex_ids.numel() for item in groups], dtype=torch.int64, device=device),
        base, beta_dirs, posedirs, weights, targets,
        torch.stack([item.left_hand_pose.reshape(-1) for item in groups]).to(device),
        torch.stack([item.right_hand_pose.reshape(-1) for item in groups]).to(device),
        torch.stack([item.scale.reshape(-1) for item in groups]).to(device),
        torch.tensor([item.frame_position for item in groups], dtype=torch.int64, device=device),
        torch.tensor([0 if item.camera == "cam2" else 1 for item in groups], dtype=torch.int64, device=device),
    )


def lbs_weight_sparsity(groups: list[Stage2PoseShapeFrameCameraCache]) -> dict[str, dict[str, float]]:
    weights = torch.cat([item.selected_lbs_weights for item in groups], 0).abs()
    summaries = {}
    for name, counts in (
        ("exact", (weights != 0).sum(1).to(torch.float64)),
        ("gt_1e-12", (weights > 1e-12).sum(1).to(torch.float64)),
        ("gt_1e-8", (weights > 1e-8).sum(1).to(torch.float64)),
    ):
        summaries[name] = {
            "min": float(counts.min()), "max": float(counts.max()),
            "mean": float(counts.mean()), "p50": float(torch.quantile(counts, .5)),
            "p95": float(torch.quantile(counts, .95)),
        }
    return summaries


def joint_depth_layout(parents: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
    depths = torch.zeros_like(parents)
    for joint in range(1, parents.numel()):
        depths[joint] = depths[int(parents[joint])] + 1
    return depths, [torch.nonzero(depths == depth, as_tuple=False).reshape(-1)
                    for depth in range(int(depths.max()) + 1)]


def batched_fk_value_by_depth_bench(
    rotations: torch.Tensor, joints: torch.Tensor, parents: torch.Tensor,
) -> torch.Tensor:
    """Batched FK value prototype; loops over tree depth, not groups/joints."""
    g, joint_count = rotations.shape[:2]
    if joints.ndim == 2:
        joints = joints.unsqueeze(0).expand(g, -1, -1)
    relative = joints.clone()
    relative[:, 1:] -= joints.index_select(1, parents[1:])
    local = torch.zeros(g, joint_count, 4, 4, dtype=rotations.dtype, device=rotations.device)
    local[:, :, :3, :3] = rotations; local[:, :, :3, 3] = relative; local[:, :, 3, 3] = 1
    global_transforms = torch.zeros_like(local)
    depths, by_depth = joint_depth_layout(parents)
    global_transforms[:, 0] = local[:, 0]
    for indices in by_depth[1:]:
        parent_indices = parents.index_select(0, indices)
        global_transforms[:, indices] = (
            global_transforms.index_select(1, parent_indices) @ local.index_select(1, indices)
        )
    return global_transforms


def _batched_rotation_values_and_jacobians(
    orientations: torch.Tensor, body_poses: torch.Tensor,
    left_hands: torch.Tensor, right_hands: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pure tensor G-batch rotation assembly; no group loop."""
    g = orientations.shape[0]
    zeros = torch.zeros(g, 3, 3, dtype=orientations.dtype, device=orientations.device)
    axis_angles = torch.cat((
        orientations[:, None], body_poses.reshape(g, 21, 3), zeros,
        left_hands.reshape(g, 15, 3), right_hands.reshape(g, 15, 3),
    ), 1)
    rotations = axis_angle_to_matrix(axis_angles)
    variables = axis_angles[:, :22]
    variable_rotations = rotations[:, :22]
    right = so3_right_jacobian(variables)
    basis = skew(right.transpose(-1, -2))
    block_derivatives = torch.einsum("...ij,...kjl->...ilk", variable_rotations, basis)
    derivatives = torch.zeros(g, 55, 3, 3, 66, dtype=orientations.dtype, device=orientations.device)
    mapping = torch.zeros(22, 66, 3, dtype=orientations.dtype, device=orientations.device)
    joints = torch.arange(22, device=orientations.device)
    components = torch.arange(3, device=orientations.device)
    mapping[joints[:, None], joints[:, None] * 3 + components[None], components[None]] = 1
    derivatives[:, :22] = torch.einsum("gjabk,jqk->gjabq", block_derivatives, mapping)
    return rotations, derivatives


def _batched_combined_fk_by_depth(
    rotations: torch.Tensor, d_rotations: torch.Tensor,
    joints: torch.Tensor, d_joints_beta: torch.Tensor, parents: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    g, joint_count = rotations.shape[:2]
    pose_dim, beta_dim = d_rotations.shape[-1], d_joints_beta.shape[-1]
    joints_g = joints.unsqueeze(0).expand(g, -1, -1)
    d_joints_g = d_joints_beta.unsqueeze(0).expand(g, -1, -1, -1)
    relative = joints_g.clone(); relative[:, 1:] -= joints_g.index_select(1, parents[1:])
    d_relative_beta = d_joints_g.clone(); d_relative_beta[:, 1:] -= d_joints_g.index_select(1, parents[1:])
    local = torch.zeros(g, joint_count, 4, 4, dtype=rotations.dtype, device=rotations.device)
    local[:, :, :3, :3] = rotations; local[:, :, :3, 3] = relative; local[:, :, 3, 3] = 1
    d_local = torch.zeros(g, joint_count, 4, 4, pose_dim+beta_dim,
                          dtype=rotations.dtype, device=rotations.device)
    d_local[:, :, :3, :3, :pose_dim] = d_rotations
    d_local[:, :, :3, 3, pose_dim:] = d_relative_beta
    globals_ = torch.zeros_like(local); d_globals = torch.zeros_like(d_local)
    globals_[:, 0] = local[:, 0]; d_globals[:, 0] = d_local[:, 0]
    _, levels = joint_depth_layout(parents)
    for indices in levels[1:]:
        parent_indices = parents.index_select(0, indices)
        parent_global = globals_.index_select(1, parent_indices)
        parent_derivative = d_globals.index_select(1, parent_indices)
        child_local = local.index_select(1, indices)
        child_derivative = d_local.index_select(1, indices)
        globals_[:, indices] = parent_global @ child_local
        d_globals[:, indices] = (
            torch.einsum("gjabq,gjbc->gjacq", parent_derivative, child_local)
            + torch.einsum("gjab,gjbcq->gjacq", parent_global, child_derivative)
        )
    rest = torch.eye(4, dtype=rotations.dtype, device=rotations.device).reshape(1,1,4,4).repeat(g,joint_count,1,1)
    rest[:, :, :3, 3] = -joints_g
    d_rest = torch.zeros_like(d_globals); d_rest[:, :, :3, 3, pose_dim:] = -d_joints_g
    relative_transforms = globals_ @ rest
    d_relative_transforms = (
        torch.einsum("gjabq,gjbc->gjacq", d_globals, rest)
        + torch.einsum("gjab,gjbcq->gjacq", globals_, d_rest)
    )
    return relative_transforms, d_relative_transforms


def batched_pose_shape_group_analytic_bench(
    valid_mask: torch.Tensor, selected_base_v: torch.Tensor,
    selected_beta_dirs: torch.Tensor, selected_posedirs: torch.Tensor,
    selected_weights: torch.Tensor, targets: torch.Tensor,
    left_hands: torch.Tensor, right_hands: torch.Tensor, scales: torch.Tensor,
    orientations: torch.Tensor, translations: torch.Tensor, body_poses: torch.Tensor,
    physical_beta: torch.Tensor, d_beta_d_latent: torch.Tensor,
    joints: torch.Tensor, d_joints_beta_latent: torch.Tensor, parents: torch.Tensor,
    image_width: int, image_height: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Padded pure-tensor prototype returning residual, active J, beta J, mask."""
    rotations, d_rotations = _batched_rotation_values_and_jacobians(
        orientations, body_poses, left_hands, right_hands,
    )
    identity = torch.eye(3, dtype=rotations.dtype, device=rotations.device)
    feature = (rotations[:, 1:] - identity).reshape(rotations.shape[0], -1)
    d_feature = d_rotations[:, 1:].reshape(rotations.shape[0], -1, 66)
    offsets = torch.einsum("gp,gpnc->gnc", feature, selected_posedirs)
    d_offsets = torch.einsum("gpq,gpnc->gncq", d_feature, selected_posedirs)
    v_shaped = selected_base_v + torch.einsum("gncb,b->gnc", selected_beta_dirs, physical_beta)
    d_v_beta = selected_beta_dirs * d_beta_d_latent.reshape(1,1,1,-1)
    v_posed = v_shaped + offsets
    relative, d_relative = _batched_combined_fk_by_depth(
        rotations, d_rotations, joints, d_joints_beta_latent, parents,
    )
    combined_v = torch.cat((d_offsets, d_v_beta), -1)
    homogeneous = torch.cat((v_posed, torch.ones_like(v_posed[..., :1])), -1)
    d_homogeneous = torch.cat((combined_v, torch.zeros_like(combined_v[..., :1, :])), -2)
    transforms_xyz = torch.einsum("gnj,gjab->gnab", selected_weights, relative[:, :, :3])
    vertices = torch.einsum("gnab,gnb->gna", transforms_xyz, homogeneous)
    d_vertices = (
        torch.einsum("gnj,gjabq,gnb->gnaq", selected_weights, d_relative[:, :, :3], homogeneous)
        + torch.einsum("gnab,gnbq->gnaq", transforms_xyz, d_homogeneous)
    )
    scale = scales.reshape(-1,1,1)
    vertices = vertices*scale + translations[:,None]
    d_vertices = d_vertices*scale.unsqueeze(-1)
    g, n = vertices.shape[:2]
    uv, projection = project_unified_erp_with_jacobian(vertices.reshape(g*n,3), image_width, image_height)
    uv=uv.reshape(g,n,2);projection=projection.reshape(g,n,2,3)
    uv_all = torch.einsum("gnab,gnbq->gnaq", projection, d_vertices)
    active = torch.cat((uv_all[..., :3], projection, uv_all[..., 3:66]), -1)
    beta_j = uv_all[..., 66:]
    residual = (uv-targets)*valid_mask[...,None]
    active = active*valid_mask[...,None,None]
    beta_j = beta_j*valid_mask[...,None,None]
    return residual, active, beta_j, valid_mask


def batched_group_hg_bench(
    kernel_args: tuple[Any, ...], row_scales: torch.Tensor,
    analytic_result: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    # Attribution callers may time this function independently.  Complete-NE
    # callers pass the already computed analytic result so geometry/J runs once.
    residual, active, beta_j, mask = (
        batched_pose_shape_group_analytic_bench(*kernel_args)
        if analytic_result is None else analytic_result
    )
    g, n = mask.shape
    row_scale = row_scales.reshape(g,n,2)
    r = (residual*row_scale).reshape(g,2*n).to(torch.float64)
    ja = (active*row_scale[...,None]).reshape(g,2*n,69).to(torch.float64)
    jb = (beta_j*row_scale[...,None]).reshape(g,2*n,beta_j.shape[-1]).to(torch.float64)
    return (
        torch.bmm(ja.transpose(1,2),ja), torch.bmm(ja.transpose(1,2),jb),
        torch.bmm(jb.transpose(1,2),jb), torch.bmm(ja.transpose(1,2),r.unsqueeze(-1)).squeeze(-1),
        torch.bmm(jb.transpose(1,2),r.unsqueeze(-1)).squeeze(-1),
    )


def batched_joint_arrowhead_hg(
    cache: Stage2PoseShapeFastCache, batch: JointGroupBatch,
    parameters: torch.Tensor, beta_abs_limit: float, row_scales: torch.Tensor,
    body_pose_anchors: list[torch.Tensor], beta_anchor: torch.Tensor,
    pose_anchor_scale: float, shape_anchor_scale: float,
) -> tuple[ArrowheadNormalEquationSystem, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Production-equivalent active-column padded arrowhead accumulation."""
    p, b = 75, cache.model.beta_dim
    framed = parameters[:-b].reshape(cache.frame_count, p).index_select(0, batch.frame_positions)
    offsets = batch.camera_ids * 6
    orient = framed.gather(1, offsets[:,None]+torch.arange(3,device=parameters.device)[None])
    translation = framed.gather(1, offsets[:,None]+3+torch.arange(3,device=parameters.device)[None])
    body = framed[:,12:]
    shape = evaluate_pose_shape_state(cache, parameters[-b:], beta_abs_limit)
    kernel_args = (
        batch.valid_mask,batch.selected_base_v,batch.selected_beta_dirs,batch.selected_posedirs,
        batch.selected_weights,batch.targets,batch.left_hand_pose,batch.right_hand_pose,batch.scale,
        orient,translation,body,shape.beta,shape.d_beta_d_latent,shape.joints,
        shape.d_joints_d_latent,cache.model.parents,cache.image_width,cache.image_height,
    )
    analytic_result = batched_pose_shape_group_analytic_bench(*kernel_args)
    residual, active, beta_j, _ = analytic_result
    h_active,cross_active,h_beta,g_active,g_beta = batched_group_hg_bench(
        kernel_args, row_scales, analytic_result,
    )
    frame_h=torch.zeros(cache.frame_count,p,p,dtype=torch.float64,device=parameters.device)
    frame_cross=torch.zeros(cache.frame_count,p,b,dtype=torch.float64,device=parameters.device)
    beta_h=h_beta.sum(0);frame_g=torch.zeros(cache.frame_count,p,dtype=torch.float64,device=parameters.device)
    beta_g=g_beta.sum(0)
    for group in range(batch.valid_mask.shape[0]):
        frame=int(batch.frame_positions[group]);camera=int(batch.camera_ids[group]);start=0 if camera==0 else 6
        indices=torch.cat((torch.arange(start,start+6,device=parameters.device),torch.arange(12,75,device=parameters.device)))
        frame_h[frame][indices[:,None],indices[None,:]]+=h_active[group]
        frame_cross[frame][indices]+=cross_active[group]
        frame_g[frame][indices]+=g_active[group]
    pose_scale2=float(pose_anchor_scale)**2
    for frame in range(cache.frame_count):
        delta=parameters[frame*p+12:(frame+1)*p]-body_pose_anchors[frame].to(parameters)
        frame_h[frame,12:,12:]+=torch.eye(63,dtype=torch.float64,device=parameters.device)*pose_scale2
        frame_g[frame,12:]+=delta.to(torch.float64)*pose_scale2
    beta,dbdz=physical_beta_from_latent(parameters[-b:],beta_abs_limit)
    shape_residual=(beta-beta_anchor.to(parameters))*shape_anchor_scale
    diagonal=(dbdz*shape_anchor_scale).to(torch.float64)
    beta_h+=torch.diag(diagonal.square());beta_g+=diagonal*shape_residual.to(torch.float64)
    system=ArrowheadNormalEquationSystem(frame_h,frame_cross,beta_h,frame_g,beta_g)
    return system,residual,active,beta_j


def batched_joint_residual_only(
    batch: JointGroupBatch, orientations: torch.Tensor, translations: torch.Tensor,
    body_poses: torch.Tensor, physical_beta: torch.Tensor, joints: torch.Tensor,
    parents: torch.Tensor, image_width: int, image_height: int,
) -> torch.Tensor:
    """Value-only padded joint kernel. No derivative tensor is constructed."""
    g=orientations.shape[0];zeros=torch.zeros(g,3,3,dtype=orientations.dtype,device=orientations.device)
    axis_angles=torch.cat((orientations[:,None],body_poses.reshape(g,21,3),zeros,
                           batch.left_hand_pose.reshape(g,15,3),batch.right_hand_pose.reshape(g,15,3)),1)
    rotations=axis_angle_to_matrix(axis_angles)
    identity=torch.eye(3,dtype=rotations.dtype,device=rotations.device)
    feature=(rotations[:,1:]-identity).reshape(g,-1)
    offsets=torch.einsum("gp,gpnc->gnc",feature,batch.selected_posedirs)
    shaped=batch.selected_base_v+torch.einsum("gncb,b->gnc",batch.selected_beta_dirs,physical_beta)
    relative=batched_fk_value_by_depth_bench(rotations,joints,parents)
    rest=torch.eye(4,dtype=rotations.dtype,device=rotations.device)[None,None].repeat(g,rotations.shape[1],1,1)
    rest[:,:,:3,3]=-joints[None]
    relative=relative@rest
    transforms=torch.einsum("gnj,gjab->gnab",batch.selected_weights,relative[:,:,:3])
    homogeneous=torch.cat((shaped+offsets,torch.ones_like(shaped[...,:1])),-1)
    vertices=torch.einsum("gnab,gnb->gna",transforms,homogeneous)
    vertices=vertices*batch.scale.reshape(-1,1,1)+translations[:,None]
    uv,_=project_unified_erp_with_jacobian(vertices.reshape(-1,3),image_width,image_height)
    return (uv.reshape(g,-1,2)-batch.targets)*batch.valid_mask[...,None]


def batched_joint_objective(
    cache: Stage2PoseShapeFastCache, batch: JointGroupBatch, parameters: torch.Tensor,
    beta_abs_limit: float, body_pose_anchors: list[torch.Tensor], beta_anchor: torch.Tensor,
    pose_anchor_scale: float, shape_anchor_scale: float, *, objective_mode: str,
    charbonnier_eps: float, rep_scale: float, aggregation_spec: dict[str,Any]|None,
) -> dict[str,torch.Tensor]:
    b=cache.model.beta_dim;framed=parameters[:-b].reshape(cache.frame_count,75).index_select(0,batch.frame_positions)
    offsets=batch.camera_ids*6
    orient=framed.gather(1,offsets[:,None]+torch.arange(3,device=parameters.device)[None])
    translation=framed.gather(1,offsets[:,None]+3+torch.arange(3,device=parameters.device)[None]);body=framed[:,12:]
    shape=evaluate_pose_shape_state(cache,parameters[-b:],beta_abs_limit)
    residual=batched_joint_residual_only(batch,orient,translation,body,shape.beta,shape.joints,cache.model.parents,cache.image_width,cache.image_height)
    if objective_mode=="raw_l2":primary=.5*torch.sum((residual*rep_scale)**2)
    else:
        if aggregation_spec is None:raise ValueError("aggregation_spec is required")
        groups=[];keys=list(aggregation_spec["terms"])
        for index in range(batch.valid_mask.shape[0]):
            count=int(batch.point_counts[index]);frame_text,camera=keys[index].split("|",1);groups.append(ReprojectionGroup(
                int(frame_text),camera,residual[index,:count]))
        primary=compute_primary_charbonnier_tensor(groups,charbonnier_eps,aggregation_spec)
    pose=.5*sum(torch.sum(((parameters[f*75+12:(f+1)*75]-body_pose_anchors[f].to(parameters))*pose_anchor_scale)**2) for f in range(cache.frame_count))
    shape_anchor=.5*torch.sum(((shape.beta-beta_anchor.to(parameters))*shape_anchor_scale)**2)
    return {"primary":primary,"pose_anchor":pose,"shape_anchor":shape_anchor,"total":primary+pose+shape_anchor}


def benchmark_callable(
    fn: Callable[[], Any], reference: torch.Tensor, warmup: int, repeats: int,
) -> dict[str, float]:
    for _ in range(warmup):
        fn()
    _sync(reference)
    samples = []
    for _ in range(repeats):
        _, elapsed = _timed(reference, fn); samples.append(elapsed)
    ordered = sorted(samples)
    p90_index = min(len(ordered)-1, int(.9*(len(ordered)-1)))
    return {
        "p50": statistics.median(samples), "p90": ordered[p90_index],
        "mean": statistics.mean(samples), "min": min(samples),
    }


def tensor_audit(tensors: dict[str, torch.Tensor]) -> list[dict[str, Any]]:
    return [{
        "name": name, "shape": list(value.shape), "dtype": str(value.dtype),
        "elements": value.numel(), "bytes": tensor_bytes(value),
    } for name, value in tensors.items()]
