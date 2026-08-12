"""Correctness tests for isolated Stage 2 joint H/g kernel candidates."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

try:
    from ._bootstrap import ensure_test_paths
except ImportError:
    from _bootstrap import ensure_test_paths
ensure_test_paths(__file__)

from tests.human_fitting.staged_lm.helpers import _cache, _parameters  # noqa:E402
from lib_pano.human_fitting.staged_lm.pose_shape import evaluate_pose_shape_state  # noqa:E402
from lib_pano.human_fitting.staged_lm.pose_shape import (  # noqa:E402
    CombinedFKResult,
    accumulate_stage2_pose_shape_arrowhead_system,
    compute_pose_shape_fk_with_combined_jacobian,
    pose_shape_group_raw_residual_and_jacobians,
)
from script.validation.staged_lm.references.pose_shape_backends import (  # noqa:E402
    compute_pose_shape_fk_with_pose_and_beta_jacobians_separate_reference,
)
from script.benchmark.staged_lm.pose_shape_kernel_candidates import (  # noqa:E402
    accumulate_active_group_hg_bench, accumulate_full_group_hg_bench,
    accumulate_stage2_pose_shape_arrowhead_fp32_bench,
    batched_fk_value_by_depth_bench, build_pose_shape_group_batch,
    batched_pose_shape_group_analytic_bench,
    compute_pose_shape_fk_with_combined_jacobian_bench, joint_depth_layout,
    lbs_weight_sparsity, profile_pose_shape_group_baseline,
    selected_lbs_direct_b2, selected_lbs_low_memory_b1,
    pack_exact_lbs_weights, selected_lbs_exact_sparse_b3,
)
from lib_pano.human_fitting.staged_lm.pose.derivatives import (  # noqa:E402
    assemble_full_pose_rotation_jacobians, compute_selected_lbs_with_jacobian,
    compute_selected_pose_offsets_with_jacobian,
)


def test_profile_reproduces_production() -> None:
    cache, parameters = _cache(), _parameters()
    shape = evaluate_pose_shape_state(cache, parameters[-2:], 3.0)
    group = cache.groups[0]
    profile = profile_pose_shape_group_baseline(cache, group, shape, parameters)
    production = pose_shape_group_raw_residual_and_jacobians(cache, group, parameters, 3.0, shape)
    torch.testing.assert_close(profile.result.raw_residual, production.raw_residual)
    torch.testing.assert_close(profile.result.local_pose_jacobian, production.local_pose_jacobian)
    torch.testing.assert_close(profile.result.shared_beta_jacobian, production.shared_beta_jacobian)
    assert set("ABCDEFGHIJKLMNOP") == {name[0] for name in profile.section_ms}


def _fk_inputs():
    cache, parameters = _cache(), _parameters()
    shape = evaluate_pose_shape_state(cache, parameters[-2:], 3.0)
    group = cache.groups[0]
    rotations, d_rotations = assemble_full_pose_rotation_jacobians(
        parameters[:3], parameters[12:75], group.left_hand_pose,
        group.right_hand_pose, cache.model,
    )
    return cache, shape, group, rotations, d_rotations


def test_combined_fk_candidate() -> None:
    cache, shape, _, rotations, d_rotations = _fk_inputs()
    baseline = compute_pose_shape_fk_with_pose_and_beta_jacobians_separate_reference(
        rotations, d_rotations, shape.joints, shape.d_joints_d_latent,
        cache.model.parents,
    )
    candidate, d_global, d_relative = compute_pose_shape_fk_with_combined_jacobian_bench(
        rotations, d_rotations, shape.joints, shape.d_joints_d_latent,
        cache.model.parents,
    )
    for left, right in (
        (candidate.global_transforms, baseline.global_transforms),
        (candidate.relative_transforms, baseline.relative_transforms),
        (candidate.d_global_transforms_d_pose, baseline.d_global_transforms_d_pose),
        (candidate.d_global_transforms_d_beta, baseline.d_global_transforms_d_beta),
        (candidate.d_relative_transforms_d_pose, baseline.d_relative_transforms_d_pose),
        (candidate.d_relative_transforms_d_beta, baseline.d_relative_transforms_d_beta),
    ):
        torch.testing.assert_close(left, right, rtol=1e-12, atol=1e-12)
    assert d_global.shape[-1] == 68 and d_relative.shape[-1] == 68
    production = compute_pose_shape_fk_with_combined_jacobian(
        rotations, d_rotations, shape.joints, shape.d_joints_d_latent,
        cache.model.parents,
    )
    for name in CombinedFKResult.__dataclass_fields__:
        torch.testing.assert_close(getattr(production, name), getattr(baseline, name), rtol=1e-12, atol=1e-12)


def test_low_memory_lbs_candidates() -> None:
    cache, shape, group, rotations, d_rotations = _fk_inputs()
    offsets, d_offsets, _, _ = compute_selected_pose_offsets_with_jacobian(
        rotations, d_rotations, group.selected_posedirs,
    )
    beta_dirs = group.selected_beta_shapedirs
    v_shaped = group.selected_base_v_shaped + torch.einsum("ncb,b->nc", beta_dirs, shape.beta)
    d_beta = beta_dirs * shape.d_beta_d_latent.reshape(1, 1, -1)
    fk = compute_pose_shape_fk_with_combined_jacobian(
        rotations, d_rotations, shape.joints, shape.d_joints_d_latent,
        cache.model.parents,
    )
    dv = torch.cat((d_offsets, d_beta), -1)
    da = torch.cat((fk.d_relative_transforms_d_pose, fk.d_relative_transforms_d_beta), -1)
    baseline = compute_selected_lbs_with_jacobian(
        v_shaped + offsets, dv, group.selected_lbs_weights,
        fk.relative_transforms, da,
    )
    for candidate in (selected_lbs_low_memory_b1, selected_lbs_direct_b2):
        vertices, derivative = candidate(
            v_shaped + offsets, dv, group.selected_lbs_weights,
            fk.relative_transforms, da,
        )
        torch.testing.assert_close(vertices, baseline[0], rtol=1e-12, atol=1e-12)
        torch.testing.assert_close(derivative, baseline[1], rtol=1e-12, atol=1e-12)
    indices, exact_weights = pack_exact_lbs_weights(group.selected_lbs_weights)
    vertices, derivative = selected_lbs_exact_sparse_b3(
        v_shaped+offsets,dv,indices,exact_weights,fk.relative_transforms,da,
    )
    torch.testing.assert_close(vertices,baseline[0],rtol=1e-12,atol=1e-12)
    torch.testing.assert_close(derivative,baseline[1],rtol=1e-12,atol=1e-12)
    assert indices.shape[1] == int((group.selected_lbs_weights != 0).sum(1).max())


def test_active_hg_and_fp32_bakeoff() -> None:
    cache, parameters = _cache(), _parameters()
    shape = evaluate_pose_shape_state(cache, parameters[-2:], 3.0)
    for group in cache.groups:
        result = pose_shape_group_raw_residual_and_jacobians(cache, group, parameters, 3.0, shape)
        scale = torch.linspace(.01, .02, result.raw_residual.numel(), dtype=parameters.dtype)
        reference = accumulate_full_group_hg_bench(result, scale, gemm_dtype=torch.float64)
        active = accumulate_active_group_hg_bench(result, group.camera, scale)
        for name in ("frame_hessian", "frame_beta_cross", "beta_hessian",
                     "frame_gradient", "beta_gradient"):
            torch.testing.assert_close(getattr(active, name), getattr(reference, name), rtol=1e-10, atol=1e-8)
        fp32 = accumulate_full_group_hg_bench(result, scale, gemm_dtype=torch.float32)
        assert torch.isfinite(fp32.frame_hessian).all()
        assert float((fp32.frame_hessian-reference.frame_hessian).norm()/reference.frame_hessian.norm()) < 1e-5
        assert float((fp32.frame_gradient-reference.frame_gradient).norm()/reference.frame_gradient.norm()) < 1e-5
    anchors=[torch.zeros(63,dtype=parameters.dtype)]
    scales=[torch.full((item.targets.numel(),),.02,dtype=parameters.dtype) for item in cache.groups]
    residual,system=accumulate_stage2_pose_shape_arrowhead_system(
        cache,parameters,3.,scales,anchors,torch.zeros(2,dtype=parameters.dtype),.2,.4)
    candidate=accumulate_stage2_pose_shape_arrowhead_fp32_bench(
        cache,parameters,3.,scales,anchors,torch.zeros(2,dtype=parameters.dtype),.2,.4)
    torch.testing.assert_close(candidate.residual,residual)
    dense=system.to_dense(count_call=False);candidate_dense=candidate.structured_system.to_dense(count_call=False)
    assert float((candidate_dense-dense).norm()/dense.norm())<1e-5


def test_padding_layout_and_depth_fk() -> None:
    cache, parameters = _cache(2), _parameters(2)
    batch = build_pose_shape_group_batch(cache.groups)
    counts = batch.point_counts.tolist()
    for index, (group, count) in enumerate(zip(cache.groups, counts)):
        assert torch.equal(batch.selected_base_v[index, :count], group.selected_base_v_shaped)
        assert torch.equal(batch.targets[index, :count], group.targets)
        assert not batch.valid_mask[index, count:].any()
        assert torch.count_nonzero(batch.selected_weights[index, count:]) == 0
    assert batch.camera_ids.tolist() == [0, 1, 0, 1]
    sparsity = lbs_weight_sparsity(cache.groups)
    assert sparsity["exact"]["min"] >= 1
    depths, by_depth = joint_depth_layout(cache.model.parents)
    assert sum(item.numel() for item in by_depth) == cache.model.joint_count
    shape = evaluate_pose_shape_state(cache, parameters[-2:], 3.0)
    rotations = []
    for group in cache.groups[:2]:
        frame = parameters[:75]
        orient = frame[:3] if group.camera == "cam2" else frame[6:9]
        value, _ = assemble_full_pose_rotation_jacobians(
            orient, frame[12:75], group.left_hand_pose, group.right_hand_pose, cache.model,
        )
        rotations.append(value)
    batched = batched_fk_value_by_depth_bench(torch.stack(rotations), shape.joints, cache.model.parents)
    for index, rotation in enumerate(rotations):
        baseline = compute_pose_shape_fk_with_combined_jacobian(
            rotation, torch.zeros(cache.model.joint_count,3,3,66,dtype=parameters.dtype),
            shape.joints, shape.d_joints_d_latent, cache.model.parents,
        )
        torch.testing.assert_close(batched[index], baseline.global_transforms)


def test_padded_batched_group_kernel() -> None:
    cache, parameters = _cache(2), _parameters(2)
    batch = build_pose_shape_group_batch(cache.groups)
    shape = evaluate_pose_shape_state(cache, parameters[-2:], 3.0)
    orientations=[];translations=[];body=[]
    for group in cache.groups:
        frame=parameters[group.frame_position*75:(group.frame_position+1)*75]
        orientations.append(frame[:3] if group.camera=="cam2" else frame[6:9])
        translations.append(frame[3:6] if group.camera=="cam2" else frame[9:12])
        body.append(frame[12:75])
    result=batched_pose_shape_group_analytic_bench(
        batch.valid_mask,batch.selected_base_v,batch.selected_beta_dirs,batch.selected_posedirs,
        batch.selected_weights,batch.targets,batch.left_hand_pose,batch.right_hand_pose,batch.scale,
        torch.stack(orientations),torch.stack(translations),torch.stack(body),shape.beta,
        shape.d_beta_d_latent,shape.joints,shape.d_joints_d_latent,cache.model.parents,
        cache.image_width,cache.image_height,
    )
    residual,active,beta_j,mask=result
    for index,group in enumerate(cache.groups):
        count=int(group.vertex_ids.numel())
        reference=pose_shape_group_raw_residual_and_jacobians(cache,group,parameters,3.,shape)
        camera_slice=slice(0,6) if group.camera=="cam2" else slice(6,12)
        reference_active=torch.cat((reference.local_pose_jacobian[:,camera_slice],reference.local_pose_jacobian[:,12:]),1)
        torch.testing.assert_close(residual[index,:count].reshape(-1),reference.raw_residual,rtol=1e-6,atol=1e-3)
        torch.testing.assert_close(active[index,:count].reshape(-1,69),reference_active,rtol=5e-6,atol=1e-4)
        torch.testing.assert_close(beta_j[index,:count].reshape(-1,2),reference.shared_beta_jacobian,rtol=5e-6,atol=1e-4)
        assert torch.count_nonzero(residual[index,count:])==0


if __name__ == "__main__":
    test_profile_reproduces_production()
    test_combined_fk_candidate()
    test_low_memory_lbs_candidates()
    test_active_hg_and_fp32_bakeoff()
    test_padding_layout_and_depth_fk()
    test_padded_batched_group_kernel()
    print("All Stage 2 joint kernel benchmark tests PASSED")
