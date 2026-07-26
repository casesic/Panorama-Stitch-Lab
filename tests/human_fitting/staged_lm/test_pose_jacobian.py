"""Pure mathematical tests for the debug-only Stage 2 pose Jacobian."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import torch
from torch.func import jacfwd

try:
    from ._bootstrap import ensure_test_paths
except ImportError:
    from _bootstrap import ensure_test_paths
ensure_test_paths(__file__)

from lib_pano.human_fitting.staged_lm.pose.cache import (  # noqa: E402
    Stage2PoseFastCache,
    Stage2PoseFrameCameraCache,
    Stage2PoseModelCache,
)
from lib_pano.human_fitting.staged_lm.pose.forward import (  # noqa: E402
    _aa_to_rotmat,
    fast_raw_groups,
    fast_raw_residual,
    fast_selected_vertices,
)
from lib_pano.human_fitting.staged_lm.pose.objective import (  # noqa: E402
    stage2_pose_fast_raw_metrics,
    stage2_pose_fast_true_objective,
)
from lib_pano.human_fitting.staged_lm.irls import build_primary_aggregation_spec, compute_primary_charbonnier_tensor  # noqa: E402
from lib_pano.human_fitting.staged_lm.residuals import flatten_reprojection_groups, raw_reproj_metrics  # noqa: E402
from lib_pano.human_fitting.staged_lm.pose.derivatives import (  # noqa: E402
    POSE_PER_FRAME_DIM,
    build_stage2_pose_analytic_linearization,
    assemble_full_pose_rotation_jacobians,
    compute_fk_with_jacobian,
    compute_pose_feature_with_jacobian,
    compute_selected_lbs_with_jacobian,
    compute_selected_pose_offsets_with_jacobian,
    fast_selected_vertices_with_jacobian,
    rotation_matrix_jacobian_additive,
    stage2_pose_anchor_residual_and_jacobian,
    stage2_pose_explicit_raw_jacobian,
    stage2_pose_group_raw_residual_and_jacobian,
    stage2_pose_raw_residual,
    stage2_pose_surrogate_residual_and_jacobian,
    validate_stage2_pose_analytic_layout,
)


DTYPE = torch.float64


def _relative_fk(rotations: torch.Tensor, joints: torch.Tensor, parents: torch.Tensor) -> torch.Tensor:
    relative = joints.clone()
    relative[1:] -= joints.index_select(0, parents[1:])
    local = torch.zeros(rotations.shape[0], 4, 4, dtype=rotations.dtype)
    local[:, :3, :3] = rotations
    local[:, :3, 3] = relative
    local[:, 3, 3] = 1
    globals_ = [local[0]]
    for joint in range(1, rotations.shape[0]):
        globals_.append(globals_[int(parents[joint])] @ local[joint])
    global_tensor = torch.stack(globals_)
    inverse = torch.eye(4, dtype=rotations.dtype).repeat(rotations.shape[0], 1, 1)
    inverse[:, :3, 3] = -joints
    return global_tensor @ inverse


def _cache(frame_count: int = 1) -> Stage2PoseFastCache:
    torch.manual_seed(41)
    joint_count, vertex_count = 55, 6
    parents = torch.arange(joint_count, dtype=torch.long) - 1
    parents[0] = -1
    joints = torch.randn(joint_count, 3, dtype=DTYPE) * .02
    model = Stage2PoseModelCache(
        v_shaped=torch.randn(vertex_count, 3, dtype=DTYPE) * .1,
        joints=joints, parents=parents, joint_count=joint_count,
        pose_feature_dim=486, full_vertex_count=vertex_count,
        dtype=DTYPE, device=torch.device("cpu"),
    )
    groups = []
    for frame in range(frame_count):
        for camera_index, camera in enumerate(("cam2", "cam4")):
            ids = torch.tensor([4, 1]) if camera == "cam2" else torch.tensor([5, 2, 0])
            weights = torch.rand(ids.numel(), joint_count, dtype=DTYPE)
            weights /= weights.sum(1, keepdim=True)
            groups.append(Stage2PoseFrameCameraCache(
                frame_position=frame, frame_index=513 + 2 * frame, camera=camera,
                vertex_ids=ids, targets=torch.randn(ids.numel(), 2, dtype=DTYPE) * 10 + 100,
                left_hand_pose=torch.randn(45, dtype=DTYPE) * (.01 + camera_index * .01),
                right_hand_pose=torch.randn(45, dtype=DTYPE) * (.02 + camera_index * .01),
                scale=torch.tensor([1.0 + camera_index * .2], dtype=DTYPE),
                selected_v_shaped=model.v_shaped.index_select(0, ids),
                selected_posedirs=torch.randn(486, ids.numel(), 3, dtype=DTYPE) * 1e-4,
                selected_lbs_weights=weights,
            ))
    return Stage2PoseFastCache(model, groups, frame_count, 63, 5760, 2880, 0.0)


def _parameters(frame_count: int = 1) -> torch.Tensor:
    torch.manual_seed(42)
    result = torch.randn(frame_count * POSE_PER_FRAME_DIM, dtype=DTYPE) * .02
    for frame in range(frame_count):
        result[frame * 75 + 3:frame * 75 + 6] += torch.tensor([.1, .2, 2.0], dtype=DTYPE)
        result[frame * 75 + 9:frame * 75 + 12] += torch.tensor([-.1, .1, 2.2], dtype=DTYPE)
    return result


def test_rotation_matrix_jacobian() -> None:
    axes = [torch.tensor([1., 0., 0.], dtype=DTYPE),
            torch.tensor([1., 2., 3.], dtype=DTYPE).div(torch.tensor(14., dtype=DTYPE).sqrt())]
    for magnitude in (0., 1e-8, 1e-5, .7, math.pi-1e-4, math.pi-1e-6,
                      math.pi+1e-6, math.pi+1e-4):
        for axis in axes:
            value = axis * magnitude
            analytic = rotation_matrix_jacobian_additive(value)
            oracle = jacfwd(lambda x: _aa_to_rotmat(x[None])[0])(value)
            assert torch.isfinite(analytic).all()
            torch.testing.assert_close(analytic, oracle, rtol=1e-8, atol=1e-8)


def test_pose_feature_and_offset_jacobians() -> None:
    cache, parameters = _cache(), _parameters()
    group = cache.groups[0]
    orient, body = parameters[:3], parameters[12:75]
    rotations, d_rotations = assemble_full_pose_rotation_jacobians(
        orient, body, group.left_hand_pose, group.right_hand_pose, cache.model,
    )
    feature, derivative = compute_pose_feature_with_jacobian(rotations, d_rotations)
    q = torch.cat((orient, body))
    def feature_fn(value: torch.Tensor) -> torch.Tensor:
        rots, _ = assemble_full_pose_rotation_jacobians(
            value[:3], value[3:], group.left_hand_pose, group.right_hand_pose, cache.model,
        )
        return (rots[1:] - torch.eye(3, dtype=DTYPE)).reshape(-1)
    torch.testing.assert_close(derivative, jacfwd(feature_fn)(q), rtol=1e-8, atol=1e-9)
    assert torch.count_nonzero(derivative[:, :3]) == 0  # root excluded
    assert torch.count_nonzero(d_rotations[22:]) == 0  # fixed jaw/eyes/hands
    offsets, d_offsets, _, _ = compute_selected_pose_offsets_with_jacobian(
        rotations, d_rotations, group.selected_posedirs,
    )
    oracle = jacfwd(lambda value: torch.einsum(
        "p,pnc->nc", feature_fn(value), group.selected_posedirs,
    ))(q)
    torch.testing.assert_close(d_offsets, oracle, rtol=1e-8, atol=1e-9)
    assert offsets.shape == (2, 3) and feature.shape == (486,)


def test_fk_parent_chain_and_relative_derivative() -> None:
    torch.manual_seed(43)
    rotations = _aa_to_rotmat(torch.randn(4, 3, dtype=DTYPE) * .1)
    joints = torch.randn(4, 3, dtype=DTYPE)
    parents = torch.tensor([-1, 0, 1, 2])
    direct = torch.zeros(4, 3, 3, 3, dtype=DTYPE)
    direct[0] = rotation_matrix_jacobian_additive(torch.zeros(3, dtype=DTYPE))
    # Use an independently parameterized root and a fixed three-level child chain.
    root = torch.zeros(3, dtype=DTYPE)
    fixed = rotations[1:].clone()
    def fn(value: torch.Tensor) -> torch.Tensor:
        return _relative_fk(torch.cat((_aa_to_rotmat(value[None]), fixed), 0), joints, parents)
    result = compute_fk_with_jacobian(
        torch.cat((_aa_to_rotmat(root[None]), fixed), 0), direct, joints, parents,
    )
    oracle = jacfwd(fn)(root)
    torch.testing.assert_close(result.relative_transforms, fn(root), rtol=1e-10, atol=1e-10)
    torch.testing.assert_close(result.d_relative_transforms, oracle, rtol=1e-8, atol=1e-9)
    assert torch.count_nonzero(result.d_relative_transforms[3]).item() > 0


def test_selected_lbs_product_rule() -> None:
    torch.manual_seed(44)
    n, j, q = 2, 4, 5
    v = torch.randn(n, 3, dtype=DTYPE)
    dv = torch.randn(n, 3, q, dtype=DTYPE)
    weights = torch.rand(n, j, dtype=DTYPE); weights /= weights.sum(1, keepdim=True)
    transforms = torch.randn(j, 4, 4, dtype=DTYPE)
    d_transforms = torch.randn(j, 4, 4, q, dtype=DTYPE)
    value, derivative, _, _ = compute_selected_lbs_with_jacobian(
        v, dv, weights, transforms, d_transforms,
    )
    def fn(x: torch.Tensor) -> torch.Tensor:
        vx = v + torch.einsum("ncq,q->nc", dv, x)
        tx = transforms + torch.einsum("jabr,r->jab", d_transforms, x)
        blended = (weights @ tx.reshape(j, 16)).reshape(n, 4, 4)
        hom = torch.cat((vx, torch.ones(n, 1, dtype=DTYPE)), 1)
        return (blended @ hom.unsqueeze(-1))[:, :3, 0]
    zero = torch.zeros(q, dtype=DTYPE)
    torch.testing.assert_close(value, fn(zero))
    torch.testing.assert_close(derivative, jacfwd(fn)(zero), rtol=1e-9, atol=1e-9)


def test_vertex_and_group_jacobian_layout() -> None:
    cache, parameters = _cache(), _parameters()
    for group in cache.groups:
        orient_slice = slice(0, 3) if group.camera == "cam2" else slice(6, 9)
        translation_slice = slice(3, 6) if group.camera == "cam2" else slice(9, 12)
        orient = parameters[orient_slice]
        translation = parameters[translation_slice]
        body = parameters[12:75]
        vertices, derivative = fast_selected_vertices_with_jacobian(
            cache, group, orient, translation, body,
        )
        q = torch.cat((orient, body))
        oracle = jacfwd(lambda value: fast_selected_vertices(
            cache, group, value[:3], translation, value[3:],
        ))(q)
        torch.testing.assert_close(vertices, fast_selected_vertices(cache, group, orient, translation, body))
        torch.testing.assert_close(derivative, oracle, rtol=2e-7, atol=2e-8)
        _, local = stage2_pose_group_raw_residual_and_jacobian(cache, group, parameters)
        full_oracle = jacfwd(lambda x: stage2_pose_raw_residual(
            Stage2PoseFastCache(cache.model, [group], 1, 63, 5760, 2880, 0.), x,
        ))(parameters)
        torch.testing.assert_close(local, full_oracle, rtol=2e-7, atol=2e-7)
        zero_slice = slice(6, 12) if group.camera == "cam2" else slice(0, 6)
        assert torch.count_nonzero(local[:, zero_slice]) == 0
        assert torch.count_nonzero(local[:, translation_slice]) > 0
        assert torch.count_nonzero(local[:, 12:75]) > 0


def test_anchor_surrogate_and_multiframe_blocks() -> None:
    cache, parameters = _cache(2), _parameters(2)
    anchors = [torch.zeros(63, dtype=DTYPE), torch.ones(63, dtype=DTYPE) * .01]
    residual, jacobian = stage2_pose_anchor_residual_and_jacobian(parameters, anchors, 2, .3)
    oracle = jacfwd(lambda x: torch.cat([
        .3 * (x[f*75+12:f*75+75] - anchors[f]) for f in range(2)
    ]))(parameters)
    torch.testing.assert_close(jacobian, oracle)
    assert residual.numel() == 126
    global_j = stage2_pose_explicit_raw_jacobian(cache, parameters)
    split = sum(group.targets.numel() for group in cache.groups[:2])
    assert torch.count_nonzero(global_j[:split, 75:]) == 0
    assert torch.count_nonzero(global_j[split:, :75]) == 0
    for mode in ("charbonnier_irls", "raw_l2"):
        result = stage2_pose_surrogate_residual_and_jacobian(
            cache, parameters, anchors, .3, objective_mode=mode,
            charbonnier_eps=1., rep_scale=.02,
        )
        raw_rows = stage2_pose_raw_residual(cache, parameters).numel()
        scale = result.residual[:raw_rows] / stage2_pose_raw_residual(cache, parameters)
        if mode == "charbonnier_irls":
            torch.testing.assert_close(scale[0::2], scale[1::2])
        else:
            torch.testing.assert_close(scale, torch.full_like(scale, .02))
            torch.testing.assert_close(result.jacobian[:raw_rows], global_j * .02)
        assert torch.isfinite(result.residual).all() and torch.isfinite(result.jacobian).all()


def test_residual_only_direct_normal_equations_and_fast_eval() -> None:
    cache, parameters = _cache(2), _parameters(2)
    anchors = [torch.zeros(63, dtype=DTYPE), torch.ones(63, dtype=DTYPE) * .01]
    cache.reset_timing()
    residual_only = fast_raw_residual(cache, parameters)
    assert cache.group_analytic_calls == 0 and cache.residual_calls == 1
    groups = fast_raw_groups(cache, parameters)
    spec = build_primary_aggregation_spec(groups)
    for is_irls in (True, False):
        cache.reset_timing()
        linearization = build_stage2_pose_analytic_linearization(
            cache, parameters, anchors, .3, is_irls=is_irls,
            charbonnier_eps=1., aggregation_spec=spec if is_irls else None,
            rep_scale=.02,
        )
        explicit_residual = linearization.surrogate(parameters)
        explicit_jacobian = linearization.jacobian(parameters)
        cache.group_analytic_calls = 0
        equations = linearization.normal_equations(parameters)
        assert cache.group_analytic_calls == len(cache.groups)
        torch.testing.assert_close(equations.residual, explicit_residual, rtol=1e-10, atol=1e-10)
        reference_j = explicit_jacobian.to(torch.float64)
        reference_r = explicit_residual.to(torch.float64)
        torch.testing.assert_close(equations.hessian, reference_j.T @ reference_j, rtol=1e-10, atol=1e-8)
        torch.testing.assert_close(equations.gradient, reference_j.T @ reference_r, rtol=1e-10, atol=1e-8)
        assert torch.count_nonzero(equations.hessian[:75, 75:]) == 0
        assert torch.count_nonzero(equations.hessian[75:, :75]) == 0
        torch.testing.assert_close(
            equations.residual[:residual_only.numel()], explicit_residual[:residual_only.numel()],
        )
        assert torch.count_nonzero(equations.hessian[:12, 12:75]) > 0

    fast_charbonnier = stage2_pose_fast_true_objective(
        cache, parameters, anchors, .3, "charbonnier_irls", 1., .02, spec,
    )
    anchor_residual, _ = stage2_pose_anchor_residual_and_jacobian(parameters, anchors, 2, .3)
    manual_charbonnier = compute_primary_charbonnier_tensor(groups, 1., spec) + .5 * torch.sum(anchor_residual ** 2)
    torch.testing.assert_close(fast_charbonnier, manual_charbonnier)
    fast_l2 = stage2_pose_fast_true_objective(
        cache, parameters, anchors, .3, "raw_l2", 1., .02, None,
    )
    manual_l2 = .5 * torch.sum((flatten_reprojection_groups(groups) * .02) ** 2) + .5 * torch.sum(anchor_residual ** 2)
    torch.testing.assert_close(fast_l2, manual_l2)
    assert stage2_pose_fast_raw_metrics(cache, parameters) == raw_reproj_metrics(
        flatten_reprojection_groups(groups),
    )
    assert cache.objective_calls == 2 and cache.metrics_calls == 1
    setup = cache.setup_ms
    cache.reset_timing()
    assert cache.objective_calls == cache.metrics_calls == cache.normal_equations_calls == 0
    assert cache.setup_ms == setup


def test_unsupported_layout_is_explicit() -> None:
    cache = _cache()
    cache.body_pose_dim = 6
    try:
        validate_stage2_pose_analytic_layout(cache)
    except RuntimeError as exc:
        assert "UNSUPPORTED_STAGE2_POSE_ANALYTIC_LAYOUT" in str(exc)
        assert "actual body_pose_dim=6" in str(exc)
    else:
        raise AssertionError("unsupported Stage 2 pose layout was accepted")


if __name__ == "__main__":
    test_rotation_matrix_jacobian()
    test_pose_feature_and_offset_jacobians()
    test_fk_parent_chain_and_relative_derivative()
    test_selected_lbs_product_rule()
    test_vertex_and_group_jacobian_layout()
    test_anchor_surrogate_and_multiframe_blocks()
    test_residual_only_direct_normal_equations_and_fast_eval()
    test_unsupported_layout_is_explicit()
    print("All Stage 2 pose Jacobian tests PASSED")
