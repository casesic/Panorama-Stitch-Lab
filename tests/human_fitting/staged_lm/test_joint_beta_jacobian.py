"""Pure mathematical tests for dynamic Stage 2 joint beta Jacobians."""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch.func import jacfwd

try:
    from ._bootstrap import ensure_test_paths
except ImportError:
    from _bootstrap import ensure_test_paths
ensure_test_paths(__file__)

from lib_pano.human_fitting.staged_lm.irls import build_primary_aggregation_spec, compute_primary_charbonnier_tensor  # noqa: E402
from lib_pano.human_fitting.staged_lm.residuals import flatten_reprojection_groups  # noqa: E402
from lib_pano.human_fitting.staged_lm.stage2_joint_fast import (  # noqa: E402
    Stage2JointFastCache, Stage2JointFrameCameraCache, Stage2JointModelCache,
    Stage2JointShapeState, evaluate_joint_shape_state, joint_fast_raw_groups,
    joint_fast_raw_residual, joint_fast_selected_vertices, physical_beta_from_latent,
)
from lib_pano.human_fitting.staged_lm.stage2_joint_beta_jacobian import (  # noqa: E402
    compute_fk_beta_jacobian, joint_explicit_raw_beta_jacobian,
    joint_explicit_raw_global_beta_block, joint_fast_selected_vertices_with_beta_jacobian,
    joint_group_raw_residual_and_beta_jacobian, joint_shape_anchor_residual_and_jacobian,
    joint_surrogate_residual_and_beta_jacobian,
)
from lib_pano.human_fitting.staged_lm.stage2_pose_fast import _aa_to_rotmat  # noqa: E402

from tests.human_fitting.staged_lm.helpers import DTYPE, _cache, _parameters


def test_beta_mapping_and_shape_state() -> None:
    latent = torch.tensor([0., 4.], dtype=DTYPE)
    beta, derivative = physical_beta_from_latent(latent, 3.)
    torch.testing.assert_close(beta, 3 * torch.tanh(latent))
    torch.testing.assert_close(derivative, jacfwd(lambda z: 3 * torch.tanh(z))(latent).diag())
    assert torch.isfinite(beta).all() and torch.isfinite(derivative).all()
    cache = _cache(); state = evaluate_joint_shape_state(cache, latent, 3.)
    oracle = jacfwd(lambda z: cache.model.base_joints + torch.einsum(
        "jcb,b->jc", cache.model.d_joints_d_beta, 3 * torch.tanh(z),
    ))(latent)
    torch.testing.assert_close(state.d_joints_d_latent, oracle)
    assert cache.shape_state_calls == 1


def test_fk_beta_complete_product_rule() -> None:
    torch.manual_seed(53)
    j, b = 4, 2
    parents = torch.tensor([-1, 0, 1, 2])
    rotations = _aa_to_rotmat(torch.randn(j, 3, dtype=DTYPE) * .1)
    base = torch.randn(j, 3, dtype=DTYPE)
    dirs = torch.randn(j, 3, b, dtype=DTYPE) * .1
    beta = torch.randn(b, dtype=DTYPE)
    result = compute_fk_beta_jacobian(rotations, base + torch.einsum("jcb,b->jc", dirs, beta), dirs, parents)
    def fn(value):
        joints = base + torch.einsum("jcb,b->jc", dirs, value)
        relative = joints.clone(); relative[1:] -= joints[parents[1:]]
        local = torch.zeros(j, 4, 4, dtype=DTYPE); local[:, :3, :3] = rotations
        local[:, :3, 3] = relative; local[:, 3, 3] = 1
        gs = [local[0]]
        for k in range(1, j): gs.append(gs[int(parents[k])] @ local[k])
        gs = torch.stack(gs)
        rest = torch.eye(4, dtype=DTYPE).repeat(j, 1, 1); rest[:, :3, 3] = -joints
        return gs, rest, gs @ rest
    global_j, rest_j, relative_j = jacfwd(fn)(beta)
    torch.testing.assert_close(result.d_global_transforms_d_beta, global_j)
    torch.testing.assert_close(result.d_rest_inverse_d_beta, rest_j)
    torch.testing.assert_close(result.d_relative_transforms_d_beta, relative_j)
    assert torch.count_nonzero(result.d_global_transforms_d_beta[3]) > 0


def test_selected_vertex_physical_and_latent_jacobian() -> None:
    cache, x = _cache(), _parameters()
    group, latent = cache.groups[0], x[-2:]
    state = evaluate_joint_shape_state(cache, latent, 3.)
    vertices, physical_j, latent_j, intermediate = joint_fast_selected_vertices_with_beta_jacobian(
        cache, group, state, x[:3], x[3:6], x[12:75], return_intermediates=True,
    )
    def physical_fn(beta):
        shape = Stage2JointShapeState(
            beta, torch.ones_like(beta),
            cache.model.base_joints + torch.einsum("jcb,b->jc", cache.model.d_joints_d_beta, beta),
            cache.model.d_joints_d_beta, cache.model.d_joints_d_beta,
        )
        return joint_fast_selected_vertices(cache, group, shape, x[:3], x[3:6], x[12:75])
    physical_oracle = jacfwd(physical_fn)(state.beta)
    latent_oracle = jacfwd(lambda z: physical_fn(3 * torch.tanh(z)))(latent)
    torch.testing.assert_close(physical_j, physical_oracle, rtol=1e-8, atol=1e-9)
    torch.testing.assert_close(latent_j, latent_oracle, rtol=1e-8, atol=1e-9)
    torch.testing.assert_close(latent_j, physical_j * state.d_beta_d_latent.reshape(1, 1, -1))
    # Pose offsets are beta-constant; v_posed inherits exactly the shape derivative.
    torch.testing.assert_close(intermediate["d_selected_v_shaped_d_beta"], group.selected_beta_shapedirs)
    assert torch.isfinite(vertices).all() and torch.isfinite(latent_j).all()


def test_projection_anchor_surrogate_and_shared_layout() -> None:
    cache, x = _cache(2), _parameters(2)
    state = evaluate_joint_shape_state(cache, x[-2:], 3.)
    for group in cache.groups:
        residual, analytic = joint_group_raw_residual_and_beta_jacobian(cache, group, x, 3., state)
        oracle = jacfwd(lambda z: joint_fast_raw_groups(
            cache, torch.cat((x[:-2], z)), 3.,
        )[cache.groups.index(group)].residual_2d.reshape(-1))(x[-2:])
        torch.testing.assert_close(analytic, oracle, rtol=1e-7, atol=1e-7)
        assert residual.shape[0] == group.targets.numel()
    beta_j = joint_explicit_raw_beta_jacobian(cache, x, 3.)
    global_j = joint_explicit_raw_global_beta_block(cache, x, 3.)
    assert torch.count_nonzero(global_j[:, :-2]) == 0
    torch.testing.assert_close(global_j[:, -2:], beta_j)
    anchor, anchor_j = joint_shape_anchor_residual_and_jacobian(x[-2:], torch.zeros(2), 3., .4)
    torch.testing.assert_close(anchor_j, jacfwd(lambda z: (3*torch.tanh(z))*.4)(x[-2:]))
    anchors = [torch.zeros(63, dtype=DTYPE), torch.zeros(63, dtype=DTYPE)]
    for mode in ("charbonnier_irls", "raw_l2"):
        result = joint_surrogate_residual_and_beta_jacobian(
            cache, x, 3., torch.zeros(2), anchors, .2, .4,
            objective_mode=mode, charbonnier_eps=1., rep_scale=.02,
        )
        raw_rows = sum(group.targets.numel() for group in cache.groups)
        assert torch.count_nonzero(result.beta_jacobian[raw_rows:raw_rows+126]) == 0
        torch.testing.assert_close(result.residual[-2:], anchor)
        torch.testing.assert_close(result.beta_jacobian[-2:], anchor_j)
        # Frozen surrogate beta gradient equals the true objective gradient.
        groups = joint_fast_raw_groups(cache, x, 3.)
        spec = build_primary_aggregation_spec(groups)
        def objective(z):
            values = torch.cat((x[:-2], z))
            gs = joint_fast_raw_groups(cache, values, 3.)
            if mode == "charbonnier_irls": primary = compute_primary_charbonnier_tensor(gs, 1., spec)
            else: primary = .5 * torch.sum((flatten_reprojection_groups(gs) * .02) ** 2)
            pose = torch.cat([(values[f*75+12:(f+1)*75]-anchors[f])*.2 for f in range(2)])
            shape = (3*torch.tanh(z))*.4
            return primary + .5*torch.sum(pose**2) + .5*torch.sum(shape**2)
        true_gradient = torch.func.grad(objective)(x[-2:])
        surrogate_gradient = result.beta_jacobian.T @ result.residual
        torch.testing.assert_close(surrogate_gradient, true_gradient, rtol=1e-7, atol=1e-7)
    # Residual-only computes one shared shape state and no analytic beta J.
    cache.reset_timing(); joint_fast_raw_residual(cache, x, 3.)
    assert cache.shape_state_calls == 1 and cache.beta_jacobian_calls == 0
    setup = cache.setup_ms; cache.reset_timing(); assert cache.setup_ms == setup


if __name__ == "__main__":
    test_beta_mapping_and_shape_state()
    test_fk_beta_complete_product_rule()
    test_selected_vertex_physical_and_latent_jacobian()
    test_projection_anchor_surrogate_and_shared_layout()
    print("All Stage 2 joint beta Jacobian tests PASSED")
