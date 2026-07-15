"""Tests for the complete dynamic Stage 2 joint analytic Jacobian."""

from __future__ import annotations

from contextlib import nullcontext
import sys
from pathlib import Path

import torch
from torch.func import jacfwd

try:
    from ._bootstrap import ensure_test_paths
except ImportError:
    from _bootstrap import ensure_test_paths
ensure_test_paths(__file__)
from tests.human_fitting.staged_lm.helpers import _cache, _parameters  # noqa: E402

from lib_pano.human_fitting.staged_lm.irls import build_primary_aggregation_spec,compute_primary_charbonnier_tensor  # noqa: E402
from lib_pano.human_fitting.staged_lm.residuals import flatten_reprojection_groups  # noqa: E402
from lib_pano.human_fitting.staged_lm.stage2_joint_fast import (  # noqa: E402
    evaluate_joint_shape_state,joint_fast_raw_groups,joint_fast_raw_residual,validate_stage2_joint_layout,
)
from lib_pano.human_fitting.staged_lm.stage2_joint_beta_jacobian import (  # noqa: E402
    compute_fk_beta_jacobian,joint_group_raw_residual_and_beta_jacobian,
)
from lib_pano.human_fitting.staged_lm.stage2_joint_jacobian import (  # noqa: E402
    accumulate_stage2_joint_arrowhead_system,
    compute_joint_fk_with_combined_jacobian,evaluate_joint_group_analytic_state,
    joint_explicit_raw_global_jacobian,joint_group_raw_residual_and_jacobians,
    joint_pose_anchor_residual_and_jacobian,joint_surrogate_residual_and_jacobian,
    build_stage2_joint_analytic_linearization,
    stage2_joint_arrowhead_normal_equations,
)
from script.validation.staged_lm.references.joint_backends import (  # noqa: E402
    build_stage2_joint_reference_linearization,
    use_separate_fk_reference,
)
from lib_pano.human_fitting.staged_lm.stage2_pose_jacobian import (  # noqa: E402
    assemble_full_pose_rotation_jacobians,compute_fk_with_jacobian,
)


def test_combined_fk_matches_existing_oracles() -> None:
    cache,x=_cache(),_parameters(); group=cache.groups[0]; shape=evaluate_joint_shape_state(cache,x[-2:],3.)
    rotations,d_rotations=assemble_full_pose_rotation_jacobians(x[:3],x[12:75],group.left_hand_pose,group.right_hand_pose,cache.model)
    combined=compute_joint_fk_with_combined_jacobian(
        rotations,d_rotations,shape.joints,shape.d_joints_d_latent,cache.model.parents,
    )
    pose=compute_fk_with_jacobian(rotations,d_rotations,shape.joints,cache.model.parents)
    beta=compute_fk_beta_jacobian(rotations,shape.joints,shape.d_joints_d_beta,cache.model.parents)
    torch.testing.assert_close(combined.global_transforms,pose.global_transforms)
    torch.testing.assert_close(combined.relative_transforms,pose.relative_transforms)
    torch.testing.assert_close(combined.d_relative_transforms_d_pose,pose.d_relative_transforms)
    torch.testing.assert_close(
        combined.d_relative_transforms_d_beta,
        beta.d_relative_transforms_d_beta*shape.d_beta_d_latent.reshape(1,1,1,-1),
    )


def test_group_dynamic_pose_and_beta_jacobian() -> None:
    cache,x=_cache(),_parameters(); shape=evaluate_joint_shape_state(cache,x[-2:],3.)
    cache.reset_timing()
    for group in cache.groups:
        result=joint_group_raw_residual_and_jacobians(cache,group,x,3.,shape,return_intermediates=True)
        group_index=cache.groups.index(group)
        oracle=jacfwd(lambda value: joint_fast_raw_groups(cache,value,3.)[group_index].residual_2d.reshape(-1))(x)
        torch.testing.assert_close(result.local_pose_jacobian,oracle[:,:75],rtol=2e-7,atol=2e-7)
        torch.testing.assert_close(result.shared_beta_jacobian,oracle[:,-2:],rtol=2e-7,atol=2e-7)
        beta_residual,beta_j=joint_group_raw_residual_and_beta_jacobian(cache,group,x,3.,shape)
        torch.testing.assert_close(result.raw_residual,beta_residual)
        torch.testing.assert_close(result.shared_beta_jacobian,beta_j,rtol=2e-7,atol=2e-7)
        zero=slice(6,12) if group.camera=="cam2" else slice(0,6)
        assert torch.count_nonzero(result.local_pose_jacobian[:,zero])==0
        assert result.intermediates is not None
    # Each combined group evaluation performs exactly one selected LBS forward.
    # Extra oracle calls above use the residual-only path and do not increment it.
    assert cache.group_analytic_calls==len(cache.groups)
    assert cache.selected_lbs_forward_calls==len(cache.groups)


def test_separate_and_combined_group_and_hg_regression() -> None:
    for frames in (1, 2):
        cache, x = _cache(frames), _parameters(frames)
        anchors = [torch.zeros(63, dtype=x.dtype) for _ in range(frames)]
        scales = [torch.linspace(.01, .02, group.targets.numel(), dtype=x.dtype)
                  for group in cache.groups]
        group_results = {}
        systems = {}
        for backend in ("separate", "combined"):
            cache.reset_timing()
            shape = evaluate_joint_shape_state(cache, x[-2:], 3.)
            manager = use_separate_fk_reference() if backend == "separate" else nullcontext()
            with manager:
                group_results[backend] = [
                    joint_group_raw_residual_and_jacobians(
                        cache, group, x, 3., shape, return_intermediates=True,
                    ) for group in cache.groups
                ]
                cache.reset_timing()
                systems[backend] = accumulate_stage2_joint_arrowhead_system(
                    cache, x, 3., scales, anchors,
                    torch.zeros(2, dtype=x.dtype), .2, .4,
                )
            assert cache.shape_state_calls == 1
            assert cache.group_analytic_calls == len(cache.groups)
            assert cache.selected_lbs_forward_calls == len(cache.groups)
        for left, right in zip(group_results["separate"], group_results["combined"]):
            torch.testing.assert_close(left.raw_residual, right.raw_residual, rtol=0, atol=1e-6)
            torch.testing.assert_close(left.local_pose_jacobian, right.local_pose_jacobian, rtol=1e-7, atol=1e-6)
            torch.testing.assert_close(left.shared_beta_jacobian, right.shared_beta_jacobian, rtol=1e-7, atol=1e-6)
            assert left.intermediates is not None and right.intermediates is not None
            for name in ("global_transforms", "relative_transforms",
                         "d_relative_transforms_d_pose", "d_relative_transforms_d_beta_latent",
                         "selected_transforms", "d_selected_transforms_d_pose",
                         "d_selected_transforms_d_beta_latent", "selected_skinned_vertices"):
                torch.testing.assert_close(left.intermediates[name], right.intermediates[name], rtol=1e-7, atol=1e-6)
        residual_s, arrow_s = systems["separate"]
        residual_c, arrow_c = systems["combined"]
        torch.testing.assert_close(residual_s, residual_c, rtol=0, atol=1e-8)
        for name in ("frame_hessians", "frame_beta_cross", "beta_hessian",
                     "frame_gradients", "beta_gradient"):
            torch.testing.assert_close(getattr(arrow_s, name), getattr(arrow_c, name), rtol=1e-10, atol=1e-8)
    cache.reset_timing(); shape=evaluate_joint_shape_state(cache,x[-2:],3.)
    for group in cache.groups:
        joint_group_raw_residual_and_jacobians(cache,group,x,3.,shape)
    assert cache.shape_state_calls==1
    assert cache.group_analytic_calls==len(cache.groups)
    assert cache.selected_lbs_forward_calls==len(cache.groups)


def test_global_layout_anchors_and_cross_block() -> None:
    cache,x=_cache(2),_parameters(2); per_frame,total=validate_stage2_joint_layout(cache,x)
    jacobian=joint_explicit_raw_global_jacobian(cache,x,3.)
    oracle=jacfwd(lambda value:joint_fast_raw_residual(cache,value,3.))(x)
    torch.testing.assert_close(jacobian,oracle,rtol=2e-7,atol=2e-7)
    split=sum(group.targets.numel() for group in cache.groups[:2])
    assert torch.count_nonzero(jacobian[:split,per_frame:2*per_frame])==0
    assert torch.count_nonzero(jacobian[split:,:per_frame])==0
    assert torch.count_nonzero(jacobian[:,-2:])>0 and jacobian.shape[1]==2*per_frame+2
    anchors=[torch.zeros(63,dtype=x.dtype),torch.ones(63,dtype=x.dtype)*.01]
    pose_r,pose_j=joint_pose_anchor_residual_and_jacobian(cache,x,anchors,.2)
    assert torch.count_nonzero(pose_j[:,-2:])==0 and pose_r.numel()==126
    cross=jacobian[:,:-2].T@jacobian[:,-2:]
    oracle_cross=oracle[:,:-2].T@oracle[:,-2:]
    torch.testing.assert_close(cross,oracle_cross,rtol=1e-8,atol=1e-8)


def test_complete_surrogate_and_true_gradient() -> None:
    cache,x=_cache(2),_parameters(2); anchors=[torch.zeros(63,dtype=x.dtype) for _ in range(2)]
    groups=joint_fast_raw_groups(cache,x,3.); spec=build_primary_aggregation_spec(groups)
    for mode in ("charbonnier_irls","raw_l2"):
        result=joint_surrogate_residual_and_jacobian(
            cache,x,3.,torch.zeros(2,dtype=x.dtype),anchors,.2,.4,
            objective_mode=mode,charbonnier_eps=1.,rep_scale=.02,
        )
        def objective(value):
            gs=joint_fast_raw_groups(cache,value,3.)
            primary=(compute_primary_charbonnier_tensor(gs,1.,spec) if mode=="charbonnier_irls"
                     else .5*torch.sum((flatten_reprojection_groups(gs)*.02)**2))
            pose=torch.cat([(value[f*75+12:(f+1)*75]-anchors[f])*.2 for f in range(2)])
            shape=(3*torch.tanh(value[-2:]))*.4
            return primary+.5*torch.sum(pose**2)+.5*torch.sum(shape**2)
        true_grad=torch.func.grad(objective)(x)
        torch.testing.assert_close(result.jacobian.T@result.residual,true_grad,rtol=2e-7,atol=2e-7)
    cache.reset_timing(); joint_fast_raw_residual(cache,x,3.)
    assert cache.shape_state_calls==1 and cache.group_analytic_calls==0


def test_layout_failure_and_saturation_finite() -> None:
    cache,x=_cache(),_parameters(); bad=x[:-1]
    try: validate_stage2_joint_layout(cache,bad)
    except RuntimeError as exc: assert "UNSUPPORTED_STAGE2_JOINT_ANALYTIC_LAYOUT" in str(exc)
    else: raise AssertionError("bad layout accepted")
    x[-2:]=torch.tensor([4.,-4.],dtype=x.dtype)
    jacobian=joint_explicit_raw_global_jacobian(cache,x,3.)
    assert torch.isfinite(jacobian).all()


def test_direct_normal_equations_match_frozen_explicit() -> None:
    for frames in (1,2):
        cache,x=_cache(frames),_parameters(frames);anchors=[torch.zeros(63,dtype=x.dtype) for _ in range(frames)]
        spec=build_primary_aggregation_spec(joint_fast_raw_groups(cache,x,3.))
        for is_irls in (True,False):
            linear=build_stage2_joint_reference_linearization(cache,x,3.,torch.zeros(2,dtype=x.dtype),anchors,.2,.4,
                is_irls=is_irls,charbonnier_eps=1.,aggregation_spec=spec if is_irls else None,rep_scale=.02,
                linear_solver_backend="dense")
            residual=linear["surrogate_fn"](x);jacobian=linear["jacobian_fn"](x)
            cache.reset_timing();equations=linear["normal_equations_fn"](x)
            schur_linear=build_stage2_joint_analytic_linearization(
                cache,x,3.,torch.zeros(2,dtype=x.dtype),anchors,.2,.4,
                is_irls=is_irls,charbonnier_eps=1.,aggregation_spec=spec if is_irls else None,
                rep_scale=.02,
            )
            structured=schur_linear["normal_equations_fn"](x)
            torch.testing.assert_close(equations.residual,residual,rtol=1e-10,atol=1e-10)
            torch.testing.assert_close(structured.residual,residual,rtol=1e-10,atol=1e-10)
            j64=jacobian.to(torch.float64);r64=residual.to(torch.float64)
            torch.testing.assert_close(equations.hessian,j64.T@j64,rtol=1e-10,atol=1e-8)
            torch.testing.assert_close(equations.gradient,j64.T@r64,rtol=1e-10,atol=1e-8)
            torch.testing.assert_close(structured.structured_system.to_dense(count_call=False),equations.hessian,rtol=1e-10,atol=1e-8)
            torch.testing.assert_close(structured.gradient,equations.gradient,rtol=1e-10,atol=1e-8)
            per=75
            if frames==2:
                assert torch.count_nonzero(equations.hessian[:per,per:2*per])==0
            cross=equations.hessian[:per,-2:]
            assert torch.count_nonzero(cross)>0
            torch.testing.assert_close(equations.hessian[-2:,:per],cross.T)
            assert torch.isfinite(equations.hessian).all()


if __name__=="__main__":
    test_combined_fk_matches_existing_oracles(); test_group_dynamic_pose_and_beta_jacobian()
    test_separate_and_combined_group_and_hg_regression()
    test_global_layout_anchors_and_cross_block(); test_complete_surrogate_and_true_gradient()
    test_layout_failure_and_saturation_finite()
    test_direct_normal_equations_match_frozen_explicit()
    print("All Stage 2 joint combined Jacobian tests PASSED")
