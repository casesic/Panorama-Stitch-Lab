"""Numerical gates for benchmark-only padded Stage1/Pose/Joint candidates."""

from __future__ import annotations

if __package__ in (None, ""):
    from _bootstrap import ensure_test_paths
    ensure_test_paths(__file__)

import torch

from helpers import synthetic_joint_cache, synthetic_joint_parameters, synthetic_stage2_context
from lib_pano.human_fitting.staged_lm.analytic_stage1 import (
    Stage1AnalyticCache, Stage1AnalyticGroupCache,
    _group_raw_residual_and_local_jacobian, stage1_normal_equations,
)
from lib_pano.human_fitting.staged_lm.stage2_shared_cache import build_stage2_shared_geometry_cache
from lib_pano.human_fitting.staged_lm.pose.cache import (
    build_stage2_pose_fast_cache,
)
from lib_pano.human_fitting.staged_lm.pose.objective import stage2_pose_fast_true_objective
from lib_pano.human_fitting.staged_lm.pose.derivatives import (
    stage2_pose_group_raw_residual_and_jacobian, stage2_pose_normal_equations,
)
from lib_pano.human_fitting.staged_lm.joint import stage2_joint_fast_objective_terms
from lib_pano.human_fitting.staged_lm.joint import (
    accumulate_stage2_joint_arrowhead_system, joint_group_raw_residual_and_jacobians,
)
from script.benchmark.staged_lm.stage1_batch_candidates import (
    build_stage1_group_batch, gather_stage1_group_parameters,
    batched_stage1_residual_and_jacobian, batched_stage1_hg,
)
from script.benchmark.staged_lm.pose_batch_candidates import (
    build_pose_group_batch, gather_pose_batch_parameters, gather_pose_group_parameters,
    batched_pose_group_analytic, batched_pose_hg, batched_pose_objective,
)
from script.benchmark.staged_lm.joint_kernel_candidates import (
    build_joint_group_batch, batched_joint_arrowhead_hg, batched_joint_objective,
)


def _stage1_fixture(frames=2):
    torch.manual_seed(81); groups=[]
    for frame in range(frames):
        for camera, count in (("cam2",2),("cam4",3)):
            groups.append(Stage1AnalyticGroupCache(frame,513+frame,camera,
                torch.randn(count,3,dtype=torch.float64)*.1,
                torch.tensor([0.,0.,2.],dtype=torch.float64),
                torch.randn(count,2,dtype=torch.float64)*10+100))
    cache=Stage1AnalyticCache(groups,frames,5760,2880,0.)
    parameters=torch.randn(frames*12,dtype=torch.float64)*.01
    return cache,parameters


def test_stage1_batch_residual_jacobian_hg():
    cache,parameters=_stage1_fixture();batch=build_stage1_group_batch(cache)
    orient,translation=gather_stage1_group_parameters(parameters,batch.layout)
    residual,jac=batched_stage1_residual_and_jacobian(batch,orient,translation,5760,2880)
    for index,group in enumerate(cache.groups):
        rr,jj=_group_raw_residual_and_local_jacobian(cache,parameters,group);count=group.targets.shape[0]
        camera=slice(0,6) if group.camera=="cam2" else slice(6,12)
        torch.testing.assert_close(residual[index,:count].reshape(-1),rr,rtol=1e-10,atol=1e-10)
        torch.testing.assert_close(jac[index,:count].reshape(-1,6),jj[:,camera],rtol=1e-10,atol=1e-10)
        assert torch.count_nonzero(residual[index,count:])==0
    scales=torch.ones(batch.layout.group_count,batch.layout.nmax,2,dtype=torch.float64)
    _,_,_,h,g=batched_stage1_hg(batch,orient,translation,scales,5760,2880)
    flat_scales=torch.ones(sum(x.targets.numel() for x in cache.groups),dtype=torch.float64)
    reference=stage1_normal_equations(cache,parameters,flat_scales)
    torch.testing.assert_close(h,reference.hessian,rtol=1e-10,atol=1e-10)
    torch.testing.assert_close(g,reference.gradient,rtol=1e-10,atol=1e-10)


def test_pose_batch_residual_active_jacobian_hg_objective():
    context=synthetic_stage2_context(frames=2);shared=build_stage2_shared_geometry_cache(context)
    cache=build_stage2_pose_fast_cache(context,shared);parameters=synthetic_joint_parameters(2)[:-2]
    batch=build_pose_group_batch(cache);orient,translation,body=gather_pose_group_parameters(parameters,batch.layout)
    pose_parameters=gather_pose_batch_parameters(parameters,batch.layout)
    residual,active,_=batched_pose_group_analytic(batch,orient,translation,body,5760,2880)
    for index,group in enumerate(cache.groups):
        rr,jj=stage2_pose_group_raw_residual_and_jacobian(cache,group,parameters);count=group.targets.shape[0]
        columns=torch.cat((torch.arange(0,6) if group.camera=="cam2" else torch.arange(6,12),torch.arange(12,75)))
        torch.testing.assert_close(residual[index,:count].reshape(-1),rr,rtol=2e-5,atol=1e-7)
        torch.testing.assert_close(active[index,:count].reshape(-1,69),jj[:,columns],rtol=2e-5,atol=1e-7)
    scales=torch.ones(batch.layout.group_count,batch.layout.nmax,2,dtype=torch.float64)
    anchors=[torch.zeros(63,dtype=torch.float64) for _ in range(2)]
    _,_,h,g=batched_pose_hg(batch,orient,translation,body,scales,anchors,.3,5760,2880,
        frame_body_poses=pose_parameters.frame_body_poses)
    reference=stage2_pose_normal_equations(cache,parameters,[torch.ones(x.targets.numel(),dtype=torch.float64) for x in cache.groups],anchors,.3)
    torch.testing.assert_close(h,reference.hessian,rtol=2e-5,atol=1e-7);torch.testing.assert_close(g,reference.gradient,rtol=2e-5,atol=1e-7)
    candidate=batched_pose_objective(batch,orient,translation,body,anchors,.3,5760,2880,objective_mode="raw_l2",charbonnier_eps=1.,rep_scale=.2,aggregation_spec=None,
        frame_body_poses=pose_parameters.frame_body_poses)
    truth=stage2_pose_fast_true_objective(cache,parameters,anchors,.3,"raw_l2",1.,.2,None)
    torch.testing.assert_close(candidate["total"],truth,rtol=1e-10,atol=1e-10)


def test_joint_batch_arrowhead_and_objective():
    cache=synthetic_joint_cache(2);parameters=synthetic_joint_parameters(2);batch=build_joint_group_batch(cache.groups)
    anchors=[torch.zeros(63,dtype=torch.float64) for _ in range(2)];beta_anchor=torch.zeros(2,dtype=torch.float64)
    scales=torch.ones(batch.valid_mask.shape+(2,),dtype=torch.float64)
    candidate, residual, active, beta_j=batched_joint_arrowhead_hg(cache,batch,parameters,1.,scales,anchors,beta_anchor,.3,.2)
    _,reference=accumulate_stage2_joint_arrowhead_system(cache,parameters,1.,[torch.ones(x.targets.numel(),dtype=torch.float64) for x in cache.groups],anchors,beta_anchor,.3,.2)
    torch.testing.assert_close(candidate.to_dense(),reference.to_dense(),rtol=2e-5,atol=1e-7)
    torch.testing.assert_close(candidate.flat_gradient(),reference.flat_gradient(),rtol=2e-5,atol=1e-7)
    terms=batched_joint_objective(cache,batch,parameters,1.,anchors,beta_anchor,.3,.2,objective_mode="raw_l2",charbonnier_eps=1.,rep_scale=.2,aggregation_spec=None)
    truth=stage2_joint_fast_objective_terms(cache,parameters,1.,beta_anchor,anchors,.3,.2,"raw_l2",1.,.2,None)
    torch.testing.assert_close(terms["total"],truth["total"],rtol=1e-10,atol=1e-10)


if __name__ == "__main__":
    test_stage1_batch_residual_jacobian_hg()
    test_pose_batch_residual_active_jacobian_hg_objective()
    test_joint_batch_arrowhead_and_objective()
    print("[PASS] batched stage candidates")
