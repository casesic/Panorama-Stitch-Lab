"""Synthetic regression tests for benchmark-only Pose LM shadow paths."""

from __future__ import annotations

if __package__ in (None, ""):
    from _bootstrap import ensure_test_paths
    ensure_test_paths(__file__)

from types import SimpleNamespace
import torch

from helpers import synthetic_joint_parameters, synthetic_stage2_context
from lib_pano.human_fitting.staged_lm.config import SolverConfig
from lib_pano.human_fitting.staged_lm.irls import build_primary_aggregation_spec
from lib_pano.human_fitting.staged_lm.stage2_shared_cache import build_stage2_shared_geometry_cache
from lib_pano.human_fitting.staged_lm.pose.cache import build_stage2_pose_fast_cache
from lib_pano.human_fitting.staged_lm.pose.forward import fast_raw_groups
from lib_pano.human_fitting.staged_lm.pose.objective import stage2_pose_fast_true_objective
from lib_pano.human_fitting.staged_lm.pose.derivatives import build_stage2_pose_analytic_linearization
from lib_pano.human_fitting.staged_lm.types import NormalEquations, StageDefinition
from script.benchmark.staged_lm.pose_batch_candidates import (
    build_pose_group_batch, gather_pose_batch_parameters,
)
from script.benchmark.staged_lm.pose_shadow_common import (
    PoseNEMode, compute_batched_pose_row_scales, evaluate_pose_equations,
    serial_pose_row_scales,
)
from script.benchmark.staged_lm.pose_shadow_lm import (
    compare_lm_results, run_pose_lm_independent_batch, run_pose_lm_lockstep,
)


def _fixture(frames=2, objective="raw_l2"):
    base=synthetic_stage2_context(frames);shared=build_stage2_shared_geometry_cache(base)
    cache=build_stage2_pose_fast_cache(base,shared);parameters=synthetic_joint_parameters(frames)[:-2]
    anchors=[torch.zeros(63,dtype=parameters.dtype) for _ in range(frames)]
    groups=fast_raw_groups(cache,parameters);spec=build_primary_aggregation_spec(groups)
    scale=.3;eps=1.;rep=.2
    stage=StageDefinition("POSE_SHADOW_TEST",frames*75,parameters,lambda x:fast_raw_groups(cache,x),None,
        cache.raw_metrics,1,analytic_cache=cache,
        linearization_builder=lambda x,s:build_stage2_pose_analytic_linearization(cache,x,anchors,scale,
            is_irls=objective=="charbonnier_irls",charbonnier_eps=eps,aggregation_spec=s,rep_scale=rep),
        lm_true_objective_fn=lambda x:stage2_pose_fast_true_objective(cache,x,anchors,scale,objective,eps,rep,spec),
        aggregation_spec=spec)
    context=SimpleNamespace(args=SimpleNamespace(reproj_objective=objective),charbonnier_eps=eps,
        scaling=SimpleNamespace(rep_scale=rep,pose_anchor_scale=scale),body_pose_anchors=anchors,
        image_width=5760,image_height=2880,solver_config=SolverConfig(lm_max_trials=3))
    return stage,build_pose_group_batch(cache),context,parameters


def test_pose_batch_parameters_and_static_layout():
    _,batch,_,parameters=_fixture()
    packed=gather_pose_batch_parameters(parameters,batch.layout)
    assert packed.group_orientations.shape==(4,3);assert packed.group_translations.shape==(4,3)
    assert packed.group_body_poses.shape==(4,63);assert packed.frame_body_poses.shape==(2,63)
    torch.testing.assert_close(packed.group_body_poses[0],packed.frame_body_poses[0])
    torch.testing.assert_close(packed.group_body_poses[1],packed.frame_body_poses[0])
    assert batch.derivative_mapping.shape==(22,66,3)
    assert len(batch.fk_depth_indices)>0
    assert batch.active_indices_cam2.shape==batch.active_indices_cam4.shape==(69,)
    assert batch.preparation_timing["total_ms"]>=batch.preparation_timing["static_mapping_depth_layout_ms"]


def test_all_equation_modes_and_error_isolation():
    stage,batch,context,parameters=_fixture()
    scales,_=serial_pose_row_scales(stage,parameters,batch,objective_mode="raw_l2",charbonnier_eps=1.,rep_scale=.2)
    values={}
    for mode in PoseNEMode:
        values[mode]=evaluate_pose_equations(mode,stage,batch,parameters,scales,context.body_pose_anchors,
            context.scaling.pose_anchor_scale,context.image_width,context.image_height)
        assert isinstance(values[mode].equations,NormalEquations)
    for mode in PoseNEMode:
        torch.testing.assert_close(values[mode].equations.hessian,values[PoseNEMode.SERIAL_PRODUCTION].equations.hessian,rtol=2e-5,atol=1e-7)
        torch.testing.assert_close(values[mode].equations.gradient,values[PoseNEMode.SERIAL_PRODUCTION].equations.gradient,rtol=2e-5,atol=1e-7)


def test_serial_and_batch_weight_sources_raw_and_charbonnier():
    for objective in ("raw_l2","charbonnier_irls"):
        stage,batch,context,parameters=_fixture(objective=objective)
        serial,serial_stats=serial_pose_row_scales(stage,parameters,batch,objective_mode=objective,charbonnier_eps=1.,rep_scale=.2)
        packed=gather_pose_batch_parameters(parameters,batch.layout)
        from script.benchmark.staged_lm.pose_batch_candidates import batched_pose_residual_only
        residual=batched_pose_residual_only(batch,packed.group_orientations,packed.group_translations,
            packed.group_body_poses,context.image_width,context.image_height)
        candidate,candidate_stats=compute_batched_pose_row_scales(residual,batch,stage.aggregation_spec,
            objective_mode=objective,charbonnier_eps=1.,rep_scale=.2)
        torch.testing.assert_close(candidate,serial,rtol=2e-5,atol=1e-7)
        if objective=="raw_l2": assert candidate_stats=={}
        else: assert candidate_stats["n_points"]==serial_stats["n_points"]


def test_lockstep_and_independent_lm_smoke():
    stage,batch,context,parameters=_fixture(frames=1)
    lockstep=run_pose_lm_lockstep(stage,batch,context,candidate_mode=PoseNEMode.FULL_BATCH,
        weight_source="serial",hg_reduction_mode="serial_group",initial_parameters=parameters)
    assert lockstep.acceptance_mismatch_count==0
    serial=run_pose_lm_independent_batch(stage,batch,context,candidate_mode=PoseNEMode.SERIAL_PRODUCTION,
        weight_source="serial",hg_reduction_mode="production_order",initial_parameters=parameters,profile_timing=False)
    candidate=run_pose_lm_independent_batch(stage,batch,context,candidate_mode=PoseNEMode.FULL_BATCH,
        weight_source="serial",hg_reduction_mode="serial_group",initial_parameters=parameters,profile_timing=False)
    comparison=compare_lm_results(serial,candidate)
    assert comparison["status_match"] and comparison["accepted_match"]
    assert comparison["first_divergent_trial"] is None


if __name__=="__main__":
    test_pose_batch_parameters_and_static_layout();test_all_equation_modes_and_error_isolation()
    test_serial_and_batch_weight_sources_raw_and_charbonnier();test_lockstep_and_independent_lm_smoke()
    print("[PASS] pose shadow LM")
