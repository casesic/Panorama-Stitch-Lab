"""Synthetic operator-boundary and readiness tests for Pose batch candidates."""

from __future__ import annotations

if __package__ in (None, ""):
    from _bootstrap import ensure_test_paths
    ensure_test_paths(__file__)

from dataclasses import replace
import torch

from helpers import synthetic_joint_parameters, synthetic_stage2_context
from lib_pano.human_fitting.staged_lm.stage2_shared_cache import build_stage2_shared_geometry_cache
from lib_pano.human_fitting.staged_lm.stage2_pose_fast import build_stage2_pose_fast_cache
from script.benchmark.staged_lm.pose_batch_candidates import (
    batched_pose_group_analytic, batched_pose_hg, batched_pose_objective,
    build_pose_group_batch, gather_pose_batch_parameters,
)
from script.benchmark.staged_lm.pose_operator_ablation import (
    PoseAblationMode, build_pose_operator_state, build_serial_pose_operator_state,
    classify_ablation, compare_operator_states, first_divergence,
    float32_ulp_distance, integration_readiness, rotation_internal_ablation,
    select_pose_group_batch, small_angle_branch_audit, validate_serial_reconstruction,
)


def _fixture(frames=2,dtype=torch.float32):
    context=synthetic_stage2_context(frames,dtype=dtype);shared=build_stage2_shared_geometry_cache(context)
    cache=build_stage2_pose_fast_cache(context,shared);parameters=synthetic_joint_parameters(frames).to(dtype)[:-2]
    return cache,build_pose_group_batch(cache),parameters


def test_serial_state_reconstructs_production_and_shapes():
    cache,batch,parameters=_fixture();state=build_serial_pose_operator_state(cache,batch,parameters)
    assert state.rotations.shape==(4,55,3,3);assert state.d_rotations.shape==(4,55,3,3,66)
    assert state.active_jacobian.shape==(4,3,2,69)
    assert validate_serial_reconstruction(cache,batch,parameters,state)["pass"]
    for field in (state.pose_offsets,state.selected_v_posed,state.final_vertices,state.uv,state.residual,state.active_jacobian):
        for group,count in enumerate(batch.layout.point_counts):assert torch.count_nonzero(field[group,int(count):])==0


def test_helpers_refactor_and_all_hybrid_connections():
    cache,batch,parameters=_fixture();packed=gather_pose_batch_parameters(parameters,batch.layout)
    old_residual,old_j,_=batched_pose_group_analytic(batch,packed.group_orientations,packed.group_translations,
        packed.group_body_poses,cache.image_width,cache.image_height)
    full=build_pose_operator_state(cache,batch,parameters,PoseAblationMode.FULL_BATCH)
    assert torch.equal(old_residual,full.residual);assert torch.equal(old_j,full.active_jacobian)
    serial=build_serial_pose_operator_state(cache,batch,parameters)
    for mode in PoseAblationMode:
        state=build_pose_operator_state(cache,batch,parameters,mode)
        assert state.residual.shape==serial.residual.shape and state.active_jacobian.shape==serial.active_jacobian.shape
    comparison=compare_operator_states(serial,full,batch.layout.valid_mask)
    assert set(("rotations","d_rotations","pose_offsets","global_transforms","selected_skinned_vertices","uv","residual","active_jacobian"))<=set(comparison)
    assert first_divergence(comparison) is not None


def test_g1_cam2_cam4_and_g2_mapping():
    cache,batch,parameters=_fixture(frames=1)
    for indices in ([0],[1],[0,1]):
        selected=select_pose_group_batch(batch,indices);selected_cache=replace(cache,groups=[cache.groups[i] for i in indices],frame_count=1)
        state=build_serial_pose_operator_state(selected_cache,selected,parameters)
        assert validate_serial_reconstruction(selected_cache,selected,parameters,state)["pass"]
        assert state.residual.shape[0]==len(indices)


def test_rotation_branch_and_internal_diagnostics():
    cache,batch,parameters=_fixture(frames=1);values=rotation_internal_ablation(cache,batch,parameters)
    assert set(("rotation_value","right_jacobian","contraction","loop_scatter","vector_scatter"))<=set(values)
    audit=small_angle_branch_audit(gather_pose_batch_parameters(parameters,batch.layout),batch)
    assert audit["threshold_theta2"]==1e-8 and audit["serial_batch_mask_equal"]


def test_frame_body_pose_required():
    cache,batch,parameters=_fixture(frames=1);packed=gather_pose_batch_parameters(parameters,batch.layout)
    scales=torch.ones(batch.layout.valid_mask.shape+(2,),dtype=parameters.dtype)
    anchors=[torch.zeros(63,dtype=parameters.dtype)]
    try:batched_pose_hg(batch,packed.group_orientations,packed.group_translations,packed.group_body_poses,scales,anchors,.3,5760,2880)
    except RuntimeError as error:assert "MISSING_FRAME_BODY_POSES" in str(error)
    else:raise AssertionError("missing frame_body_poses must fail")
    try:batched_pose_objective(batch,packed.group_orientations,packed.group_translations,packed.group_body_poses,anchors,.3,5760,2880,
        objective_mode="raw_l2",charbonnier_eps=1.,rep_scale=.2,aggregation_spec=None)
    except RuntimeError as error:assert "MISSING_FRAME_BODY_POSES" in str(error)
    else:raise AssertionError("missing frame_body_poses must fail")


def test_float32_ulp_signed_zero_and_nonfinite():
    a=torch.tensor([1.,-1.,0.,-0.],dtype=torch.float32)
    b=torch.nextafter(a,torch.tensor([2.,-2.,1.,-1.],dtype=torch.float32))
    distance=float32_ulp_distance(a,b);assert distance[0]==1 and distance[1]==1
    assert float32_ulp_distance(torch.tensor([0.]),torch.tensor([-0.]))[0]==0
    try:float32_ulp_distance(torch.tensor([float("inf")]),torch.tensor([float("inf")]))
    except ValueError:pass
    else:raise AssertionError("nonfinite ULP input must fail")


def test_classification_and_gates_are_independent():
    assert classify_ablation(fp64_reduction_ratio=.001,branch_index_padding_ok=True,solver_gate=True,ulp_le_2_fraction=.99)=="EXPECTED_FP32_OPERATION_ORDER_DIFFERENCE"
    assert classify_ablation(fp64_reduction_ratio=1.,branch_index_padding_ok=True,solver_gate=True,ulp_le_2_fraction=.99)=="FORMULA_OR_INDEX_BUG"
    assert integration_readiness(classification="EXPECTED_FP32_OPERATION_ORDER_DIFFERENCE",solver_gate=True,full_batch_performance_gate=True)=="READY_TO_INTEGRATE_FULL_BATCH_NE_SERIAL_WEIGHTS"
    assert integration_readiness(classification="UNRESOLVED_NUMERICAL_DIFFERENCE",solver_gate=True,full_batch_performance_gate=True)=="DO_NOT_INTEGRATE"


if __name__=="__main__":
    test_serial_state_reconstructs_production_and_shapes();test_helpers_refactor_and_all_hybrid_connections()
    test_g1_cam2_cam4_and_g2_mapping();test_rotation_branch_and_internal_diagnostics();test_frame_body_pose_required()
    test_float32_ulp_signed_zero_and_nonfinite();test_classification_and_gates_are_independent()
    print("[PASS] Pose operator ablation")

