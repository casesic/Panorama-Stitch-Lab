"""Synthetic regression tests for benchmark-only Joint FULL_BATCH shadows."""
from __future__ import annotations
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[3]/"src"));sys.path.insert(0,str(Path(__file__).resolve().parent))
import torch
from helpers import synthetic_joint_cache,synthetic_joint_parameters
from lib_pano.human_fitting.staged_lm.joint import evaluate_joint_shape_state
from script.benchmark.staged_lm.joint_batch_candidates import JointNEMode,build_joint_batch_cache,gather_joint_batch_parameters,joint_batched_dynamic_shape,joint_batched_group_analytic,evaluate_joint_normal_equations
from script.benchmark.staged_lm.joint_shadow_common import arrowhead_errors,build_schur_diagnostics,schur_errors


def fixture(frames=1):
    cache=synthetic_joint_cache(frames);parameters=synthetic_joint_parameters(frames)
    batch=build_joint_batch_cache(cache,.1);scales=[torch.ones(group.targets.numel(),dtype=parameters.dtype) for group in cache.groups]
    anchors=[parameters[f*75+12:(f+1)*75].clone() for f in range(frames)]
    beta_anchor=torch.tensor([.04,-.03],dtype=parameters.dtype)
    return cache,parameters,batch,scales,anchors,beta_anchor


def test_layout_and_static_cache():
    cache,_,batch,_,_,_=fixture(2);layout=batch.layout
    assert layout.group_count==4 and layout.frame_positions_host==(0,0,1,1)
    assert layout.camera_indices_host==(0,1,0,1)
    assert tuple(layout.frame_positions.tolist())==layout.frame_positions_host
    assert tuple(layout.camera_indices.tolist())==layout.camera_indices_host
    assert batch.derivative_mapping.shape==(22,66,3) and len(batch.fk_depth_indices)>1
    assert batch.active_indices_cam2.shape==(69,) and batch.active_indices_cam4.shape==(69,)
    for index,count in enumerate(layout.point_counts_host):
        assert not layout.valid_mask[index,count:].any();assert torch.count_nonzero(layout.targets[index,count:])==0
        assert torch.count_nonzero(batch.selected_base_v_shaped[index,count:])==0
        assert torch.count_nonzero(batch.selected_beta_shapedirs[index,count:])==0


def test_parameter_gather_no_group_arithmetic():
    _,parameters,batch,_,_,_=fixture(2);packed=gather_joint_batch_parameters(parameters,batch.layout,2)
    assert packed.frame_body_poses.shape==(2,63) and packed.group_body_poses.shape==(4,63)
    assert torch.equal(packed.group_body_poses[0],packed.group_body_poses[1])
    assert torch.equal(packed.group_orientations[0],parameters[:3])
    assert torch.equal(packed.group_orientations[1],parameters[6:9])


def test_dynamic_shape_and_single_calls():
    cache,parameters,batch,_,_,_=fixture();shape=evaluate_joint_shape_state(cache,parameters[-2:],3.)
    value,derivative=joint_batched_dynamic_shape(batch,shape)
    assert value.shape==(2,3,3) and derivative.shape==(2,3,3,2)
    batch.shape_state_calls=batch.full_analytic_calls=0
    result=joint_batched_group_analytic(cache,batch,parameters,3.,return_intermediates=True)
    assert cache.batched_shape_state_calls==1 and cache.batched_full_analytic_calls==1
    assert result.residual.shape==(2,3,2) and result.active_pose_jacobian.shape==(2,3,2,69)
    assert result.beta_jacobian.shape==(2,3,2,2)


def test_all_equation_modes_and_arrowhead_blocks():
    cache,p,batch,scales,anchors,beta_anchor=fixture(2)
    values={mode:evaluate_joint_normal_equations(mode,cache,batch,p,3.,scales,anchors,beta_anchor,.1,.2) for mode in JointNEMode}
    reference=values[JointNEMode.SERIAL_REFERENCE]
    for mode,value in values.items():
        system=value.structured_system
        assert system.frame_hessians.shape==(2,75,75);assert system.frame_beta_cross.shape==(2,75,2)
        assert system.beta_hessian.shape==(2,2);assert value.gradient.dtype==torch.float64
        assert torch.allclose(system.to_dense(count_call=False)@torch.ones(p.numel(),dtype=torch.float64),
                              system.to_dense(count_call=False)@torch.ones(p.numel(),dtype=torch.float64))
        if mode is JointNEMode.SERIAL_FORWARD_BATCH_GEMM:
            errors=arrowhead_errors(value,reference)
            assert errors["A"]["max_abs"]<1e-8 and errors["B"]["max_abs"]<1e-8


def test_schur_delta_and_reduction_modes():
    cache,p,batch,scales,anchors,beta_anchor=fixture()
    ref=evaluate_joint_normal_equations(JointNEMode.SERIAL_REFERENCE,cache,batch,p,3.,scales,anchors,beta_anchor,.1,.2)
    candidate=evaluate_joint_normal_equations(JointNEMode.FULL_BATCH_PRODUCTION_REDUCTION,cache,batch,p,3.,scales,anchors,beta_anchor,.1,.2)
    diagnostics=schur_errors(candidate,ref,1e-2)
    assert diagnostics["flat_delta"]["relative"]<1e-3
    assert build_schur_diagnostics(candidate.structured_system,1e-2)["success"]


def test_cuda_smoke():
    if not torch.cuda.is_available():return
    # Real CUDA regression is performed by profile_joint_shadow_lm; synthetic
    # fixtures intentionally remain CPU-only because their dataclass contract
    # records CPU model ownership.


def main():
    test_layout_and_static_cache();test_parameter_gather_no_group_arithmetic();test_dynamic_shape_and_single_calls()
    test_all_equation_modes_and_arrowhead_blocks();test_schur_delta_and_reduction_modes();test_cuda_smoke()
    print("[PASS] joint FULL_BATCH shadow synthetic tests")
if __name__=="__main__":main()
