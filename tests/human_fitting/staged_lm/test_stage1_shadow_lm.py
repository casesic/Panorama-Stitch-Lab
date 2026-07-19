"""Synthetic contracts for Stage-1 FULL_BATCH shadow normal equations."""
from __future__ import annotations

if __package__ in (None, ""):
    from _bootstrap import ensure_test_paths
    ensure_test_paths(__file__)

import torch

from lib_pano.human_fitting.staged_lm.analytic_stage1 import (
    Stage1AnalyticCache, Stage1AnalyticGroupCache,
    _group_raw_residual_and_local_jacobian,
)
from script.benchmark.staged_lm.stage1_batch_candidates import (
    build_stage1_batch_cache, gather_stage1_batch_parameters,
    stage1_batched_group_analytic,
)
from script.benchmark.staged_lm.stage1_shadow_common import (
    Stage1NEMode, build_stage1_shadow_linearization_common,
    equation_errors, evaluate_stage1_equations,
)


def fixture(frames=2,dtype=torch.float64):
    torch.manual_seed(81);groups=[]
    for frame in range(frames):
        for camera,count in (("cam2",2),("cam4",3)):
            groups.append(Stage1AnalyticGroupCache(
                frame,513+frame,camera,torch.randn(count,3,dtype=dtype)*.1,
                torch.tensor([0.,0.,2.],dtype=dtype),
                torch.randn(count,2,dtype=dtype)*10+100))
    cache=Stage1AnalyticCache(groups,frames,5760,2880,0.)
    return cache,torch.randn(frames*12,dtype=dtype)*.01


def common(cache,batch,parameters):
    return build_stage1_shadow_linearization_common(
        cache,batch,parameters,is_irls=False,charbonnier_eps=1.,
        aggregation_spec=None,rep_scale=.2)


def test_layout_cache_padding_and_mapping():
    cache,parameters=fixture();batch=build_stage1_batch_cache(cache);layout=batch.layout
    assert layout.frame_positions_host==(0,0,1,1) and layout.camera_indices_host==(0,1,0,1)
    assert tuple(layout.frame_positions.tolist())==layout.frame_positions_host
    assert tuple(layout.rigid_parameter_indices[0].tolist())==tuple(range(6))
    assert tuple(layout.rigid_parameter_indices[1].tolist())==tuple(range(6,12))
    for i,count in enumerate(layout.point_counts_host):
        assert not layout.valid_mask[i,count:].any()
        assert torch.count_nonzero(layout.targets[i,count:])==0
        assert torch.count_nonzero(batch.local_vertices[i,count:])==0
    packed=gather_stage1_batch_parameters(parameters,layout)
    assert torch.equal(packed.group_orientations[0],parameters[:3])
    assert torch.equal(packed.group_orientations[1],parameters[6:9])


def test_analytic_residual_jacobian_and_single_call():
    cache,p=fixture();batch=build_stage1_batch_cache(cache);batch.full_analytic_calls=0
    result=stage1_batched_group_analytic(batch,p,cache.image_width,cache.image_height)
    assert batch.full_analytic_calls==1 and result.active_jacobian.shape[-1]==6
    for i,(group,count,camera) in enumerate(zip(cache.groups,batch.layout.point_counts_host,batch.layout.camera_indices_host)):
        rr,jj=_group_raw_residual_and_local_jacobian(cache,p,group)
        torch.testing.assert_close(result.residual[i,:count].reshape(-1),rr)
        torch.testing.assert_close(result.active_jacobian[i,:count].reshape(-1,6),jj[:,camera*6:camera*6+6])
        assert torch.count_nonzero(result.residual[i,count:])==0
        assert torch.count_nonzero(result.active_jacobian[i,count:])==0


def test_all_equation_modes_fp64_blocks_and_order():
    cache,p=fixture();batch=build_stage1_batch_cache(cache);weights=common(cache,batch,p)
    values={mode:evaluate_stage1_equations(mode,cache,batch,p,weights.padded_row_scales,weights.flat_row_scales) for mode in Stage1NEMode}
    reference=values[Stage1NEMode.SERIAL_REFERENCE]
    for mode,value in values.items():
        assert value.equations.hessian.dtype==torch.float64
        assert value.equations.gradient.dtype==torch.float64
        assert value.equations.residual.shape==reference.equations.residual.shape
        for frame in range(cache.frame_count):
            block=value.equations.hessian[frame*12:(frame+1)*12,frame*12:(frame+1)*12]
            assert torch.count_nonzero(block[:6,6:])==0 and torch.count_nonzero(block[6:,:6])==0
        if mode is not Stage1NEMode.SERIAL_REFERENCE:assert value.full_analytic_calls in (0,1)
    errors=equation_errors(values[Stage1NEMode.FULL_BATCH_GROUP_GEMM],reference)
    assert errors["H"]["relative_frobenius"]<1e-10
    assert errors["g"]["relative_frobenius"]<1e-10


def test_raw_l2_scales_and_padding():
    cache,p=fixture();batch=build_stage1_batch_cache(cache);weights=common(cache,batch,p)
    assert torch.all(weights.flat_row_scales==.2)
    for i,count in enumerate(batch.layout.point_counts_host):
        assert torch.all(weights.padded_row_scales[i,:count]==.2)
        assert torch.count_nonzero(weights.padded_row_scales[i,count:])==0


def test_padding_nmax_does_not_change_valid_result():
    cache,p=fixture(1);batch=build_stage1_batch_cache(cache)
    original=stage1_batched_group_analytic(batch,p,cache.image_width,cache.image_height)
    extra=4;batch.local_vertices=torch.nn.functional.pad(batch.local_vertices,(0,0,0,extra))
    batch.layout=type(batch.layout)(
        batch.layout.frame_positions,batch.layout.camera_indices,batch.layout.point_counts,
        torch.nn.functional.pad(batch.layout.valid_mask,(0,extra)),
        torch.nn.functional.pad(batch.layout.targets,(0,0,0,extra)),batch.layout.rigid_parameter_indices,
        batch.layout.frame_positions_host,batch.layout.camera_indices_host,batch.layout.point_counts_host,
        batch.layout.group_count,batch.layout.frame_count,batch.layout.nmax+extra,
        batch.layout.padding_ratio,batch.layout.padded_bytes)
    padded=stage1_batched_group_analytic(batch,p,cache.image_width,cache.image_height)
    for i,count in enumerate(batch.layout.point_counts_host):
        torch.testing.assert_close(padded.residual[i,:count],original.residual[i,:count])
        torch.testing.assert_close(padded.active_jacobian[i,:count],original.active_jacobian[i,:count])


def main():
    test_layout_cache_padding_and_mapping();test_analytic_residual_jacobian_and_single_call()
    test_all_equation_modes_fp64_blocks_and_order();test_raw_l2_scales_and_padding()
    test_padding_nmax_does_not_change_valid_result();print("[PASS] Stage1 FULL_BATCH shadow synthetic tests")


if __name__=="__main__":main()
