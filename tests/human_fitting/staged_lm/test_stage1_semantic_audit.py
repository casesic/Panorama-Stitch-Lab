#!/usr/bin/env python3
"""Synthetic contracts for the Stage-1 semantic/Hessian audit."""
from __future__ import annotations
import math, sys
from pathlib import Path
import torch

ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT/"src"))
from script.benchmark.staged_lm.stage1_semantic_audit import (
    _point_audit, _uv_delta, evaluate_gates, float32_ulp_distance,
    hessian_audit, parameter_labels, raw_parameter_audit,
    rotation_geodesic,
)


def labels(n=4):return parameter_labels([513+i for i in range(n)])


def test_labels_and_raw():
    ls=labels();assert len(ls)==48
    assert (ls[0].camera,ls[0].category,ls[0].axis)==("cam2","rotation","x")
    assert (ls[11].camera,ls[11].category,ls[11].axis)==("cam4","translation","z")
    s=torch.zeros(48);c=s.clone();c[34]=1e-4
    audit=raw_parameter_audit(s,c,ls);assert audit.worst["global_index"]==34
    assert audit.worst["frame_index"]==515 and audit.worst["camera"]=="cam4"
    assert audit.rotation_component_max_abs==0 and audit.translation_component_max_abs>0


def test_rotation_translation_and_ulp():
    zero=torch.zeros(3);assert rotation_geodesic(zero,zero)["geodesic_rad"]==0
    value=rotation_geodesic(zero,torch.tensor([1e-3,0.,0.]))["geodesic_rad"]
    assert abs(value-1e-3)<3e-5
    # Axis-angle representations differing by 2*pi describe the same rotation.
    equivalent=rotation_geodesic(torch.tensor([math.pi,0.,0.]),torch.tensor([-math.pi,0.,0.]))
    assert equivalent["geodesic_rad"]<1e-3
    assert float32_ulp_distance(1.0,torch.nextafter(torch.tensor(1.),torch.tensor(2.)).item())==1


def test_wrap_and_geometry_aggregation():
    s=torch.tensor([[5759.9,4.],[2.,5.]]);c=torch.tensor([[.1,4.],[3.,5.]])
    raw,wrapped,changed=_uv_delta(s,c,5760)
    assert raw[0,0].abs()>5000 and wrapped[0,0].abs()<1 and int(changed.sum())==1
    ps=torch.tensor([[0.,0.,0.],[1.,0.,0.],[2.,0.,0.]])
    pc=ps.clone();pc[2,0]+=.3
    stats,_,_=_point_audit(ps,pc)
    assert stats["worst_point_index"]==2 and stats["max"]>.29 and stats["rms"]<stats["max"]
    assert stats["normalized_max_by_rms_radius"]>0


def _hessian(correlation,dx):
    h=torch.eye(12,dtype=torch.float64);h[0,1]=h[1,0]=correlation
    return hessian_audit(h,torch.zeros(12,dtype=torch.float64),dx,labels(1))


def test_hessian_scaling_flatness_and_psd():
    dx=torch.zeros(12,dtype=torch.float64);dx[0]=1;dx[1]=-1
    strong=_hessian(1-1e-6,dx);assert strong["flat_direction_classification"]=="STRONGLY_FLAT"
    assert strong["jacobi_scaled"]["quadratic_identity_relative_error"]<1e-9
    weak=_hessian(1-1e-3,dx);assert weak["flat_direction_classification"]=="WEAKLY_FLAT"
    notflat=_hessian(0.,dx);assert notflat["flat_direction_classification"]=="NOT_FLAT"
    bad=_hessian(1.1,dx);assert not bad["psd_gate"] and bad["spectrum"]["jacobi_scaled"]["substantial_negative_count"]==1
    tiny=_hessian(1+1e-12,dx);assert tiny["psd_gate"]
    assert strong["dx_energy_bands"]["le_1e-04"]["coordinate_energy_fraction"]>.99
    assert strong["per_frame"][0]["cross_camera_max_abs"]==0
    assert strong["top_dx_eigendirections"][0]["top_parameter_components"]


def _gate_inputs(flat="STRONGLY_FLAT",geometry_failure=False,downstream=True):
    class Raw:max_abs=1e-4
    value=2e-3 if geometry_failure else 1e-8
    geometry={"rigid_transforms":[{"geodesic_rad":0.}],
      "geometry_3d":{"global":{"normalized_max_by_rms_radius":value,"normalized_rms_by_rms_radius":0.}},
      "uv_projection":{"global":{"max":0.,"rms":0.,"p95":0.}},
      "raw_residual":{"global":{"max":0.,"rms":0.}}}
    objective={"relative_difference":0.}
    comparison={"metrics_max_abs":0.,"first_divergent_trial":None,"status_match":True,"accepted_match":True,
      "rejected_match":True,"trial_count_match":True,"solve_path_match":True,"final_lambda_relative":0.}
    return Raw(),geometry,objective,comparison,{"gate_pass":downstream},{"flat_direction_classification":flat,"psd_gate":True}


def test_independent_gates():
    gates=evaluate_gates(*_gate_inputs())
    assert gates["raw_parameter_strict"]["status"]=="FAIL"
    assert gates["solver_trajectory"]["status"]=="PASS"
    assert gates["integration_readiness"]=="READY_TO_INTEGRATE_STAGE1_FULL_BATCH_NE_SERIAL_WEIGHTS"
    unresolved=evaluate_gates(*_gate_inputs(flat="NOT_FLAT"));assert unresolved["classification"]=="HESSIAN_FLATNESS_NOT_CONFIRMED"
    mismatch=evaluate_gates(*_gate_inputs(geometry_failure=True));assert mismatch["classification"]=="GEOMETRIC_MISMATCH"
    downstream=evaluate_gates(*_gate_inputs(downstream=False));assert downstream["downstream_semantic"]["status"]=="FAIL"


def test_production_module_untouched_architecture():
    text=(ROOT/"src/script/benchmark/staged_lm/stage1_semantic_audit.py").read_text()
    assert "script.benchmark" not in (ROOT/"src/lib_pano/human_fitting/staged_lm/analytic_stage1.py").read_text()
    assert "hybrid" not in text.lower() and "float64 rotation" not in text.lower()


if __name__=="__main__":
    test_labels_and_raw();test_rotation_translation_and_ulp();test_wrap_and_geometry_aggregation()
    test_hessian_scaling_flatness_and_psd();test_independent_gates();test_production_module_untouched_architecture()
    print("[PASS] Stage1 semantic audit synthetic tests")
