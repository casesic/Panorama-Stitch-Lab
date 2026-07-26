"""Semantic/geometric and Hessian audit for one Stage-1 shadow failure."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import struct
from typing import Any, Iterable

import torch

from lib_pano.human_fitting.staged_lm.analytic_stage1 import (
    Stage1AnalyticCache, _group_transform,
)
from lib_pano.human_fitting.staged_lm.geometry.erp import project_unified_erp
from lib_pano.human_fitting.staged_lm.geometry.so3 import axis_angle_to_matrix
from script.benchmark.staged_lm.stage1_shadow_common import (
    Stage1NEMode, build_stage1_shadow_linearization_common,
    evaluate_stage1_equations,
)


@dataclass(frozen=True)
class Stage1ParameterLabel:
    global_index: int
    frame_position: int
    frame_index: int
    camera: str
    category: str
    axis: str
    local_index: int


@dataclass
class Stage1RawParameterAudit:
    max_abs: float
    l2: float
    relative_l2: float
    worst: dict[str, Any]
    top_components: list[dict[str, Any]]
    rotation_component_max_abs: float
    translation_component_max_abs: float
    per_frame_camera: list[dict[str, Any]]


def parameter_labels(frame_indices: Iterable[int]) -> list[Stage1ParameterLabel]:
    labels=[]
    for position, frame_index in enumerate(frame_indices):
        for local in range(12):
            camera="cam2" if local < 6 else "cam4"; rigid=local % 6
            category="rotation" if rigid < 3 else "translation"
            labels.append(Stage1ParameterLabel(
                position*12+local, position, int(frame_index), camera, category,
                "xyz"[rigid % 3], local,
            ))
    return labels


def _component(label, serial, candidate, delta):
    return {**asdict(label), "serial_value":float(serial),
            "candidate_value":float(candidate), "signed_difference":float(delta),
            "absolute_difference":abs(float(delta))}


def raw_parameter_audit(serial, candidate, labels, top_k=10):
    s=serial.to(torch.float64); c=candidate.to(torch.float64); d=c-s
    order=torch.argsort(d.abs(),descending=True)
    top=[_component(labels[int(i)],s[i],c[i],d[i]) for i in order[:top_k]]
    groups=[]
    for position in range(len(labels)//12):
        for camera_index,camera in enumerate(("cam2","cam4")):
            sl=slice(position*12+camera_index*6,position*12+camera_index*6+6)
            groups.append({"frame_position":position,"frame_index":labels[position*12].frame_index,
                           "camera":camera,"rotation_component_max_abs":float(d[sl][:3].abs().max()),
                           "translation_component_max_abs":float(d[sl][3:].abs().max()),
                           "component_l2":float(d[sl].norm())})
    rotation=torch.cat([d.reshape(-1,12)[:,:3],d.reshape(-1,12)[:,6:9]],dim=1)
    translation=torch.cat([d.reshape(-1,12)[:,3:6],d.reshape(-1,12)[:,9:12]],dim=1)
    return Stage1RawParameterAudit(float(d.abs().max()),float(d.norm()),
        float(d.norm())/max(float(s.norm()),1e-30),top[0],top,
        float(rotation.abs().max()),float(translation.abs().max()),groups)


def rotation_geodesic(serial_axis_angle, candidate_axis_angle):
    rs=axis_angle_to_matrix(serial_axis_angle).to(torch.float64)
    rc=axis_angle_to_matrix(candidate_axis_angle).to(torch.float64)
    relative=rs.transpose(-1,-2)@rc
    cosine=torch.clamp((torch.trace(relative)-1.0)/2.0,-1.0,1.0)
    angle=torch.acos(cosine); delta=rc-rs
    return {"geodesic_rad":float(angle),"geodesic_deg":math.degrees(float(angle)),
            "axis_angle_component_max_abs":float((candidate_axis_angle-serial_axis_angle).abs().max()),
            "axis_angle_component_l2":float((candidate_axis_angle-serial_axis_angle).norm()),
            "rotation_matrix_max_abs":float(delta.abs().max()),
            "rotation_matrix_frobenius":float(delta.norm())}


def _distribution(values):
    values=values.to(torch.float64)
    return {"max":float(values.max()) if values.numel() else 0.0,
            "rms":float(torch.sqrt(torch.mean(values.square()))) if values.numel() else 0.0,
            "mean":float(values.mean()) if values.numel() else 0.0,
            "p50":float(torch.quantile(values,.5)) if values.numel() else 0.0,
            "p95":float(torch.quantile(values,.95)) if values.numel() else 0.0}


def _point_audit(serial_points, candidate_points):
    delta=(candidate_points-serial_points).to(torch.float64); norms=delta.norm(dim=-1)
    worst=int(norms.argmax()) if norms.numel() else 0
    centroid=serial_points.to(torch.float64).mean(0)
    radius=torch.sqrt(torch.mean((serial_points.to(torch.float64)-centroid).square().sum(-1)))
    bbox=(serial_points.to(torch.float64).amax(0)-serial_points.to(torch.float64).amin(0)).norm()
    stats=_distribution(norms); stats.update({
        "max_abs_coordinate":float(delta.abs().max()),"worst_point_index":worst,
        "worst_serial_xyz":serial_points[worst].tolist(),"worst_candidate_xyz":candidate_points[worst].tolist(),
        "worst_delta_xyz":delta[worst].tolist(),"rms_radius":float(radius),"bbox_diagonal":float(bbox),
        "normalized_max_by_rms_radius":stats["max"]/max(float(radius),1e-30),
        "normalized_rms_by_rms_radius":stats["rms"]/max(float(radius),1e-30),
        "normalized_max_by_bbox":stats["max"]/max(float(bbox),1e-30),
        "normalized_rms_by_bbox":stats["rms"]/max(float(bbox),1e-30)})
    return stats,delta,norms


def _uv_delta(serial_uv,candidate_uv,width):
    raw=(candidate_uv-serial_uv).to(torch.float64)
    wrapped=raw.clone();wrapped[:,0]=torch.remainder(raw[:,0]+width/2,width)-width/2
    changed=(raw[:,0]-wrapped[:,0]).abs()>1e-6
    return raw,wrapped,changed


def geometry_audit(cache: Stage1AnalyticCache, serial, candidate):
    rigid=[];groups=[];all_s=[];all_c=[];all_du=[];all_uvraw=[];all_raw=[];meta=[]
    framed_s=serial.reshape(-1,12);framed_c=candidate.reshape(-1,12)
    for group_index,group in enumerate(cache.groups):
        camera_index=0 if group.camera=="cam2" else 1; start=camera_index*6
        rot=rotation_geodesic(framed_s[group.frame_position,start:start+3],framed_c[group.frame_position,start:start+3])
        td=(framed_c[group.frame_position,start+3:start+6]-framed_s[group.frame_position,start+3:start+6]).to(torch.float64)
        rigid.append({"frame_position":group.frame_position,"frame_index":group.frame_index,"camera":group.camera,
                      **rot,"translation_signed_delta":td.tolist(),"translation_max_abs":float(td.abs().max()),
                      "translation_l2":float(td.norm())})
        ps,_=_group_transform(serial,group);pc,_=_group_transform(candidate,group)
        stats,_,_= _point_audit(ps,pc)
        us=project_unified_erp(ps,cache.image_width,cache.image_height)
        uc=project_unified_erp(pc,cache.image_width,cache.image_height)
        raw,wrapped,changed=_uv_delta(us,uc,cache.image_width);uvnorm=wrapped.norm(dim=-1)
        target=group.targets.to(us);rs=us-target;rc=uc-target;res_delta=(rc-rs).to(torch.float64)
        uvstats=_distribution(uvnorm);worst=int(uvnorm.argmax())
        uvstats.update({"max_abs_u_raw":float(raw[:,0].abs().max()),"max_abs_u_wrapped":float(wrapped[:,0].abs().max()),
                        "max_abs_v":float(wrapped[:,1].abs().max()),"seam_wrap_changed_count":int(changed.sum()),
                        "worst_point_index":worst,"serial_uv":us[worst].tolist(),"candidate_uv":uc[worst].tolist(),
                        "target_uv":target[worst].tolist(),"wrapped_uv_delta":wrapped[worst].tolist()})
        rnorm=res_delta.norm(dim=-1);rstats=_distribution(rnorm);rw=int(rnorm.argmax())
        wrapped_residual_stats=_distribution(wrapped.norm(dim=-1))
        rstats.update({"max_coordinate_abs":float(res_delta.abs().max()),"worst_point_index":rw,
                       "serial_residual":rs[rw].tolist(),"candidate_residual":rc[rw].tolist(),
                       "wrap_aware_max":wrapped_residual_stats["max"],
                       "wrap_aware_rms":wrapped_residual_stats["rms"]})
        groups.append({"frame_position":group.frame_position,"frame_index":group.frame_index,"camera":group.camera,
                       "point_count":int(ps.shape[0]),"geometry_3d":stats,"uv":uvstats,"raw_residual":rstats})
        all_s.append(ps);all_c.append(pc);all_du.append(wrapped);all_uvraw.append(raw);all_raw.append(res_delta)
        meta.extend((group_index,i) for i in range(ps.shape[0]))
    gs=torch.cat(all_s);gc=torch.cat(all_c);global3d,_,_= _point_audit(gs,gc)
    du=torch.cat(all_du);uvraw=torch.cat(all_uvraw);rawres=torch.cat(all_raw);uvnorm=du.norm(dim=-1);rnorm=rawres.norm(dim=-1)
    ui=int(uvnorm.argmax());ri=int(rnorm.argmax())
    uvglobal=_distribution(uvnorm);uvglobal.update({"max_abs_u_raw":float(uvraw[:,0].abs().max()),"max_abs_u_wrapped":float(du[:,0].abs().max()),
        "max_abs_v":float(du[:,1].abs().max()),"worst_group_index":meta[ui][0],"worst_point_index":meta[ui][1],
        "seam_wrap_changed_count":sum(g["uv"]["seam_wrap_changed_count"] for g in groups)})
    wrapped_residual=_distribution(uvnorm)
    rglobal=_distribution(rnorm);rglobal.update({"max_coordinate_abs":float(rawres.abs().max()),
        "wrap_aware_max":wrapped_residual["max"],"wrap_aware_rms":wrapped_residual["rms"],
        "worst_group_index":meta[ri][0],"worst_point_index":meta[ri][1]})
    return {"rigid_transforms":rigid,"geometry_3d":{"per_group":groups,"global":global3d},
            "uv_projection":{"per_group":groups,"global":uvglobal},
            "raw_residual":{"per_group":groups,"global":rglobal}}


def float32_ulp_distance(left: float, right: float) -> int | None:
    if not math.isfinite(left) or not math.isfinite(right): return None
    def ordered(value):
        bits=struct.unpack("!I",struct.pack("!f",float(value)))[0]
        return 0x80000000-bits if bits&0x80000000 else bits+0x80000000
    return abs(ordered(left)-ordered(right))


def objective_audit(stage, serial, candidate):
    fs=stage.lm_true_objective_fn(serial);fc=stage.lm_true_objective_fn(candidate)
    left=float(fs);right=float(fc);delta=right-left
    path=[];dx=candidate-serial
    for alpha in (0.,.25,.5,.75,1.):
        value=float(stage.lm_true_objective_fn(serial+alpha*dx))
        path.append({"alpha":alpha,"objective":value,"delta_from_serial":value-left,
                     "relative_from_serial":abs(value-left)/max(abs(left),1e-30)})
    return {"serial":left,"candidate":right,"absolute_difference":abs(delta),
            "relative_difference":abs(delta)/max(abs(left),1e-30),"torch_equal":bool(torch.equal(fs,fc)),
            "ulp_difference":float32_ulp_distance(left,right),"path":path}


def _spectrum(h, displacement, labels=None):
    eigenvalues,eigenvectors=torch.linalg.eigh(h);maximum=max(float(eigenvalues.abs().max()),1e-30)
    coeff=eigenvectors.T@displacement;coordinate=coeff.square();curvature=eigenvalues.clamp_min(0)*coordinate
    cs=max(float(coordinate.sum()),1e-30);ks=max(float(curvature.sum()),1e-30)
    bands={}
    for threshold in (1e-8,1e-6,1e-4,1e-3,1e-2):
        mask=eigenvalues/maximum<=threshold
        bands[f"le_{threshold:.0e}"]={"eigenvector_count":int(mask.sum()),
            "coordinate_energy_fraction":float(coordinate[mask].sum())/cs,
            "curvature_energy_fraction":float(curvature[mask].sum())/ks}
    top=[]
    for index in torch.argsort(coordinate,descending=True)[:min(10,coordinate.numel())]:
        i=int(index);vector=eigenvectors[:,i];components=[]
        if labels is not None:
            for j in torch.argsort(vector.abs(),descending=True)[:min(5,vector.numel())]:
                components.append({**asdict(labels[int(j)]),"value":float(vector[j])})
        top.append({"eigen_index":i,"eigenvalue":float(eigenvalues[i]),
                    "eigenvalue_over_lambda_max":float(eigenvalues[i])/maximum,"coefficient":float(coeff[i]),
                    "coordinate_energy_fraction":float(coordinate[i])/cs,
                    "curvature_energy_fraction":float(curvature[i])/ks,"top_parameter_components":components})
    return eigenvalues,bands,top


def hessian_audit(hessian,gradient,dx,labels):
    h=hessian.to(torch.float64);g=gradient.to(torch.float64);dx=dx.to(torch.float64)
    hs=.5*(h+h.T);asym=h-h.T;hdx=hs@dx;quad=float(dx@hdx);linear=float(g@dx)
    diagonal=hs.diag().abs();floor=max(float(diagonal.max())*1e-12,1e-12)
    scale=diagonal.clamp_min(floor).sqrt();z=scale*dx
    scaled=hs/scale[:,None]/scale[None,:];gs=g/scale
    reconstructed=float(z@(scaled@z));identity_abs=abs(reconstructed-quad)
    raw_eig=torch.linalg.eigvalsh(hs)
    eig,bands,top=_spectrum(scaled,z,labels);lambda_abs=max(float(eig.abs().max()),1e-30)
    negative_threshold=-max(lambda_abs*1e-10,1e-12);substantial=int((eig<negative_threshold).sum())
    scaled_rayleigh=reconstructed/max(float(z@z),1e-30);normalized=scaled_rayleigh/lambda_abs
    if substantial==0 and bands["le_1e-04"]["coordinate_energy_fraction"]>=.8 and normalized<=1e-4: flat="STRONGLY_FLAT"
    elif substantial==0 and bands["le_1e-03"]["coordinate_energy_fraction"]>=.8 and normalized<=1e-3: flat="WEAKLY_FLAT"
    else: flat="NOT_FLAT"
    per_frame=[];per_camera=[]
    for frame in range(len(labels)//12):
        sl=slice(frame*12,(frame+1)*12);hf=hs[sl,sl];df=dx[sl]
        sf=hf.diag().abs().clamp_min(max(float(hf.diag().abs().max())*1e-12,1e-12)).sqrt();zf=sf*df;hfs=hf/sf[:,None]/sf[None,:]
        ef,bf,_=_spectrum(hfs,zf)
        per_frame.append({"frame_position":frame,"frame_index":labels[frame*12].frame_index,
            "dxT_H_dx":float(df@(hf@df)),"Hdx_norm":float((hf@df).norm()),
            "raw_rayleigh":float(df@(hf@df))/max(float(df@df),1e-30),"eigenvalues":ef.tolist(),
            "normalized_scaled_rayleigh":float(zf@(hfs@zf))/max(float(zf@zf)*max(float(ef.abs().max()),1e-30),1e-30),
            "energy_bands":bf,"cross_camera_max_abs":float(hf[:6,6:].abs().max())})
        for camera_index,camera in enumerate(("cam2","cam4")):
            cs=slice(frame*12+camera_index*6,frame*12+camera_index*6+6);hc=hs[cs,cs];dc=dx[cs]
            sc=hc.diag().abs().clamp_min(max(float(hc.diag().abs().max())*1e-12,1e-12)).sqrt();zc=sc*dc;hcs=hc/sc[:,None]/sc[None,:]
            ec,bc,_=_spectrum(hcs,zc)
            per_camera.append({"frame_position":frame,"frame_index":labels[frame*12].frame_index,"camera":camera,
                "raw_dx_l2":float(dc.norm()),"dxT_H_dx":float(dc@(hc@dc)),"Hdx_norm":float((hc@dc).norm()),
                "raw_rayleigh":float(dc@(hc@dc))/max(float(dc@dc),1e-30),
                "normalized_scaled_rayleigh":float(zc@(hcs@zc))/max(float(zc@zc)*max(float(ec.abs().max()),1e-30),1e-30),
                "eigenvalues":ec.tolist(),"energy_bands":bc})
    return {"symmetry":{"frobenius":float(asym.norm()),"max_abs":float(asym.abs().max())},
      "raw_coordinates":{"dx_norm":float(dx.norm()),"Hdx_norm":float(hdx.norm()),"Hdx_max_abs":float(hdx.abs().max()),
        "dxT_H_dx":quad,"gT_dx":linear,"local_quadratic_model_delta":linear+.5*quad,
        "rayleigh_raw":quad/max(float(dx@dx),1e-30)},
      "jacobi_scaled":{"diag_floor":floor,"quadratic_reconstructed":reconstructed,
        "quadratic_identity_absolute_error":identity_abs,"quadratic_identity_relative_error":identity_abs/max(abs(quad),1e-30),
        "scaled_rayleigh":scaled_rayleigh,"normalized_scaled_rayleigh":normalized,
        "H_scaled_z_norm":float((scaled@z).norm()),"g_scaled_norm":float(gs.norm()),"g_scaledT_z":float(gs@z)},
      "spectrum":{"raw_coordinates":{"minimum":float(raw_eig.min()),"maximum":float(raw_eig.max()),
        "smallest_10":raw_eig[:10].tolist(),"largest_10":raw_eig[-10:].tolist()},
        "jacobi_scaled":{"minimum":float(eig.min()),"maximum":float(eig.max()),"smallest_10":eig[:10].tolist(),
        "largest_10":eig[-10:].tolist(),"negative_count":int((eig<0).sum()),
        "substantial_negative_count":substantial,"negative_threshold":negative_threshold}},
      "dx_energy_bands":bands,"top_dx_eigendirections":top,"per_frame":per_frame,"per_camera":per_camera,
      "flat_direction_classification":flat,"psd_gate":substantial==0}


def evaluate_gates(raw,geometry,objective,comparison,downstream,hessian):
    g3=geometry["geometry_3d"]["global"];uv=geometry["uv_projection"]["global"];rr=geometry["raw_residual"]["global"]
    max_rotation=max(x["geodesic_rad"] for x in geometry["rigid_transforms"])
    metric_error=comparison.get("metrics_max_abs",0.0)
    checks=[("rotation_geodesic",max_rotation,1e-3),("vertex_normalized_max",g3["normalized_max_by_rms_radius"],1e-4),
      ("vertex_normalized_rms",g3["normalized_rms_by_rms_radius"],2e-5),("uv_max",uv["max"],1e-3),
      ("uv_rms",uv["rms"],1e-4),("uv_p95",uv["p95"],5e-4),("residual_max",rr["max"],1e-3),
      ("residual_rms",rr["rms"],1e-4),("objective_relative",objective["relative_difference"],1e-6),
      ("metrics_max_abs",metric_error,1e-4)]
    failed=next(({"name":n,"value":v,"threshold":t} for n,v,t in checks if v>t),None)
    trajectory=(comparison["first_divergent_trial"] is None and comparison["status_match"] and comparison["accepted_match"]
        and comparison["rejected_match"] and comparison["trial_count_match"] and comparison["solve_path_match"]
        and comparison["final_lambda_relative"]<=1e-5)
    downstream_pass=bool(downstream and downstream.get("gate_pass"))
    flat=hessian["flat_direction_classification"]
    if failed: classification="GEOMETRIC_MISMATCH";readiness="DO_NOT_INTEGRATE_GEOMETRIC_MISMATCH"
    elif not hessian["psd_gate"]: classification="UNRESOLVED_SEMANTIC_DIFFERENCE";readiness="UNRESOLVED_SEMANTIC_DIFFERENCE"
    elif flat=="NOT_FLAT": classification="HESSIAN_FLATNESS_NOT_CONFIRMED";readiness="UNRESOLVED_SEMANTIC_DIFFERENCE"
    elif not downstream_pass: classification="GEOMETRIC_MISMATCH";readiness="DO_NOT_INTEGRATE_GEOMETRIC_MISMATCH"
    elif trajectory: classification=("SEMANTICALLY_EQUIVALENT_FLAT_DIRECTION" if flat=="STRONGLY_FLAT" else "SEMANTICALLY_EQUIVALENT_WEAK_DIRECTION");readiness="READY_TO_INTEGRATE_STAGE1_FULL_BATCH_NE_SERIAL_WEIGHTS"
    else: classification="UNRESOLVED_SEMANTIC_DIFFERENCE";readiness="UNRESOLVED_SEMANTIC_DIFFERENCE"
    return {"raw_parameter_strict":{"status":"PASS" if raw.max_abs<=5e-5 else "FAIL","value":raw.max_abs,"threshold":5e-5},
      "solver_trajectory":{"status":"PASS" if trajectory else "FAIL"},
      "geometric_equivalence":{"status":"PASS" if failed is None else "FAIL","first_failure":failed,"checks":[{"name":n,"value":v,"threshold":t,"pass":v<=t} for n,v,t in checks]},
      "downstream_semantic":{"status":"PASS" if downstream_pass else "FAIL"},
      "hessian_psd":{"status":"PASS" if hessian["psd_gate"] else "FAIL"},
      "classification":classification,"integration_readiness":readiness}
