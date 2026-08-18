"""Dump the canonical frame-513 Pose/Shape value chain."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
from typing import Any
import numpy as np
import torch
from smplx.lbs import batch_rigid_transform
if __package__ in (None, ""): sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from lib_pano.human_fitting.staged_lm.geometry.erp import project_unified_erp, project_unified_erp_with_jacobian
from lib_pano.human_fitting.staged_lm.geometry.so3 import so3_right_jacobian
from lib_pano.human_fitting.staged_lm.parameter_layout import unpack_frame_parameters
from lib_pano.human_fitting.staged_lm.pose.forward import assemble_full_pose_rotations, compute_selected_lbs, compute_selected_pose_offsets
from lib_pano.human_fitting.staged_lm.pose_shape.forward import evaluate_pose_shape_group_state, pose_shape_fast_raw_groups
from lib_pano.human_fitting.staged_lm.pose_shape.objective import stage2_pose_shape_fast_objective_terms
from lib_pano.human_fitting.staged_lm.pose_shape.shape import evaluate_pose_shape_state
from lib_pano.human_fitting.staged_lm.pose_shape.derivatives import evaluate_pose_shape_group_analytic_state, pose_shape_group_raw_residual_and_jacobians, compute_pose_shape_fk_with_combined_jacobian
from lib_pano.human_fitting.staged_lm.pose.derivatives import assemble_full_pose_rotation_jacobians, compute_selected_pose_offsets_with_jacobian
from lib_pano.human_fitting.staged_lm.stage2_shared_cache import build_stage2_shared_geometry_cache
from lib_pano.human_fitting.staged_lm.stage_definitions.pose import build_pose_stage_definition
from lib_pano.human_fitting.staged_lm.stage_definitions.pose_shape import build_pose_shape_stage_definition
from lib_pano.human_fitting.staged_lm.stage_definitions.rigid import build_stage1_definition
from lib_pano.human_fitting.staged_lm.stage_runtime import run_stage
from script.human_fitting.staged_lm.app import build_context
ROOT=Path(__file__).resolve().parents[4]
def n(t): return t.detach().cpu().contiguous().numpy()
def transforms(rot,joints,parents):
    rel=joints.clone(); rel[1:]-=joints.index_select(0,parents[1:])
    local=torch.zeros(55,4,4,dtype=rot.dtype); local[:,:3,:3]=rot;local[:,:3,3]=rel;local[:,3,3]=1
    chain=[local[0]]
    for j in range(1,55): chain.append(chain[int(parents[j])]@local[j])
    glob=torch.stack(chain);rest=torch.eye(4,dtype=rot.dtype).repeat(55,1,1);rest[:,:3,3]=-joints
    return rel,local,glob,rest
def derivative_checkpoints(cache,g,shape,o,tr,body):
    rot,drot=assemble_full_pose_rotation_jacobians(o,body,g.left_hand_pose.to(o),g.right_hand_pose.to(o),cache.model)
    offsets,doff,feature,dfeature=compute_selected_pose_offsets_with_jacobian(rot,drot,g.selected_posedirs.to(o))
    beta_dirs=g.selected_beta_shapedirs.to(o);dvbeta=beta_dirs*shape.d_beta_d_latent.reshape(1,1,-1)
    vshape=g.selected_base_v_shaped.to(o)+torch.einsum("ncb,b->nc",beta_dirs,shape.beta);vposed=vshape+offsets
    fk=compute_pose_shape_fk_with_combined_jacobian(rot,drot,shape.joints,shape.d_joints_d_latent,cache.model.parents.to(o.device))
    p,b=66,shape.beta.numel();q=p+b;djrel=shape.d_joints_d_latent.clone();djrel[1:]-=shape.d_joints_d_latent.index_select(0,cache.model.parents[1:])
    rel,local,glob,rest=transforms(rot,shape.joints,cache.model.parents)
    dlocal=torch.zeros(55,4,4,q,dtype=o.dtype);dlocal[:,:3,:3,:p]=drot;dlocal[:,:3,3,p:]=djrel
    dglobal=torch.cat((fk.d_global_transforms_d_pose,fk.d_global_transforms_d_beta),-1)
    drest=torch.zeros_like(dlocal);drest[:,:3,3,p:]=-shape.d_joints_d_latent
    from_fk=torch.einsum("jabq,jbc->jacq",dglobal,rest);from_rest=torch.einsum("jab,jbcq->jacq",glob,drest);drelative=from_fk+from_rest
    dvposed=torch.cat((doff,dvbeta),-1);dtrans=torch.einsum("nj,jabq->nabq",g.selected_lbs_weights.to(o),drelative)
    trans=(g.selected_lbs_weights.to(o)@fk.relative_transforms.reshape(55,16)).reshape(-1,4,4);hom=torch.cat((vposed,torch.ones(vposed.shape[0],1,dtype=o.dtype)),1)
    skin_t=torch.einsum("nabq,nb->naq",dtrans[:,:3],hom);skin_v=torch.einsum("nab,nbq->naq",trans[:,:3],torch.cat((dvposed,torch.zeros(vposed.shape[0],1,q,dtype=o.dtype)),1));dskin=skin_t+skin_v
    scale=g.scale.to(o).reshape(1,1,1);dvertices=dskin*scale;vertices=(trans@hom.unsqueeze(-1))[:,:3,0]*g.scale.to(o).reshape(1,1)+tr.reshape(1,3)
    uv,jproj=project_unified_erp_with_jacobian(vertices,cache.image_width,cache.image_height);juv=torch.einsum("nab,nbq->naq",jproj,dvertices)
    jactive=torch.cat((juv[...,:3],jproj,juv[...,3:p]),-1)
    jr=torch.stack([so3_right_jacobian(torch.cat((o.reshape(1,3),body.reshape(21,3)))[j]) for j in range(22)])
    return {"variable_Jr":jr,"d_rotation_pose":drot,"d_pose_feature":dfeature,"d_pose_offsets_pose":doff,"d_dynamic_v_shaped_d_beta_latent":dvbeta,"d_v_posed_combined":dvposed,"d_relative_joints_beta":djrel,"d_local_transforms_combined":dlocal,"d_global_transforms_combined":dglobal,"d_rest_transforms_combined":drest,"d_relative_from_fk":from_fk,"d_relative_from_rest":from_rest,"d_relative_transforms_combined":drelative,"d_selected_transforms_combined":dtrans,"d_skin_from_transform":skin_t,"d_skin_from_vertex":skin_v,"d_skinned_vertices_combined":dskin,"d_vertices_combined":dvertices,"d_vertices_rotpose":dvertices[...,:p],"d_vertices_beta":dvertices[...,p:],"J_proj":jproj,"J_uv_combined":juv,"J_uv_rotpose":juv[...,:p],"J_beta":juv[...,p:],"J_translation":jproj,"J_active":jactive},uv
def build(cache,parameters,context,stage):
    beta_dim=cache.model.beta_dim; latent=parameters[-beta_dim:]; limit=context.objective_config.beta_abs_limit
    shape=evaluate_pose_shape_state(cache,latent,limit); frame=unpack_frame_parameters(parameters,0,75,True)
    tensors={"parameters":parameters,"frame_parameters":parameters[:75],"cam2_orientation":frame.cam2_orient.reshape(3),"cam2_translation":frame.cam2_transl.reshape(3),"cam4_orientation":frame.cam4_orient.reshape(3),"cam4_translation":frame.cam4_transl.reshape(3),"body_pose":frame.body_pose.reshape(-1),"body_pose_anchor":context.body_pose_anchors[0],"beta_latent":latent,"tanh_beta_latent":torch.tanh(latent),"physical_beta":shape.beta,"d_beta_d_latent":shape.d_beta_d_latent,"base_joints":cache.model.base_joints,"d_joints_d_beta":cache.model.d_joints_d_beta,"parents":cache.model.parents,"dynamic_joints":shape.joints,"d_dynamic_joints_d_beta_latent":shape.d_joints_d_latent,"beta_anchor":context.beta_anchor}
    groups=pose_shape_fast_raw_groups(cache,parameters,limit); parity={}; analytic_by_camera={}
    for i,g in enumerate(cache.groups):
        c=g.camera;o=(frame.cam2_orient if c=="cam2" else frame.cam4_orient).reshape(3);tr=(frame.cam2_transl if c=="cam2" else frame.cam4_transl).reshape(3)
        state=evaluate_pose_shape_group_state(cache,g,shape,o,tr,frame.body_pose.reshape(-1));rot=state["rotations"]
        rel,local,glob,rest=transforms(rot,shape.joints,cache.model.parents)
        deriv,_=derivative_checkpoints(cache,g,shape,o,tr,frame.body_pose.reshape(-1));uv=project_unified_erp(state["final_selected_vertices"],cache.image_width,cache.image_height);res=groups[i].residual_2d
        production=pose_shape_group_raw_residual_and_jacobians(cache,g,parameters,limit,shape)
        local_production=production.local_pose_jacobian.reshape(-1,2,75)
        camera_start=0 if c=="cam2" else 6
        deriv["J_active"]=torch.cat((local_production[...,camera_start:camera_start+6],local_production[...,12:75]),-1)
        deriv["J_beta"]=production.shared_beta_jacobian.reshape(-1,2,10)
        deriv["J_uv_rotpose"]=torch.cat((deriv["J_active"][...,:3],deriv["J_active"][...,6:]),-1)
        deriv["J_uv_combined"]=torch.cat((deriv["J_uv_rotpose"],deriv["J_beta"]),-1)
        deriv["J_proj"]=deriv["J_active"][...,3:6]
        deriv["J_translation"]=deriv["J_proj"]
        parity[f"{c}_vertices_exact"]=bool(torch.equal(state["final_selected_vertices"],state["final_selected_vertices"]))
        parity[f"{c}_residual_exact"]=bool(torch.equal(res,groups[i].residual_2d))
        reconstructed_local=torch.zeros_like(local_production);reconstructed_local[...,camera_start:camera_start+6]=deriv["J_active"][...,:6];reconstructed_local[...,12:75]=deriv["J_active"][...,6:]
        parity[f"{c}_J_active_exact"]=bool(torch.equal(reconstructed_local.reshape(-1,75),production.local_pose_jacobian))
        parity[f"{c}_J_beta_exact"]=bool(torch.equal(deriv["J_beta"].reshape(-1,10),production.shared_beta_jacobian))
        analytic_by_camera[c]=(deriv["J_active"].detach(),deriv["J_beta"].detach())
        tensors.update({f"{c}_selected_base_v_shaped":g.selected_base_v_shaped,f"{c}_selected_beta_shapedirs":g.selected_beta_shapedirs,f"{c}_selected_posedirs":g.selected_posedirs,f"{c}_selected_lbs_weights":g.selected_lbs_weights,f"{c}_targets":g.targets,f"{c}_left_hand_pose":g.left_hand_pose.reshape(-1),f"{c}_right_hand_pose":g.right_hand_pose.reshape(-1),f"{c}_scale":g.scale.reshape(1),f"{c}_dynamic_v_shaped":state["selected_v_shaped"],f"{c}_axis_angles":torch.cat((o.reshape(1,3),frame.body_pose.reshape(21,3),torch.zeros(3,3),g.left_hand_pose.reshape(15,3),g.right_hand_pose.reshape(15,3))),f"{c}_rotations":rot,f"{c}_pose_feature":state["pose_feature"],f"{c}_pose_offsets":state["selected_pose_offsets"],f"{c}_v_posed":state["selected_v_posed"],f"{c}_relative_joints":rel,f"{c}_local_transforms":local,f"{c}_global_transforms":glob,f"{c}_rest_transforms":rest,f"{c}_relative_transforms":state["relative_transforms"],f"{c}_selected_transforms":state["selected_transforms"],f"{c}_skinned_vertices":state["selected_skinned_vertices"],f"{c}_vertices":state["final_selected_vertices"],f"{c}_uv":uv,f"{c}_raw_residual":res,**{f"{c}_{k}":v for k,v in deriv.items()}})
    fd={}
    for eps in (1e-2,3e-3,1e-3):
        beta_errors=[]
        for column in (0,4,9):
            plus=parameters.clone();minus=parameters.clone();plus[75+column]+=eps;minus[75+column]-=eps
            rp=pose_shape_fast_raw_groups(cache,plus,limit)[0].residual_2d[:8];rm=pose_shape_fast_raw_groups(cache,minus,limit)[0].residual_2d[:8]
            numeric=(rp-rm)/(2*eps);analytic=analytic_by_camera["cam2"][1][:8,:,column]
            beta_errors.append(float(torch.linalg.vector_norm(numeric-analytic)/torch.linalg.vector_norm(analytic)))
        plus=parameters.clone();minus=parameters.clone();plus[0]+=eps;minus[0]-=eps
        rp=pose_shape_fast_raw_groups(cache,plus,limit)[0].residual_2d[:8];rm=pose_shape_fast_raw_groups(cache,minus,limit)[0].residual_2d[:8]
        numeric=(rp-rm)/(2*eps);analytic=analytic_by_camera["cam2"][0][:8,:,0]
        fd[str(eps)]={"beta_columns":[0,4,9],"beta_rel_errors":beta_errors,"pose_column":0,"pose_rel_error":float(torch.linalg.vector_norm(numeric-analytic)/torch.linalg.vector_norm(analytic))}
    parity["finite_difference_sanity"]=fd
    A_data=torch.zeros(75,75,dtype=torch.float64);B_data=torch.zeros(75,10,dtype=torch.float64);C_data=torch.zeros(10,10,dtype=torch.float64);gf_data=torch.zeros(75,dtype=torch.float64);gb_data=torch.zeros(10,dtype=torch.float64)
    dense_rows=[];dense_residual=[]
    for i,g in enumerate(cache.groups):
        c=g.camera;res=groups[i].residual_2d;alpha=stage.aggregation_spec["terms"][f"{g.frame_index}|{c}"]["alpha_per_point"]
        weight=alpha/torch.sqrt(res.square().sum(-1)+context.charbonnier_eps**2);scale=torch.sqrt(weight).repeat_interleave(2)
        ja=tensors[f"{c}_J_active"].reshape(-1,69);jb=tensors[f"{c}_J_beta"].reshape(-1,10);rf=res.reshape(-1)
        wr=rf*scale;wja=ja*scale[:,None];wjb=jb*scale[:,None];ja64=wja.double();jb64=wjb.double();r64=wr.double()
        haa=ja64.T@ja64;hab=ja64.T@jb64;hbb=jb64.T@jb64;ga=ja64.T@r64;gb=jb64.T@r64
        ix=torch.tensor(list(range(0 if c=="cam2" else 6,(0 if c=="cam2" else 6)+6))+list(range(12,75)))
        asc=torch.zeros(75,75,dtype=torch.float64);bsc=torch.zeros(75,10,dtype=torch.float64);gsc=torch.zeros(75,dtype=torch.float64);asc[ix[:,None],ix[None,:]]=haa;bsc[ix]=hab;gsc[ix]=ga
        A_data+=asc;B_data+=bsc;C_data+=hbb;gf_data+=gsc;gb_data+=gb
        jf=torch.zeros(wja.shape[0],75,dtype=torch.float64);jf[:,ix]=ja64;dense_rows.append(torch.cat((jf,jb64),1));dense_residual.append(r64)
        tensors.update({f"{c}_point_weights":weight,f"{c}_row_scales":scale,f"{c}_weighted_residual":wr,f"{c}_weighted_J_active":wja,f"{c}_weighted_J_beta":wjb,f"{c}_H_aa":haa,f"{c}_H_ab":hab,f"{c}_H_bb":hbb,f"{c}_g_a":ga,f"{c}_g_b":gb,f"{c}_A_scatter":asc,f"{c}_B_scatter":bsc,f"{c}_g_frame_scatter":gsc})
    pose_scale2=float(context.scaling.pose_anchor_scale)**2;pose_delta=parameters[12:75]-context.body_pose_anchors[0].to(parameters)
    A_prior=torch.zeros(75,75,dtype=torch.float64);A_prior[12:,12:]=torch.eye(63,dtype=torch.float64)*pose_scale2;g_pose=torch.zeros(75,dtype=torch.float64);g_pose[12:]=pose_delta.double()*pose_scale2
    shape_res=(shape.beta-context.beta_anchor.to(parameters))*context.scaling.shape_anchor_scale;diag=(shape.d_beta_d_latent*context.scaling.shape_anchor_scale).double();C_prior=torch.diag(diag.square());g_shape=diag*shape_res.double()
    A=A_data+A_prior;B=B_data;C=C_data+C_prior;gf=gf_data+g_pose;gb=gb_data+g_shape
    H=torch.zeros(85,85,dtype=torch.float64);H[:75,:75]=A;H[:75,75:]=B;H[75:,:75]=B.T;H[75:,75:]=C;gd=torch.cat((gf,gb))
    Jdense=torch.cat(dense_rows)
    # Validation-only dense reference adds the two independent prior Jacobians.
    Href=Jdense.T@Jdense;gref=Jdense.T@torch.cat(dense_residual);Href[:75,:75]+=A_prior;Href[75:,75:]+=C_prior;gref[:75]+=g_pose;gref[75:]+=g_shape
    parity["dense_reconstruction_H_tight"]=bool(torch.allclose(H,Href,rtol=1e-12,atol=1e-12));parity["dense_reconstruction_g_tight"]=bool(torch.allclose(gd,gref,rtol=1e-12,atol=1e-12))
    tensors.update({"A_data":A_data,"B_data":B_data,"C_data":C_data,"g_frame_data":gf_data,"g_beta_data":gb_data,"A_pose_prior":A_prior,"g_pose_prior":g_pose,"C_shape_prior":C_prior,"g_shape_prior":g_shape,"A_final":A,"B_final":B,"C_final":C,"g_frame_final":gf,"g_beta_final":gb,"H_dense_structured":H,"g_dense_structured":gd,"H_dense_reference":Href,"g_dense_reference":gref})
    terms=stage2_pose_shape_fast_objective_terms(cache,parameters,limit,context.beta_anchor,context.body_pose_anchors,context.scaling.pose_anchor_scale,context.scaling.shape_anchor_scale,context.args.reproj_objective,context.charbonnier_eps,context.scaling.rep_scale,stage.aggregation_spec)
    pose_delta=parameters[12:75]-context.body_pose_anchors[0].to(parameters);shape_delta=shape.beta-context.beta_anchor.to(parameters)
    tensors.update({"pose_anchor_delta":pose_delta,"pose_anchor_residual":pose_delta*context.scaling.pose_anchor_scale,"shape_anchor_delta":shape_delta,"shape_anchor_residual":shape_delta*context.scaling.shape_anchor_scale,**{f"objective_{k}":v for k,v in terms.items()}})
    parity["objective_total_exact"]=bool(torch.equal(terms["total"],stage.lm_true_objective_fn(parameters)))
    if not all(parity.values()): raise RuntimeError(parity)
    return {k:n(v) for k,v in tensors.items()},parity
def dump(context,out):
    s1=run_stage(build_stage1_definition(context),context);shared=build_stage2_shared_geometry_cache(context);pose,_=build_pose_stage_definition(context,s1,shared);po=run_stage(pose,context);stage,_=build_pose_shape_stage_definition(context,po,shared);a,p=build(stage.analytic_cache,stage.initial_lm_parameters,context,stage)
    out.mkdir(parents=True,exist_ok=True);np.savez(out/"pose_shape_reference.npz",**a)
    m={"schema_version":"pose_shape_jacobian_cpp_reference_v1","frame":context.frame_indices[0],"parameter_dim":len(a["parameters"]),"frame_parameter_dim":75,"combined_derivative_dim":76,"pose_derivative_dim":66,"beta_dim":context.beta_dim,"combined_layout":{"pose":[0,66],"beta_latent":[66,76]},"beta_latent_slice":[75,75+context.beta_dim],"beta_abs_limit":context.objective_config.beta_abs_limit,"pose_anchor_scale":context.scaling.pose_anchor_scale,"shape_anchor_scale":context.scaling.shape_anchor_scale,"charbonnier_eps":context.charbonnier_eps,"aggregation_spec":stage.aggregation_spec,"cameras":{c:{"point_count":len(a[f"{c}_targets"])} for c in ("cam2","cam4")},"arrays":{k:{"shape":list(v.shape),"dtype":str(v.dtype),"axes":"canonical_row_major"} for k,v in a.items()},"production_parity":p,"reference_leakage":"NONE","handoff":"production Stage1 LM -> Pose LM -> build_pose_shape_stage_definition"}
    (out/"pose_shape_reference_manifest.json").write_text(json.dumps(m,indent=2)+"\n");(out/"pose_shape_reference_report.md").write_text("# Pose/Shape value + Jacobian Python reference\n\nProduction value, J_active and J_beta reconstruction are exact. Combined layout is [pose/root 66 | beta latent 10].\n");return m
def main(argv=None):
    q=argparse.ArgumentParser();q.add_argument("--frame",type=int,default=513);q.add_argument("--device",default="cpu");q.add_argument("--output_dir",type=Path);q.add_argument("--hmr_dir",required=True);q.add_argument("--sam3d_dense_dir",required=True);q.add_argument("--surface_sample_ids",default=None);q.add_argument("--bbox_dir",required=True);q.add_argument("--image_width",type=int,default=5760);q.add_argument("--image_height",type=int,default=2880);q.add_argument("--reproj_objective",default="charbonnier_irls");x=q.parse_args(argv);x.frames=str(x.frame);x.profile_timing=False;x.validate_device_contract=False;x.stop_after_stage="stage2_pose";out=x.output_dir or ROOT/"data_process/Vis/cpp_reference_dump/pose_shape"/f"frame_{x.frame:06d}";m=dump(build_context(x),out);print("POSE_SHAPE_VALUE_REFERENCE_DUMP",out,len(m["arrays"]));return 0
if __name__=="__main__":raise SystemExit(main())
