"""Stage 2 joint beta and combined analytic Jacobian validations.

Mechanically consolidated from the historical per-check scripts.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
import numpy as np
import torch
from torch.func import jacfwd, jvp
from smplx.lbs import batch_rigid_transform, blend_shapes, vertices2joints
from script.human_fitting.staged_lm.app import build_context
from lib_pano.human_fitting.staged_lm.irls import build_primary_aggregation_spec, compute_primary_charbonnier_tensor
from lib_pano.human_fitting.staged_lm.residuals import flatten_reprojection_groups
from lib_pano.human_fitting.staged_lm.stage2_joint_fast import build_stage2_joint_fast_cache, evaluate_joint_shape_state, joint_fast_raw_groups, joint_fast_selected_vertices, physical_beta_from_latent
from lib_pano.human_fitting.staged_lm.stage2_joint_beta_jacobian import joint_fast_selected_vertices_with_beta_jacobian, joint_group_raw_residual_and_beta_jacobian, joint_surrogate_residual_and_beta_jacobian
from lib_pano.human_fitting.staged_lm.stage2_pose_fast import assemble_full_pose_rotations
from lib_pano.human_fitting.staged_lm.stages import build_pose_stage_definition, build_stage1_definition, run_stage
from lib_pano.human_fitting.smplx_sequence import forward_smplx_tensor_batch
import argparse, time
from lib_pano.human_fitting.staged_lm.types import ReprojectionGroup
from lib_pano.human_fitting.staged_lm.stage2_joint_fast import build_stage2_joint_fast_cache, joint_fast_raw_groups, joint_fast_raw_residual
from lib_pano.human_fitting.staged_lm.stage2_joint_beta_jacobian import joint_explicit_raw_beta_jacobian
from lib_pano.human_fitting.staged_lm.stage2_joint_jacobian import joint_explicit_raw_global_jacobian, joint_group_raw_residual_and_jacobians, joint_surrogate_residual_and_jacobian
from lib_pano.human_fitting.staged_lm.stage2_pose_fast import build_stage2_pose_fast_cache
from lib_pano.human_fitting.staged_lm.stage2_shared_cache import build_stage2_shared_geometry_cache
from lib_pano.human_fitting.staged_lm.stage2_pose_jacobian import stage2_pose_group_raw_residual_and_jacobian


def joint_beta_context_args(cli):
    return argparse.Namespace(
        frames=cli.frames, hmr_dir=cli.hmr_dir, sam3d_dense_dir=cli.sam3d_dense_dir,
        bbox_dir=cli.bbox_dir, image_width=cli.image_width, image_height=cli.image_height,
        output_dir="/tmp/debug115", reproj_objective="charbonnier_irls",
        stop_after_stage="stage2_pose", run_lbfgs_oracle=False,
        jacobian_backend="analytic_stage1", stage1_eval_backend="analytic",
        validate_stage1_fast_eval=False, stage2_pose_linearization_backend="analytic",
        stage2_pose_eval_backend="fast", validate_stage2_pose_fast_eval=False,
        validate_irls_first_order=False,
        profile_timing=cli.profile,
    )


def joint_beta_error(actual, expected):
    delta = actual - expected
    return {"rel_l2": float(delta.norm()/expected.norm().clamp_min(1e-16)),
            "max_abs": float(delta.abs().max()), "mean_abs": float(delta.abs().mean())}


def full_selected(context, group, local_pose, beta):
    frozen = context.frozen_params_by_frame[group.frame_position]
    orient = local_pose[:3] if group.camera == "cam2" else local_pose[6:9]
    translation = local_pose[3:6] if group.camera == "cam2" else local_pose[9:12]
    left = getattr(frozen, f"{group.camera}_left_hand_pose").to(orient)
    right = getattr(frozen, f"{group.camera}_right_hand_pose").to(orient)
    scale = getattr(frozen, f"{group.camera}_scale").to(orient)
    vertices, _, _ = forward_smplx_tensor_batch(
        context.model, orient.reshape(1, 3), local_pose[12:75].reshape(1, -1), left, right,
        beta.reshape(1, -1), translation.reshape(1, 3), scale.expand(1),
    )
    return vertices[0].index_select(0, group.vertex_ids.to(vertices.device))


def reference_intermediates(context, cache, group, local, beta):
    model = context.model; reference = local
    expression = torch.zeros(1, model.num_expression_coeffs, dtype=reference.dtype, device=reference.device)
    components = torch.cat((beta.reshape(1, -1), expression), 1)
    dirs = torch.cat((model.shapedirs.to(reference), model.expr_dirs.to(reference)), -1)
    v_shaped = model.v_template.to(reference) + blend_shapes(components, dirs).squeeze(0)
    joints = vertices2joints(model.J_regressor.to(reference), v_shaped.unsqueeze(0)).squeeze(0)
    orient = local[:3] if group.camera == "cam2" else local[6:9]
    translation = local[3:6] if group.camera == "cam2" else local[9:12]
    rotations = assemble_full_pose_rotations(
        orient, local[12:75], group.left_hand_pose.to(reference), group.right_hand_pose.to(reference), cache.model,
    )
    identity = torch.eye(3, dtype=reference.dtype, device=reference.device)
    pose_feature = (rotations[1:] - identity).reshape(-1)
    pose_offsets = (pose_feature @ model.posedirs.to(reference)).reshape(-1, 3)
    v_posed = v_shaped + pose_offsets
    _, relative = batch_rigid_transform(
        rotations.unsqueeze(0), joints.unsqueeze(0), model.parents.to(reference.device), dtype=reference.dtype,
    )
    relative = relative[0]
    relative_joints = joints.clone(); relative_joints[1:] -= joints[model.parents[1:].to(reference.device)]
    local_transforms = torch.zeros(cache.model.joint_count, 4, 4, dtype=reference.dtype, device=reference.device)
    local_transforms[:, :3, :3] = rotations; local_transforms[:, :3, 3] = relative_joints
    local_transforms[:, 3, 3] = 1
    globals_ = [local_transforms[0]]
    for joint in range(1, cache.model.joint_count):
        globals_.append(globals_[int(model.parents[joint])] @ local_transforms[joint])
    global_transforms = torch.stack(globals_)
    transforms = (model.lbs_weights.to(reference) @ relative.reshape(cache.model.joint_count, 16)).reshape(-1, 4, 4)
    homogeneous = torch.cat((v_posed, torch.ones(v_posed.shape[0], 1, dtype=reference.dtype, device=reference.device)), 1)
    skinned = (transforms @ homogeneous.unsqueeze(-1))[:, :3, 0]
    ids = group.vertex_ids.to(reference.device); scale = group.scale.to(reference)
    return {
        "physical_beta": beta,
        "selected_v_shaped": v_shaped.index_select(0, ids), "joints": joints,
        "rotations": rotations, "pose_feature": pose_feature,
        "selected_pose_offsets": pose_offsets.index_select(0, ids),
        "selected_v_posed": v_posed.index_select(0, ids),
        "relative_joints": relative_joints, "global_transforms": global_transforms,
        "relative_transforms": relative,
        "selected_transforms": transforms.index_select(0, ids),
        "selected_skinned_vertices": skinned.index_select(0, ids),
        "final_selected_vertices": skinned.index_select(0, ids)*scale + translation,
    }


def joint_beta_points(anchor_latent, archive, beta_dim):
    generator = torch.Generator(device=anchor_latent.device).manual_seed(115)
    result = [("zero", torch.zeros_like(anchor_latent)), ("anchor", anchor_latent)]
    for dimension in range(beta_dim):
        plus = anchor_latent.clone(); plus[dimension] += 1e-3
        minus = anchor_latent.clone(); minus[dimension] -= 1e-3
        result.extend(((f"dim{dimension}_plus", plus), (f"dim{dimension}_minus", minus)))
    result.extend((
        ("random_1e-2", anchor_latent + torch.randn(beta_dim, generator=generator, device=anchor_latent.device)*1e-2),
        ("random_0.1", anchor_latent + torch.randn(beta_dim, generator=generator, device=anchor_latent.device)*.1),
        ("large", torch.linspace(-2, 2, beta_dim, device=anchor_latent.device)),
        ("saturation", torch.cat((anchor_latent.new_tensor([4., -4.]), anchor_latent[2:]))),
    ))
    if archive is not None:
        for key in ("beta_latent0", "x0_stage2_joint_lm", "x_stage2_joint_lm", "x_stage2_joint_lbfgs"):
            if key in archive:
                value = torch.as_tensor(archive[key]).to(anchor_latent)
                result.append((key, value[-beta_dim:]))
    return result


def check_forward_and_jacobian(context, cache, pose, beta_points, limit, check_intermediates):
    for label, latent in beta_points:
        shape = evaluate_joint_shape_state(cache, latent, limit)
        for group in cache.groups:
            local = pose[group.frame_position*75:(group.frame_position+1)*75]
            orient = local[:3] if group.camera == "cam2" else local[6:9]
            translation = local[3:6] if group.camera == "cam2" else local[9:12]
            fast, physical_j, latent_j, intermediate = joint_fast_selected_vertices_with_beta_jacobian(
                cache, group, shape, orient, translation, local[12:75], return_intermediates=True,
            )
            full = full_selected(context, group, local, shape.beta)
            vertex_error = joint_beta_error(fast, full)
            if vertex_error["max_abs"] > 1e-5 or vertex_error["rel_l2"] > 1e-6:
                raise RuntimeError(f"STAGE2_JOINT_DYNAMIC_FORWARD_MISMATCH: {label}/{group.camera} {vertex_error}")
            fast_uv = context.fitter._project_cam2(fast, context.image_width, context.image_height)
            full_uv = context.fitter._project_cam2(full, context.image_width, context.image_height)
            torch.testing.assert_close(fast_uv, full_uv, rtol=1e-6, atol=1e-3)
            physical_oracle = jacfwd(lambda beta: full_selected(context, group, local, beta))(shape.beta)
            full_latent_oracle = jacfwd(lambda z: full_selected(
                context, group, local, limit*torch.tanh(z),
            ))(latent)
            fast_latent_oracle = jacfwd(lambda z: joint_fast_selected_vertices(
                cache, group, evaluate_joint_shape_state(cache, z, limit),
                orient, translation, local[12:75],
            ))(latent)
            reports = {
                "physical_full": joint_beta_error(physical_j, physical_oracle),
                "latent_full": joint_beta_error(latent_j, full_latent_oracle),
                "latent_fast": joint_beta_error(latent_j, fast_latent_oracle),
                "chain": joint_beta_error(latent_j, physical_j*shape.d_beta_d_latent.reshape(1,1,-1)),
            }
            saturation = label == "saturation"
            if (reports["physical_full"]["rel_l2"] > 1e-4
                    or (reports["latent_full"]["rel_l2"] > 1e-4 and not saturation)
                    or reports["latent_full"]["max_abs"] > 1e-4):
                raise RuntimeError(f"STAGE2_JOINT_BETA_VERTEX_J_MISMATCH: {label}/{group.camera} {reports}")
            parameters = torch.cat((pose, latent))
            residual, analytic_uv_j = joint_group_raw_residual_and_beta_jacobian(
                cache, group, parameters, limit, shape,
            )
            full_residual_oracle = jacfwd(lambda z: (
                context.fitter._project_cam2(full_selected(context, group, local, limit*torch.tanh(z)),
                                             context.image_width, context.image_height)
                - group.targets.to(z)
            ).reshape(-1))(latent)
            uv_report = joint_beta_error(analytic_uv_j, full_residual_oracle)
            if uv_report["rel_l2"] > 5e-4 and uv_report["max_abs"] > 1e-3:
                raise RuntimeError(f"STAGE2_JOINT_BETA_RESIDUAL_J_MISMATCH: {label}/{group.camera} {uv_report}")
            per_beta = [joint_beta_error(analytic_uv_j[:, b], full_residual_oracle[:, b]) for b in range(cache.model.beta_dim)]
            worst = max(range(cache.model.beta_dim), key=lambda b: per_beta[b]["rel_l2"])
            print(f"[{label}] frame={group.frame_index} camera={group.camera} vertex={vertex_error} J={reports} uvJ={uv_report} worst_beta={worst}:{per_beta[worst]}")
            if check_intermediates:
                reference_values = reference_intermediates(context, cache, group, local, shape.beta)
                intermediate_errors = {name: joint_beta_error(intermediate[name], expected)
                                       for name, expected in reference_values.items()}
                worst_name = max(intermediate_errors, key=lambda name: intermediate_errors[name]["max_abs"])
                print(f"[INTERMEDIATE/{label}/{group.camera}] worst={worst_name} {intermediate_errors[worst_name]}")
                if intermediate_errors[worst_name]["max_abs"] > 1e-5:
                    raise RuntimeError(f"STAGE2_JOINT_INTERMEDIATE_MISMATCH: {worst_name}")
                if not all(torch.isfinite(v).all() for v in intermediate.values()):
                    raise RuntimeError("STAGE2_JOINT_NONFINITE_INTERMEDIATE")


def gradient_jvp_difference(cache, pose, latent, limit, beta_anchor, body_anchors, pose_scale, shape_scale):
    parameters = torch.cat((pose, latent))
    groups = joint_fast_raw_groups(cache, parameters, limit)
    spec = build_primary_aggregation_spec(groups)
    for mode in ("charbonnier_irls", "raw_l2"):
        surrogate = joint_surrogate_residual_and_beta_jacobian(
            cache, parameters, limit, beta_anchor, body_anchors, pose_scale, shape_scale,
            objective_mode=mode, charbonnier_eps=1., rep_scale=.02,
        )
        def objective(z):
            x = torch.cat((pose, z)); gs = joint_fast_raw_groups(cache, x, limit)
            primary = (compute_primary_charbonnier_tensor(gs, 1., spec) if mode == "charbonnier_irls"
                       else .5*torch.sum((flatten_reprojection_groups(gs)*.02)**2))
            pose_anchor = torch.cat([(pose[f*75+12:(f+1)*75]-body_anchors[f].to(pose))*pose_scale for f in range(cache.frame_count)])
            beta = limit*torch.tanh(z)
            return primary + .5*torch.sum(pose_anchor**2) + .5*torch.sum(((beta-beta_anchor.to(beta))*shape_scale)**2)
        true_grad = torch.func.grad(objective)(latent)
        surrogate_grad = surrogate.beta_jacobian.T @ surrogate.residual
        report = joint_beta_error(surrogate_grad, true_grad)
        print(f"[TRUE_GRADIENT/{mode}] {report}")
        if report["rel_l2"] > 5e-4: raise RuntimeError("STAGE2_JOINT_BETA_TRUE_GRADIENT_MISMATCH")
    analytic = joint_group_raw_residual_and_beta_jacobian(cache, cache.groups[0], parameters, limit)[1]
    fn = lambda z: joint_fast_raw_groups(cache, torch.cat((pose, z)), limit)[0].residual_2d.reshape(-1)
    generator = torch.Generator(device=latent.device).manual_seed(116)
    directions = [torch.eye(latent.numel(), device=latent.device)[0], torch.randn(latent.shape, generator=generator, device=latent.device),
                  torch.ones_like(latent), torch.linspace(-1, 1, latent.numel(), device=latent.device)]
    for index, direction in enumerate(directions):
        _, oracle = jvp(fn, (latent,), (direction,)); report = joint_beta_error(analytic@direction, oracle)
        print(f"[JVP/{index}] {report}")
        if report["rel_l2"] > 5e-4: raise RuntimeError("STAGE2_JOINT_BETA_JVP_MISMATCH")
    for epsilon in (3e-4, 1e-3, 3e-3, 1e-2):
        columns=[]
        for b in range(latent.numel()):
            step=torch.zeros_like(latent); step[b]=epsilon
            columns.append((fn(latent+step)-fn(latent-step))/(2*epsilon))
        finite=torch.stack(columns,1); print(f"[FINITE eps={epsilon:g}] {joint_beta_error(analytic, finite)}")


def joint_beta_sync(t):
    if t.is_cuda: torch.cuda.synchronize(t.device)


def joint_beta_profile(context, cache, pose, latent, limit):
    cache.reset_timing()
    for groups_label, selected in (("cam2", cache.groups[:1]), ("cam4", cache.groups[1:2]), ("complete_frame", cache.groups[:2])):
        def full_fn():
            return torch.cat([jacfwd(lambda z, group=g: context.fitter._project_cam2(
                full_selected(context, group, pose[:75], limit*torch.tanh(z)), context.image_width, context.image_height
            ).reshape(-1))(latent) for g in selected])
        def fast_fn():
            return torch.cat([jacfwd(lambda z, group=g: context.fitter._project_cam2(
                joint_fast_selected_vertices(cache, group, evaluate_joint_shape_state(cache,z,limit),
                    pose[:3] if group.camera=="cam2" else pose[6:9],
                    pose[3:6] if group.camera=="cam2" else pose[9:12], pose[12:75]),
                context.image_width, context.image_height).reshape(-1))(latent) for g in selected])
        parameters=torch.cat((pose,latent))
        def analytic_fn():
            shape=evaluate_joint_shape_state(cache,latent,limit)
            return torch.cat([joint_group_raw_residual_and_beta_jacobian(cache,g,parameters,limit,shape)[1] for g in selected])
        for _ in range(10): full_fn(); fast_fn(); analytic_fn()
        def measure(fn):
            joint_beta_sync(latent); start=time.perf_counter()
            for _ in range(20): fn()
            joint_beta_sync(latent); return (time.perf_counter()-start)*50
        full_ms, fast_ms, analytic_ms=measure(full_fn),measure(fast_fn),measure(analytic_fn)
        print(f"[PROFILE/{groups_label}] cache_setup={cache.setup_ms:.3f} full_jacfwd={full_ms:.3f} fast_jacfwd={fast_ms:.3f} analytic={analytic_ms:.3f} speedup_full={full_ms/analytic_ms:.2f} speedup_fast={fast_ms/analytic_ms:.2f} beta_dim={cache.model.beta_dim} rows={sum(g.targets.numel() for g in selected)}")
    print(f"[PROFILE/CACHE] {cache.timing_dict()}")


def run_joint_beta_validation(cli=None):
    cli=cli or parse_args(); context=build_context(joint_beta_context_args(cli)); archive=np.load(cli.joint_npz) if cli.joint_npz else None
    if archive is not None and "x_stage2_pose_lm" in archive:
        pose=torch.as_tensor(archive["x_stage2_pose_lm"]).to(context.beta_anchor)
    else:
        stage1=run_stage(build_stage1_definition(context),context)
        shared=build_stage2_shared_geometry_cache(context)
        pose=run_stage(build_pose_stage_definition(context,stage1,shared)[0],context).lm_result.parameters
    shared=build_stage2_shared_geometry_cache(context)
    cache=build_stage2_joint_fast_cache(context,shared); limit=context.objective_config.beta_abs_limit
    anchor_latent=torch.atanh(context.beta_anchor/limit)
    beta_points=joint_beta_points(anchor_latent,archive,context.beta_dim)
    check_forward_and_jacobian(context,cache,pose,beta_points,limit,cli.check_intermediates)
    gradient_jvp_difference(cache,pose,anchor_latent,limit,context.beta_anchor,context.body_pose_anchors,
                            context.scaling.pose_anchor_scale,context.scaling.shape_anchor_scale)
    if cli.profile: joint_beta_profile(context,cache,pose,anchor_latent,limit)
    print("[PASS] Stage 2 joint dynamic-beta forward and beta Jacobian")
    return 0


def joint_jacobian_context_args(c):
    return argparse.Namespace(frames=c.frames,hmr_dir=c.hmr_dir,sam3d_dense_dir=c.sam3d_dense_dir,bbox_dir=c.bbox_dir,
        image_width=c.image_width,image_height=c.image_height,output_dir="/tmp/debug116",reproj_objective="charbonnier_irls",
        stop_after_stage="stage2_pose",run_lbfgs_oracle=False,jacobian_backend="analytic_stage1",stage1_eval_backend="analytic",
        validate_stage1_fast_eval=False,stage2_pose_linearization_backend="analytic",stage2_pose_eval_backend="fast",
        validate_stage2_pose_fast_eval=False,validate_irls_first_order=False,
        profile_timing=c.profile,device=getattr(c,"device","auto"))


def joint_jacobian_error(a,b):
    d=a-b;return {"rel":float(d.norm()/b.norm().clamp_min(1e-16)),"max":float(d.abs().max()),"mean":float(d.abs().mean())}


def full_group(context,cache,group,parameters,limit):
    per=12+cache.model.body_pose_dim;local=parameters[group.frame_position*per:(group.frame_position+1)*per]
    orient=local[:3] if group.camera=="cam2" else local[6:9];translation=local[3:6] if group.camera=="cam2" else local[9:12]
    frozen=context.frozen_params_by_frame[group.frame_position]
    left=getattr(frozen,f"{group.camera}_left_hand_pose").to(parameters);right=getattr(frozen,f"{group.camera}_right_hand_pose").to(parameters)
    scale=getattr(frozen,f"{group.camera}_scale").to(parameters);beta=limit*torch.tanh(parameters[-cache.model.beta_dim:])
    vertices,_,_=forward_smplx_tensor_batch(context.model,orient.reshape(1,3),local[12:].reshape(1,-1),left,right,
        beta.reshape(1,-1),translation.reshape(1,3),scale.expand(1))
    selected=vertices[0].index_select(0,group.vertex_ids.to(vertices.device))
    uv=context.fitter._project_cam2(selected,context.image_width,context.image_height)
    return (uv-group.targets.to(parameters)).reshape(-1)


def full_raw(context,cache,parameters,limit): return torch.cat([full_group(context,cache,g,parameters,limit) for g in cache.groups])


def joint_jacobian_points(archive,anchor,pose,beta_dim):
    result=[]
    for key in ("x0_stage2_joint_lm","x_stage2_joint_lm","x_stage2_joint_lbfgs"):
        if archive is not None and key in archive:result.append((key,torch.as_tensor(archive[key]).to(pose)))
    base=torch.cat((pose,anchor));result.extend((("latent_zero",torch.cat((pose,torch.zeros_like(anchor)))),
        ("random_1e-2",base+torch.cat((torch.zeros_like(pose),torch.randn_like(anchor)*1e-2))),
        ("random_0.1",base+torch.cat((torch.zeros_like(pose),torch.randn_like(anchor)*.1)))))
    saturation=base.clone();saturation[-beta_dim:-beta_dim+2]=base.new_tensor([4.,-4.]);result.append(("saturation",saturation))
    combined=base.clone();combined[:12]+=.01;combined[12:75]+=.01;combined[-beta_dim:]+=.02;result.append(("pose_beta",combined))
    return result


def report_blocks(actual,oracle,cache,label):
    per=12+cache.model.body_pose_dim
    for frame in range(cache.frame_count):
        base=frame*per
        for name,sl in (("c2o",slice(base,base+3)),("c2t",slice(base+3,base+6)),("c4o",slice(base+6,base+9)),
                        ("c4t",slice(base+9,base+12)),("body",slice(base+12,base+per))): print(f"[{label}/{frame}/{name}] {joint_jacobian_error(actual[:,sl],oracle[:,sl])}")
    body=[]
    for joint in range(21):
        sl=slice(12+joint*3,15+joint*3);body.append((joint_jacobian_error(actual[:,sl],oracle[:,sl]),joint+1))
    beta=[(joint_jacobian_error(actual[:,-cache.model.beta_dim+i],oracle[:,-cache.model.beta_dim+i]),i) for i in range(cache.model.beta_dim)]
    print(f"[{label}/WORST_BODY] {max(body,key=lambda x:x[0]['rel'])}");print(f"[{label}/WORST_BETA] {max(beta,key=lambda x:x[0]['rel'])}")


def checks(context,cache,pose_cache,test_points,limit):
    for label,x in test_points:
        analytic=joint_explicit_raw_global_jacobian(cache,x,limit);fast=jacfwd(lambda value:joint_fast_raw_residual(cache,value,limit))(x)
        full=jacfwd(lambda value:full_raw(context,cache,value,limit))(x)
        ea,ef=joint_jacobian_error(analytic,fast),joint_jacobian_error(analytic,full);print(f"[{label}] fast={ea} full={ef}")
        if ef["rel"]>5e-4:raise RuntimeError("STAGE2_JOINT_COMPLETE_J_MISMATCH")
        report_blocks(analytic,full,cache,label)
        beta_only=joint_explicit_raw_beta_jacobian(cache,x,limit)
        print(f"[{label}/BETA_ORACLE] {joint_jacobian_error(analytic[:,-cache.model.beta_dim:],beta_only)}")
        # Cross-block evidence for the future arrowhead normal equations.
        cross_a=analytic[:,:-cache.model.beta_dim].T@analytic[:,-cache.model.beta_dim:]
        cross_f=full[:,:-cache.model.beta_dim].T@full[:,-cache.model.beta_dim:]
        print(f"[{label}/H_POSE_BETA] norm={float(cross_a.norm()):.3e} max={float(cross_a.abs().max()):.3e} error={joint_jacobian_error(cross_a,cross_f)}")
        if joint_jacobian_error(cross_a,cross_f)["rel"]>1e-5:raise RuntimeError("STAGE2_JOINT_CROSS_H_MISMATCH")
        if label=="x_stage2_joint_lm" and pose_cache is not None:
            dynamic=joint_group_raw_residual_and_jacobians(cache,cache.groups[0],x,limit).local_pose_jacobian
            fixed=stage2_pose_group_raw_residual_and_jacobian(pose_cache,pose_cache.groups[0],x[:-cache.model.beta_dim])[1]
            print(f"[DYNAMIC_VS_FIXED_ANCHOR_POSE_J] {joint_jacobian_error(dynamic,fixed)}")
    return test_points[0][1]


def surrogate_gradient(context,cache,x,limit):
    spec=build_primary_aggregation_spec(joint_fast_raw_groups(cache,x,limit))
    for mode in ("charbonnier_irls","raw_l2"):
        result=joint_surrogate_residual_and_jacobian(cache,x,limit,context.beta_anchor,context.body_pose_anchors,
            context.scaling.pose_anchor_scale,context.scaling.shape_anchor_scale,objective_mode=mode,charbonnier_eps=context.charbonnier_eps,
            rep_scale=context.scaling.rep_scale)
        def objective(value):
            groups=[]
            for group in cache.groups:
                residual=full_group(context,cache,group,value,limit)
                groups.append(ReprojectionGroup(group.frame_index,group.camera,residual.reshape(-1,2)))
            primary=(compute_primary_charbonnier_tensor(groups,context.charbonnier_eps,spec) if mode=="charbonnier_irls"
                     else .5*torch.sum((torch.cat([g.residual_2d.reshape(-1) for g in groups])*context.scaling.rep_scale)**2))
            per=12+cache.model.body_pose_dim;pose_a=torch.cat([(value[f*per+12:(f+1)*per]-context.body_pose_anchors[f])*context.scaling.pose_anchor_scale for f in range(cache.frame_count)])
            beta=limit*torch.tanh(value[-cache.model.beta_dim:]);shape=(beta-context.beta_anchor)*context.scaling.shape_anchor_scale
            return primary+.5*torch.sum(pose_a**2)+.5*torch.sum(shape**2)
        true=torch.func.grad(objective)(x);approx=result.jacobian.T@result.residual
        print(f"[TRUE_GRAD/{mode}] {joint_jacobian_error(approx,true)}")
        if joint_jacobian_error(approx,true)["rel"]>context.solver_config.irls_gradient_rel_tol:raise RuntimeError("STAGE2_JOINT_COMPLETE_GRAD_MISMATCH")


def directional(context,cache,x,limit):
    analytic=joint_explicit_raw_global_jacobian(cache,x,limit);gen=torch.Generator(device=x.device).manual_seed(116)
    per=12+cache.model.body_pose_dim;masks=[("c2o",slice(0,3)),("c2t",slice(3,6)),("c4o",slice(6,9)),("c4t",slice(9,12)),
        ("body",slice(12,per)),("beta",slice(-cache.model.beta_dim,None)),("all",slice(None))]
    for label,sl in masks:
        direction=torch.zeros_like(x);direction[sl]=torch.randn(direction[sl].shape,generator=gen,device=x.device)
        _,fast=jvp(lambda value:joint_fast_raw_residual(cache,value,limit),(x,),(direction,));_,full=jvp(lambda value:full_raw(context,cache,value,limit),(x,),(direction,))
        print(f"[JVP/{label}] fast={joint_jacobian_error(analytic@direction,fast)} full={joint_jacobian_error(analytic@direction,full)}")
        reports=[]
        for eps in (3e-4,1e-3,3e-3,1e-2):
            finite=(joint_fast_raw_residual(cache,x+eps*direction,limit)-joint_fast_raw_residual(cache,x-eps*direction,limit))/(2*eps)
            reports.append((joint_jacobian_error(analytic@direction,finite),eps))
        print(f"[FINITE/{label}] best={min(reports,key=lambda z:z[0]['rel'])}")


def joint_jacobian_sync(x):
    if x.is_cuda:torch.cuda.synchronize(x.device)


def joint_jacobian_profile(context,cache,x,limit):
    full=lambda:jacfwd(lambda value:full_raw(context,cache,value,limit))(x)
    fast=lambda:jacfwd(lambda value:joint_fast_raw_residual(cache,value,limit))(x)
    combined=lambda:joint_explicit_raw_global_jacobian(cache,x,limit)
    separate=lambda:torch.cat((combined()[:,:-cache.model.beta_dim],joint_explicit_raw_beta_jacobian(cache,x,limit)),1)
    for _ in range(10):full();fast();combined();separate()
    def measure(fn):
        joint_jacobian_sync(x);start=time.perf_counter()
        for _ in range(20):fn()
        joint_jacobian_sync(x);return (time.perf_counter()-start)*50
    values={name:measure(fn) for name,fn in (("full",full),("fast",fast),("combined",combined),("separate",separate))}
    print(f"[PROFILE] cache={cache.setup_ms:.3f} {values} combined_vs_separate={values['separate']/values['combined']:.2f} full_speedup={values['full']/values['combined']:.2f} fast_speedup={values['fast']/values['combined']:.2f} shape={(sum(g.targets.numel() for g in cache.groups),x.numel())}")


def run_joint_analytic_validation(cli=None):
    cli=cli or parse_args();context=build_context(joint_jacobian_context_args(cli));archive=np.load(cli.joint_npz) if cli.joint_npz else None
    if archive is None or "x_stage2_pose_lm" not in archive:raise RuntimeError("debug116 requires --joint_npz with x_stage2_pose_lm")
    pose=torch.as_tensor(archive["x_stage2_pose_lm"]).to(context.beta_anchor);limit=context.objective_config.beta_abs_limit
    anchor=torch.atanh(context.beta_anchor/limit);shared=build_stage2_shared_geometry_cache(context);cache=build_stage2_joint_fast_cache(context,shared);pose_cache=build_stage2_pose_fast_cache(context,shared)
    test_points=joint_jacobian_points(archive,anchor,pose,context.beta_dim);reference=checks(context,cache,pose_cache,test_points,limit)
    surrogate_gradient(context,cache,reference,limit);directional(context,cache,reference,limit)
    if cli.profile:joint_jacobian_profile(context,cache,reference,limit)
    print("[PASS] Complete dynamic Stage 2 joint analytic Jacobian");return 0

def run_all_joint_jacobian_validations(args) -> None:
    run_joint_beta_validation(args)
    run_joint_analytic_validation(args)


def run_joint_production_shadow(args) -> int:
    """Concise serial-Joint versus FULL_BATCH shadow solver gate."""
    import contextlib, io
    from lib_pano.human_fitting.staged_lm.lm_solver import run_lm_optimization
    from lib_pano.human_fitting.staged_lm.stage2_joint_jacobian import build_stage2_joint_analytic_linearization
    from lib_pano.human_fitting.staged_lm.stage2_shared_cache import build_stage2_shared_geometry_cache
    from lib_pano.human_fitting.staged_lm.stages import build_stage1_definition, build_pose_stage_definition, build_joint_stage_definition, run_stage
    from script.benchmark.staged_lm.joint_batch_candidates import JointNEMode, build_joint_batch_cache
    from script.benchmark.staged_lm.joint_shadow_lm import compare_joint_lm_results, run_joint_lm_independent_batch
    context=build_context(joint_jacobian_context_args(args))
    with contextlib.redirect_stdout(io.StringIO()):
        s1=run_stage(build_stage1_definition(context),context);shared=build_stage2_shared_geometry_cache(context)
        pose,_=build_pose_stage_definition(context,s1,shared);pose_out=run_stage(pose,context)
        joint,_=build_joint_stage_definition(context,pose_out,shared)
        def serial_linearization(x):
            return build_stage2_joint_analytic_linearization(
                joint.analytic_cache, x, context.objective_config.beta_abs_limit,
                context.beta_anchor, context.body_pose_anchors,
                context.scaling.pose_anchor_scale, context.scaling.shape_anchor_scale,
                is_irls=context.is_irls, charbonnier_eps=context.charbonnier_eps,
                aggregation_spec=joint.aggregation_spec,
                rep_scale=context.scaling.rep_scale,
            )
        serial=run_lm_optimization("JOINT_SERIAL_REFERENCE",serial_linearization,
            joint.raw_metrics_fn,joint.initial_lm_parameters,joint.parameter_dimension,joint.lm_max_accepted_steps,
            context.solver_config.lm_max_trials,context.solver_config,True,joint.lm_true_objective_fn)
        batch=joint.analytic_cache.batch_cache
        if batch is None:
            batch=build_joint_batch_cache(joint.analytic_cache,context.scaling.pose_anchor_scale)
        candidate=run_joint_lm_independent_batch(joint,batch,context,mode=JointNEMode.FULL_BATCH_PRODUCTION_REDUCTION)
    comparison=compare_joint_lm_results(serial,candidate,context.objective_config.beta_abs_limit,context.beta_dim)
    ok=(comparison["status_match"] and comparison["accepted_match"] and comparison["rejected_match"]
        and comparison["trial_count_match"] and comparison["schur_match"]
        and comparison["weight_history_structure_match"]
        and comparison["final_parameter_max_abs"] <= (5e-5 if context.device.type=="cuda" else 1e-5)
        and comparison["physical_beta_max_abs"] <= (5e-6 if context.device.type=="cuda" else 2e-6)
        and comparison["objective_relative"] <= 1e-6 and comparison["metrics_max_abs"] <= 1e-3)
    print(f"[JOINT_PRODUCTION_SHADOW] reference_backend=analytic_stage2_joint_serial "
          f"production_backend=analytic_stage2_joint_batched {comparison}")
    if not ok:raise RuntimeError("JOINT_FULL_BATCH_SHADOW_GATE_FAILED")
    print("[PASS] Joint serial-reference vs FULL_BATCH shadow");return 0
