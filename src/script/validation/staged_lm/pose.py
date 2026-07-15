"""Stage 2 pose forward, Jacobian, and solver validations.

Mechanically consolidated from the historical per-check scripts.
"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path
import numpy as np
import torch
from torch.func import jvp
from smplx.lbs import batch_rigid_transform, blend_shapes, vertices2joints
from script.human_fitting.staged_lm.app import build_context
from lib_pano.human_fitting.staged_lm.config import STAGE1_PER_FRAME_DIM
from lib_pano.human_fitting.staged_lm.parameter_layout import unpack_frame_parameters
from lib_pano.human_fitting.staged_lm.stage2_pose_fast import build_stage2_pose_fast_cache, assemble_full_pose_rotations, fast_selected_vertices, project_cam2_erp
from lib_pano.human_fitting.staged_lm.stage2_shared_cache import build_stage2_shared_geometry_cache
from lib_pano.human_fitting.staged_lm.stages import build_stage1_definition, handoff_s1_to_pose, run_stage
from lib_pano.human_fitting.smplx_sequence import forward_smplx_tensor_batch
import dataclasses
from torch.func import jacfwd, jvp
from smplx.lbs import batch_rigid_transform
from lib_pano.human_fitting.staged_lm.stage2_pose_fast import Stage2PoseFastCache, assemble_full_pose_rotations, build_stage2_pose_fast_cache, fast_selected_vertices, project_cam2_erp
from lib_pano.human_fitting.staged_lm.stage2_pose_jacobian import POSE_PER_FRAME_DIM, assemble_full_pose_rotation_jacobians, compute_fk_with_jacobian, compute_pose_feature_with_jacobian, compute_selected_lbs_with_jacobian, compute_selected_pose_offsets_with_jacobian, fast_selected_vertices_with_jacobian, rotation_matrix_jacobian_additive, stage2_pose_anchor_residual_and_jacobian, stage2_pose_group_raw_residual_and_jacobian, stage2_pose_surrogate_residual_and_jacobian
from lib_dynamic_hvip.smpl.smpl_lib import _aa_to_rotmat
import statistics
from dataclasses import dataclass
from lib_pano.human_fitting.staged_lm.residuals import flatten_reprojection_groups
from script.validation.staged_lm.references.stage_backends import (
    _build_pose_warmup_definition,
    build_stage1_definition as reference_build_stage1_definition,
    run_stage as reference_run_stage,
)


def pose_forward_context_args(cli: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(
        frames=cli.frames, hmr_dir=cli.hmr_dir,
        sam3d_dense_dir=cli.sam3d_dense_dir, bbox_dir=cli.bbox_dir,
        image_width=cli.image_width, image_height=cli.image_height,
        output_dir="/tmp/debug112", reproj_objective="charbonnier_irls",
        stop_after_stage="stage1", run_lbfgs_oracle=False,
        jacobian_backend="autograd", stage1_eval_backend="full_forward",
        validate_stage1_fast_eval=False, validate_irls_first_order=False,
        validate_irls_each_accepted=False, profile_timing=False,
        progress_every_accepted=0,
    )


def error(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, float]:
    difference = actual - expected
    return {
        "max_abs": float(difference.abs().max()),
        "mean_abs": float(difference.abs().mean()),
        "rel_l2": float(difference.norm() / expected.norm().clamp_min(1e-16)),
    }


def pose_forward_full_selected(context, group, orient, translation, body_pose):
    frozen = context.frozen_params_by_frame[group.frame_position]
    left = getattr(frozen, f"{group.camera}_left_hand_pose").to(orient)
    right = getattr(frozen, f"{group.camera}_right_hand_pose").to(orient)
    scale = getattr(frozen, f"{group.camera}_scale").to(orient)
    vertices, _, _ = forward_smplx_tensor_batch(
        context.model, orient.reshape(1, 3), body_pose.reshape(1, -1),
        left, right, context.beta_anchor.to(orient).reshape(1, -1),
        translation.reshape(1, 3), scale.expand(1),
    )
    return vertices.squeeze(0).index_select(0, group.vertex_ids.to(orient.device)), vertices.squeeze(0)


def reference_intermediates(context, cache, group, orient, translation, body_pose):
    """Reproduce full LBS intermediates in debug code without modifying SMPL-X."""
    model = context.model
    expression = torch.zeros(1, model.num_expression_coeffs, dtype=orient.dtype, device=orient.device)
    shape_components = torch.cat((context.beta_anchor.to(orient).reshape(1, -1), expression), dim=1)
    shapedirs = torch.cat((model.shapedirs.to(orient), model.expr_dirs.to(orient)), dim=-1)
    v_shaped = model.v_template.to(orient) + blend_shapes(shape_components, shapedirs).squeeze(0)
    joints = vertices2joints(model.J_regressor.to(orient), v_shaped.unsqueeze(0)).squeeze(0)
    rotations = assemble_full_pose_rotations(
        orient, body_pose, group.left_hand_pose.to(orient), group.right_hand_pose.to(orient), cache.model,
    )
    identity = torch.eye(3, dtype=orient.dtype, device=orient.device)
    pose_feature = (rotations[1:] - identity).reshape(-1)
    pose_offsets = (pose_feature @ model.posedirs.to(orient)).reshape(-1, 3)
    v_posed = v_shaped + pose_offsets
    _, transforms = batch_rigid_transform(
        rotations.unsqueeze(0), joints.unsqueeze(0), model.parents.to(orient.device), dtype=orient.dtype,
    )
    transforms = transforms.squeeze(0)
    full_vertex_transforms = (
        model.lbs_weights.to(orient) @ transforms.reshape(cache.model.joint_count, 16)
    ).reshape(-1, 4, 4)
    homogeneous = torch.cat((v_posed, torch.ones(v_posed.shape[0], 1, dtype=orient.dtype, device=orient.device)), 1)
    skinned = (full_vertex_transforms @ homogeneous.unsqueeze(-1))[:, :3, 0]
    ids = group.vertex_ids.to(orient.device)
    scale = group.scale.to(orient)
    return {
        "v_shaped": v_shaped,
        "joints": joints,
        "rotation_matrices": rotations,
        "pose_feature": pose_feature,
        "selected_pose_offsets": pose_offsets.index_select(0, ids),
        "fk_transforms": transforms,
        "selected_v_posed": v_posed.index_select(0, ids),
        "selected_skinned_vertices": skinned.index_select(0, ids),
        "root_center": joints[0] * scale,
        "final_selected_vertices": skinned.index_select(0, ids) * scale + translation.reshape(1, 3),
    }


def pose_forward_parameter_points(initial: torch.Tensor, body_pose_dim: int) -> list[tuple[str, torch.Tensor]]:
    generator = torch.Generator(device=initial.device).manual_seed(112)
    points = [("handoff_initial", initial.clone())]
    per_frame = STAGE1_PER_FRAME_DIM + body_pose_dim
    for name, magnitude in (("body_random", 0.02), ("body_small", 1e-3),
                            ("body_medium", 1e-2), ("body_large", 5e-2)):
        value = initial.clone()
        for frame in range(initial.numel() // per_frame):
            pose_slice = slice(frame * per_frame + STAGE1_PER_FRAME_DIM, (frame + 1) * per_frame)
            value[pose_slice] += torch.randn(body_pose_dim, generator=generator) * magnitude
        points.append((name, value))
    orient = initial.clone(); orient[0:3] += torch.tensor([0.01, -0.02, 0.015]); points.append(("orient", orient))
    translation = initial.clone(); translation[3:6] += torch.tensor([0.02, -0.01, 0.03]); points.append(("translation", translation))
    combined = points[-2][1].clone(); combined[3:6] += torch.tensor([-0.01, 0.02, 0.01]); combined[12:12 + body_pose_dim] += 0.01
    points.append(("combined", combined))
    return points


def extract_group_parameters(parameters, group, body_pose_dim):
    per_frame = STAGE1_PER_FRAME_DIM + body_pose_dim
    frame = unpack_frame_parameters(parameters, group.frame_position, per_frame, True)
    orient = frame.cam2_orient if group.camera == "cam2" else frame.cam4_orient
    translation = frame.cam2_transl if group.camera == "cam2" else frame.cam4_transl
    return orient.reshape(3), translation.reshape(3), frame.body_pose.reshape(-1)


def check_points(context, cache, points, check_intermediates):
    for label, parameters in points:
        for group in cache.groups:
            orient, translation, body_pose = extract_group_parameters(parameters, group, cache.body_pose_dim)
            full, _ = pose_forward_full_selected(context, group, orient, translation, body_pose)
            fast_result = fast_selected_vertices(
                cache, group, orient, translation, body_pose,
                return_intermediates=check_intermediates,
            )
            fast, intermediates = fast_result if check_intermediates else (fast_result, None)
            vertex_error = error(fast, full)
            if vertex_error["max_abs"] > 1e-5 or vertex_error["rel_l2"] > 1e-6:
                raise RuntimeError(f"STAGE2_POSE_FAST_VERTEX_MISMATCH: {label}/{group.camera} {vertex_error}")
            full_uv = context.fitter._project_cam2(full, context.image_width, context.image_height)
            fast_uv = project_cam2_erp(fast, context.image_width, context.image_height)
            uv_error = error(fast_uv, full_uv)
            torch.testing.assert_close(fast_uv, full_uv, rtol=1e-6, atol=1e-3)
            full_residual = full_uv - group.targets.to(full_uv)
            fast_residual = fast_uv - group.targets.to(fast_uv)
            torch.testing.assert_close(fast_residual, full_residual, rtol=1e-6, atol=1e-3)
            fixed_rotation = torch.tensor(
                [1, 0, 0, 0, 0, 1, 0, -1, 0], dtype=fast.dtype, device=fast.device,
            ).reshape(3, 3)
            camera_points = fast @ fixed_rotation.T
            horizontal = torch.sqrt(camera_points[:, 0] ** 2 + camera_points[:, 2] ** 2)
            seam_distance = torch.minimum(fast_uv[:, 0], context.image_width - fast_uv[:, 0]).min()
            latitude_ratio = (-camera_points[:, 1] / torch.norm(camera_points, dim=1).clamp_min(1e-6)).abs()
            intermediate_text = ""
            if intermediates is not None:
                reference = reference_intermediates(
                    context, cache, group, orient, translation, body_pose,
                )
                intermediate_errors = {
                    name: error(intermediates[name], expected)
                    for name, expected in reference.items()
                }
                for name, intermediate_error in intermediate_errors.items():
                    if intermediate_error["max_abs"] > 1e-5:
                        raise RuntimeError(
                            f"STAGE2_POSE_FAST_INTERMEDIATE_MISMATCH: {name} {intermediate_error}"
                        )
                worst_name = max(intermediate_errors, key=lambda name: intermediate_errors[name]["max_abs"])
                intermediate_text = (
                    f" worst_intermediate={worst_name}:"
                    f"{intermediate_errors[worst_name]['max_abs']:.3e}"
                    f" transform_shape={tuple(intermediates['selected_transforms'].shape)}"
                )
            print(
                f"[{label}] frame={group.frame_index} camera={group.camera} vertices={vertex_error} "
                f"uv={uv_error} seam_distance={float(seam_distance):.3f} "
                f"min_horizontal={float(horizontal.min()):.3e} "
                f"latitude_clamped={bool((latitude_ratio >= 1 - 1e-6).any())}{intermediate_text}"
            )


def check_jvp(context, cache, parameters):
    group = cache.groups[0]
    orient, translation, body_pose = extract_group_parameters(parameters, group, cache.body_pose_dim)
    generator = torch.Generator(device=parameters.device).manual_seed(113)
    directions = {
        "body_pose": (torch.zeros_like(orient), torch.zeros_like(translation), torch.randn(body_pose.shape, generator=generator)),
        "global_orient": (torch.randn(orient.shape, generator=generator), torch.zeros_like(translation), torch.zeros_like(body_pose)),
        "translation": (torch.zeros_like(orient), torch.randn(translation.shape, generator=generator), torch.zeros_like(body_pose)),
    }
    def full_fn(o, t, b): return pose_forward_full_selected(context, group, o, t, b)[0]
    def fast_fn(o, t, b): return fast_selected_vertices(cache, group, o, t, b)
    for name, direction in directions.items():
        _, full_direction = jvp(full_fn, (orient, translation, body_pose), direction)
        _, fast_direction = jvp(fast_fn, (orient, translation, body_pose), direction)
        direction_error = error(fast_direction, full_direction)
        print(f"[JVP {name}] {direction_error}")
        if direction_error["rel_l2"] > 5e-4:
            raise RuntimeError(f"STAGE2_POSE_FAST_JVP_MISMATCH: {name}")


def pose_forward_synchronize(reference):
    if reference.is_cuda: torch.cuda.synchronize(reference.device)


def pose_forward_profile(context, cache, parameters):
    group = cache.groups[0]
    orient, translation, body_pose = extract_group_parameters(parameters, group, cache.body_pose_dim)
    def full_forward_call(): return pose_forward_full_selected(context, group, orient, translation, body_pose)[1]
    full_vertices = full_forward_call()
    def full_projection_call():
        projected = context.fitter._project_cam2(full_vertices, context.image_width, context.image_height)
        return projected.index_select(0, group.vertex_ids.to(projected.device))
    def fast_forward_call(): return fast_selected_vertices(cache, group, orient, translation, body_pose)
    fast_vertices = fast_forward_call()
    def fast_projection_call(): return project_cam2_erp(fast_vertices, context.image_width, context.image_height)
    def full_call(): return full_projection_call() if full_forward_call() is not None else None
    def fast_call(): return project_cam2_erp(fast_forward_call(), context.image_width, context.image_height)
    for _ in range(10): full_call(); fast_call()
    repeats = 50
    def measure(function):
        pose_forward_synchronize(orient); start = time.perf_counter()
        for _ in range(repeats): function()
        pose_forward_synchronize(orient); return (time.perf_counter() - start) * 1000 / repeats
    full_forward_ms = measure(full_forward_call)
    full_projection_ms = measure(full_projection_call)
    fast_forward_ms = measure(fast_forward_call)
    fast_projection_ms = measure(fast_projection_call)
    full_ms = measure(full_call)
    fast_ms = measure(fast_call)
    full_peak_memory = fast_peak_memory = 0
    if orient.is_cuda:
        torch.cuda.reset_peak_memory_stats(orient.device); full_call(); pose_forward_synchronize(orient)
        full_peak_memory = torch.cuda.max_memory_allocated(orient.device)
        torch.cuda.reset_peak_memory_stats(orient.device); fast_call(); pose_forward_synchronize(orient)
        fast_peak_memory = torch.cuda.max_memory_allocated(orient.device)
    print(
        f"[PROFILE] cache_setup_ms={cache.setup_ms:.3f} full_total_ms/call={full_ms:.3f} "
        f"fast_total_ms/call={fast_ms:.3f} speedup={full_ms/fast_ms:.2f}x "
        f"full_forward_ms/call={full_forward_ms:.3f} "
        f"full_projection_selection_ms/call={full_projection_ms:.3f} "
        f"fast_forward_ms/call={fast_forward_ms:.3f} "
        f"fast_projection_ms/call={fast_projection_ms:.3f} "
        f"selected_points={group.vertex_ids.numel()} full_vertices={cache.model.full_vertex_count} "
        f"joints={cache.model.joint_count} pose_feature_dim={cache.model.pose_feature_dim} "
        f"full_peak_cuda_bytes={full_peak_memory} fast_peak_cuda_bytes={fast_peak_memory}"
    )


def run_pose_forward_validation(cli=None) -> int:
    cli = cli or parse_args()
    context = build_context(pose_forward_context_args(cli))
    stage1_definition = build_stage1_definition(context)
    stage1_outcome = run_stage(stage1_definition, context)
    initial = handoff_s1_to_pose(
        stage1_outcome.lm_result.parameters, context.body_pose_anchors,
        context.frame_count, context.body_pose_dim,
    )
    cache = build_stage2_pose_fast_cache(
        context, build_stage2_shared_geometry_cache(context),
    )
    points = pose_forward_parameter_points(initial, context.body_pose_dim)
    if cli.stage2_pose_npz:
        with np.load(cli.stage2_pose_npz) as archive:
            for key, label in (("x_stage2_pose_lm", "npz_lm"), ("x_stage2_pose_lbfgs", "npz_lbfgs")):
                if key in archive:
                    points.append((label, torch.as_tensor(archive[key]).to(initial)))
    check_points(context, cache, points, cli.check_intermediates)
    check_jvp(context, cache, initial)
    if cli.profile:
        pose_forward_profile(context, cache, initial)
    print("[PASS] Stage 2 selected-vertex forward and directional derivatives")
    return 0


def pose_jacobian_context_args(cli: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(
        frames=cli.frames, hmr_dir=cli.hmr_dir, sam3d_dense_dir=cli.sam3d_dense_dir,
        bbox_dir=cli.bbox_dir, image_width=cli.image_width, image_height=cli.image_height,
        output_dir="/tmp/debug113", reproj_objective="charbonnier_irls",
        stop_after_stage="stage1", run_lbfgs_oracle=False, jacobian_backend="autograd",
        stage1_eval_backend="full_forward", validate_stage1_fast_eval=False,
        validate_irls_first_order=False, validate_irls_each_accepted=False,
        profile_timing=False, progress_every_accepted=0,
    )


def errors(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, float]:
    delta = actual - expected
    return {
        "rel_frob": float(delta.norm() / expected.norm().clamp_min(1e-16)),
        "max_abs": float(delta.abs().max()),
        "mean_abs": float(delta.abs().mean()),
    }


def require(label: str, actual: torch.Tensor, expected: torch.Tensor, rel: float, absolute: float) -> None:
    report = errors(actual, expected)
    print(f"[{label}] {report}")
    if not torch.isfinite(actual).all() or report["rel_frob"] > rel or report["max_abs"] > absolute:
        raise RuntimeError(f"STAGE2_POSE_ANALYTIC_{label}_MISMATCH: {report}")


def local_parameters(parameters: torch.Tensor, frame_position: int) -> torch.Tensor:
    return parameters[frame_position * POSE_PER_FRAME_DIM:(frame_position + 1) * POSE_PER_FRAME_DIM]


def camera_values(local: torch.Tensor, camera: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if camera == "cam2":
        return local[:3], local[3:6], local[12:75]
    return local[6:9], local[9:12], local[12:75]


def one_group_cache(cache: Stage2PoseFastCache, group) -> Stage2PoseFastCache:
    return Stage2PoseFastCache(
        cache.model, [dataclasses.replace(group, frame_position=0)], 1,
        cache.body_pose_dim, cache.image_width, cache.image_height, cache.setup_ms,
    )


def pose_jacobian_full_selected(context, group, orient, translation, body_pose):
    frozen = context.frozen_params_by_frame[group.frame_position]
    left = getattr(frozen, f"{group.camera}_left_hand_pose").to(orient)
    right = getattr(frozen, f"{group.camera}_right_hand_pose").to(orient)
    scale = getattr(frozen, f"{group.camera}_scale").to(orient)
    vertices, _, _ = forward_smplx_tensor_batch(
        context.model, orient.reshape(1, 3), body_pose.reshape(1, -1), left, right,
        context.beta_anchor.to(orient).reshape(1, -1), translation.reshape(1, 3), scale.expand(1),
    )
    return vertices[0].index_select(0, group.vertex_ids.to(orient.device))


def fast_group_residual(cache, group, local):
    single = one_group_cache(cache, group)
    residual, _ = stage2_pose_group_raw_residual_and_jacobian(single, single.groups[0], local)
    return residual


def full_group_residual(context, group, local):
    orient, translation, body = camera_values(local, group.camera)
    uv = context.fitter._project_cam2(
        pose_jacobian_full_selected(context, group, orient, translation, body),
        context.image_width, context.image_height,
    )
    return (uv - group.targets.to(uv)).reshape(-1)


def pose_jacobian_parameter_points(initial: torch.Tensor) -> list[tuple[str, torch.Tensor]]:
    generator = torch.Generator(device=initial.device).manual_seed(113)
    points = [("initial", initial.clone())]
    for name, magnitude in (("pose_1e-3", 1e-3), ("pose_1e-2", 1e-2), ("pose_5e-2", 5e-2)):
        value = initial.clone()
        for frame in range(initial.numel() // 75):
            value[frame*75+12:frame*75+75] += torch.randn(63, generator=generator, device=initial.device) * magnitude
        points.append((name, value))
    orient = initial.clone(); orient[:3] += initial.new_tensor([.01, -.02, .015]); points.append(("orient", orient))
    translation = initial.clone(); translation[3:6] += initial.new_tensor([.02, -.01, .03]); points.append(("translation", translation))
    combined = points[-2][1].clone(); combined[3:6] += initial.new_tensor([-.01, .02, .01]); combined[12:75] += .01
    points.append(("combined", combined))
    return points


def module_checks(cache: Stage2PoseFastCache, parameters: torch.Tensor) -> None:
    group = cache.groups[0]
    local = local_parameters(parameters, group.frame_position)
    orient, translation, body = camera_values(local, group.camera)
    q = torch.cat((orient, body))
    rotations, d_rotations = assemble_full_pose_rotation_jacobians(
        orient, body, group.left_hand_pose.to(orient), group.right_hand_pose.to(orient), cache.model,
    )
    require("ROTATION_J", rotation_matrix_jacobian_additive(orient),
            jacfwd(lambda x: _aa_to_rotmat(x[None])[0])(orient), 5e-5, 5e-5)
    rotation_oracle = jacfwd(lambda value: assemble_full_pose_rotations(
        value[:3], value[3:], group.left_hand_pose.to(value), group.right_hand_pose.to(value), cache.model,
    ))(q)
    require("ASSEMBLED_ROTATION_J", d_rotations, rotation_oracle, 5e-5, 5e-5)
    feature, d_feature = compute_pose_feature_with_jacobian(rotations, d_rotations)
    feature_fn = lambda value: (assemble_full_pose_rotations(
        value[:3], value[3:], group.left_hand_pose.to(value), group.right_hand_pose.to(value), cache.model,
    )[1:] - torch.eye(3, dtype=value.dtype, device=value.device)).reshape(-1)
    require("POSE_FEATURE_J", d_feature, jacfwd(feature_fn)(q), 5e-5, 5e-5)
    offsets, d_offsets, _, _ = compute_selected_pose_offsets_with_jacobian(
        rotations, d_rotations, group.selected_posedirs.to(q),
    )
    require("POSE_OFFSET_J", d_offsets, jacfwd(lambda value: torch.einsum(
        "p,pnc->nc", feature_fn(value), group.selected_posedirs.to(value),
    ))(q), 5e-5, 5e-5)
    fk = compute_fk_with_jacobian(
        rotations, d_rotations, cache.model.joints.to(q), cache.model.parents.to(q.device),
    )
    def fk_fn(value):
        rots = assemble_full_pose_rotations(
            value[:3], value[3:], group.left_hand_pose.to(value), group.right_hand_pose.to(value), cache.model,
        )
        posed, relative = batch_rigid_transform(
            rots.unsqueeze(0), cache.model.joints.to(value).unsqueeze(0),
            cache.model.parents.to(value.device), dtype=value.dtype,
        )
        return posed[0], relative[0]
    def global_fk_fn(value):
        rots = assemble_full_pose_rotations(
            value[:3], value[3:], group.left_hand_pose.to(value),
            group.right_hand_pose.to(value), cache.model,
        )
        joints = cache.model.joints.to(value)
        relative_joints = joints.clone()
        relative_joints[1:] -= joints.index_select(0, cache.model.parents[1:].to(value.device))
        local_transforms = torch.zeros(
            cache.model.joint_count, 4, 4, dtype=value.dtype, device=value.device,
        )
        local_transforms[:, :3, :3] = rots
        local_transforms[:, :3, 3] = relative_joints
        local_transforms[:, 3, 3] = 1
        transforms = [local_transforms[0]]
        for joint in range(1, cache.model.joint_count):
            transforms.append(
                transforms[int(cache.model.parents[joint])] @ local_transforms[joint]
            )
        return torch.stack(transforms)
    posed_oracle, relative_oracle = fk_fn(q)
    require("FK_POSED_JOINTS", fk.posed_joints, posed_oracle, 1e-6, 1e-6)
    require("FK_RELATIVE_FORWARD", fk.relative_transforms, relative_oracle, 1e-6, 1e-6)
    require("FK_GLOBAL_J", fk.d_global_transforms, jacfwd(global_fk_fn)(q), 5e-5, 5e-5)
    require("FK_POSED_JOINT_J", fk.d_global_transforms[:, :3, 3],
            jacfwd(lambda value: fk_fn(value)[0])(q), 5e-5, 5e-5)
    require("FK_RELATIVE_J", fk.d_relative_transforms,
            jacfwd(lambda value: fk_fn(value)[1])(q), 5e-5, 5e-5)
    vertices, d_vertices, intermediates = fast_selected_vertices_with_jacobian(
        cache, group, orient, translation, body, return_intermediates=True,
    )
    vertex_oracle = jacfwd(lambda value: fast_selected_vertices(
        cache, group, value[:3], translation, value[3:],
    ))(q)
    require("SELECTED_VERTEX_J", d_vertices, vertex_oracle, 1e-4, 1e-4)
    if not torch.isfinite(torch.cat(tuple(value.reshape(-1) for value in intermediates.values()))).all():
        raise RuntimeError("STAGE2_POSE_ANALYTIC_INTERMEDIATE_NONFINITE")
    print(f"[MODULES] vertices={tuple(vertices.shape)} pose_feature={tuple(feature.shape)}")


def report_blocks(actual: torch.Tensor, oracle: torch.Tensor, label: str) -> None:
    blocks = {
        "cam2_orient": slice(0, 3), "cam2_translation": slice(3, 6),
        "cam4_orient": slice(6, 9), "cam4_translation": slice(9, 12),
        "body_pose": slice(12, 75),
    }
    for name, columns in blocks.items():
        print(f"[{label}/{name}] {errors(actual[:, columns], oracle[:, columns])}")
    joint_reports = []
    for joint in range(21):
        columns = slice(12 + joint * 3, 15 + joint * 3)
        joint_reports.append((errors(actual[:, columns], oracle[:, columns]), joint))
    worst, joint = max(joint_reports, key=lambda item: item[0]["rel_frob"])
    print(f"[{label}/WORST_BODY_JOINT] joint={joint + 1} columns={12+joint*3}:{15+joint*3} {worst}")


def residual_checks(context, cache, points) -> None:
    full_labels = {"initial", "npz_lm", "npz_lbfgs"}
    for label, parameters in points:
        for group in cache.groups:
            local = local_parameters(parameters, group.frame_position)
            single = one_group_cache(cache, group)
            analytic_residual, analytic_jacobian = stage2_pose_group_raw_residual_and_jacobian(
                single, single.groups[0], local,
            )
            fast_oracle = jacfwd(lambda value: fast_group_residual(cache, group, value))(local)
            require(f"FAST_RESIDUAL_J/{label}/{group.frame_index}/{group.camera}",
                    analytic_jacobian, fast_oracle, 5e-4, 1e-2)
            report_blocks(analytic_jacobian, fast_oracle, f"{label}/{group.camera}")
            if label in full_labels:
                full_oracle = jacfwd(lambda value: full_group_residual(context, group, value))(local)
                require(f"FULL_RESIDUAL_J/{label}/{group.frame_index}/{group.camera}",
                        analytic_jacobian, full_oracle, 5e-4, 1e-2)
                torch.testing.assert_close(analytic_residual, full_group_residual(context, group, local), rtol=1e-6, atol=1e-3)


def directional_and_difference_checks(context, cache, parameters) -> None:
    group = cache.groups[0]
    local = local_parameters(parameters, group.frame_position)
    single = one_group_cache(cache, group)
    _, analytic = stage2_pose_group_raw_residual_and_jacobian(single, single.groups[0], local)
    generator = torch.Generator(device=local.device).manual_seed(114)
    masks = {
        "cam2_orient": slice(0, 3), "cam2_translation": slice(3, 6),
        "cam4_orient": slice(6, 9), "cam4_translation": slice(9, 12),
        "body_pose": slice(12, 75), "all": slice(0, 75),
    }
    for label, active in masks.items():
        direction = torch.zeros_like(local)
        direction[active] = torch.randn(direction[active].shape, generator=generator, device=local.device)
        _, fast_jvp = jvp(lambda value: fast_group_residual(cache, group, value), (local,), (direction,))
        report = errors(analytic @ direction, fast_jvp)
        print(f"[JVP/{label}] {report}")
        if fast_jvp.norm() > 0 and report["rel_frob"] > (1e-5 if "translation" in label else 5e-4):
            raise RuntimeError(f"STAGE2_POSE_ANALYTIC_JVP_MISMATCH: {label}")
    full_direction = torch.randn(local.shape, generator=generator, device=local.device)
    _, full_jvp = jvp(lambda value: full_group_residual(context, group, value), (local,), (full_direction,))
    require("FULL_DIRECTIONAL_JVP", analytic @ full_direction, full_jvp, 5e-4, 1e-2)
    finite_reports = []
    for epsilon in (3e-4, 1e-3, 3e-3, 1e-2):
        columns = []
        for column in range(local.numel()):
            direction = torch.zeros_like(local); direction[column] = epsilon
            columns.append((fast_group_residual(cache, group, local + direction)
                            - fast_group_residual(cache, group, local - direction)) / (2 * epsilon))
        finite = torch.stack(columns, 1)
        report = errors(analytic, finite)
        finite_reports.append((report["rel_frob"], epsilon, report))
        print(f"[CENTRAL_DIFFERENCE eps={epsilon:g}] {report}")
    _, best_epsilon, best_report = min(finite_reports)
    print(f"[CENTRAL_DIFFERENCE_BEST] eps={best_epsilon:g} {best_report}")


def surrogate_checks(cache, parameters, anchors) -> None:
    # One frame keeps the debug-only dense oracle affordable and makes frozen
    # point weights explicit in the jacfwd closure.
    groups = [dataclasses.replace(group, frame_position=0) for group in cache.groups[:2]]
    single = Stage2PoseFastCache(cache.model, groups, 1, 63, cache.image_width, cache.image_height, cache.setup_ms)
    local = parameters[:75]
    for mode in ("charbonnier_irls", "raw_l2"):
        result = stage2_pose_surrogate_residual_and_jacobian(
            single, local, anchors[:1], .1, objective_mode=mode,
            charbonnier_eps=1., rep_scale=.02,
        )
        raw_rows = sum(group.targets.numel() for group in groups)
        frozen_scales = result.residual[:raw_rows] / torch.cat([
            fast_group_residual(single, group, local) for group in groups
        ])
        def frozen_fn(value):
            raw = torch.cat([fast_group_residual(single, group, value) for group in groups])
            anchor = .1 * (value[12:75] - anchors[0].to(value))
            return torch.cat((raw * frozen_scales.detach(), anchor))
        oracle = jacfwd(frozen_fn)(local)
        require(f"SURROGATE_{mode}", result.jacobian, oracle, 5e-4, 1e-2)


def pose_jacobian_synchronize(tensor):
    if tensor.is_cuda:
        torch.cuda.synchronize(tensor.device)


def pose_jacobian_profile(context, cache, parameters) -> None:
    cases = []
    for group in cache.groups[:2]:
        local = local_parameters(parameters, group.frame_position)
        single = one_group_cache(cache, group)
        cases.append((
            group.camera, local, group.targets.numel(),
            lambda group=group, local=local: jacfwd(
                lambda value: full_group_residual(context, group, value)
            )(local),
            lambda group=group, local=local: jacfwd(
                lambda value: fast_group_residual(cache, group, value)
            )(local),
            lambda single=single, local=local: stage2_pose_group_raw_residual_and_jacobian(
                single, single.groups[0], local,
            )[1],
        ))
    frame_groups = cache.groups[:2]
    frame_local = parameters[:75]
    cases.append((
        "complete_frame", frame_local, sum(group.targets.numel() for group in frame_groups),
        lambda: jacfwd(lambda value: torch.cat([
            full_group_residual(context, group, value) for group in frame_groups
        ]))(frame_local),
        lambda: jacfwd(lambda value: torch.cat([
            fast_group_residual(cache, group, value) for group in frame_groups
        ]))(frame_local),
        lambda: torch.cat([
            stage2_pose_group_raw_residual_and_jacobian(
                one_group_cache(cache, group), one_group_cache(cache, group).groups[0], frame_local,
            )[1] for group in frame_groups
        ]),
    ))
    for label, local, row_count, full_fn, fast_fn, analytic_fn in cases:
        for _ in range(10):
            full_fn(); fast_fn(); analytic_fn()
        def measure(fn):
            pose_jacobian_synchronize(local); start = time.perf_counter()
            for _ in range(20): fn()
            pose_jacobian_synchronize(local); return (time.perf_counter() - start) * 50
        full_ms, fast_ms, analytic_ms = measure(full_fn), measure(fast_fn), measure(analytic_fn)
        full_peak = fast_peak = analytic_peak = 0
        if local.is_cuda:
            peaks = []
            for function in (full_fn, fast_fn, analytic_fn):
                torch.cuda.reset_peak_memory_stats(local.device)
                function(); pose_jacobian_synchronize(local)
                peaks.append(torch.cuda.max_memory_allocated(local.device))
            full_peak, fast_peak, analytic_peak = peaks
        print(
            f"[PROFILE/{label}] cache_setup_ms={cache.setup_ms:.3f} "
            f"full_jacfwd_ms={full_ms:.3f} fast_jacfwd_ms={fast_ms:.3f} "
            f"analytic_ms={analytic_ms:.3f} speedup_full={full_ms/analytic_ms:.2f}x "
            f"speedup_fast={fast_ms/analytic_ms:.2f}x jacobian_shape={(row_count, 75)} "
            f"peak_cuda_bytes=full:{full_peak},fast:{fast_peak},analytic:{analytic_peak}"
        )


def run_pose_jacobian_validation(cli=None) -> int:
    cli = cli or parse_args()
    context = build_context(pose_jacobian_context_args(cli))
    stage1 = run_stage(build_stage1_definition(context), context)
    initial = handoff_s1_to_pose(
        stage1.lm_result.parameters, context.body_pose_anchors,
        context.frame_count, context.body_pose_dim,
    )
    cache = build_stage2_pose_fast_cache(
        context, build_stage2_shared_geometry_cache(context),
    )
    points = pose_jacobian_parameter_points(initial)
    if cli.stage2_pose_npz:
        with np.load(cli.stage2_pose_npz) as archive:
            for key, label in (("x_stage2_pose_lm", "npz_lm"), ("x_stage2_pose_lbfgs", "npz_lbfgs")):
                if key in archive:
                    points.append((label, torch.as_tensor(archive[key]).to(initial)))
    module_checks(cache, initial)
    residual_checks(context, cache, points)
    surrogate_checks(cache, initial, context.body_pose_anchors)
    directional_and_difference_checks(context, cache, initial)
    if cli.profile:
        pose_jacobian_profile(context, cache, initial)
    print("[PASS] Stage 2 pose explicit analytic Jacobian diagnostics")
    return 0


@dataclass
class ModeResult:
    label: str
    outcome: object
    raw_residual: torch.Tensor


def pose_solver_context_args(cli: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(
        frames=cli.frames, hmr_dir=cli.hmr_dir, sam3d_dense_dir=cli.sam3d_dense_dir,
        bbox_dir=cli.bbox_dir, image_width=cli.image_width, image_height=cli.image_height,
        output_dir="/tmp/debug114", reproj_objective="charbonnier_irls",
        stop_after_stage="stage2_pose", run_lbfgs_oracle=False,
        jacobian_backend="analytic_stage1", stage1_eval_backend="analytic",
        validate_stage1_fast_eval=False, stage2_pose_linearization_backend="autograd",
        stage2_pose_eval_backend="full_forward", validate_stage2_pose_fast_eval=False,
        validate_irls_first_order=False, validate_irls_each_accepted=False,
        profile_timing=cli.profile, progress_every_accepted=0,
    )


def metric_close(actual: float, expected: float) -> bool:
    absolute = abs(actual - expected)
    relative = absolute / max(abs(expected), 1e-16)
    return absolute <= 1e-3 or relative <= 1e-5


def run_mode(context, stage1, label: str, linearization: str, evaluation: str) -> ModeResult:
    context.args.stage2_pose_linearization_backend = linearization
    context.args.stage2_pose_eval_backend = evaluation
    context.args.validate_stage2_pose_fast_eval = evaluation == "fast"
    definition, _ = _build_pose_warmup_definition(context, stage1)
    outcome = reference_run_stage(definition, context)
    raw = flatten_reprojection_groups(
        definition.raw_groups_fn(outcome.lm_result.parameters),
    ).detach()
    return ModeResult(label, outcome, raw)


def compare(reference: ModeResult, candidate: ModeResult) -> None:
    ref, got = reference.outcome.lm_result, candidate.outcome.lm_result
    objective_relative = abs(got.objective - ref.objective) / max(abs(ref.objective), 1e-16)
    if objective_relative > 1e-5:
        raise RuntimeError(f"STAGE2_POSE_SOLVER_OBJECTIVE_MISMATCH: {candidate.label}")
    torch.testing.assert_close(candidate.raw_residual, reference.raw_residual, rtol=1e-6, atol=1e-3)
    torch.testing.assert_close(got.parameters, ref.parameters, rtol=1e-4, atol=1e-4)
    if not all(metric_close(got.rms_metrics[name], ref.rms_metrics[name])
               for name in ("rms", "median", "p95")):
        raise RuntimeError(f"STAGE2_POSE_SOLVER_METRICS_MISMATCH: {candidate.label}")
    print(
        f"[REGRESSION {reference.label}->{candidate.label}] obj_rel={objective_relative:.3e} "
        f"raw_max={float((candidate.raw_residual-reference.raw_residual).abs().max()):.3e} "
        f"param_max={float((got.parameters-ref.parameters).abs().max()):.3e} "
        f"acc/rej={got.counters.accepted_steps}/{got.counters.rejected_trials} "
        f"status={got.status} lambda={got.final_lambda:.6e}"
    )


def profile_summary(results: list[ModeResult]) -> None:
    for result in results:
        lm = result.outcome.lm_result
        timing = lm.profile_timing or {}
        print(
            f"[TIMING {result.label}] wall={lm.wall_time_ms:.3f} "
            f"solver_jacobian={timing.get('solver_jacobian_ms', 0.):.3f} "
            f"objective_eval={timing.get('objective_eval_ms', 0.):.3f} "
            f"final_metrics={timing.get('final_metrics_ms', 0.):.3f} "
            f"linear_solve={timing.get('linear_solve_ms', 0.):.3f} "
            f"normal_eq={lm.counters.normal_equation_count} "
            f"fast_residual={timing.get('stage2_pose_residual_ms', 0.):.3f} "
            f"direct_hg={timing.get('stage2_pose_normal_equations_ms', 0.):.3f} "
            f"fast_objective={timing.get('stage2_pose_objective_ms', 0.):.3f} "
            f"fast_metrics={timing.get('stage2_pose_metrics_ms', 0.):.3f}"
        )


def run_pose_solver_regression(cli=None) -> int:
    cli = cli or parse_args()
    context = build_context(pose_solver_context_args(cli))
    stage1 = reference_run_stage(reference_build_stage1_definition(context), context)
    modes = (
        ("A_AUTOGRAD_FULL", "autograd", "full_forward"),
        ("B_ANALYTIC_FULL", "analytic", "full_forward"),
        ("C_ANALYTIC_FAST", "analytic", "fast"),
    )
    repeats = 3 if cli.profile else 1
    runs: dict[str, list[ModeResult]] = {label: [] for label, _, _ in modes}
    for _ in range(repeats):
        for label, linearization, evaluation in modes:
            runs[label].append(run_mode(context, stage1, label, linearization, evaluation))
    representatives = [values[0] for values in runs.values()]
    compare(representatives[0], representatives[1])
    compare(representatives[0], representatives[2])
    profile_summary(representatives)
    if cli.profile:
        for label, values in runs.items():
            def median_timing(key: str) -> float:
                return statistics.median(
                    (value.outcome.lm_result.profile_timing or {}).get(key, 0.)
                    for value in values
                )
            print(
                f"[MEDIAN {label}] "
                f"wall_ms={statistics.median(v.outcome.lm_result.wall_time_ms for v in values):.3f} "
                f"solver_jacobian_ms={median_timing('solver_jacobian_ms'):.3f} "
                f"objective_eval_ms={median_timing('objective_eval_ms'):.3f} "
                f"final_metrics_ms={median_timing('final_metrics_ms'):.3f} "
                f"linear_solve_ms={median_timing('linear_solve_ms'):.3f}"
            )
    print("[PASS] Stage 2 pose A/B/C solver regression")
    return 0

def run_all_pose_validations(args) -> None:
    run_pose_forward_validation(args)
    run_pose_jacobian_validation(args)
    run_pose_solver_regression(args)
