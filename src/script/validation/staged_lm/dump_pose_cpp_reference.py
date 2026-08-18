"""Dump the frame-513 Stage 2 Pose value path for the standalone C++ port."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from smplx.lbs import batch_rigid_transform

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from lib_pano.human_fitting.staged_lm.geometry.erp import project_unified_erp
from lib_pano.human_fitting.staged_lm.parameter_layout import unpack_frame_parameters
from lib_pano.human_fitting.staged_lm.pose.forward import (
    assemble_full_pose_rotations,
    compute_selected_lbs,
    compute_selected_pose_offsets,
    fast_raw_groups,
    fast_selected_vertices,
)
from lib_pano.human_fitting.staged_lm.pose.derivatives import (
    assemble_full_pose_rotation_jacobians,
    compute_fk_with_jacobian,
    compute_selected_lbs_with_jacobian,
    compute_selected_pose_offsets_with_jacobian,
    stage2_pose_group_raw_residual_and_jacobian,
)
from lib_pano.human_fitting.staged_lm.irls import (
    build_primary_aggregation_spec,
    compute_irls_point_weights,
)
from lib_pano.human_fitting.staged_lm.pose.batched import (
    build_stage2_pose_batch_cache,
    gather_pose_batch_parameters,
    pad_stage2_pose_row_scales,
    stage2_pose_batched_accumulate_normal_equations,
    stage2_pose_batched_group_analytic,
    stage2_pose_batched_normal_equations,
)
from lib_pano.human_fitting.staged_lm.geometry.erp import (
    project_unified_erp_with_jacobian,
)
from lib_pano.human_fitting.staged_lm.geometry.so3 import so3_right_jacobian
from lib_pano.human_fitting.staged_lm.stage2_shared_cache import (
    build_stage2_shared_geometry_cache,
)
from lib_pano.human_fitting.staged_lm.stage_definitions.pose import (
    build_pose_stage_definition,
)
from lib_pano.human_fitting.staged_lm.stage_definitions.rigid import (
    build_stage1_definition,
)
from lib_pano.human_fitting.staged_lm.stage_runtime import run_stage
from script.human_fitting.staged_lm.app import build_context


REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
CAMERAS = ("cam2", "cam4")
SCHEMA_VERSION = "pose_value_cpp_reference_v1"
JOINT_LAYOUT = "root,body[21],jaw,left_eye,right_eye,left_hand[15],right_hand[15]"


def _numpy(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().cpu().contiguous().numpy()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPOSITORY_ROOT,
            text=True, stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _axis_angles(
    orientation: torch.Tensor,
    body_pose: torch.Tensor,
    left_hand_pose: torch.Tensor,
    right_hand_pose: torch.Tensor,
) -> torch.Tensor:
    zeros = torch.zeros(3, dtype=orientation.dtype, device=orientation.device)
    return torch.cat((
        orientation.reshape(1, 3), body_pose.reshape(21, 3),
        zeros.reshape(1, 3), zeros.reshape(1, 3), zeros.reshape(1, 3),
        left_hand_pose.reshape(15, 3), right_hand_pose.reshape(15, 3),
    ), dim=0)


def _fk_intermediates(
    rotations: torch.Tensor,
    joints: torch.Tensor,
    parents: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Materialize the exact local/global ordering used by batch_rigid_transform."""
    relative_joints = joints.clone()
    relative_joints[1:] -= joints.index_select(0, parents[1:])
    local = torch.zeros(
        rotations.shape[0], 4, 4, dtype=rotations.dtype, device=rotations.device,
    )
    local[:, :3, :3] = rotations
    local[:, :3, 3] = relative_joints
    local[:, 3, 3] = 1
    chain = [local[0]]
    for joint in range(1, int(parents.numel())):
        chain.append(torch.matmul(chain[int(parents[joint])], local[joint]))
    return relative_joints, local, torch.stack(chain)


def build_pose_reference_arrays(
    cache: Any,
    parameters: torch.Tensor,
    *,
    body_pose_anchor: torch.Tensor | None = None,
    pose_anchor_scale: float = 1.0,
    charbonnier_eps: float = 1.0,
    aggregation_spec: dict[str, Any] | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    if cache.frame_count != 1 or parameters.numel() != 75:
        raise ValueError("POSE_CPP_REFERENCE_REQUIRES_SINGLE_FRAME_75D")
    frame = unpack_frame_parameters(parameters, 0, 75, include_body_pose=True)
    body_pose = frame.body_pose.reshape(-1)
    tensors: dict[str, torch.Tensor] = {
        "parameters": parameters,
        "body_pose": body_pose,
        "joints": cache.model.joints.to(parameters),
        "parents": cache.model.parents.to(device=parameters.device),
    }
    canonical_groups = fast_raw_groups(cache, parameters)
    parity: dict[str, Any] = {}
    finite_difference: dict[str, Any] = {}
    for index, group in enumerate(cache.groups):
        camera = group.camera
        orientation = (
            frame.cam2_orient if camera == "cam2" else frame.cam4_orient
        ).reshape(3)
        translation = (
            frame.cam2_transl if camera == "cam2" else frame.cam4_transl
        ).reshape(3)
        left = group.left_hand_pose.to(parameters).reshape(-1)
        right = group.right_hand_pose.to(parameters).reshape(-1)
        axis_angles = _axis_angles(orientation, body_pose, left, right)
        rotations = assemble_full_pose_rotations(
            orientation, body_pose, left, right, cache.model,
        )
        offsets, feature = compute_selected_pose_offsets(
            rotations, group.selected_posedirs.to(parameters),
        )
        v_posed = group.selected_v_shaped.to(parameters) + offsets
        relative_joints, local, global_transforms = _fk_intermediates(
            rotations, cache.model.joints.to(parameters),
            cache.model.parents.to(device=parameters.device),
        )
        posed_joints, relative_transforms_batch = batch_rigid_transform(
            rotations.unsqueeze(0), cache.model.joints.to(parameters).unsqueeze(0),
            cache.model.parents.to(device=parameters.device), dtype=parameters.dtype,
        )
        relative_transforms = relative_transforms_batch.squeeze(0)
        skinned, selected_transforms = compute_selected_lbs(
            v_posed, group.selected_lbs_weights.to(parameters), relative_transforms,
        )
        vertices = skinned * group.scale.to(parameters).reshape(1, 1) + translation.reshape(1, 3)
        uv = project_unified_erp(vertices, cache.image_width, cache.image_height)
        residual = uv - group.targets.to(parameters)
        canonical_vertices, canonical_intermediates = fast_selected_vertices(
            cache, group, orientation, translation, body_pose,
            return_intermediates=True,
        )
        parity[f"{camera}_vertices_exact"] = bool(torch.equal(vertices, canonical_vertices))
        parity[f"{camera}_uv_exact"] = bool(torch.equal(
            uv, canonical_groups[index].residual_2d + group.targets.to(parameters),
        ))
        parity[f"{camera}_residual_exact"] = bool(torch.equal(
            residual, canonical_groups[index].residual_2d,
        ))
        parity[f"{camera}_relative_transforms_exact"] = bool(torch.equal(
            relative_transforms, canonical_intermediates["fk_transforms"],
        ))
        if not all(parity[key] for key in parity if key.startswith(camera)):
            raise RuntimeError(f"POSE_REFERENCE_PRODUCTION_PARITY_FAILED: {camera}")
        tensors.update({
            f"{camera}_orientation": orientation,
            f"{camera}_translation": translation,
            f"{camera}_left_hand_pose": left,
            f"{camera}_right_hand_pose": right,
            f"{camera}_scale": group.scale.to(parameters).reshape(1),
            f"{camera}_selected_v_shaped": group.selected_v_shaped.to(parameters),
            f"{camera}_selected_posedirs": group.selected_posedirs.to(parameters),
            f"{camera}_selected_lbs_weights": group.selected_lbs_weights.to(parameters),
            f"{camera}_targets": group.targets.to(parameters),
            f"{camera}_axis_angles": axis_angles,
            f"{camera}_rotations": rotations,
            f"{camera}_pose_feature": feature,
            f"{camera}_pose_offsets": offsets,
            f"{camera}_v_posed": v_posed,
            f"{camera}_relative_joints": relative_joints,
            f"{camera}_local_transforms": local,
            f"{camera}_global_transforms": global_transforms,
            f"{camera}_relative_transforms": relative_transforms,
            f"{camera}_posed_joints": posed_joints.squeeze(0),
            f"{camera}_selected_transforms": selected_transforms,
            f"{camera}_skinned_vertices": skinned,
            f"{camera}_vertices": vertices,
            f"{camera}_uv": uv,
            f"{camera}_raw_residual": residual,
        })

        # Canonical serial analytic derivative path. These calls are the same
        # primitives used by stage2_pose_group_raw_residual_and_jacobian.
        analytic_rotations, d_rotations = assemble_full_pose_rotation_jacobians(
            orientation, body_pose, left, right, cache.model,
        )
        analytic_offsets, d_offsets, _, d_feature = (
            compute_selected_pose_offsets_with_jacobian(
                analytic_rotations, d_rotations,
                group.selected_posedirs.to(parameters),
            )
        )
        analytic_v_posed = group.selected_v_shaped.to(parameters) + analytic_offsets
        fk = compute_fk_with_jacobian(
            analytic_rotations, d_rotations, cache.model.joints.to(parameters),
            cache.model.parents.to(parameters.device),
        )
        analytic_skinned, d_skinned, analytic_transforms, d_transforms = (
            compute_selected_lbs_with_jacobian(
                analytic_v_posed, d_offsets,
                group.selected_lbs_weights.to(parameters),
                fk.relative_transforms, fk.d_relative_transforms,
            )
        )
        scale = group.scale.to(parameters).reshape(1, 1)
        analytic_vertices = analytic_skinned * scale + translation.reshape(1, 3)
        d_vertices = d_skinned * scale.unsqueeze(-1)
        _, j_proj = project_unified_erp_with_jacobian(
            analytic_vertices, cache.image_width, cache.image_height,
        )
        j_uv_rotpose = torch.einsum("nab,nbq->naq", j_proj, d_vertices)
        j_active = torch.cat((
            j_uv_rotpose[..., :3], j_proj, j_uv_rotpose[..., 3:]
        ), dim=-1)
        j_frame = torch.zeros(
            group.targets.shape[0], 2, 75,
            dtype=parameters.dtype, device=parameters.device,
        )
        camera_column = 0 if camera == "cam2" else 6
        j_frame[..., camera_column:camera_column + 6] = j_active[..., :6]
        j_frame[..., 12:75] = j_active[..., 6:69]

        homogeneous = torch.cat((
            analytic_v_posed,
            torch.ones(analytic_v_posed.shape[0], 1, dtype=parameters.dtype),
        ), dim=1)
        d_homogeneous = torch.cat((
            d_offsets,
            torch.zeros(d_offsets.shape[0], 1, 66, dtype=parameters.dtype),
        ), dim=1)
        d_skin_transform = torch.einsum(
            "nabq,nb->naq", d_transforms[:, :3], homogeneous,
        )
        d_skin_vertex = torch.einsum(
            "nab,nbq->naq", analytic_transforms[:, :3], d_homogeneous,
        )
        canonical_residual, canonical_j_frame = (
            stage2_pose_group_raw_residual_and_jacobian(cache, group, parameters)
        )
        parity[f"{camera}_analytic_residual_exact"] = bool(torch.equal(
            canonical_residual, (project_unified_erp(
                analytic_vertices, cache.image_width, cache.image_height,
            ) - group.targets.to(parameters)).reshape(-1),
        ))
        parity[f"{camera}_j_frame_exact"] = bool(torch.equal(
            canonical_j_frame, j_frame.reshape(-1, 75),
        ))
        if not parity[f"{camera}_j_frame_exact"]:
            raise RuntimeError(f"POSE_JACOBIAN_REFERENCE_PARITY_FAILED: {camera}")
        tensors.update({
            f"{camera}_variable_right_jacobians": so3_right_jacobian(axis_angles[:22]),
            f"{camera}_d_rotation": d_rotations,
            f"{camera}_d_pose_feature": d_feature,
            f"{camera}_d_pose_offsets": d_offsets,
            f"{camera}_d_v_posed": d_offsets,
            f"{camera}_d_local_transforms": torch.nn.functional.pad(
                d_rotations.permute(0, 3, 1, 2), (0, 1, 0, 1)
            ).permute(0, 2, 3, 1),
            f"{camera}_d_global_transforms": fk.d_global_transforms,
            f"{camera}_d_relative_transforms": fk.d_relative_transforms,
            f"{camera}_d_selected_transforms": d_transforms,
            f"{camera}_d_skin_from_transform": d_skin_transform,
            f"{camera}_d_skin_from_vertex": d_skin_vertex,
            f"{camera}_d_skinned_vertices": d_skinned,
            f"{camera}_d_vertices_rotpose": d_vertices,
            f"{camera}_J_proj": j_proj,
            f"{camera}_J_uv_rotpose": j_uv_rotpose,
            f"{camera}_J_translation": j_proj,
            f"{camera}_J_active": j_active,
            f"{camera}_J_active_flat": j_active.reshape(-1, 69),
            f"{camera}_J_frame": j_frame.reshape(-1, 75),
        })

        # Diagnostic-only central finite-difference sweep on a small point and
        # parameter subset. The formal gate remains analytic C++ vs Python.
        sample_count = min(8, int(group.targets.shape[0]))
        sampled_parameters = (0, 1, 2, 3, 15, 33, 65)
        camera_fd: dict[str, Any] = {}
        for epsilon in (1e-2, 3e-3, 1e-3):
            columns = []
            for q in sampled_parameters:
                plus = parameters.clone()
                minus = parameters.clone()
                parameter_column = camera_column + q if q < 3 else 12 + (q - 3)
                plus[parameter_column] += epsilon
                minus[parameter_column] -= epsilon
                plus_frame = unpack_frame_parameters(plus, 0, 75, include_body_pose=True)
                minus_frame = unpack_frame_parameters(minus, 0, 75, include_body_pose=True)
                plus_orient = (
                    plus_frame.cam2_orient if camera == "cam2" else plus_frame.cam4_orient
                ).reshape(3)
                minus_orient = (
                    minus_frame.cam2_orient if camera == "cam2" else minus_frame.cam4_orient
                ).reshape(3)
                plus_translation = (
                    plus_frame.cam2_transl if camera == "cam2" else plus_frame.cam4_transl
                ).reshape(3)
                minus_translation = (
                    minus_frame.cam2_transl if camera == "cam2" else minus_frame.cam4_transl
                ).reshape(3)
                plus_vertices = fast_selected_vertices(
                    cache, group, plus_orient, plus_translation,
                    plus_frame.body_pose.reshape(-1),
                )
                minus_vertices = fast_selected_vertices(
                    cache, group, minus_orient, minus_translation,
                    minus_frame.body_pose.reshape(-1),
                )
                plus_uv = project_unified_erp(
                    plus_vertices[:sample_count], cache.image_width, cache.image_height,
                )
                minus_uv = project_unified_erp(
                    minus_vertices[:sample_count], cache.image_width, cache.image_height,
                )
                columns.append((plus_uv - minus_uv) / (2.0 * epsilon))
            fd = torch.stack(columns, dim=-1)
            analytic = j_uv_rotpose[:sample_count, :, list(sampled_parameters)]
            difference = (fd - analytic).to(torch.float64)
            relative = float(difference.norm() / max(float(analytic.to(torch.float64).norm()), 1e-30))
            camera_fd[f"eps_{epsilon:g}"] = {
                "relative_l2": relative,
                "max_abs": float(difference.abs().max()),
                "sample_points": sample_count,
                "internal_columns": list(sampled_parameters),
            }
        finite_difference[camera] = camera_fd
    parity["finite_difference"] = finite_difference

    # Normal-equation checkpoints follow the production FULL_BATCH path:
    # serial frozen IRLS weights, padded row scales, FP32 scaling, FP64 bmm,
    # active-to-frame scatter, then the pose anchor.
    groups = fast_raw_groups(cache, parameters)
    if aggregation_spec is None:
        aggregation_spec = build_primary_aggregation_spec(groups)
    weights = compute_irls_point_weights(groups, charbonnier_eps, aggregation_spec)
    row_scales = [torch.sqrt(weight).repeat_interleave(2) for weight in weights]
    batch_cache = getattr(cache, "batch_cache", None)
    if batch_cache is None:
        batch_cache = build_stage2_pose_batch_cache(
            cache, pose_anchor_scale=pose_anchor_scale,
        )
    padded_scales = pad_stage2_pose_row_scales(batch_cache, row_scales)
    gathered = gather_pose_batch_parameters(parameters, batch_cache.layout)
    batched_residual, batched_active, _ = stage2_pose_batched_group_analytic(
        cache, batch_cache, gathered,
    )
    scaled_residual = batched_residual * padded_scales.to(parameters)
    scaled_jacobian = batched_active * padded_scales.to(parameters)[..., None]
    residual_64 = scaled_residual.reshape(batch_cache.layout.group_count, -1).to(torch.float64)
    jacobian_64 = scaled_jacobian.reshape(batch_cache.layout.group_count, -1, 69).to(torch.float64)
    active_hessian = torch.bmm(jacobian_64.transpose(1, 2), jacobian_64)
    active_gradient = torch.bmm(
        jacobian_64.transpose(1, 2), residual_64[..., None],
    ).squeeze(-1)

    h_data = torch.zeros(75, 75, dtype=torch.float64, device=parameters.device)
    g_data = torch.zeros(75, dtype=torch.float64, device=parameters.device)
    active_indices = {
        "cam2": batch_cache.active_indices_cam2,
        "cam4": batch_cache.active_indices_cam4,
    }
    for index, group in enumerate(cache.groups):
        count = int(group.targets.shape[0])
        camera = group.camera
        indices = active_indices[camera]
        h_data[indices[:, None], indices[None, :]] += active_hessian[index]
        g_data.index_add_(0, indices, active_gradient[index])
        tensors.update({
            f"{camera}_raw_residual_2d": groups[index].residual_2d,
            f"{camera}_raw_residual_flat": groups[index].residual_2d.reshape(-1),
            f"{camera}_point_weights": weights[index],
            f"{camera}_row_scales": row_scales[index],
            f"{camera}_weighted_residual": scaled_residual[index, :count].reshape(-1),
            f"{camera}_weighted_J_active": scaled_jacobian[index, :count].reshape(-1, 69),
            f"{camera}_H_active": active_hessian[index],
            f"{camera}_g_active": active_gradient[index],
        })

    body_current = parameters[12:75]
    anchor = (
        torch.zeros(63, dtype=parameters.dtype, device=parameters.device)
        if body_pose_anchor is None else body_pose_anchor.to(parameters)
    )
    anchor_delta = body_current - anchor
    anchor_residual = anchor_delta * pose_anchor_scale
    anchor_scale_squared = float(pose_anchor_scale) ** 2
    h_anchor = torch.zeros_like(h_data)
    g_anchor = torch.zeros_like(g_data)
    h_anchor[12:, 12:] = torch.eye(
        63, dtype=torch.float64, device=parameters.device,
    ) * anchor_scale_squared
    g_anchor[12:] = anchor_delta.to(torch.float64) * anchor_scale_squared
    h_frame = h_data + h_anchor
    g_frame = g_data + g_anchor
    tensors.update({
        "body_pose_current": body_current,
        "body_pose_anchor": anchor,
        "pose_anchor_delta": anchor_delta,
        "pose_anchor_residual": anchor_residual,
        "H_data_frame": h_data,
        "g_data_frame": g_data,
        "H_pose_anchor": h_anchor,
        "g_pose_anchor": g_anchor,
        "H_frame": h_frame,
        "g_frame": g_frame,
    })
    production = stage2_pose_batched_normal_equations(
        cache, batch_cache, parameters, padded_scales, [anchor], pose_anchor_scale,
    )
    reconstructed = stage2_pose_batched_accumulate_normal_equations(
        batch_cache, parameters, gathered, batched_residual, batched_active,
        padded_scales, [anchor], pose_anchor_scale,
    )
    parity.update({
        "normal_equations_hessian_exact": bool(torch.equal(h_frame, production.hessian)),
        "normal_equations_gradient_exact": bool(torch.equal(g_frame, production.gradient)),
        "normal_equations_reconstruction_hessian_exact": bool(torch.equal(
            reconstructed.hessian, production.hessian,
        )),
        "normal_equations_reconstruction_gradient_exact": bool(torch.equal(
            reconstructed.gradient, production.gradient,
        )),
    })
    if not all(parity[key] for key in (
        "normal_equations_hessian_exact", "normal_equations_gradient_exact",
        "normal_equations_reconstruction_hessian_exact",
        "normal_equations_reconstruction_gradient_exact",
    )):
        raise RuntimeError("POSE_NORMAL_EQUATIONS_REFERENCE_PARITY_FAILED")
    return {name: _numpy(tensor) for name, tensor in tensors.items()}, parity


def _array_metadata(array: np.ndarray) -> dict[str, Any]:
    return {
        "shape": list(array.shape), "dtype": str(array.dtype),
        "device": "cpu", "contiguous": bool(array.flags.c_contiguous),
    }


def _self_check(npz_path: Path, manifest: dict[str, Any]) -> None:
    with np.load(npz_path, allow_pickle=False) as arrays:
        if set(arrays.files) != set(manifest["arrays"]):
            raise RuntimeError("POSE_REFERENCE_RELOAD_KEY_MISMATCH")
        for name in arrays.files:
            if list(arrays[name].shape) != manifest["arrays"][name]["shape"]:
                raise RuntimeError(f"POSE_REFERENCE_SHAPE_MISMATCH: {name}")
            if arrays[name].dtype.kind == "f" and not np.isfinite(arrays[name]).all():
                raise RuntimeError(f"POSE_REFERENCE_NONFINITE: {name}")


def _report(manifest: dict[str, Any]) -> str:
    return f"""# Pose value C++ reference

## Scope

- Frame: {manifest['frame']}, CPU, cam2 + cam4
- Points: cam2 {manifest['cameras']['cam2']['point_count']}, cam4 {manifest['cameras']['cam4']['point_count']}
- Parameters: `[cam2 orient(3), cam2 translation(3), cam4 orient(3), cam4 translation(3), body pose(63)]`
- Joint layout: `{JOINT_LAYOUT}`

## Value chain

`parameters → full axis angles → rotations → pose feature → selected pose offsets → v_posed → FK → production rest correction → selected LBS → scale/translation → ERP → residual`.

The fixture-facing static inputs are selected frozen-beta shaped vertices, selected posedirs, selected LBS weights, joints, parents, frozen hand poses, camera scale and targets. Jaw and both eye axis angles are production zeros.

## Production parity

All reconstructed final vertices, ERP values, residuals and rest-corrected transforms are bitwise equal to `fast_selected_vertices` / `fast_raw_groups`: `{manifest['parity']}`.

## Reference-only intermediates

`full_axis_angles`, `relative_joints`, `local_transforms`, and `global_transforms` are materialized by the validation exporter in the same joint traversal and operation order as installed SMPL-X `batch_rigid_transform`. No full SMPL-X forward, autograd, finite difference, or Jacobian path is used.

## Documentation discrepancies

The serial production value path uses `_aa_to_rotmat`, while the batched linearization path uses `geometry.so3.axis_angle_to_matrix`; the C++ value reference intentionally reuses its already validated Stage1 SO(3) kernel and applies the documented `1e-7` rotation gate. Also, installed SMPL-X computes rest correction as `global - pad(global @ joint_homogeneous)`, not by directly evaluating `global @ Rest`; C++ follows the production subtraction order.
"""


def write_pose_reference(
    cache: Any,
    parameters: torch.Tensor,
    output_directory: Path,
    *,
    frame: int,
    image_width: int,
    image_height: int,
    body_pose_anchor: torch.Tensor | None = None,
    pose_anchor_scale: float = 1.0,
    charbonnier_eps: float = 1.0,
    aggregation_spec: dict[str, Any] | None = None,
) -> tuple[Path, Path, Path, dict[str, Any]]:
    arrays, parity = build_pose_reference_arrays(
        cache, parameters, body_pose_anchor=body_pose_anchor,
        pose_anchor_scale=pose_anchor_scale, charbonnier_eps=charbonnier_eps,
        aggregation_spec=aggregation_spec,
    )
    output_directory.mkdir(parents=True, exist_ok=True)
    npz_path = output_directory / "pose_reference.npz"
    manifest_path = output_directory / "pose_reference_manifest.json"
    report_path = output_directory / "pose_reference_report.md"
    np.savez(npz_path, **arrays)
    sources = (
        "src/lib_pano/human_fitting/staged_lm/pose/cache.py",
        "src/lib_pano/human_fitting/staged_lm/pose/forward.py",
        "src/lib_pano/human_fitting/staged_lm/stage_definitions/pose.py",
        "src/lib_pano/human_fitting/staged_lm/stage2_shared_cache.py",
        "src/lib_pano/human_fitting/staged_lm/parameter_layout.py",
    )
    manifest = {
        "schema_version": SCHEMA_VERSION, "stage": "STAGE2_POSE_VALUE",
        "frame": frame, "device": str(parameters.device), "dtype": str(parameters.dtype),
        "parameter_dim": 75,
        "parameter_layout": [
            "cam2_orientation[0:3]", "cam2_translation[3:6]",
            "cam4_orientation[6:9]", "cam4_translation[9:12]",
            "body_pose[12:75]",
        ],
        "joint_count": 55, "joint_layout": JOINT_LAYOUT,
        "pose_feature_dim": 486, "image_width": image_width,
        "image_height": image_height,
        "cameras": {
            camera: {"point_count": int(arrays[f"{camera}_targets"].shape[0])}
            for camera in CAMERAS
        },
        "normal_equations": {
            "objective_mode": "charbonnier_irls",
            "charbonnier_eps": charbonnier_eps,
            "pose_anchor_scale": pose_anchor_scale,
            "aggregation_spec": aggregation_spec or build_primary_aggregation_spec(
                fast_raw_groups(cache, parameters),
            ),
            "group_order": [group.camera for group in cache.groups],
            "residual_flatten_order": "point-major interleaved [point0_u, point0_v, ...]",
            "point_weights_source": "directly_from_production: compute_irls_point_weights",
            "row_scales_source": "derived_for_reference_only: sqrt(point_weight).repeat_interleave(2)",
            "dtype_before_hg": "float32",
            "dtype_after_cast": "float64",
            "active_to_frame": {
                "cam2": "active[0:6]->frame[0:6], active[6:69]->frame[12:75]",
                "cam4": "active[0:6]->frame[6:12], active[6:69]->frame[12:75]",
            },
        },
        "arrays": {name: _array_metadata(array) for name, array in arrays.items()},
        "parity": parity,
        "production_sources": list(sources),
        "source_hashes": {source: _sha256(REPOSITORY_ROOT / source) for source in sources},
        "git_commit": _git_commit(),
        "reference_only_materialized": [
            "full_axis_angles", "relative_joints", "local_transforms",
            "global_transforms",
        ],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    report_path.write_text(_report(manifest))
    _self_check(npz_path, manifest)
    return npz_path, manifest_path, report_path, manifest


def dump_from_context(context: Any, output_directory: Path):
    if context.frame_count != 1:
        raise ValueError("POSE_CPP_REFERENCE_REQUIRES_SINGLE_FRAME")
    stage1_definition = build_stage1_definition(context)
    stage1_outcome = run_stage(stage1_definition, context)
    shared_cache = build_stage2_shared_geometry_cache(context)
    pose_definition, _ = build_pose_stage_definition(
        context, stage1_outcome, shared_cache,
    )
    return write_pose_reference(
        pose_definition.analytic_cache,
        pose_definition.initial_lm_parameters,
        output_directory, frame=int(context.frame_indices[0]),
        image_width=context.image_width, image_height=context.image_height,
        body_pose_anchor=context.body_pose_anchors[0],
        pose_anchor_scale=context.scaling.pose_anchor_scale,
        charbonnier_eps=context.charbonnier_eps,
        aggregation_spec=pose_definition.aggregation_spec,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frame", type=int, default=513)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument("--hmr_dir", required=True)
    parser.add_argument("--sam3d_dense_dir", required=True)
    parser.add_argument("--surface_sample_ids", default=None)
    parser.add_argument("--bbox_dir", required=True)
    parser.add_argument("--image_width", type=int, default=5760)
    parser.add_argument("--image_height", type=int, default=2880)
    parser.add_argument("--reproj_objective", default="charbonnier_irls")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.frames = str(args.frame)
    args.profile_timing = False
    args.validate_device_contract = False
    args.stop_after_stage = "stage2_pose"
    output = args.output_dir or (
        REPOSITORY_ROOT / "data_process/Vis/cpp_reference_dump/pose"
        / f"frame_{args.frame:06d}"
    )
    result = dump_from_context(build_context(args), output)
    print(f"[POSE_VALUE_CPP_REFERENCE_DUMP] frame={args.frame} output={output}")
    print(f"  arrays={len(result[3]['arrays'])} parity={result[3]['parity']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
