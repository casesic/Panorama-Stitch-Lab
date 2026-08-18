"""Export a single-frame Stage 1 intermediate reference for the C++ port."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from lib_pano.human_fitting.staged_lm.analytic_stage1 import (
    Stage1AnalyticCache,
    Stage1AnalyticGroupCache,
    build_stage1_analytic_linearization,
)
from lib_pano.human_fitting.staged_lm.config import STAGE1_PER_FRAME_DIM
from lib_pano.human_fitting.staged_lm.geometry.erp import (
    project_unified_erp_with_jacobian,
)
from lib_pano.human_fitting.staged_lm.geometry.so3 import (
    axis_angle_to_matrix,
    rotated_point_jacobian,
    so3_right_jacobian,
)
from lib_pano.human_fitting.staged_lm.irls import compute_irls_point_weights
from lib_pano.human_fitting.staged_lm.stage_definitions.rigid import (
    build_stage1_definition,
)
from script.human_fitting.staged_lm.app import build_context


SCHEMA_VERSION = "stage1_cpp_reference_v1"
CAMERAS = ("cam2", "cam4")
REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
PRODUCTION_SOURCES = (
    ("src/lib_pano/human_fitting/staged_lm/stage_definitions/rigid.py", "build_stage1_definition"),
    ("src/lib_pano/human_fitting/staged_lm/analytic_stage1.py", "build_stage1_analytic_cache"),
    ("src/lib_pano/human_fitting/staged_lm/analytic_stage1.py", "build_stage1_analytic_linearization"),
    ("src/lib_pano/human_fitting/staged_lm/analytic_stage1.py", "stage1_raw_groups"),
    ("src/lib_pano/human_fitting/staged_lm/analytic_stage1.py", "stage1_normal_equations"),
    ("src/lib_pano/human_fitting/staged_lm/parameter_layout.py", "build_stage1_initial_parameters"),
    ("src/lib_pano/human_fitting/staged_lm/geometry/so3.py", "axis_angle_to_matrix"),
    ("src/lib_pano/human_fitting/staged_lm/geometry/so3.py", "so3_right_jacobian"),
    ("src/lib_pano/human_fitting/staged_lm/geometry/so3.py", "rotated_point_jacobian"),
    ("src/lib_pano/human_fitting/staged_lm/geometry/erp.py", "project_unified_erp_with_jacobian"),
    ("src/lib_pano/human_fitting/staged_lm/irls.py", "compute_irls_point_weights"),
)
REFERENCE_ONLY_STEPS = (
    "materialize broadcast translation derivative as a 3x3 identity",
    "materialize per-camera point weights and row scales",
    "materialize per-camera weighted residual/J and H/g contributions",
    "materialize R and SO(3) right Jacobian",
)


def _numpy(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().cpu().numpy()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPOSITORY_ROOT, text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _array_description(name: str) -> str:
    suffixes = {
        "orientation": "axis-angle orientation",
        "translation": "camera translation",
        "local_vertices": "selected vertices relative to the root rotation center",
        "rotation_center": "root joint used as rotation center",
        "targets": "observed ERP pixel targets",
        "R": "axis-angle rotation matrix",
        "Jr": "SO(3) right Jacobian",
        "points": "transformed selected 3D points",
        "uv": "projected ERP pixels",
        "J_proj": "ERP pixel Jacobian with respect to input 3D point",
        "dpoint_dorient": "3D point Jacobian with respect to axis-angle orientation",
        "dpoint_dtranslation": "broadcast translation derivative identity",
        "J_orientation": "ERP Jacobian with respect to orientation",
        "J_translation": "ERP Jacobian with respect to translation",
        "J_camera": "per-point local rigid Jacobian [orientation, translation]",
        "J_frame": "flattened rigid Jacobian scattered into the 12D frame layout",
        "raw_residual_2d": "uv minus target, point-major with (u,v) columns",
        "raw_residual_flat": "production row ordering: u0,v0,u1,v1,...",
        "point_weights": "IRLS per-point weight",
        "row_scales": "frozen production row scale repeated for u and v",
        "weighted_residual": "row-scaled residual",
        "weighted_J": "row-scaled frame Jacobian",
        "H": "per-camera FP64 normal-equation Hessian contribution",
        "g": "per-camera FP64 normal-equation gradient contribution",
    }
    if name in ("parameters",):
        return "single-frame Stage 1 parameter vector"
    if name in ("H_frame", "H_global"):
        return "canonical production FP64 Stage 1 Hessian"
    if name in ("g_frame", "g_global"):
        return "canonical production FP64 Stage 1 gradient"
    return next((text for suffix, text in suffixes.items() if name.endswith(suffix)), name)


def _tensor_metadata(array: np.ndarray, description: str) -> dict[str, Any]:
    return {
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "device": "cpu",
        "contiguous": bool(array.flags.c_contiguous),
        "description": description,
    }


def _camera_arrays(
    cache: Stage1AnalyticCache,
    group: Stage1AnalyticGroupCache,
    parameters: torch.Tensor,
    point_weights: torch.Tensor,
    row_scales: torch.Tensor,
) -> dict[str, torch.Tensor]:
    camera = group.camera
    offset = 0 if camera == "cam2" else 6
    orientation = parameters[offset:offset + 3]
    translation = parameters[offset + 3:offset + 6]
    local = group.local_vertices.to(parameters)
    center = group.rotation_center.to(parameters)
    rotation = axis_angle_to_matrix(orientation)
    right_jacobian = so3_right_jacobian(orientation)
    points = local @ rotation.T + center + translation
    uv, projection_jacobian = project_unified_erp_with_jacobian(
        points, cache.image_width, cache.image_height,
    )
    point_orient_jacobian = rotated_point_jacobian(orientation, local)
    orient_jacobian = torch.einsum(
        "nij,njk->nik", projection_jacobian, point_orient_jacobian,
    )
    camera_jacobian = torch.cat(
        (orient_jacobian, projection_jacobian), dim=-1,
    )
    residual_2d = uv - group.targets.to(parameters)
    residual_flat = residual_2d.reshape(-1)
    frame_jacobian = torch.zeros(
        residual_flat.numel(), STAGE1_PER_FRAME_DIM,
        dtype=parameters.dtype, device=parameters.device,
    )
    frame_jacobian[:, offset:offset + 6] = camera_jacobian.reshape(-1, 6)
    weighted_residual = residual_flat * row_scales.to(parameters)
    weighted_jacobian = frame_jacobian * row_scales.to(parameters).unsqueeze(1)
    weighted_residual_64 = weighted_residual.to(torch.float64)
    weighted_jacobian_64 = weighted_jacobian.to(torch.float64)
    return {
        f"{camera}_orientation": orientation,
        f"{camera}_translation": translation,
        f"{camera}_local_vertices": local,
        f"{camera}_rotation_center": center,
        f"{camera}_targets": group.targets.to(parameters),
        f"{camera}_R": rotation,
        f"{camera}_Jr": right_jacobian,
        f"{camera}_points": points,
        f"{camera}_uv": uv,
        f"{camera}_J_proj": projection_jacobian,
        f"{camera}_dpoint_dorient": point_orient_jacobian,
        f"{camera}_dpoint_dtranslation": torch.eye(3, dtype=parameters.dtype, device=parameters.device),
        f"{camera}_J_orientation": orient_jacobian,
        f"{camera}_J_translation": projection_jacobian,
        f"{camera}_J_camera": camera_jacobian,
        f"{camera}_J_frame": frame_jacobian,
        f"{camera}_raw_residual_2d": residual_2d,
        f"{camera}_raw_residual_flat": residual_flat,
        f"{camera}_point_weights": point_weights,
        f"{camera}_row_scales": row_scales,
        f"{camera}_weighted_residual": weighted_residual,
        f"{camera}_weighted_J": weighted_jacobian,
        f"{camera}_H": weighted_jacobian_64.T @ weighted_jacobian_64,
        f"{camera}_g": weighted_jacobian_64.T @ weighted_residual_64,
    }


def build_reference_arrays(
    cache: Stage1AnalyticCache,
    parameters: torch.Tensor,
    *,
    is_irls: bool,
    charbonnier_eps: float,
    aggregation_spec: dict[str, Any],
    rep_scale: float,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Build reference arrays and compare them against production equations."""
    if cache.frame_count != 1 or parameters.numel() != STAGE1_PER_FRAME_DIM:
        raise ValueError("STAGE1_CPP_REFERENCE_REQUIRES_SINGLE_FRAME")
    groups_by_camera = {group.camera: group for group in cache.groups}
    if set(groups_by_camera) != set(CAMERAS) or len(cache.groups) != 2:
        raise ValueError("STAGE1_CPP_REFERENCE_REQUIRES_CAM2_AND_CAM4")

    linearization = build_stage1_analytic_linearization(
        cache, parameters, is_irls=is_irls,
        charbonnier_eps=charbonnier_eps,
        aggregation_spec=aggregation_spec, rep_scale=rep_scale,
    )
    production_equations = linearization.normal_equations(parameters)

    raw_groups = cache.raw_groups(parameters)
    if is_irls:
        weights = compute_irls_point_weights(
            raw_groups, charbonnier_eps, aggregation_spec,
        )
        scales = [torch.sqrt(weight).repeat_interleave(2) for weight in weights]
    else:
        weights = [
            torch.full((group.valid_point_count,), rep_scale * rep_scale,
                       dtype=parameters.dtype, device=parameters.device)
            for group in raw_groups
        ]
        scales = [
            torch.full((group.valid_point_count * 2,), rep_scale,
                       dtype=parameters.dtype, device=parameters.device)
            for group in raw_groups
        ]

    tensors: dict[str, torch.Tensor] = {"parameters": parameters}
    for group, point_weights, row_scales in zip(cache.groups, weights, scales):
        tensors.update(_camera_arrays(
            cache, group, parameters, point_weights, row_scales,
        ))
    tensors["H_frame"] = production_equations.hessian
    tensors["g_frame"] = production_equations.gradient
    tensors["H_global"] = production_equations.hessian
    tensors["g_global"] = production_equations.gradient

    weighted_residual = torch.cat([
        tensors[f"{camera}_weighted_residual"] for camera in CAMERAS
    ])
    h_sum = tensors["cam2_H"] + tensors["cam4_H"]
    g_sum = tensors["cam2_g"] + tensors["cam4_g"]
    parity = {
        "H_exact": bool(torch.equal(tensors["H_frame"], production_equations.hessian)),
        "g_exact": bool(torch.equal(tensors["g_frame"], production_equations.gradient)),
        "residual_exact": bool(torch.equal(weighted_residual, production_equations.residual)),
        "H_camera_sum_exact": bool(torch.equal(h_sum, tensors["H_frame"])),
        "g_camera_sum_exact": bool(torch.equal(g_sum, tensors["g_frame"])),
        "H_camera_sum_max_abs": float((h_sum - tensors["H_frame"]).abs().max()),
        "g_camera_sum_max_abs": float((g_sum - tensors["g_frame"]).abs().max()),
        "analytic_fd_check": "NOT_RUN",
    }
    if not (parity["H_exact"] and parity["g_exact"] and parity["residual_exact"]):
        raise RuntimeError("STAGE1_CPP_REFERENCE_PRODUCTION_PARITY_FAILED")
    return {name: _numpy(value) for name, value in tensors.items()}, parity


def _self_check(npz_path: Path, manifest: dict[str, Any]) -> None:
    with np.load(npz_path, allow_pickle=False) as loaded:
        if set(loaded.files) != set(manifest["arrays"]):
            raise RuntimeError("REFERENCE_RELOAD_KEY_MISMATCH")
        for name, metadata in manifest["arrays"].items():
            array = loaded[name]
            if list(array.shape) != metadata["shape"] or str(array.dtype) != metadata["dtype"]:
                raise RuntimeError(f"REFERENCE_RELOAD_METADATA_MISMATCH: {name}")
            if not np.isfinite(array).all():
                raise RuntimeError(f"REFERENCE_NONFINITE: {name}")
        for camera in CAMERAS:
            n = manifest["cameras"][camera]["point_count"]
            expected = {
                f"{camera}_points": (n, 3), f"{camera}_uv": (n, 2),
                f"{camera}_J_proj": (n, 2, 3),
                f"{camera}_J_orientation": (n, 2, 3),
                f"{camera}_J_frame": (2 * n, 12),
                f"{camera}_row_scales": (2 * n,),
                f"{camera}_weighted_residual": (2 * n,),
                f"{camera}_weighted_J": (2 * n, 12),
                f"{camera}_H": (12, 12), f"{camera}_g": (12,),
            }
            for name, shape in expected.items():
                if loaded[name].shape != shape:
                    raise RuntimeError(f"REFERENCE_SHAPE_MISMATCH: {name}")
        if loaded["H_frame"].shape != (12, 12) or loaded["g_frame"].shape != (12,):
            raise RuntimeError("REFERENCE_NORMAL_EQUATION_SHAPE_MISMATCH")
        if not np.array_equal(loaded["H_frame"], loaded["H_frame"].T):
            raise RuntimeError("REFERENCE_HESSIAN_NOT_EXACTLY_SYMMETRIC")
        if not np.array_equal(loaded["H_frame"], loaded["H_global"]):
            raise RuntimeError("REFERENCE_SINGLE_FRAME_GLOBAL_H_MISMATCH")
        if not np.array_equal(loaded["g_frame"], loaded["g_global"]):
            raise RuntimeError("REFERENCE_SINGLE_FRAME_GLOBAL_G_MISMATCH")
        if np.any(loaded["cam2_J_frame"][:, 6:] != 0):
            raise RuntimeError("REFERENCE_CAM2_SCATTER_MISMATCH")
        if np.any(loaded["cam4_J_frame"][:, :6] != 0):
            raise RuntimeError("REFERENCE_CAM4_SCATTER_MISMATCH")


def _report(manifest: dict[str, Any]) -> str:
    arrays = manifest["arrays"]
    shape_rows = "\n".join(
        f"| `{name}` | `{meta['shape']}` | `{meta['dtype']}` |"
        for name, meta in arrays.items()
        if name in {"parameters", "cam2_points", "cam2_uv", "cam2_J_proj",
                    "cam2_J_frame", "cam4_points", "cam4_uv", "cam4_J_proj",
                    "cam4_J_frame", "H_frame", "g_frame"}
    )
    parity = manifest["parity"]
    return f"""# Stage 1 C++ reference dump

## Scope

- Stage: Stage 1 Rigid
- Frame: {manifest['frame']}
- Cameras: cam2 ({manifest['cameras']['cam2']['point_count']} points), cam4 ({manifest['cameras']['cam4']['point_count']} points)
- Device: {manifest['device']}
- Parameter order: `[cam2_orientation(3), cam2_translation(3), cam4_orientation(3), cam4_translation(3)]`

## Production computation chain

`parameters → axis_angle_to_matrix → transformed points → project_unified_erp_with_jacobian → rotated_point_jacobian → rigid 2D Jacobian → residual → frozen IRLS row scales → weighted residual/J → FP64 H/g`.

The selected local vertices, root rotation centers, targets, frozen body/hand parameters and scale are static Stage 1 cache inputs. Rotation, points, ERP values/Jacobians, residuals and weighted normal-equation inputs are recomputed at each LM linearisation point.

## Main arrays

| Array | Shape | Dtype |
|---|---:|---|
{shape_rows}

Geometry, residual and Jacobian intermediates retain the production parameter dtype. Weighted residual/J are converted to FP64 only after row scaling, immediately before the canonical frame-level matrix products that form H/g.

## Reference-only materialisation

{chr(10).join(f'- {item}' for item in REFERENCE_ONLY_STEPS)}

## Production parity

- H exact: `{parity['H_exact']}`
- g exact: `{parity['g_exact']}`
- weighted residual exact: `{parity['residual_exact']}`
- finite difference: `{parity['analytic_fd_check']}`
- Per-camera contribution sum exact: H=`{parity['H_camera_sum_exact']}`, g=`{parity['g_camera_sum_exact']}`. Production concatenates cam2/cam4 rows and performs one FP64 matrix product; separately reduced camera contributions can differ only by floating-point accumulation order. The authoritative `H_frame/g_frame` arrays are copied from the canonical production normal equations.

## Documentation discrepancies

No Stage 1 formula discrepancy was found. The computation-graph document describes the same operations; it does not spell out that per-camera H/g reductions performed separately need not be bitwise identical to production's concatenated frame reduction.

## Suggested C++ comparison order

Compare parameters/static cache first, then `R`, `Jr`, points, ERP `uv`, `J_proj`, `dpoint_dorient`, 2D orientation/translation Jacobians, scattered frame Jacobians, raw residuals, IRLS row scales, weighted residual/J, per-camera contributions, and finally authoritative frame H/g.
"""


def write_reference_dump(
    cache: Stage1AnalyticCache,
    parameters: torch.Tensor,
    output_directory: Path,
    *,
    frame: int,
    is_irls: bool,
    charbonnier_eps: float,
    aggregation_spec: dict[str, Any],
    rep_scale: float,
    objective_name: str,
    relevant_config: dict[str, Any] | None = None,
) -> tuple[Path, Path, Path, dict[str, Any]]:
    arrays, parity = build_reference_arrays(
        cache, parameters, is_irls=is_irls,
        charbonnier_eps=charbonnier_eps,
        aggregation_spec=aggregation_spec, rep_scale=rep_scale,
    )
    output_directory.mkdir(parents=True, exist_ok=True)
    npz_path = output_directory / "stage1_reference.npz"
    manifest_path = output_directory / "stage1_reference_manifest.json"
    report_path = output_directory / "stage1_reference_report.md"
    np.savez(npz_path, **arrays)

    source_hashes = {}
    for source, _ in PRODUCTION_SOURCES:
        source_path = REPOSITORY_ROOT / source
        source_hashes[source] = _sha256(source_path)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "stage": "STAGE1_RIGID",
        "stage_name": "Stage1 Rigid",
        "frame": frame,
        "frame_id": frame,
        "device": str(parameters.device),
        "dtype": str(parameters.dtype),
        "parameter_dim": STAGE1_PER_FRAME_DIM,
        "parameter_layout": [
            "cam2_orientation[0:3]", "cam2_translation[3:6]",
            "cam4_orientation[6:9]", "cam4_translation[9:12]",
        ],
        "image_width": cache.image_width,
        "image_height": cache.image_height,
        "cameras": {
            camera: {
                "point_count": int(next(g for g in cache.groups if g.camera == camera).targets.shape[0]),
                "translation_derivative": "3x3 identity broadcast over points; materialized for reference only",
            } for camera in CAMERAS
        },
        "normal_equation_dtype": str(arrays["H_frame"].dtype),
        "arrays": {
            name: _tensor_metadata(array, _array_description(name))
            for name, array in arrays.items()
        },
        "config": {
            "objective": objective_name,
            "is_irls": is_irls,
            "charbonnier_eps": charbonnier_eps,
            "rep_scale": rep_scale,
            "aggregation_spec": aggregation_spec,
            "image_width": cache.image_width,
            "image_height": cache.image_height,
            "relevant_stage1_config": relevant_config or {},
        },
        "production_sources": [
            {"file": source, "function": function} for source, function in PRODUCTION_SOURCES
        ],
        "reused_production_functions": [function for _, function in PRODUCTION_SOURCES],
        "reconstructed_reference_only_steps": list(REFERENCE_ONLY_STEPS),
        "git_commit": _git_commit(),
        "source_hashes": source_hashes,
        "parity": parity,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    report_path.write_text(_report(manifest))
    _self_check(npz_path, manifest)
    return npz_path, manifest_path, report_path, manifest


def dump_from_context(context: Any, output_directory: Path) -> tuple[Path, Path, Path, dict[str, Any]]:
    """Build the canonical Stage 1 definition and dump its initial point."""
    if context.frame_count != 1:
        raise ValueError("STAGE1_CPP_REFERENCE_REQUIRES_SINGLE_FRAME")
    stage = build_stage1_definition(context)
    relevant_config = {
        "solver_config": asdict(context.solver_config) if is_dataclass(context.solver_config) else {},
        "objective_config": asdict(context.objective_config) if is_dataclass(context.objective_config) else {},
        "linearization_backend": stage.linearization_backend,
        "evaluation_backend": stage.eval_backend,
    }
    return write_reference_dump(
        stage.analytic_cache, stage.initial_lm_parameters, output_directory,
        frame=int(context.frame_indices[0]), is_irls=context.is_irls,
        charbonnier_eps=context.charbonnier_eps,
        aggregation_spec=stage.aggregation_spec,
        rep_scale=context.scaling.rep_scale,
        objective_name=context.args.reproj_objective,
        relevant_config=relevant_config,
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
    parser.add_argument(
        "--reproj_objective", choices=("raw_l2", "charbonnier_irls"),
        default="charbonnier_irls",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.frames = str(args.frame)
    args.profile_timing = False
    args.validate_device_contract = False
    output = args.output_dir or (
        REPOSITORY_ROOT / "data_process/Vis/cpp_reference_dump/stage1"
        / f"frame_{args.frame:06d}"
    )
    context = build_context(args)
    npz_path, manifest_path, report_path, manifest = dump_from_context(context, output)
    print(f"[STAGE1_CPP_REFERENCE_DUMP] frame={args.frame} device={args.device}")
    print(f"  npz={npz_path}")
    print(f"  manifest={manifest_path}")
    print(f"  report={report_path}")
    print(f"  arrays={len(manifest['arrays'])} parity={manifest['parity']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
