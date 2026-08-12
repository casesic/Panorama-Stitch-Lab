"""Check C++ Pose IRLS and normal equations against Python production."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from script.validation.staged_lm.check_pose_cpp_jacobian import check_pose_jacobian


CAMERA_FIELDS = (
    ("point_weights", "f32", "irls_weight"),
    ("row_scales", "f32", "irls_scale"),
    ("weighted_residual", "f32", "weighted_residual"),
    ("weighted_J_active", "f32", "weighted_jacobian"),
    ("H_active", "f64", "intermediate_h_active"),
    ("g_active", "f64", "gradient"),
)
FRAME_FIELDS = (
    ("H_data_frame", "f64", "hg"), ("g_data_frame", "f64", "gradient"),
    ("pose_anchor_delta", "f32", "anchor_value"),
    ("pose_anchor_residual", "f32", "anchor_value"),
    ("H_pose_anchor", "f64", "anchor_hg"),
    ("g_pose_anchor", "f64", "anchor_hg"),
    ("H_frame", "f64", "hg"), ("g_frame", "f64", "gradient"),
)
INTERMEDIATE_H_ACTIVE_REL_TOL = 1.1e-6
FINAL_H_REL_TOL = 1e-6
TOLERANCES = {"hg": FINAL_H_REL_TOL, "anchor_value": 0.0, "anchor_hg": 1e-7}


def _metadata(path: Path) -> dict[str, str]:
    return dict(line.split("=", 1) for line in path.read_text().splitlines() if line)


def _metrics(
    reference: np.ndarray, actual: np.ndarray, category: str,
    *, upstream_pass: bool = True,
) -> dict[str, Any]:
    difference = actual.astype(np.float64) - reference.astype(np.float64)
    denominator = max(float(np.linalg.norm(reference.astype(np.float64).ravel())), 1e-300)
    relative = float(np.linalg.norm(difference.ravel()) / denominator)
    exact = bool(np.array_equal(reference, actual))
    max_abs = float(np.max(np.abs(difference))) if difference.size else 0.0
    if category == "irls_weight":
        tolerance, passed = 1e-7, upstream_pass and max_abs <= 1e-7
        gate = "upstream residual PASS and max_abs <= 1e-7"
    elif category == "irls_scale":
        tolerance, passed = 1e-6, upstream_pass and max_abs <= 1e-6
        gate = "upstream weights PASS and max_abs <= 1e-6"
    elif category == "weighted_residual":
        tolerance, passed = 1e-5, upstream_pass and max_abs <= 1e-5
        gate = "upstream residual/row-scale PASS and max_abs <= 1e-5"
    elif category == "weighted_jacobian":
        tolerance = 1e-5
        passed = upstream_pass and relative <= 1e-5 and max_abs <= 1e-3
        gate = "upstream J/row-scale PASS, rel <= 1e-5, max_abs <= 1e-3"
    elif category == "gradient":
        tolerance = 5e-6
        passed = upstream_pass and relative <= 5e-6 and max_abs <= 1e-4
        gate = "upstream weighted J/r PASS, rel <= 5e-6, max_abs <= 1e-4"
    elif category == "intermediate_h_active":
        tolerance = INTERMEDIATE_H_ACTIVE_REL_TOL
        passed = relative <= tolerance
        gate = "INTERMEDIATE_H_ACTIVE_REL_TOL=1.1e-6"
    else:
        tolerance = TOLERANCES[category]
        passed = exact if tolerance == 0 else relative <= tolerance
        gate = f"rel <= {tolerance:g}" if tolerance else "exact"
    passed = bool(np.isfinite(actual).all() and passed)
    strong_gate = (
        5e-7 if category in ("hg", "gradient", "intermediate_h_active")
        else min(tolerance, 1e-7)
    )
    return {
        "shape_match": actual.shape == reference.shape,
        "dtype_match": actual.dtype == reference.dtype,
        "all_finite": bool(np.isfinite(actual).all()), "exact": exact,
        "max_abs": max_abs,
        "mean_abs": float(np.mean(np.abs(difference))) if difference.size else 0.0,
        "relative_frobenius": relative, "tolerance": tolerance,
        "gate": gate, "upstream_pass": upstream_pass,
        "classification": "FAIL" if not passed else "STRONG_PASS" if exact or relative <= strong_gate else "PASS",
        "pass": bool(actual.shape == reference.shape and actual.dtype == reference.dtype and passed),
    }


def check_pose_normal_equations(
    reference_path: Path, cpp_output: Path, report_directory: Path | None = None,
) -> dict[str, Any]:
    jacobian = check_pose_jacobian(reference_path, cpp_output, cpp_output)
    if not jacobian["pass"]:
        raise RuntimeError("POSE_JACOBIAN_REFERENCE_MUST_PASS_BEFORE_NORMAL_EQUATIONS")
    meta = _metadata(cpp_output / "cpp_output_meta.txt")
    value = json.loads((cpp_output / "pose_cpp_parity.json").read_text())
    counts = {camera: int(meta[f"{camera}_count"]) for camera in ("cam2", "cam4")}
    shapes = {
        "point_weights": lambda n: (n,), "row_scales": lambda n: (2 * n,),
        "weighted_residual": lambda n: (2 * n,),
        "weighted_J_active": lambda n: (2 * n, 69),
        "H_active": lambda _n: (69, 69), "g_active": lambda _n: (69,),
        "H_data_frame": (75, 75), "g_data_frame": (75,),
        "pose_anchor_delta": (63,), "pose_anchor_residual": (63,),
        "H_pose_anchor": (75, 75), "g_pose_anchor": (75,),
        "H_frame": (75, 75), "g_frame": (75,),
    }
    comparisons: dict[str, Any] = {}
    order: list[str] = []
    memory_bytes: dict[str, int] = {}
    with np.load(reference_path, allow_pickle=False) as reference:
        value_residual_pass = all(
            value["comparisons"][f"{camera}_raw_residual"]["pass"]
            for camera in ("cam2", "cam4")
        )
        value_jacobian_pass = all(
            jacobian["comparisons"][f"{camera}_J_active"]["pass"]
            for camera in ("cam2", "cam4")
        )
        camera_upstream: dict[str, dict[str, bool]] = {
            camera: {
                "point_weights": value_residual_pass,
                "row_scales": False,
                "weighted_residual": False,
                "weighted_J_active": False,
                "H_active": True,
                "g_active": False,
            } for camera in ("cam2", "cam4")
        }
        for field, suffix, category in CAMERA_FIELDS:
            for camera in ("cam2", "cam4"):
                name = f"{camera}_{field}"
                dtype = "<f4" if suffix == "f32" else "<f8"
                actual = np.fromfile(cpp_output / f"{name}.{suffix}", dtype=dtype).reshape(
                    shapes[field](counts[camera])
                )
                if field == "row_scales":
                    camera_upstream[camera][field] = comparisons[
                        f"{camera}_point_weights"
                    ]["pass"]
                elif field == "weighted_residual":
                    camera_upstream[camera][field] = (
                        value_residual_pass
                        and comparisons[f"{camera}_row_scales"]["pass"]
                    )
                elif field == "weighted_J_active":
                    camera_upstream[camera][field] = (
                        value_jacobian_pass
                        and comparisons[f"{camera}_row_scales"]["pass"]
                    )
                elif field == "g_active":
                    camera_upstream[camera][field] = (
                        comparisons[f"{camera}_weighted_residual"]["pass"]
                        and comparisons[f"{camera}_weighted_J_active"]["pass"]
                    )
                comparisons[name] = _metrics(
                    reference[name], actual, category,
                    upstream_pass=camera_upstream[camera][field],
                )
                order.append(name)
                memory_bytes[name] = actual.nbytes
        for field, suffix, category in FRAME_FIELDS:
            dtype = "<f4" if suffix == "f32" else "<f8"
            actual = np.fromfile(cpp_output / f"{field}.{suffix}", dtype=dtype).reshape(shapes[field])
            upstream = True
            if category == "gradient":
                upstream = all(
                    comparisons[f"{camera}_weighted_residual"]["pass"]
                    and comparisons[f"{camera}_weighted_J_active"]["pass"]
                    for camera in ("cam2", "cam4")
                )
            comparisons[field] = _metrics(
                reference[field], actual, category, upstream_pass=upstream,
            )
            order.append(field)
            memory_bytes[field] = actual.nbytes

        symmetry = {}
        psd = {}
        for name in ("H_data_frame", "H_pose_anchor", "H_frame"):
            value = np.fromfile(cpp_output / f"{name}.f64", dtype="<f8").reshape(75, 75)
            symmetry[name] = float(np.max(np.abs(value - value.T)))
            diagonal = np.diag(value)
            psd[name] = {
                "diagonal_finite": bool(np.isfinite(diagonal).all()),
                "min_diagonal": float(diagonal.min()),
                "min_eigenvalue_diagnostic": float(np.linalg.eigvalsh((value + value.T) * 0.5).min()),
            }
        h_anchor = np.fromfile(cpp_output / "H_pose_anchor.f64", dtype="<f8").reshape(75, 75)
        g_anchor = np.fromfile(cpp_output / "g_pose_anchor.f64", dtype="<f8")
        anchor_invariant = bool(
            np.count_nonzero(h_anchor[:12]) == 0
            and np.count_nonzero(h_anchor[:, :12]) == 0
            and np.count_nonzero(g_anchor[:12]) == 0
        )
    first = next((name for name in order if not comparisons[name]["pass"]), "NONE")
    symmetry_pass = all(value <= 1e-10 for value in symmetry.values())
    if first == "NONE" and not symmetry_pass:
        first = "H_symmetry"
    if first == "NONE" and not anchor_invariant:
        first = "pose_anchor_block"
    timing = {key: float(value) for key, value in meta.items() if key.endswith("_ms")}
    ne_memory_names = (
        "cam2_weighted_J_active", "cam4_weighted_J_active",
        "cam2_weighted_residual", "cam4_weighted_residual",
        "cam2_H_active", "cam4_H_active", "cam2_g_active", "cam4_g_active",
    )
    # Peak model: both camera FP32 weighted arrays, their FP64 J/r casts, and
    # both active H/g coexist in this correctness-first reference.
    fp64_cast_bytes = sum(
        counts[camera] * 2 * (69 + 1) * 8 for camera in ("cam2", "cam4")
    )
    peak_bytes = sum(memory_bytes[name] for name in ne_memory_names) + fp64_cast_bytes
    result = {
        "schema_version": "pose_normal_equations_cpp_parity_v1",
        "POSE_NORMAL_EQUATIONS_CPP_CPU_REFERENCE_READY": first == "NONE",
        "value_ready": True, "jacobian_ready": True,
        "comparisons": comparisons, "first_divergence": first,
        "pass": first == "NONE", "symmetry_max_abs": symmetry,
        "psd_diagnostic": psd, "pose_anchor_rigid_zero_invariant": anchor_invariant,
        "shared_body_pose_scatter": "cam2/cam4 active[6:69] accumulate into frame[12:75]",
        "dtype_boundary": {"before_hg": meta["ne_dtype_before_hg"], "after_cast": meta["ne_dtype_after_cast"]},
        "parity_policy": {
            "INTERMEDIATE_H_ACTIVE_REL_TOL": INTERMEDIATE_H_ACTIVE_REL_TOL,
            "FINAL_H_REL_TOL": FINAL_H_REL_TOL,
        },
        "cpp_timing_ms": timing, "ne_memory_bytes": memory_bytes,
        "ne_additional_peak_memory_bytes": peak_bytes,
        "ne_additional_peak_memory_mib": peak_bytes / 2**20,
    }
    report_directory = report_directory or cpp_output
    json_path = report_directory / "pose_normal_equations_cpp_parity.json"
    md_path = report_directory / "pose_normal_equations_cpp_parity.md"
    json_path.write_text(json.dumps(result, indent=2) + "\n")
    rows = "\n".join(
        f"| `{name}` | {metric['exact']} | {metric['max_abs']:.6e} | "
        f"{metric['relative_frobenius']:.6e} | {metric['classification']} |"
        for name, metric in comparisons.items()
    )
    md_path.write_text(f"""# Pose normal equations C++ CPU parity

| Array | Exact | Max abs | Relative norm | Classification |
|---|---:|---:|---:|---:|
{rows}

`FIRST_DIVERGENCE={first}`

`POSE_NORMAL_EQUATIONS_CPP_CPU_REFERENCE_READY={'PASS' if result['pass'] else 'FAIL'}`

Symmetry max abs: `{symmetry}`. Additional NE peak memory estimate: `{peak_bytes / 2**20:.3f} MiB`.
""")
    print(f"FIRST_DIVERGENCE={first}")
    print(f"POSE_NORMAL_EQUATIONS_CPP_PARITY={'PASS' if result['pass'] else 'FAIL'}")
    print(
        "POSE_NORMAL_EQUATIONS_CPP_CPU_REFERENCE_READY="
        f"{'PASS' if result['pass'] else 'FAIL'}"
    )
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--cpp_output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = check_pose_normal_equations(args.reference, args.cpp_output)
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
