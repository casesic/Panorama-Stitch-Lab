"""Check the standalone C++ Pose analytic Jacobian against Python production."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from script.validation.staged_lm.check_pose_cpp_reference import check_pose_reference


FIELDS = (
    ("variable_right_jacobians", "so3"), ("d_rotation", "so3"),
    ("d_pose_feature", "feature"), ("d_pose_offsets", "feature"),
    ("d_v_posed", "feature"), ("d_local_transforms", "fk"),
    ("d_global_transforms", "fk"), ("d_relative_transforms", "fk"),
    ("d_selected_transforms", "lbs"),
    ("d_skin_from_transform", "lbs"), ("d_skin_from_vertex", "lbs"),
    ("d_skinned_vertices", "lbs"), ("d_vertices_rotpose", "lbs"),
    ("J_proj", "erp"), ("J_uv_rotpose", "final"),
    ("J_translation", "final"), ("J_active", "final"),
    ("J_frame", "final"),
)


def _metadata(path: Path) -> dict[str, str]:
    return dict(line.split("=", 1) for line in path.read_text().splitlines() if line)


def _metrics(reference: np.ndarray, actual: np.ndarray, category: str) -> dict[str, Any]:
    difference = actual.astype(np.float64) - reference.astype(np.float64)
    max_abs = float(np.max(np.abs(difference))) if difference.size else 0.0
    mean_abs = float(np.mean(np.abs(difference))) if difference.size else 0.0
    relative = float(
        np.linalg.norm(difference.ravel())
        / max(np.linalg.norm(reference.astype(np.float64).ravel()), 1e-300)
    )
    tolerance = {"so3": 1e-6, "feature": 2e-6, "fk": 2e-6,
                 "lbs": 2e-6, "erp": 1e-6, "final": 2e-6}[category]
    passed = relative <= tolerance and np.isfinite(actual).all()
    return {
        "shape_match": actual.shape == reference.shape,
        "dtype_match": actual.dtype == reference.dtype,
        "all_finite": bool(np.isfinite(actual).all()),
        "exact": bool(np.array_equal(actual, reference)),
        "max_abs": max_abs, "mean_abs": mean_abs,
        "relative_frobenius": relative, "tolerance": tolerance,
        "classification": "FAIL" if not passed else "STRONG_PASS" if relative <= 1e-7 else "PASS",
        "pass": bool(actual.shape == reference.shape and actual.dtype == reference.dtype and passed),
    }


def check_pose_jacobian(
    reference_path: Path, cpp_output: Path,
    report_directory: Path | None = None,
) -> dict[str, Any]:
    value_result = check_pose_reference(reference_path, cpp_output, cpp_output)
    if not value_result["pass"]:
        raise RuntimeError("POSE_VALUE_REFERENCE_MUST_PASS_BEFORE_JACOBIAN")
    meta = _metadata(cpp_output / "cpp_output_meta.txt")
    counts = {camera: int(meta[f"{camera}_count"]) for camera in ("cam2", "cam4")}
    shapes = {
        "variable_right_jacobians": lambda n: (22, 3, 3),
        "d_rotation": lambda n: (55, 3, 3, 66),
        "d_pose_feature": lambda n: (486, 66),
        "d_pose_offsets": lambda n: (n, 3, 66),
        "d_v_posed": lambda n: (n, 3, 66),
        "d_local_transforms": lambda n: (55, 4, 4, 66),
        "d_global_transforms": lambda n: (55, 4, 4, 66),
        "d_relative_transforms": lambda n: (55, 4, 4, 66),
        "d_selected_transforms": lambda n: (n, 4, 4, 66),
        "d_skin_from_transform": lambda n: (n, 3, 66),
        "d_skin_from_vertex": lambda n: (n, 3, 66),
        "d_skinned_vertices": lambda n: (n, 3, 66),
        "d_vertices_rotpose": lambda n: (n, 3, 66),
        "J_proj": lambda n: (n, 2, 3),
        "J_uv_rotpose": lambda n: (n, 2, 66),
        "J_translation": lambda n: (n, 2, 3),
        "J_active": lambda n: (n, 2, 69),
        "J_frame": lambda n: (2 * n, 75),
    }
    comparisons = {}
    order = []
    memory = {}
    with np.load(reference_path, allow_pickle=False) as reference:
        for camera in ("cam2", "cam4"):
            for field, category in FIELDS:
                name = f"{camera}_{field}"
                actual = np.fromfile(cpp_output / f"{name}.f32", dtype="<f4").reshape(
                    shapes[field](counts[camera])
                )
                comparisons[name] = _metrics(reference[name], actual, category)
                order.append(name)
                memory[name] = int(actual.nbytes)
            root_feature_zero = bool(np.count_nonzero(
                np.fromfile(cpp_output / f"{camera}_d_pose_feature.f32", dtype="<f4")
                .reshape(486, 66)[:, :3]
            ) == 0)
            comparisons[f"{camera}_root_feature_zero"] = {
                "pass": root_feature_zero, "exact": root_feature_zero,
            }
            if not root_feature_zero:
                order.append(f"{camera}_root_feature_zero")
    first = next((name for name in order if not comparisons[name]["pass"]), "NONE")
    manifest_path = reference_path.with_name("pose_reference_manifest.json")
    finite_difference = "NOT_RUN"
    if manifest_path.is_file():
        finite_difference = json.loads(manifest_path.read_text()).get(
            "parity", {}
        ).get("finite_difference", "NOT_RUN")
    result = {
        "schema_version": "pose_jacobian_cpp_parity_v1",
        "value_ready": True,
        "comparisons": comparisons, "first_divergence": first,
        "pass": first == "NONE",
        "finite_difference": finite_difference,
        "memory_bytes": memory,
        "total_derivative_intermediate_mb": {
            camera: sum(value for name, value in memory.items() if name.startswith(camera)) / 2**20
            for camera in ("cam2", "cam4")
        },
        "cpp_timing_ms": {
            key: float(value) for key, value in meta.items() if key.endswith("_ms")
        },
    }
    report_directory = report_directory or cpp_output
    json_path = report_directory / "pose_jacobian_cpp_parity.json"
    md_path = report_directory / "pose_jacobian_cpp_parity.md"
    json_path.write_text(json.dumps(result, indent=2) + "\n")
    rows = "\n".join(
        f"| `{name}` | {metric.get('exact')} | {metric.get('max_abs', 0):.6e} | "
        f"{metric.get('relative_frobenius', 0):.6e} | {metric['pass']} |"
        for name, metric in comparisons.items()
    )
    md_path.write_text(f"""# Pose analytic Jacobian C++ CPU parity

| Array | Exact | Max abs | Relative norm | Pass |
|---|---:|---:|---:|---:|
{rows}

`FIRST_DIVERGENCE={first}`

Total derivative intermediates: cam2 `{result['total_derivative_intermediate_mb']['cam2']:.2f} MiB`, cam4 `{result['total_derivative_intermediate_mb']['cam4']:.2f} MiB`.
""")
    print(f"FIRST_DIVERGENCE={first}")
    print(f"POSE_JACOBIAN_CPP_PARITY={'PASS' if result['pass'] else 'FAIL'}")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--cpp_output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = check_pose_jacobian(args.reference, args.cpp_output)
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
