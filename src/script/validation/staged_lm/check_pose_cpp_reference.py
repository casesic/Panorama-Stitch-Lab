"""Compare C++ Pose value outputs with the canonical Python reference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


FIELDS = (
    ("axis_angles", "exact"), ("rotations", "rotation"),
    ("pose_feature", "rotation"), ("pose_offsets", "float"),
    ("v_posed", "float"), ("relative_joints", "float"),
    ("local_transforms", "float"), ("global_transforms", "float"),
    ("relative_transforms", "float"), ("selected_transforms", "float"),
    ("skinned_vertices", "float"), ("vertices", "float"),
    ("uv", "uv"), ("raw_residual", "residual"),
)

UV_RELATIVE_TOLERANCE = 1e-6
PIXEL_ABSOLUTE_TOLERANCE = 1e-3


def _read_meta(path: Path) -> dict[str, str]:
    return dict(
        line.split("=", 1) for line in path.read_text().splitlines()
        if line and not line.startswith("#")
    )


def _metrics(reference: np.ndarray, actual: np.ndarray, policy: str) -> dict[str, Any]:
    shape_match = reference.shape == actual.shape
    finite = bool(np.isfinite(actual).all())
    if shape_match:
        difference = actual.astype(np.float64) - reference.astype(np.float64)
        max_abs = float(np.max(np.abs(difference))) if difference.size else 0.0
        mean_abs = float(np.mean(np.abs(difference))) if difference.size else 0.0
        relative = float(
            np.linalg.norm(difference.ravel())
            / max(np.linalg.norm(reference.astype(np.float64).ravel()), 1e-300)
        )
        exact = bool(np.array_equal(actual, reference))
    else:
        max_abs = mean_abs = relative = float("inf")
        exact = False
    threshold = {
        "exact": 0.0, "rotation": 1e-7, "float": 1e-6,
        "uv": UV_RELATIVE_TOLERANCE, "residual": PIXEL_ABSOLUTE_TOLERANCE,
    }[policy]
    if policy == "exact":
        passed = exact
    elif policy == "uv":
        passed = relative <= UV_RELATIVE_TOLERANCE and max_abs <= PIXEL_ABSOLUTE_TOLERANCE
    elif policy == "residual":
        # The upstream UV gate is applied after all arrays have been loaded.
        passed = max_abs <= PIXEL_ABSOLUTE_TOLERANCE
    else:
        passed = relative <= threshold
    classification = (
        "FAIL" if not passed else "STRONG_PASS" if relative <= 5e-7 else "PASS"
    )
    return {
        "shape_match": shape_match, "dtype_match": actual.dtype == reference.dtype,
        "all_finite": finite, "exact": exact, "max_abs": max_abs,
        "mean_abs": mean_abs, "relative_frobenius": relative,
        "gate": policy, "threshold": threshold,
        "classification": classification,
        "pass": bool(shape_match and actual.dtype == reference.dtype and finite and passed),
    }


def check_pose_reference(
    reference_path: Path, cpp_output: Path,
    report_directory: Path | None = None,
) -> dict[str, Any]:
    meta = _read_meta(cpp_output / "cpp_output_meta.txt")
    counts = {camera: int(meta[f"{camera}_count"]) for camera in ("cam2", "cam4")}
    shapes = {
        "axis_angles": lambda n: (55, 3), "rotations": lambda n: (55, 3, 3),
        "pose_feature": lambda n: (486,), "pose_offsets": lambda n: (n, 3),
        "v_posed": lambda n: (n, 3), "relative_joints": lambda n: (55, 3),
        "local_transforms": lambda n: (55, 4, 4),
        "global_transforms": lambda n: (55, 4, 4),
        "relative_transforms": lambda n: (55, 4, 4),
        "selected_transforms": lambda n: (n, 4, 4),
        "skinned_vertices": lambda n: (n, 3), "vertices": lambda n: (n, 3),
        "uv": lambda n: (n, 2), "raw_residual": lambda n: (n, 2),
    }
    comparisons = {}
    order = []
    with np.load(reference_path, allow_pickle=False) as reference:
        for camera in ("cam2", "cam4"):
            for field, policy in FIELDS:
                name = f"{camera}_{field}"
                actual = np.fromfile(cpp_output / f"{name}.f32", dtype="<f4").reshape(
                    shapes[field](counts[camera])
                )
                comparisons[name] = _metrics(reference[name], actual, policy)
                order.append(name)
            residual_name = f"{camera}_raw_residual"
            uv_name = f"{camera}_uv"
            residual = comparisons[residual_name]
            residual["upstream_uv_pass"] = comparisons[uv_name]["pass"]
            residual["relative_is_diagnostic_only"] = True
            residual["pass"] = bool(
                residual["pass"] and residual["upstream_uv_pass"]
            )
            residual["classification"] = (
                "FAIL" if not residual["pass"] else
                "STRONG_PASS" if residual["max_abs"] <= 5e-4 else "PASS"
            )
    first_divergence = next(
        (name for name in order if not comparisons[name]["pass"]), "NONE",
    )
    result = {
        "schema_version": "pose_cpp_parity_v1",
        "reference": str(reference_path), "cpp_output": str(cpp_output),
        "policy": {
            "fixture_and_axis_angles": "exact",
            "rotations_and_pose_feature_relative": 1e-7,
            "float32_value_chain_relative": 1e-6,
            "strong_pass_relative": 5e-7,
            "uv_relative": UV_RELATIVE_TOLERANCE,
            "uv_max_abs_px": PIXEL_ABSOLUTE_TOLERANCE,
            "residual_max_abs_px": PIXEL_ABSOLUTE_TOLERANCE,
            "residual_requires_upstream_uv_pass": True,
            "residual_relative": "diagnostic_only",
        },
        "comparisons": comparisons, "first_divergence": first_divergence,
        "pass": first_divergence == "NONE",
        "cpp_timing_ms": {
            key: float(value) for key, value in meta.items() if key.endswith("_ms")
        },
    }
    report_directory = report_directory or cpp_output
    report_directory.mkdir(parents=True, exist_ok=True)
    json_path = report_directory / "pose_cpp_parity.json"
    md_path = report_directory / "pose_cpp_parity.md"
    json_path.write_text(json.dumps(result, indent=2) + "\n")
    rows = "\n".join(
        f"| `{name}` | {value['exact']} | {value['max_abs']:.6e} | "
        f"{value['mean_abs']:.6e} | {value['relative_frobenius']:.6e} | "
        f"{value['classification']} |"
        for name, value in comparisons.items()
    )
    timing_rows = "\n".join(
        f"| `{name}` | {value:.6f} |"
        for name, value in result["cpp_timing_ms"].items()
    )
    md_path.write_text(f"""# Pose value C++ CPU parity

| Array | Exact | Max abs | Mean abs | Relative norm | Result |
|---|---:|---:|---:|---:|---|
{rows}

`FIRST_DIVERGENCE={first_divergence}`

Overall pass: `{result['pass']}`.

## Release timing

| Section | ms |
|---|---:|
{timing_rows}
""")
    print(f"FIRST_DIVERGENCE={first_divergence}")
    print(f"POSE_VALUE_CPP_PARITY={'PASS' if result['pass'] else 'FAIL'}")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--cpp_output", type=Path, required=True)
    parser.add_argument("--report_dir", type=Path, default=None)
    args = parser.parse_args(argv)
    result = check_pose_reference(args.reference, args.cpp_output, args.report_dir)
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
