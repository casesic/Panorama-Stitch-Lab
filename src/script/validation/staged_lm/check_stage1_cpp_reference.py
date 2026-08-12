"""Compare standalone C++ Stage 1 outputs with the Python reference NPZ."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


CAMERAS = ("cam2", "cam4")
FLOAT32_CROSS_BACKEND_REL_TOL = 1e-6
HG_CROSS_BACKEND_REL_TOL = 1e-6
STRONG_PASS_RELATIVE_TOL = 5e-7
NORMAL_EQUATION_PASS_RELATIVE = 1e-6
NORMAL_EQUATION_WARNING_RELATIVE = 5e-7
CAMERA_FIELDS = (
    ("R", "R", (3, 3), "geometry"),
    ("Jr", "Jr", (3, 3), "geometry"),
    ("points", "points", None, "geometry"),
    ("uv", "uv", None, "projection"),
    ("J_proj", "J_proj", None, "projection"),
    ("dpoint_dorient", "dpoint_dorient", None, "geometry"),
    ("J_orientation", "J_orientation", None, "projection"),
    ("J_translation", "J_translation", None, "projection"),
    ("J_frame", "J_frame", None, "projection"),
    ("raw_residual", "raw_residual_flat", None, "irls"),
    ("point_weights", "point_weights", None, "irls"),
    ("row_scales", "row_scales", None, "irls"),
    ("weighted_residual", "weighted_residual", None, "irls"),
    ("weighted_J", "weighted_J", None, "irls"),
)


def _meta(path: Path) -> dict[str, str]:
    result = {}
    for line in path.read_text().splitlines():
        if line and not line.startswith("#"):
            key, value = line.split("=", 1)
            result[key] = value
    return result


def _shape(field: str, count: int) -> tuple[int, ...]:
    return {
        "points": (count, 3), "uv": (count, 2), "J_proj": (count, 2, 3),
        "dpoint_dorient": (count, 3, 3),
        "J_orientation": (count, 2, 3), "J_translation": (count, 2, 3),
        "J_frame": (2 * count, 12), "raw_residual": (2 * count,),
        "point_weights": (count,), "row_scales": (2 * count,),
        "weighted_residual": (2 * count,), "weighted_J": (2 * count, 12),
    }[field]


def _metrics(reference: np.ndarray, actual: np.ndarray, category: str) -> dict[str, Any]:
    shape_match = actual.shape == reference.shape
    dtype_match = actual.dtype == reference.dtype
    finite = bool(np.isfinite(actual).all())
    if shape_match:
        difference = actual.astype(np.float64) - reference.astype(np.float64)
        max_abs = float(np.max(np.abs(difference))) if difference.size else 0.0
        mean_abs = float(np.mean(np.abs(difference))) if difference.size else 0.0
        denominator = max(float(np.linalg.norm(reference.astype(np.float64).ravel())), 1e-300)
        relative = float(np.linalg.norm(difference.ravel()) / denominator)
        exact = bool(np.array_equal(actual, reference))
    else:
        max_abs = mean_abs = relative = float("inf")
        exact = False
    if category == "geometry":
        passed = relative <= 1e-6 and max_abs <= 1e-5
    elif category == "projection":
        passed = relative <= 1e-5 and max_abs <= 1e-3
    elif category == "irls":
        passed = relative <= FLOAT32_CROSS_BACKEND_REL_TOL
    elif category == "normal_equations":
        passed = relative <= NORMAL_EQUATION_PASS_RELATIVE
    else:
        raise ValueError(category)
    warning = bool(
        category == "normal_equations"
        and relative > NORMAL_EQUATION_WARNING_RELATIVE
    )
    classification = (
        "FAIL" if not passed else
        "STRONG_PASS" if relative <= STRONG_PASS_RELATIVE_TOL else "PASS"
    )
    return {
        "shape_match": shape_match, "dtype_match": dtype_match,
        "exact": exact, "max_abs": max_abs, "mean_abs": mean_abs,
        "relative_frobenius": relative,
        "relative_l2": relative if reference.ndim == 1 else None,
        "all_finite": finite,
        "warning": warning,
        "classification": classification,
        "pass": bool(shape_match and dtype_match and finite and passed),
    }


def check_reference(
    reference_path: Path, cpp_output: Path,
    report_directory: Path | None = None,
) -> dict[str, Any]:
    output_meta = _meta(cpp_output / "cpp_output_meta.txt")
    counts = {camera: int(output_meta[f"{camera}_count"]) for camera in CAMERAS}
    comparisons: dict[str, Any] = {}
    ordered_names: list[str] = []
    with np.load(reference_path, allow_pickle=False) as reference:
        for camera in CAMERAS:
            for output_field, reference_field, fixed_shape, category in CAMERA_FIELDS:
                name = f"{camera}_{output_field}"
                shape = fixed_shape or _shape(output_field, counts[camera])
                actual = np.fromfile(cpp_output / f"{name}.f32", dtype="<f4").reshape(shape)
                comparisons[name] = _metrics(reference[f"{camera}_{reference_field}"], actual, category)
                ordered_names.append(name)
        for name, shape in (("H_frame", (12, 12)), ("g_frame", (12,))):
            actual = np.fromfile(cpp_output / f"{name}.f64", dtype="<f8").reshape(shape)
            comparisons[name] = _metrics(reference[name], actual, "normal_equations")
            ordered_names.append(name)

    first_divergence = next(
        (name for name in ordered_names if not comparisons[name]["pass"]), "NONE",
    )
    result = {
        "schema_version": "stage1_cpp_parity_v1",
        "reference": str(reference_path),
        "cpp_output": str(cpp_output),
        "comparisons": comparisons,
        "first_divergence": first_divergence,
        "pass": first_divergence == "NONE",
        "finite_difference": "NOT_RUN",
        "policy": {
            "name": "cross_backend_float32_parity_v1",
            "normal_equation_pass_relative": NORMAL_EQUATION_PASS_RELATIVE,
            "normal_equation_warning_relative": NORMAL_EQUATION_WARNING_RELATIVE,
            "float32_cross_backend_relative_tolerance": FLOAT32_CROSS_BACKEND_REL_TOL,
            "hg_cross_backend_relative_tolerance": HG_CROSS_BACKEND_REL_TOL,
            "strong_pass_relative_tolerance": STRONG_PASS_RELATIVE_TOL,
            "rationale": (
                "H/g accumulate in float64, but their weighted J/r inputs retain "
                "expected float32 cross-backend transcendental differences."
            ),
        },
        "cpp_timing_ms": {
            key: float(value) for key, value in output_meta.items() if key.endswith("_ms")
        },
    }
    report_directory = report_directory or cpp_output
    report_directory.mkdir(parents=True, exist_ok=True)
    json_path = report_directory / "stage1_cpp_parity.json"
    md_path = report_directory / "stage1_cpp_parity.md"
    json_path.write_text(json.dumps(result, indent=2) + "\n")
    rows = "\n".join(
        f"| `{name}` | {metric['exact']} | {metric['max_abs']:.6e} | "
        f"{metric['mean_abs']:.6e} | {metric['relative_frobenius']:.6e} | {metric['pass']} |"
        for name, metric in comparisons.items()
    )
    md_path.write_text(f"""# Stage 1 C++ CPU parity

| Array | Exact | Max abs | Mean abs | Relative norm | Pass |
|---|---:|---:|---:|---:|---:|
{rows}

`FIRST_DIVERGENCE={first_divergence}`

Overall pass: `{result['pass']}`. Finite difference: `NOT_RUN`.

Cross-backend H/g policy: relative norm pass `<= {NORMAL_EQUATION_PASS_RELATIVE:.1e}`;
warning when `> {NORMAL_EQUATION_WARNING_RELATIVE:.1e}`. Float64 accumulation does not
remove float32 differences already present in weighted J/r.
""")
    print(f"FIRST_DIVERGENCE={first_divergence}")
    print(f"STAGE1_CPP_PARITY={'PASS' if result['pass'] else 'FAIL'}")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--cpp_output", type=Path, required=True)
    parser.add_argument("--report_dir", type=Path, default=None)
    args = parser.parse_args(argv)
    result = check_reference(args.reference, args.cpp_output, args.report_dir)
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
