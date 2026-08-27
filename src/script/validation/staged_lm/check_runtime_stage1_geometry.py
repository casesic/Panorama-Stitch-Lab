"""Compare stored/Python Stage1 geometry with C++ runtime preparation dumps."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _stats(reference: np.ndarray, actual: np.ndarray) -> dict[str, object]:
    if reference.shape != actual.shape:
        return {"shape_reference": list(reference.shape), "shape_actual": list(actual.shape),
                "finite": False, "pass": False, "reason": "shape_mismatch"}
    difference = actual.astype(np.float64) - reference.astype(np.float64)
    denominator = max(float(np.linalg.norm(reference.astype(np.float64))), 1e-30)
    return {
        "shape": list(reference.shape),
        "max_abs": float(np.max(np.abs(difference), initial=0.0)),
        "mean_abs": float(np.mean(np.abs(difference))),
        "rel_frob": float(np.linalg.norm(difference) / denominator),
        "finite": bool(np.isfinite(reference).all() and np.isfinite(actual).all()),
    }


def _load(path: Path, columns: int) -> np.ndarray:
    value = np.fromfile(path, dtype="<f4")
    return value.reshape(-1, columns) if columns > 1 else value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--cpp-debug", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--python-oracle", type=Path)
    parser.add_argument("--rel-tol", type=float, default=2e-6)
    parser.add_argument("--abs-tol", type=float, default=2e-6)
    args = parser.parse_args()

    report: dict[str, object] = {"cameras": {}}
    all_pass = True
    python_values = np.load(args.python_oracle) if args.python_oracle else None
    pipeline_metadata = dict(
        line.split("=", 1)
        for line in (args.fixture / "pipeline_meta.txt").read_text().splitlines()
        if line
    )
    person_runtime = pipeline_metadata.get("input_mode") == "person_runtime_v1"
    report["fixture_mode"] = pipeline_metadata.get("input_mode", "legacy")
    for camera in ("cam2", "cam4"):
        stored_local = _load(
            args.fixture / f"stage1_{camera}_local_vertices.f32", 3)
        cpp_local = _load(
            args.cpp_debug / f"{camera}_stage1_local_vertices.f32", 3)
        stored_center = _load(
            args.fixture / f"stage1_{camera}_rotation_center.f32", 1)
        cpp_center = _load(args.cpp_debug / f"{camera}_rotation_center.f32", 1)
        values = {}
        if not person_runtime:
            values.update({
                "stored_vs_cpp_stage1_local_vertices": _stats(stored_local, cpp_local),
                "stored_vs_cpp_rotation_center": _stats(stored_center, cpp_center),
            })
        if python_values is not None:
            for quantity, filename in (
                ("selected_vertices", "selected_vertices"),
                ("support_local_vertices", "support_local_vertices"),
                ("rotation_center", "rotation_center"),
            ):
                cpp = _load(args.cpp_debug / f"{camera}_{filename}.f32",
                            3 if quantity != "rotation_center" else 1)
                values[f"python_vs_cpp_{quantity}"] = _stats(
                    python_values[f"{camera}_{quantity}"], cpp)
        for value in values.values():
            passed = bool(value.get("finite", False)) and (
                float(value.get("rel_frob", float("inf"))) <= args.rel_tol
                or float(value.get("max_abs", float("inf"))) <= args.abs_tol
            )
            value["pass"] = passed
            all_pass &= passed
        report["cameras"][camera] = values
    report.update({"rel_tolerance": args.rel_tol, "abs_tolerance": args.abs_tol,
                   "pass": all_pass})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"RUNTIME_STAGE1_GEOMETRY_MATCH={'PASS' if all_pass else 'FAIL'}")
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
