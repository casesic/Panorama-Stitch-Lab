"""Production DenseKP observation export for the C++ staged-LM runtime.

This module only normalizes observations and injects them into an already
prepared 204-support C++ model bundle. It never runs SMPL-X or Python LM.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from lib_pano.fitting_io.densekp_delivery_io import (
    DenseKPDeliveryObservation,
    load_densekp_delivery_observation,
)


DEFAULT_ASSETS = PROJECT_ROOT / "cpp/staged_lm/assets"


def _metadata(path: Path) -> dict[str, str]:
    return dict(
        line.split("=", 1) for line in path.read_text(encoding="utf-8").splitlines()
        if line and not line.startswith("#")
    )


def _write_metadata(path: Path, values: dict[str, str]) -> None:
    path.write_text(
        "".join(f"{key}={value}\n" for key, value in sorted(values.items())),
        encoding="utf-8",
    )


def write_runtime_observation(
    path: Path, observation: DenseKPDeliveryObservation,
) -> None:
    pairs = np.column_stack(
        (observation.points_cam2, observation.points_cam4)
    )
    payload = {
        "coord_space": "erp_rotate",
        "point_ids": observation.point_ids.astype(int).tolist(),
        "pairs": pairs.astype(np.float32).tolist(),
        "valid": observation.valid.astype(np.uint8).tolist(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def materialize_runtime_fixture(
    base_fixture: Path,
    output_fixture: Path,
    observation: DenseKPDeliveryObservation,
    assets_directory: Path = DEFAULT_ASSETS,
    *,
    link_static_files: bool = False,
) -> None:
    """Clone a 204-support model bundle and inject DenseKP observations.

    The forward model always evaluates 204 supporting SMPL-X vertices.  The
    observation operator regresses those vertices to at most 138 valid DenseKP
    semantic rows; these counts describe different spaces and must not be
    conflated.
    """
    if output_fixture.exists():
        raise FileExistsError(f"runtime fixture already exists: {output_fixture}")
    if link_static_files:
        shutil.copytree(
            base_fixture, output_fixture,
            copy_function=lambda source, destination: os.symlink(
                Path(source).resolve(), destination,
            ),
        )
    else:
        shutil.copytree(base_fixture, output_fixture)

    regressor = np.fromfile(
        assets_directory / "densekp_regressor_138x204.f32", dtype="<f4",
    ).reshape(138, 204)
    support_ids = np.fromfile(
        assets_directory / "densekp_support_vertex_ids.i32", dtype="<i4",
    )
    if support_ids.shape != (204,):
        raise RuntimeError("invalid DenseKP support asset")

    stage_meta_path = output_fixture / "stage1_fixture_meta.txt"
    model_meta_path = output_fixture / "model_fixture_meta.txt"
    stage_meta = _metadata(stage_meta_path)
    model_meta = _metadata(model_meta_path)
    if any(int(model_meta[f"{camera}_count"]) != 204 for camera in ("cam2", "cam4")):
        raise RuntimeError("production base fixture must contain 204 support vertices")
    densekp_ready = model_meta.get("observation_mode") == "densekp"
    expected_stage_count = 138 if densekp_ready else 204
    if any(
        int(stage_meta[f"{camera}_count"]) != expected_stage_count
        for camera in ("cam2", "cam4")
    ):
        raise RuntimeError(
            "production base fixture Stage1 must contain either 204 support "
            "points or all 138 DenseKP points"
        )

    valid_rows = np.flatnonzero(observation.valid)
    densekp_ids = np.ascontiguousarray(
        observation.point_ids[valid_rows], dtype="<i4",
    )
    if densekp_ids.size == 0 or np.any(densekp_ids < 0) or np.any(densekp_ids >= 138):
        raise RuntimeError("invalid runtime DenseKP IDs")

    for camera, points in (
        ("cam2", observation.points_cam2),
        ("cam4", observation.points_cam4),
    ):
        targets = np.ascontiguousarray(points[valid_rows], dtype="<f4")
        local_path = output_fixture / f"stage1_{camera}_local_vertices.f32"
        base_local = np.fromfile(local_path, dtype="<f4").reshape(
            expected_stage_count, 3
        )
        densekp_local = np.ascontiguousarray(
            base_local[densekp_ids]
            if densekp_ready else regressor[densekp_ids] @ base_local,
            dtype="<f4",
        )
        if local_path.is_symlink():
            local_path.unlink()
        densekp_local.tofile(local_path)
        for name in (
            f"stage1_{camera}_targets.f32",
            f"model_{camera}_targets.f32",
            f"model_{camera}_densekp_ids.i32",
        ):
            path = output_fixture / name
            if path.is_symlink():
                path.unlink()
        targets.tofile(output_fixture / f"stage1_{camera}_targets.f32")
        targets.tofile(output_fixture / f"model_{camera}_targets.f32")
        densekp_ids.tofile(output_fixture / f"model_{camera}_densekp_ids.i32")
        stage_meta[f"{camera}_count"] = str(densekp_ids.size)
        stage_meta[f"{camera}_alpha_per_point"] = repr(1.0 / densekp_ids.size)
        model_meta[f"{camera}_support_count"] = "204"
        model_meta[f"{camera}_observation_count"] = str(densekp_ids.size)
        model_meta[f"{camera}_alpha"] = repr(1.0 / densekp_ids.size)

    model_meta.update({
        "observation_mode": "densekp",
        "densekp_support_count": "204",
        "densekp_row_count": "138",
    })
    for path in (stage_meta_path, model_meta_path):
        if path.is_symlink():
            path.unlink()
    _write_metadata(stage_meta_path, stage_meta)
    _write_metadata(model_meta_path, model_meta)
    for name in (
        "densekp_support_vertex_ids.i32", "densekp_regressor_138x204.f32",
    ):
        destination = output_fixture / name
        if destination.is_symlink():
            destination.unlink()
        shutil.copyfile(assets_directory / name, destination)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--densekp-observation", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--camera-group", choices=("old", "new"), default="old")
    parser.add_argument("--base-fixture", type=Path)
    parser.add_argument("--output-fixture", type=Path)
    parser.add_argument("--assets-directory", type=Path, default=DEFAULT_ASSETS)
    args = parser.parse_args()
    if (args.base_fixture is None) != (args.output_fixture is None):
        parser.error("--base-fixture and --output-fixture must be provided together")

    observation = load_densekp_delivery_observation(
        args.densekp_observation, camera_group=args.camera_group,
    )
    write_runtime_observation(args.output_json, observation)
    if args.base_fixture is not None:
        materialize_runtime_fixture(
            args.base_fixture, args.output_fixture, observation,
            args.assets_directory,
        )
    print("DENSEKP_CPP_RUNTIME_INPUT_READY=PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
