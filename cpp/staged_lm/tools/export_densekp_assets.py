#!/usr/bin/env python3
"""Export the fixed 204-support DenseKP operator used by C++ staged-LM."""

from __future__ import annotations

import argparse
import hashlib
import pickle
from pathlib import Path

import numpy as np


EXPECTED_SHA256 = {
    "downsample_mat.pkl": "088237aa367ce47dbcc3643d8e16d0322a5f7c7754c09d1ccff6f2d60f186889",
    "smplx2smpl.pkl": "c1d912d121ad98132e4492d8e7a0f1a8cf4412811e14a7ef8cb337bb48eef99e",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-dir", type=Path,
        default=Path("tools/CameraHMR/data/train-eval-utils"),
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("cpp/staged_lm/assets"),
    )
    args = parser.parse_args()
    downsample_path = args.source_dir / "downsample_mat.pkl"
    mapping_path = args.source_dir / "smplx2smpl.pkl"
    for path in (downsample_path, mapping_path):
        actual = sha256(path)
        if actual != EXPECTED_SHA256[path.name]:
            raise RuntimeError(f"SHA-256 mismatch for {path}: {actual}")

    cached_full = args.source_dir / "densekp_regressor_R.npy"
    if cached_full.exists():
        full = np.asarray(np.load(cached_full), dtype=np.float32)
    else:
        try:
            with downsample_path.open("rb") as stream:
                downsample = pickle.load(stream, encoding="latin1")
            with mapping_path.open("rb") as stream:
                mapping = pickle.load(stream, encoding="latin1")["matrix"]
        except ModuleNotFoundError as error:
            raise RuntimeError(
                "loading the original pickle requires its PyTorch environment; "
                "generate densekp_regressor_R.npy with load_densekp_regressor() first"
            ) from error
        if hasattr(downsample, "to_dense"):
            downsample = downsample.to_dense().cpu().numpy()
        if hasattr(mapping, "cpu"):
            mapping = mapping.cpu().numpy()
        full = np.asarray(downsample @ mapping, dtype=np.float32)
    support_ids = np.flatnonzero(np.any(full != 0.0, axis=0)).astype("<i4")
    reduced = np.ascontiguousarray(full[:, support_ids], dtype="<f4")
    if full.shape != (138, 10475) or support_ids.shape != (204,):
        raise RuntimeError(
            f"unexpected DenseKP shapes full={full.shape} support={support_ids.shape}"
        )

    rng = np.random.default_rng(20260819)
    vertices = rng.standard_normal((10475, 3), dtype=np.float32)
    lhs = full @ vertices
    rhs = reduced @ vertices[support_ids]
    if not np.array_equal(lhs, rhs):
        max_abs = float(np.max(np.abs(lhs - rhs)))
        if max_abs > 1e-6:
            raise RuntimeError(f"reduced regressor parity failed: max_abs={max_abs}")
    model_path = Path("tools/smpl_models/smplx/SMPLX_NEUTRAL.npz")
    real_vertices = np.asarray(np.load(model_path)["v_template"], dtype=np.float32)
    real_lhs = full @ real_vertices
    real_rhs = reduced @ real_vertices[support_ids]
    real_max_abs = float(np.max(np.abs(real_lhs - real_rhs)))
    if real_max_abs > 1e-6:
        raise RuntimeError(
            f"real SMPL-X reduced regressor parity failed: max_abs={real_max_abs}"
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    ids_path = args.output_dir / "densekp_support_vertex_ids.i32"
    regressor_path = args.output_dir / "densekp_regressor_138x204.f32"
    support_ids.tofile(ids_path)
    reduced.tofile(regressor_path)
    np.save(args.output_dir / "densekp_support_vertex_ids.npy", support_ids)
    print(f"DENSEKP_SUPPORT_VERTEX_COUNT={support_ids.size}")
    print(f"DENSEKP_REDUCED_REGRESSOR_MAX_ABS={np.max(np.abs(lhs-rhs)):.9e}")
    print(f"DENSEKP_REAL_SMPLX_REGRESSOR_MAX_ABS={real_max_abs:.9e}")
    print("DENSEKP_REDUCED_REGRESSOR_PARITY=PASS")
    print(f"DENSEKP_SUPPORT_IDS_SHA256={sha256(ids_path)}")
    print(f"DENSEKP_REGRESSOR_SHA256={sha256(regressor_path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
