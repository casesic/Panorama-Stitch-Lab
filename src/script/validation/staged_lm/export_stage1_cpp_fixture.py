"""Export only the raw Stage 1 inputs needed by the standalone C++ reference."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


INPUT_KEYS = (
    "parameters",
    "cam2_local_vertices", "cam2_rotation_center", "cam2_targets",
    "cam4_local_vertices", "cam4_rotation_center", "cam4_targets",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def export_fixture(reference_directory: Path, output_directory: Path | None = None) -> Path:
    npz_path = reference_directory / "stage1_reference.npz"
    manifest_path = reference_directory / "stage1_reference_manifest.json"
    output_directory = output_directory or reference_directory / "cpp_fixture"
    manifest = json.loads(manifest_path.read_text())
    if manifest["stage"] != "STAGE1_RIGID" or manifest["parameter_dim"] != 12:
        raise ValueError("UNSUPPORTED_STAGE1_REFERENCE")
    config = manifest["config"]
    if config["objective"] != "charbonnier_irls":
        raise ValueError("CPP_REFERENCE_REQUIRES_CHARBONNIER_IRLS")
    output_directory.mkdir(parents=True, exist_ok=True)

    hashes: dict[str, str] = {}
    with np.load(npz_path, allow_pickle=False) as arrays:
        for name in INPUT_KEYS:
            array = np.ascontiguousarray(arrays[name], dtype="<f4")
            path = output_directory / f"{name}.f32"
            path.write_bytes(array.tobytes(order="C"))
            hashes[path.name] = _sha256(path)

    terms = config["aggregation_spec"]["terms"]
    metadata = {
        "schema_version": "1",
        "frame": str(manifest["frame"]),
        "cam2_count": str(manifest["cameras"]["cam2"]["point_count"]),
        "cam4_count": str(manifest["cameras"]["cam4"]["point_count"]),
        "parameter_dim": str(manifest["parameter_dim"]),
        "image_width": str(manifest["image_width"]),
        "image_height": str(manifest["image_height"]),
        "reprojection_objective": config["objective"],
        "irls_epsilon": repr(config["charbonnier_eps"]),
        "camera_weight": repr(terms[f"{manifest['frame']}|cam2"]["camera_weight"]),
        "cam2_alpha_per_point": repr(terms[f"{manifest['frame']}|cam2"]["alpha_per_point"]),
        "cam4_alpha_per_point": repr(terms[f"{manifest['frame']}|cam4"]["alpha_per_point"]),
        "binary_endianness": "little",
        "binary_dtype": "float32",
        "row_order": "cam2_then_cam4",
    }
    for filename, digest in hashes.items():
        metadata[f"sha256_{filename}"] = digest
    meta_path = output_directory / "fixture_meta.txt"
    meta_path.write_text("".join(f"{key}={value}\n" for key, value in metadata.items()))
    print(f"[STAGE1_CPP_FIXTURE] output={output_directory}")
    for filename, digest in hashes.items():
        print(f"  {filename} sha256={digest}")
    return output_directory


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, default=None)
    args = parser.parse_args(argv)
    export_fixture(args.reference_dir, args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
