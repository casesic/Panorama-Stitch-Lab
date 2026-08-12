"""Export input-only raw binaries for the standalone Pose value reference."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


FLOAT_INPUTS = (
    "parameters", "joints", "body_pose_anchor",
    "cam2_left_hand_pose", "cam2_right_hand_pose", "cam2_scale",
    "cam2_selected_v_shaped", "cam2_selected_posedirs",
    "cam2_selected_lbs_weights", "cam2_targets",
    "cam4_left_hand_pose", "cam4_right_hand_pose", "cam4_scale",
    "cam4_selected_v_shaped", "cam4_selected_posedirs",
    "cam4_selected_lbs_weights", "cam4_targets",
)
INTEGER_INPUTS = ("parents",)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def export_pose_fixture(reference_directory: Path, output_directory: Path | None = None) -> Path:
    manifest = json.loads((reference_directory / "pose_reference_manifest.json").read_text())
    if manifest["stage"] != "STAGE2_POSE_VALUE" or manifest["parameter_dim"] != 75:
        raise ValueError("UNSUPPORTED_POSE_VALUE_REFERENCE")
    output_directory = output_directory or reference_directory / "cpp_fixture"
    output_directory.mkdir(parents=True, exist_ok=True)
    hashes = {}
    with np.load(reference_directory / "pose_reference.npz", allow_pickle=False) as arrays:
        for name in FLOAT_INPUTS:
            path = output_directory / f"{name}.f32"
            path.write_bytes(np.ascontiguousarray(arrays[name], dtype="<f4").tobytes())
            hashes[path.name] = _sha256(path)
        for name in INTEGER_INPUTS:
            path = output_directory / f"{name}.i64"
            path.write_bytes(np.ascontiguousarray(arrays[name], dtype="<i8").tobytes())
            hashes[path.name] = _sha256(path)
    metadata = {
        "schema_version": "1", "frame": str(manifest["frame"]),
        "parameter_dim": "75", "joint_count": str(manifest["joint_count"]),
        "pose_feature_dim": str(manifest["pose_feature_dim"]),
        "image_width": str(manifest["image_width"]),
        "image_height": str(manifest["image_height"]),
        "cam2_count": str(manifest["cameras"]["cam2"]["point_count"]),
        "cam4_count": str(manifest["cameras"]["cam4"]["point_count"]),
        "joint_layout": manifest["joint_layout"],
        "binary_endianness": "little", "float_dtype": "float32",
        "integer_dtype": "int64", "matrix_storage": "canonical_row_major",
        "objective_mode": manifest["normal_equations"]["objective_mode"],
        "charbonnier_eps": str(manifest["normal_equations"]["charbonnier_eps"]),
        "pose_anchor_scale": str(manifest["normal_equations"]["pose_anchor_scale"]),
        "cam2_alpha_per_point": str(
            manifest["normal_equations"]["aggregation_spec"]["terms"]
            [f"{manifest['frame']}|cam2"]["alpha_per_point"]
        ),
        "cam4_alpha_per_point": str(
            manifest["normal_equations"]["aggregation_spec"]["terms"]
            [f"{manifest['frame']}|cam4"]["alpha_per_point"]
        ),
        "group_order": ",".join(manifest["normal_equations"]["group_order"]),
        "residual_flatten_order": manifest["normal_equations"]["residual_flatten_order"],
        "lm_max_accepted_steps": "5",
        "lm_max_trials": "8",
        "lm_rel_objective_tolerance": "1e-7",
        "lm_scaled_gradient_tolerance": "1e-5",
        "lm_step_tolerance": "1e-5",
    }
    metadata.update({f"sha256_{name}": digest for name, digest in hashes.items()})
    (output_directory / "fixture_meta.txt").write_text(
        "".join(f"{key}={value}\n" for key, value in metadata.items())
    )
    print(f"[POSE_CPP_FIXTURE] output={output_directory} files={len(hashes)}")
    for name, digest in hashes.items():
        print(f"  {name} sha256={digest}")
    return output_directory


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, default=None)
    args = parser.parse_args(argv)
    export_pose_fixture(args.reference_dir, args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
