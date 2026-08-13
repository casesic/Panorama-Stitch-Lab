"""Export real frame-513 pose-offset inputs for the experimental GEMM bakeoff."""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np


FEATURE_TOLERANCE = 2e-6  # Imported policy value documented by current Jacobian checkers.


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--model_fixture", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    arrays = np.load(args.reference / "pose_shape_reference.npz")
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": "pose_offset_gemm_bakeoff_v1",
        "frame": 513, "feature_tolerance": FEATURE_TOLERANCE,
        "contexts": {}, "reference_leakage": "validation_fixture_only",
    }
    for stage in ("pose", "pose_shape"):
        for camera, count in (("cam2", 1023), ("cam4", 927)):
            context = f"{stage}_{camera}"
            target = args.output / context
            target.mkdir(exist_ok=True)
            shutil.copy2(
                args.model_fixture / f"{camera}_selected_posedirs.f32",
                target / "selected_posedirs.f32",
            )
            np.asarray(arrays[f"{camera}_d_pose_feature"], "<f4").tofile(
                target / "d_pose_feature.f32"
            )
            np.asarray(arrays[f"{camera}_d_pose_offsets_pose"], "<f4").tofile(
                target / "python_d_pose_offsets.f32"
            )
            manifest["contexts"][context] = {
                "point_count": count,
                "boundary_state": "pose_final_equals_pose_shape_initial",
                "posedirs_shape": [486, count, 3],
                "d_pose_feature_shape": [486, 66],
                "output_shape": [count, 3, 66],
            }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print("POSE_OFFSET_GEMM_FIXTURE_WRITTEN")


if __name__ == "__main__":
    main()
