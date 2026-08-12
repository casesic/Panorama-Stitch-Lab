"""Build an input-only fixture for the C++ staged pipeline."""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage1_fixture", type=Path, required=True)
    parser.add_argument("--pose_shape_fixture", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    for source in args.stage1_fixture.iterdir():
        if source.name == "parameters.f32":
            target = args.output / "stage1_initial_parameters.f32"
        else:
            target = args.output / f"stage1_{source.name}"
        shutil.copy2(source, target)
    for source in args.pose_shape_fixture.iterdir():
        if source.name == "parameters.f32":
            continue
        shutil.copy2(source, args.output / f"model_{source.name}")
    (args.output / "pipeline_meta.txt").write_text(
        "schema_version=1\n"
        "frame=513\n"
        "stage_order=stage1,pose,pose_shape\n"
        "stage1_max_accepted_steps=20\n"
        "pose_max_accepted_steps=5\n"
        "pose_shape_max_accepted_steps=1\n"
        "lm_max_trials=8\n"
        "rel_objective_tolerance=1e-7\n"
        "scaled_gradient_tolerance=1e-5\n"
        "step_tolerance=1e-5\n"
        "reference_leakage=NONE\n"
    )
    print("STAGED_LM_PIPELINE_FIXTURE_WRITTEN")


if __name__ == "__main__":
    main()
