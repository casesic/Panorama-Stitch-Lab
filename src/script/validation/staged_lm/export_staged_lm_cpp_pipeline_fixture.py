"""Build an input-only fixture for the C++ staged pipeline."""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def read_metadata(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        key, separator, value = line.partition("=")
        if separator:
            values[key.strip()] = value.strip()
    return values


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage1_fixture", type=Path, required=True)
    parser.add_argument("--pose_shape_fixture", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stage1_max_accepted_steps", type=int, default=20)
    parser.add_argument("--pose_max_accepted_steps", type=int, default=5)
    parser.add_argument("--pose_shape_max_accepted_steps", type=int, default=1)
    parser.add_argument("--lm_max_trials", type=int, default=8)
    parser.add_argument("--rel_objective_tolerance", type=float, default=1e-7)
    parser.add_argument("--scaled_gradient_tolerance", type=float, default=1e-5)
    parser.add_argument("--step_tolerance", type=float, default=1e-5)
    args = parser.parse_args()
    stage1_meta = read_metadata(args.stage1_fixture / "fixture_meta.txt")
    pose_shape_meta = read_metadata(args.pose_shape_fixture / "fixture_meta.txt")
    frame = int(stage1_meta["frame"])
    pose_shape_frame = int(pose_shape_meta.get("frame", frame))
    if pose_shape_frame != frame:
        raise ValueError(
            "PIPELINE_FIXTURE_FRAME_MISMATCH: "
            f"stage1={frame} pose_shape={pose_shape_frame}"
        )
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
        f"frame={frame}\n"
        "stage_order=stage1,pose,pose_shape\n"
        f"stage1_max_accepted_steps={args.stage1_max_accepted_steps}\n"
        f"pose_max_accepted_steps={args.pose_max_accepted_steps}\n"
        f"pose_shape_max_accepted_steps={args.pose_shape_max_accepted_steps}\n"
        f"lm_max_trials={args.lm_max_trials}\n"
        f"rel_objective_tolerance={args.rel_objective_tolerance}\n"
        f"scaled_gradient_tolerance={args.scaled_gradient_tolerance}\n"
        f"step_tolerance={args.step_tolerance}\n"
        "reference_leakage=NONE\n"
    )
    print("STAGED_LM_PIPELINE_FIXTURE_WRITTEN")


if __name__ == "__main__":
    main()
