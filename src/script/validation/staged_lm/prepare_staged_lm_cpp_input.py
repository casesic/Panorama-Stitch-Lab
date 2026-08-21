"""Prepare a single-frame input-only fixture for the C++ staged-LM pipeline."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[4]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frame", type=int, required=True)
    parser.add_argument("--hmr_dir", type=Path, required=True)
    parser.add_argument("--sam3d_dense_dir", type=Path, required=True)
    parser.add_argument("--bbox_dir", type=Path, required=True)
    parser.add_argument("--output_root", type=Path, required=True)
    parser.add_argument("--surface_sample_ids", type=Path, default=None)
    parser.add_argument("--densekp_dir", type=Path, default=None)
    parser.add_argument("--densekp_pairs_json", type=Path, default=None)
    parser.add_argument("--camera_group", choices=("old", "new"), default="old")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--image_width", type=int, default=5760)
    parser.add_argument("--image_height", type=int, default=2880)
    parser.add_argument("--stage1_max_accepted_steps", type=int, default=20)
    parser.add_argument("--pose_max_accepted_steps", type=int, default=5)
    parser.add_argument("--pose_shape_max_accepted_steps", type=int, default=1)
    parser.add_argument("--lm_max_trials", type=int, default=8)
    return parser.parse_args()


def run(command: list[str]) -> None:
    print("[RUN]", " ".join(command), flush=True)
    subprocess.run(command, cwd=REPOSITORY_ROOT, check=True)


def main() -> int:
    args = parse_args()
    scripts = REPOSITORY_ROOT / "src/script/validation/staged_lm"
    frame_directory = f"frame_{args.frame:06d}"
    stage1_reference = args.output_root / "stage1" / frame_directory
    pose_shape_reference = args.output_root / "pose_shape" / frame_directory
    stage1_fixture = stage1_reference / "cpp_fixture"
    pose_shape_fixture = pose_shape_reference / "cpp_fixture"
    pipeline_fixture = (
        args.output_root / "staged_lm_pipeline" / frame_directory / "cpp_fixture"
    )

    common = [
        "--frame", str(args.frame),
        "--hmr_dir", str(args.hmr_dir),
        "--sam3d_dense_dir", str(args.sam3d_dense_dir),
        "--bbox_dir", str(args.bbox_dir),
        "--device", args.device,
        "--image_width", str(args.image_width),
        "--image_height", str(args.image_height),
        "--reproj_objective", "charbonnier_irls",
    ]
    surface_sample_ids = args.surface_sample_ids
    densekp_requested = (
        args.densekp_dir is not None or args.densekp_pairs_json is not None
    )
    if args.densekp_dir is not None and args.densekp_pairs_json is not None:
        raise ValueError("DenseKP directory and delivery JSON are mutually exclusive")
    if densekp_requested:
        if surface_sample_ids is not None:
            raise ValueError("--densekp_dir and --surface_sample_ids are mutually exclusive")
        surface_sample_ids = (
            REPOSITORY_ROOT / "cpp/staged_lm/assets/densekp_support_vertex_ids.npy"
        )
        common.append("--cpp_include_all_surface_samples")
    if surface_sample_ids is not None:
        common.extend(("--surface_sample_ids", str(surface_sample_ids)))

    run([
        sys.executable, str(scripts / "dump_stage1_cpp_reference.py"),
        *common, "--output_dir", str(stage1_reference),
    ])
    run([
        sys.executable, str(scripts / "export_stage1_cpp_fixture.py"),
        "--reference_dir", str(stage1_reference),
        "--output_dir", str(stage1_fixture),
    ])
    run([
        sys.executable, str(scripts / "dump_pose_shape_cpp_reference.py"),
        *common, "--output_dir", str(pose_shape_reference),
    ])
    run([
        sys.executable, str(scripts / "export_pose_shape_cpp_fixture.py"),
        "--reference_dir", str(pose_shape_reference),
        "--output_dir", str(pose_shape_fixture),
    ])
    run([
        sys.executable, str(scripts / "export_staged_lm_cpp_pipeline_fixture.py"),
        "--stage1_fixture", str(stage1_fixture),
        "--pose_shape_fixture", str(pose_shape_fixture),
        "--output", str(pipeline_fixture),
        "--stage1_max_accepted_steps", str(args.stage1_max_accepted_steps),
        "--pose_max_accepted_steps", str(args.pose_max_accepted_steps),
        "--pose_shape_max_accepted_steps", str(args.pose_shape_max_accepted_steps),
        "--lm_max_trials", str(args.lm_max_trials),
    ])
    if densekp_requested:
        command = [
            sys.executable,
            str(scripts / "convert_cpp_fixture_to_densekp.py"),
            "--fixture", str(pipeline_fixture),
            "--frame", str(args.frame),
            "--image-width", str(args.image_width),
            "--image-height", str(args.image_height),
        ]
        if args.densekp_pairs_json is not None:
            command.extend((
                "--densekp-pairs-json", str(args.densekp_pairs_json),
                "--camera-group", args.camera_group,
            ))
        else:
            command.extend((
                "--densekp-dir", str(args.densekp_dir),
                "--bbox-dir", str(args.bbox_dir),
            ))
        run(command)
    print(f"[PASS] STAGED_LM_CPP_INPUT_READY frame={args.frame}")
    print(f"[FIXTURE] {pipeline_fixture}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

r'''
python src/script/validation/staged_lm/prepare_staged_lm_cpp_input.py \
  --frame 514 \
  --hmr_dir data_process/Output/20260311_riding/hmr_cache \
  --sam3d_dense_dir data_process/Output/20260311_riding/sam3dbody/smplx_dense \
  --bbox_dir data_process/Output/20260311_riding/image/equirect_crop_rotate \
  --surface_sample_ids src/lib_pano/human3d/assets/smplx_surface_ids_region_fps_100.npy \
  --output_root data_process/Vis/cpp_reference_dump_s100\
  --device cpu
'''
