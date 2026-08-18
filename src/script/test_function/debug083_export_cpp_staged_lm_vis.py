"""Convert one C++ staged-LM result to the pickle layout used by debug030.

This is deliberately an adapter rather than a second visualizer: the C++
runner stores only the optimized flat parameter vectors, while debug030 uses
the repository's standard ``SMPLX_Param_Seq`` representation.  Hand poses and
camera scales are frozen during staged LM and are therefore restored from the
C++ fixture that produced the result.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import config_script  # noqa: F401  (initialises repository import paths)

from lib_pano.fitting_io.fitting_result_io import save_segment_result
from lib_pano.human_fitting.smplx_sequence import SMPLX_Seq_Forward_Module


FRAME_PARAMETER_DIM = 75
BETA_DIM = 10
POSE_SHAPE_PARAMETER_DIM = FRAME_PARAMETER_DIM + BETA_DIM


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export a single-frame C++ staged-LM result for debug030 visualization.",
    )
    parser.add_argument(
        "--cpp_output", type=Path, required=True,
        help="Directory containing pose_shape_final_parameters.f32.",
    )
    parser.add_argument(
        "--fixture", type=Path, required=True,
        help="cpp_fixture directory used by the C++ run.",
    )
    parser.add_argument(
        "--output_dir", type=Path, required=True,
        help="Output root; cam2/ and cam4/ are created below it.",
    )
    parser.add_argument(
        "--frame", type=int, default=None,
        help="Image frame number. By default read fixture/pipeline_meta.txt.",
    )
    return parser.parse_args()


def read_f32(path: Path, expected_count: int) -> torch.Tensor:
    if not path.is_file():
        raise FileNotFoundError(path)
    values = np.fromfile(path, dtype="<f4")
    if values.size != expected_count:
        raise ValueError(
            f"{path}: expected {expected_count} float32 values, got {values.size}",
        )
    return torch.from_numpy(values.copy())


def read_frame_from_fixture(fixture: Path) -> int:
    meta_path = fixture / "pipeline_meta.txt"
    if not meta_path.is_file():
        raise FileNotFoundError(
            f"Cannot infer --frame because {meta_path} does not exist",
        )
    fields: dict[str, str] = {}
    for line in meta_path.read_text(encoding="utf-8").splitlines():
        key, separator, value = line.partition("=")
        if separator:
            fields[key.strip()] = value.strip()
    if "frame" not in fields:
        raise ValueError(f"Missing frame=... in {meta_path}")
    return int(fields["frame"])


def build_camera_sequence(
    parameters: torch.Tensor,
    beta: torch.Tensor,
    fixture: Path,
    camera: str,
) -> object:
    camera_offset = 0 if camera == "cam2" else 6
    sequence = SMPLX_Seq_Forward_Module.SMPLX_Param_Seq(
        num_batch=1, gender="neutral",
    )
    sequence.orient[0] = parameters[camera_offset:camera_offset + 3]
    sequence.transl[0] = parameters[camera_offset + 3:camera_offset + 6]
    sequence.body_pose[0] = parameters[12:75]
    sequence.left_hand_pose[0] = read_f32(
        fixture / f"model_{camera}_left_hand_pose.f32", 45,
    )
    sequence.right_hand_pose[0] = read_f32(
        fixture / f"model_{camera}_right_hand_pose.f32", 45,
    )
    sequence.beta = beta.clone()
    sequence.scale = read_f32(fixture / f"model_{camera}_scale.f32", 1)
    return sequence


def main() -> int:
    args = parse_args()
    frame = args.frame if args.frame is not None else read_frame_from_fixture(args.fixture)

    parameters = read_f32(
        args.cpp_output / "pose_shape_final_parameters.f32",
        POSE_SHAPE_PARAMETER_DIM,
    )
    beta_path = args.cpp_output / "final_physical_beta.f32"
    beta = read_f32(beta_path, BETA_DIM) if beta_path.is_file() else 2.5 * torch.tanh(parameters[75:85])

    cam2 = build_camera_sequence(parameters, beta, args.fixture, "cam2")
    cam4 = build_camera_sequence(parameters, beta, args.fixture, "cam4")
    save_segment_result(
        result_c2=cam2,
        result_c4=cam4,
        shared_beta=beta,
        output_dir=args.output_dir,
        frame_idxs=[frame],
        valid_frame_count_c2=1,
        valid_frame_count_c4=1,
    )

    for camera in ("cam2", "cam4"):
        meta_path = args.output_dir / camera / f"seg_{frame}_{frame}_meta.json"
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        metadata.update({
            "export_source": "cpp_staged_lm",
            "cpp_output": str(args.cpp_output.resolve()),
            "cpp_fixture": str(args.fixture.resolve()),
        })
        meta_path.write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    print(f"[PASS] CPP_STAGED_LM_VIS_EXPORT frame={frame} output={args.output_dir}")
    print(
        "[NEXT] python src/script/test_function/debug030_vis_dual_erp.py "
        f"--result_dir {args.output_dir} --segment {frame}_{frame} "
        "--image_dir <ERP_IMAGE_ROOT> --output_dir <VIS_OUTPUT> --mesh",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
