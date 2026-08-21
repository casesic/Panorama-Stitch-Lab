"""Compare frame-local and persistent DenseKP ORT runtimes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time


ROOT = Path(__file__).resolve().parents[3]
DELIVERY_ROOT = ROOT / "Circular_Video/01_recv/20260708"
if str(DELIVERY_ROOT) not in sys.path:
    sys.path.insert(0, str(DELIVERY_ROOT))

from densekp_delivery.cli import _parse_args as parse_delivery_args  # noqa: E402
from densekp_delivery.pipeline import run_pipeline  # noqa: E402
from densekp_delivery.runtime import DenseKPRuntime  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", type=int, nargs="+", required=True)
    parser.add_argument("--fisheye_root", type=Path, required=True)
    parser.add_argument("--onnx_path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--calibration_cache_dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--camera", choices=("old", "new"), default="old")
    return parser.parse_args()


def delivery_args(args: argparse.Namespace, frame: int, output: Path):
    return parse_delivery_args([
        "--backend", "ort",
        "--onnx_path", str(args.onnx_path),
        "--fisheye_cam2", str(args.fisheye_root / "cam2" / f"{frame:05d}.jpg"),
        "--fisheye_cam4", str(args.fisheye_root / "cam4" / f"{frame:05d}.jpg"),
        "--out_dir", str(output),
        "--calibration_cache_dir", str(args.calibration_cache_dir),
        "--cache_mode", "read_write",
        "--output_coord_space", "erp_rotate",
        "--vis_option", "none",
        "--save_option", "none",
        "--device", args.device,
        "--camera", args.camera,
        "--profile_runtime",
    ])


INFERENCE_PROFILE_KEYS = {
    "preprocess_ms", "detector_ms", "bbox_postprocess_ms",
    "densekp_model_ms", "decode_ms", "coordinate_ms",
}


def read_outputs(root: Path, frame: int) -> tuple[bytes, dict, dict]:
    result = root / f"frame_{frame:05d}" / "result"
    return (
        (result / "densekp_pairs_erp_rotate.json").read_bytes(),
        json.loads((result / "densekp_runtime_profile.json").read_text()),
        json.loads((result / "densekp_inference_profile.json").read_text()),
    )


def main() -> int:
    args = parse_args()
    old_root = args.output / "frame_local"
    persistent_root = args.output / "persistent"
    old_wall_ms = []
    for frame in args.frames:
        frame_args = delivery_args(args, frame, old_root / f"frame_{frame:05d}")
        start = time.perf_counter()
        run_pipeline(frame_args)
        old_wall_ms.append((time.perf_counter() - start) * 1000.0)

    first_args = delivery_args(
        args, args.frames[0], persistent_root / f"frame_{args.frames[0]:05d}"
    )
    runtime = DenseKPRuntime.from_args(first_args)
    runtime.load()
    persistent_wall_ms = []
    try:
        for frame in args.frames:
            frame_args = delivery_args(
                args, frame, persistent_root / f"frame_{frame:05d}"
            )
            start = time.perf_counter()
            runtime.process(frame_args)
            persistent_wall_ms.append((time.perf_counter() - start) * 1000.0)
    finally:
        session_count = runtime.session_creation_count
        detector_count = runtime.detector_initialization_count
        startup_ms = runtime.startup_ms
        runtime.close()

    frames = []
    all_exact = True
    all_profiles_valid = True
    for index, frame in enumerate(args.frames):
        old_json, old_profile, _ = read_outputs(old_root, frame)
        new_json, new_profile, inference_profile = read_outputs(
            persistent_root, frame
        )
        exact = old_json == new_json
        all_exact &= exact
        profile_valid = (
            set(inference_profile) == INFERENCE_PROFILE_KEYS
            and all(float(value) >= 0.0 for value in inference_profile.values())
        )
        all_profiles_valid &= profile_valid
        frames.append({
            "frame": frame,
            "json_bitwise_equal": exact,
            "points_max_abs": 0.0 if exact else None,
            "old_wall_ms": old_wall_ms[index],
            "persistent_per_frame_wall_ms": persistent_wall_ms[index],
            "old_ort_init_ms": old_profile.get("ort_init_ms", 0.0),
            "old_detector_init_ms": old_profile.get("detector_init_ms", 0.0),
            "persistent_ort_init_ms": new_profile.get("ort_init_ms", 0.0),
            "persistent_detector_init_ms": new_profile.get("detector_init_ms", 0.0),
            "persistent_densekp_inference_ms": new_profile.get(
                "densekp_inference_ms", 0.0
            ),
            "inference_profile_valid": profile_valid,
            "inference_profile": inference_profile,
        })

    passed = (
        all_exact and all_profiles_valid
        and session_count == 1 and detector_count == 1
    )
    report = {
        "status": "PASS" if passed else "FAIL",
        "runtime_startup_ms": startup_ms,
        "session_creation_count": session_count,
        "detector_initialization_count": detector_count,
        "frames": frames,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    report_path = args.output / "densekp_session_persistence_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(
        "DENSEKP_RUNTIME_SESSION_PERSISTENCE_READY="
        + ("PASS" if passed else "FAIL")
    )
    print(
        "DENSEKP_INFERENCE_PROFILING_READY="
        + ("PASS" if passed else "FAIL")
    )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
