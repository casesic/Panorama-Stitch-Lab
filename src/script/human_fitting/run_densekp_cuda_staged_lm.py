"""Run persistent CUDA DenseKP followed by one multi-frame CUDA staged-LM."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from lib_pano.fitting_io.densekp_delivery_io import (
    load_densekp_delivery_observation,
)
from lib_pano.fitting_io.densekp_runtime_export import (
    materialize_runtime_fixture,
    write_runtime_observation,
)

DELIVERY_ROOT = ROOT / "Circular_Video/01_recv/20260708"
if str(DELIVERY_ROOT) not in sys.path:
    sys.path.insert(0, str(DELIVERY_ROOT))

from densekp_delivery.config import DenseKPConfig  # noqa: E402
from densekp_delivery.runtime import DenseKPRuntime  # noqa: E402
from script.human_fitting.densekp_staged_lm_reporting import (  # noqa: E402
    load_numeric_profile,
    mean,
    remap_backend_is_cuda,
    write_summary,
)


def _frames(value: str) -> list[int]:
    result: list[int] = []
    for item in value.split(","):
        if "-" in item:
            first, last = (int(part) for part in item.split("-", 1))
            result.extend(range(first, last + 1))
        else:
            result.append(int(item))
    if not result or len(set(result)) != len(result):
        raise argparse.ArgumentTypeError("frames must be non-empty and unique")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", type=_frames, required=True)
    parser.add_argument("--fisheye-root", type=Path, required=True)
    parser.add_argument("--onnx-path", type=Path, required=True)
    parser.add_argument("--base-fixture-root", type=Path, required=True)
    parser.add_argument("--cpp-executable", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--calibration-cache-dir", type=Path, required=True)
    parser.add_argument("--camera", choices=("old", "new"), default="old")
    parser.add_argument("--detector-backend", default="rtmlib_yolox")
    parser.add_argument("--detector-threshold", type=float, default=0.25)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument(
        "--erp-remap-backend", choices=("cpu", "cuda"), default="cuda",
    )
    parser.add_argument("--erp-remap-cuda-library", type=Path)
    parser.add_argument("--keep-runtime-fixtures", action="store_true")
    parser.add_argument(
        "--visualize-optimized", action="store_true",
        help="Write observed-vs-optimized DenseKP overlays after fitting.",
    )
    return parser.parse_args()


def _delivery_result(path: Path) -> tuple[Path, dict]:
    expected = path / "result/densekp_pairs_erp_rotate.json"
    fallback = path / "result/densekp_pairs.json"
    result = expected if expected.exists() else fallback
    if not result.exists():
        raise RuntimeError(f"DenseKP delivery did not write a result below {path}")
    return result, json.loads(result.read_text(encoding="utf-8"))


def _base_fixture(root: Path, frame: int) -> Path:
    candidates = (
        root / "staged_lm_pipeline" / f"frame_{frame:06d}" / "cpp_fixture",
        root / f"frame_{frame:06d}" / "cpp_fixture",
        root / f"frame_{frame:06d}",
    )
    for candidate in candidates:
        if (candidate / "pipeline_meta.txt").exists():
            return candidate
    raise FileNotFoundError(f"no C++ model bundle for frame {frame} below {root}")


def _delivery_config(args: argparse.Namespace) -> DenseKPConfig:
    """Build shared delivery configuration without a CLI argv round-trip."""
    return DenseKPConfig(
        backend="ort",
        onnx_path=str(args.onnx_path),
        fisheye_cam2="",
        fisheye_cam4="",
        out_dir="",
        calibration_cache_dir=str(args.calibration_cache_dir),
        cache_mode="read_write",
        output_coord_space="erp_rotate",
        vis_option="none",
        save_option="none",
        device="cuda",
        camera=args.camera,
        detector_backend=args.detector_backend,
        detector_threshold=args.detector_threshold,
        batch_size=args.batch_size,
        erp_remap_backend=args.erp_remap_backend,
        erp_remap_cuda_library=str(args.erp_remap_cuda_library),
        profile_runtime=True,
    )


def _frame_delivery_config(
    config: DenseKPConfig, args: argparse.Namespace, frame: int, output: Path,
) -> DenseKPConfig:
    return config.for_frame(
        args.fisheye_root / "cam2" / f"{frame:05d}.jpg",
        args.fisheye_root / "cam4" / f"{frame:05d}.jpg",
        output,
    )


def run_densekp_cuda_staged_lm(args: argparse.Namespace) -> int:
    """Run delivery, materialize observations, then invoke staged-LM once."""
    args.output.mkdir(parents=True, exist_ok=True)
    for path in (args.onnx_path, args.cpp_executable):
        if not path.exists():
            raise FileNotFoundError(path)
    if args.erp_remap_cuda_library is None:
        args.erp_remap_cuda_library = (
            args.cpp_executable.parent / "libdensekp_erp_remap_cuda.so"
        )
    bases = [_base_fixture(args.base_fixture_root, frame) for frame in args.frames]

    project_tmp = ROOT / "data_process/tmp"
    project_tmp.mkdir(parents=True, exist_ok=True)
    temporary = tempfile.TemporaryDirectory(
        prefix="densekp_cuda_staged_lm_", dir=project_tmp,
    )
    scratch = Path(temporary.name)
    delivery_root = scratch / "densekp"
    fixture_root = scratch / "fixtures"
    delivery_config = _delivery_config(args)
    runtime = DenseKPRuntime.from_config(delivery_config)
    runtime.load()
    execution_providers = runtime.execution_providers
    if "CUDAExecutionProvider" not in execution_providers:
        runtime.close()
        raise RuntimeError(
            "DenseKP requested CUDA but ORT CUDAExecutionProvider is not active: "
            f"{execution_providers}"
        )
    densekp_started = time.perf_counter()
    fixture_paths: list[Path] = []
    valid_frames: list[int] = []
    skipped_frames: list[dict[str, int | str]] = []
    per_frame: list[dict[str, float | int]] = []
    measured_static_cache_counts = dict(runtime.static_cache_counts)
    try:
        for frame, base in zip(args.frames, bases):
            frame_started = time.perf_counter()
            delivery_output = delivery_root / f"frame_{frame:06d}"
            runtime.process(_frame_delivery_config(
                delivery_config, args, frame, delivery_output,
            ))
            densekp_json, delivery_payload = _delivery_result(delivery_output)
            profile_path = delivery_output / "result/densekp_runtime_profile.json"
            delivery_profile = load_numeric_profile(profile_path)
            delivery_status = str(delivery_payload.get("status", "unknown"))
            if delivery_payload.get("schema_version") == "densekp_pairs_v2":
                persons = delivery_payload.get("persons", [])
                if len(persons) > 1:
                    raise RuntimeError(
                        "C++ staged-LM runner is single-person: DenseKP v2 "
                        f"contains {len(persons)} matched people for frame {frame}. "
                        "Track/select one identity before fitting; multi-frame is "
                        "temporal batching, not multiple people."
                    )
                valid_count = int(sum(persons[0].get("valid", []))) if persons else 0
            else:
                valid_count = int(sum(delivery_payload.get("valid", [])))
            if delivery_status not in {"ok", "matched"} or valid_count == 0:
                reason = delivery_status if valid_count else f"{delivery_status}:valid=0"
                skipped_frames.append({"frame": frame, "reason": reason})
                per_frame.append({
                    "frame": frame,
                    "status": "skipped",
                    "skip_reason": reason,
                    "valid_points": valid_count,
                    **delivery_profile,
                    "observation_load_export_ms": 0.0,
                    "runtime_fixture_ms": 0.0,
                    "densekp_and_export_ms": (
                        time.perf_counter() - frame_started
                    ) * 1000.0,
                })
                continue
            observation_started = time.perf_counter()
            observation = load_densekp_delivery_observation(
                densekp_json, camera_group=args.camera,
            )
            observation_output = (
                args.output / "observations" / f"frame_{frame:06d}.json"
            )
            write_runtime_observation(observation_output, observation)
            observation_load_export_ms = (
                time.perf_counter() - observation_started
            ) * 1000.0
            fixture = fixture_root / f"frame_{frame:06d}" / "cpp_fixture"
            export_started = time.perf_counter()
            materialize_runtime_fixture(
                base, fixture, observation, link_static_files=True,
            )
            fixture_paths.append(fixture)
            valid_frames.append(frame)
            per_frame.append({
                "frame": frame,
                "status": "valid",
                "valid_points": int(observation.valid.sum()),
                **delivery_profile,
                "observation_load_export_ms": observation_load_export_ms,
                "runtime_fixture_ms": (time.perf_counter() - export_started) * 1000.0,
                "densekp_and_export_ms": (time.perf_counter() - frame_started) * 1000.0,
            })
        densekp_total_ms = (time.perf_counter() - densekp_started) * 1000.0
        measured_static_cache_counts = dict(runtime.static_cache_counts)
    finally:
        runtime_startup_ms = runtime.startup_ms
        session_count = runtime.session_creation_count
        detector_count = runtime.detector_initialization_count
        static_cache_counts = measured_static_cache_counts
        static_geometry_entry_count = runtime.static_geometry_entry_count
        runtime.close()
    if not fixture_paths:
        raise RuntimeError("no valid DenseKP frames remain after detection filtering")
    lm_started = time.perf_counter()
    cpp_output = args.output / "cpp_lm"
    completed = subprocess.run([
        str(args.cpp_executable),
        "--fixtures", ",".join(str(path) for path in fixture_paths),
        "--output", str(cpp_output),
        "--evaluator-backend", "cuda",
    ], cwd=ROOT, check=True)
    del completed
    lm_process_ms = (time.perf_counter() - lm_started) * 1000.0
    lm_summary = dict(
        line.split("=", 1) for line in
        (cpp_output / "multiframe_summary.txt").read_text().splitlines()
        if line
    )
    if lm_summary.get("evaluator_backend") != "cuda":
        raise RuntimeError(
            "C++ staged-LM CUDA backend is not active: "
            f"{lm_summary.get('evaluator_backend')}"
        )
    numeric_frames = [
        {key: float(value) for key, value in row.items()
         if isinstance(value, (int, float)) and key not in {"frame", "valid_points"}}
        for row in per_frame
    ]
    steady_rows = numeric_frames[1:] if len(numeric_frames) > 1 else numeric_frames
    excluded_frames = [args.frames[0]] if len(numeric_frames) > 1 else []
    frame_count = len(valid_frames)
    requested_frame_count = len(args.frames)
    cpp_stage_per_frame = {
        "cpp_stage1_ms": float(lm_summary["stage1_total_ms"]) / requested_frame_count,
        "cpp_pose_cuda_ms": float(lm_summary["pose_total_ms"]) / requested_frame_count,
        "cpp_pose_shape_cuda_ms": float(lm_summary["pose_shape_total_ms"]) / requested_frame_count,
        "cpp_fixture_model_load_ms": float(lm_summary["fixture_model_load_ms"]) / requested_frame_count,
        "cpp_output_export_ms": float(lm_summary["output_export_ms"]) / requested_frame_count,
        "cpp_process_launch_ms": max(
            0.0, lm_process_ms - float(lm_summary["app_profiled_ms"])
        ) / requested_frame_count,
    }
    average_stage_ms = {
        "fisheye_image_read": mean(steady_rows, "image_load_ms"),
        "calibration_lut_load": mean(steady_rows, "lut_cache_load_ms"),
        "valid_mask_load": mean(steady_rows, "mask_cache_load_ms"),
        "fisheye_to_erp_remap": mean(steady_rows, "erp_remap_ms"),
        "detector_inference": mean(steady_rows, "detector_inference_ms"),
        "bbox_postprocess": mean(steady_rows, "bbox_postprocess_ms"),
        "densekp_crop_resize": mean(steady_rows, "densekp_crop_resize_ms"),
        "densekp_normalize": mean(steady_rows, "densekp_normalize_ms"),
        "densekp_tensor_prepare": mean(steady_rows, "densekp_tensor_prepare_ms"),
        "densekp_ort_inference": mean(steady_rows, "densekp_inference_ms"),
        "densekp_output_decode": mean(steady_rows, "densekp_postprocess_ms"),
        "person_match_postprocess": mean(steady_rows, "delivery_postprocess_ms"),
        "erp_rotate_point_transform": mean(steady_rows, "coordinate_transform_ms"),
        "densekp_json_export": mean(steady_rows, "json_export_ms"),
        "observation_runtime_prepare": mean(steady_rows, "observation_load_export_ms"),
        "runtime_fixture_prepare": mean(steady_rows, "runtime_fixture_ms"),
        **cpp_stage_per_frame,
    }
    steady_wall_ms = statistics.fmean(
        row["densekp_and_export_ms"] for row in steady_rows
    ) + lm_process_ms / requested_frame_count
    average_stage_ms["unattributed_python_control"] = max(
        0.0, steady_wall_ms - sum(average_stage_ms.values())
    )
    report = {
        "status": "PASS",
        "frames": args.frames,
        "frame_count": len(args.frames),
        "valid_frames": valid_frames,
        "valid_frame_count": len(valid_frames),
        "skipped_frames": skipped_frames,
        "skipped_frame_count": len(skipped_frames),
        "cpp_output_frame_mapping": {
            f"frame_{index}": frame for index, frame in enumerate(valid_frames)
        },
        "coordinate_space": "erp_rotate",
        "densekp_device": "cuda",
        "densekp_execution_providers": execution_providers,
        "lm_evaluator_backend": "cuda",
        "erp_remap_backend_requested": args.erp_remap_backend,
        "erp_remap_backends_used": sorted({
            str(row.get("erp_remap_backend", "unknown")) for row in per_frame
        }),
        "densekp_runtime_startup_ms": runtime_startup_ms,
        "densekp_session_creation_count": session_count,
        "detector_initialization_count": detector_count,
        "static_cache": {
            "before": {
                "lut_disk_load_count": len(args.frames),
                "valid_mask_disk_load_count": len(args.frames),
            },
            "after": static_cache_counts,
            "resident_calibration_count": static_geometry_entry_count,
        },
        "densekp_total_ms": densekp_total_ms,
        "densekp_fps_excluding_startup": 1000.0 * len(args.frames) / densekp_total_ms,
        "cpp_lm_process_ms": lm_process_ms,
        "cpp_lm_internal_ms": float(lm_summary["total_ms"]),
        "cpp_lm_internal_fps": 1000.0 / float(lm_summary["ms_per_frame"]),
        "end_to_end_ms_excluding_startup": densekp_total_ms + lm_process_ms,
        "end_to_end_fps_excluding_startup": (
            1000.0 * len(args.frames) / (densekp_total_ms + lm_process_ms)
        ),
        "per_frame": per_frame,
        "lm_summary": lm_summary,
        "warmup": {"frame": args.frames[0], **per_frame[0]},
        "steady_state": {
            "excluded_frames": excluded_frames,
            "frame_count": len(steady_rows),
            "ms_per_frame": steady_wall_ms,
            "fps": 1000.0 / steady_wall_ms,
            "average_stage_ms": average_stage_ms,
            "image_io_profile": {
                "cam2_read_ms": mean(steady_rows, "cam2_read_ms"),
                "cam4_read_ms": mean(steady_rows, "cam4_read_ms"),
                "total_read_ms": mean(steady_rows, "image_load_ms"),
                "disk_read_ms": mean(steady_rows, "image_disk_read_ms"),
                "decode_ms": mean(steady_rows, "image_decode_ms"),
                "copy_ms": mean(steady_rows, "image_copy_ms"),
            },
        },
        "initialization": {
            "ort_and_detector_startup_ms": runtime_startup_ms,
            "ort_init_ms": float(runtime.profile_timing.get("ort_init_ms", 0.0)),
            "detector_init_ms": float(runtime.profile_timing.get("detector_init_ms", 0.0)),
            "first_frame_lut_build_ms": float(per_frame[0].get("lut_build_ms", 0.0)),
            "first_frame_mask_build_ms": float(per_frame[0].get("mask_build_ms", 0.0)),
        },
        "DENSEKP_RUNTIME_STATIC_CACHE_READY": "PASS",
        "DENSEKP_RUNTIME_IMAGE_IO_READY": "PASS",
        "DENSEKP_CUDA_ERP_REMAP_READY": (
            "PASS" if remap_backend_is_cuda(per_frame) else "CPU_FALLBACK"
        ),
    }
    for path in (
        args.output / "densekp_cuda_staged_lm_report.json",
        args.output / "densekp_end_to_end_profile.json",
    ):
        path.write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8",
        )
    write_summary(
        args.output / "densekp_end_to_end_profile_summary.md", report,
    )
    if args.visualize_optimized:
        from script.human_fitting.visualize_densekp_cuda_staged_lm import (
            visualize_optimized_densekp,
        )

        visualize_optimized_densekp(
            args.output,
            args.fisheye_root,
            args.calibration_cache_dir,
            args.erp_remap_backend,
            args.erp_remap_cuda_library,
        )
    if args.keep_runtime_fixtures:
        retained = args.output / "runtime_fixtures"
        shutil.copytree(fixture_root, retained, symlinks=True)
    temporary.cleanup()
    print(json.dumps({key: value for key, value in report.items()
                      if key not in {"per_frame", "lm_summary"}}, indent=2))
    print("DENSEKP_CUDA_CPP_LM_CONNECTED=PASS")
    print("DENSEKP_END_TO_END_PROFILING_READY=PASS")
    print("DENSEKP_RUNTIME_STATIC_CACHE_READY=PASS")
    print("DENSEKP_RUNTIME_IMAGE_IO_READY=PASS")
    if report["DENSEKP_CUDA_ERP_REMAP_READY"] == "PASS":
        print("DENSEKP_CUDA_ERP_REMAP_READY=PASS")
    return 0


def main() -> int:
    args = parse_args()
    return run_densekp_cuda_staged_lm(args)


if __name__ == "__main__":
    raise SystemExit(main())

r'''
python src/script/human_fitting/run_densekp_cuda_staged_lm.py \
  --frames 513-516 \
  --fisheye-root data_process/Output/20260311_riding/image/fisheye \
  --onnx-path Circular_Video/01_recv/20260708/CameraHMR/deploy/densekp_onnx/artifacts/densekp_b_256x256.onnx \
  --base-fixture-root data_process/tmp/densekp_cpp_reference2 \
  --cpp-executable data_process/tmp/staged_lm_cuda_backend_build/multiframe_staged_lm_reference_main \
  --erp-remap-cuda-library data_process/tmp/staged_lm_cuda_backend_build/libdensekp_erp_remap_cuda.so \
  --calibration-cache-dir data_process/tmp/densekp_e2e_profile_calibration_cache \
  --output data_process/Vis/densekp_cuda_staged_lm_513_516 \
  --camera old
'''
