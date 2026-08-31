"""Formal Phase 4.1 track-guided production sequence entry.

Consumes a prepared normal-ERP frame sequence (the upstream fisheye -> ERP
remap happens before this entry), runs the verified FULL / TRACK_GUIDED sparse
delivery with the Phase 4.1 identity gate, temporal tracking, HMR2 cold start
/ previous LM warm start, and per-person C++ staged-LM fitting.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", required=True, type=_frame_list,
                        help="e.g. 2213-2221 or comma separated list")
    parser.add_argument("--erp-root", type=Path, required=True,
                        help="prepared ERP root with cam2/ and cam4/ subdirs")
    parser.add_argument("--valid-masks", type=Path, required=True,
                        help="npz file holding valid_masks_cam2 / valid_masks_cam4")
    parser.add_argument("--densekp-backend", choices=("ort", "trt"), default="ort")
    parser.add_argument("--onnx-path", type=Path)
    parser.add_argument("--trt-engine-path", type=Path)
    parser.add_argument("--hmr2-checkpoint", type=Path, required=True)
    parser.add_argument("--base-fixture", type=Path, required=True)
    parser.add_argument("--cpp-executable", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--detect-interval", type=int, default=3,
                        help="K; 1 reproduces the Phase 2.2 full-detection path")
    parser.add_argument("--detector-backend", default="rtmlib_yolox")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--cpp-backend", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument(
        "--cpp-execution-backend",
        choices=("subprocess", "persistent_worker"), default="subprocess",
    )
    parser.add_argument("--cpp-worker-executable", type=Path)
    parser.add_argument("--cpp-worker-timeout-seconds", type=float, default=30.0)
    parser.add_argument("--cpp-worker-debug-dump-dir", type=Path)
    args = parser.parse_args()
    if args.densekp_backend == "ort" and args.onnx_path is None:
        parser.error("--onnx-path is required with --densekp-backend ort")
    if args.densekp_backend == "trt" and args.trt_engine_path is None:
        parser.error("--trt-engine-path is required with --densekp-backend trt")
    if (args.cpp_execution_backend == "persistent_worker"
            and args.cpp_worker_executable is None):
        parser.error(
            "--cpp-worker-executable is required with "
            "--cpp-execution-backend persistent_worker")
    if args.cpp_execution_backend == "persistent_worker" and args.cpp_backend != "cpu":
        parser.error("persistent_worker requires --cpp-backend cpu")
    return args


def _frame_list(value: str) -> list[int]:
    frames: list[int] = []
    for part in value.split(","):
        if "-" in part:
            first, last = (int(item) for item in part.split("-", 1))
            frames.extend(range(first, last + 1))
        else:
            frames.append(int(part))
    if not frames or len(set(frames)) != len(frames):
        raise argparse.ArgumentTypeError("frames must be non-empty and unique")
    return sorted(frames)


def main() -> int:
    from lib_pano.delivery_bootstrap import ensure_importable
    from lib_pano.fitting_io.densekp_delivery_boundary import load_erp_frame_pair
    from lib_pano.human3d.hmr2_initializer import HMR2Initializer
    from lib_pano.human_fitting.per_person_cpp import (
        CppPersonFitterConfig, create_cpp_per_person_fitter,
    )
    from lib_pano.human_fitting.sequence_orchestrator import (
        ErpHmr2ColdStartBatch, TrackGuidedSequenceConfig,
        TrackGuidedSequenceOrchestrator,
    )

    args = parse_args()
    ensure_importable()
    from densekp_delivery.config import DenseKPConfig
    from densekp_delivery.runtime import DenseKPRuntime
    masks = dict(np.load(args.valid_masks).items())
    valid_masks = (masks["valid_masks_cam2"], masks["valid_masks_cam4"])
    height, width = np.asarray(valid_masks[0]).shape[:2]

    template = DenseKPConfig(
        backend=args.densekp_backend, fisheye_cam2="", fisheye_cam4="", out_dir="",
        onnx_path=(str(args.onnx_path) if args.onnx_path else None),
        trt_engine_path=(str(args.trt_engine_path) if args.trt_engine_path else None),
        detector_backend=args.detector_backend,
        device=args.device, batch_size=1, profile_runtime=True)
    runtime = DenseKPRuntime.from_config(template).load()
    hmr = HMR2Initializer(args.hmr2_checkpoint, device=args.device)
    cold_start = ErpHmr2ColdStartBatch(hmr, width, height)
    fitter = create_cpp_per_person_fitter(CppPersonFitterConfig(
        args.base_fixture, args.cpp_executable,
        args.output_dir / "per_person_cpp", args.cpp_backend,
        execution_backend=args.cpp_execution_backend,
        worker_executable=args.cpp_worker_executable,
        worker_timeout_seconds=args.cpp_worker_timeout_seconds,
        worker_debug_dump_dir=args.cpp_worker_debug_dump_dir))

    try:
        orchestrator = TrackGuidedSequenceOrchestrator(
            runtime, cold_start, fitter, valid_masks,
            TrackGuidedSequenceConfig(detection_interval=args.detect_interval),
            sequence_start_frame=args.frames[0])

        def erp_path(camera: str, frame: int) -> Path:
            for suffix in (".png", ".jpg", ".jpeg"):
                candidate = args.erp_root / camera / f"{frame:05d}{suffix}"
                if candidate.exists():
                    return candidate
            raise FileNotFoundError(f"no ERP image for {camera} frame {frame}")

        for frame in args.frames:
            erp_cam2, erp_cam4 = load_erp_frame_pair(
                erp_path("cam2", frame), erp_path("cam4", frame))
            record = orchestrator.process_frame(frame, erp_cam2, erp_cam4)
            persons = len(record.payload["persons"])
            print(f"frame {frame}: mode={record.actual_mode.value} "
                  f"persons={persons} detector={record.detector_calls} "
                  f"densekp={record.densekp_calls}", flush=True)

        result = orchestrator.sequence_result({
            "erp_root": str(args.erp_root),
            "requested_frames": args.frames,
            "detect_interval": args.detect_interval,
            "densekp_backend": args.densekp_backend,
            "trt_engine_path": (
                str(args.trt_engine_path) if args.trt_engine_path else None),
            "cpp_execution_backend": args.cpp_execution_backend,
            "cpp_evaluator_backend": args.cpp_backend,
        })
        worker = getattr(fitter, "worker", None)
        if worker is not None:
            result["worker_diagnostics"] = {
                "process_start_count": worker.start_count,
                "restart_count": worker.restart_count,
                "static_model_load_count": worker.static_model_load_count,
                "worker_startup_ms": worker.worker_startup_ms,
                "static_model_load_ms": worker.static_model_load_ms,
                "request_count": worker.request_count,
                "request_profiles": worker.profile_rows,
                "recent_stderr": worker.recent_stderr,
            }
        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir / "sequence_result.json").write_text(
            json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print("DENSEKP_TRACK_GUIDED_SEQUENCE=PASS "
              f"frames={result['num_frames']} tracks={result['unique_tracks']} "
              f"observations={result['matched_person_observations']} "
              f"lm_successes={result['lm_successes']}")
        return 0
    finally:
        fitter.close()
        runtime.close()


if __name__ == "__main__":
    raise SystemExit(main())

r'''
  python src/script/human_fitting/run_densekp_track_guided_sequence.py \
    --frames 2213-2221 \
    --erp-root data_process/tmp/densekp_phaseA_erp_input \
    --valid-masks data_process/tmp/densekp_phaseA_erp_input/valid_masks.npz \
    --onnx-path Circular_Video/01_recv/20260708/CameraHMR/deploy/densekp_onnx/artifacts/densekp_b1_256x256.onnx \
    --hmr2-checkpoint ~/.cache/4DHumans/logs/train/multiruns/hmr2/0/checkpoints/epoch=35-step=1000000.ckpt \
    --base-fixture data_process/tmp/densekp_cpp_reference2/staged_lm_pipeline/frame_000513/cpp_fixture \
    --cpp-executable data_process/tmp/staged_lm_phaseA_build/multiframe_staged_lm_reference_main \
    --output-dir data_process/Vis/densekp_phaseA_sequence_2213_2221_k3 \
    --detect-interval 3

  python src/script/test_function/debug090_phase41_sequence_visualization.py \
    --frames 2213-2221 \
    --erp-root data_process/tmp/densekp_phaseA_erp_input \
    --valid-masks data_process/tmp/densekp_phaseA_erp_input/valid_masks.npz \
    --onnx-path Circular_Video/01_recv/20260708/CameraHMR/deploy/densekp_onnx/artifacts/densekp_b1_256x256.onnx \
    --hmr2-checkpoint ~/.cache/4DHumans/logs/train/multiruns/hmr2/0/checkpoints/epoch=35-step=1000000.ckpt \
    --base-fixture data_process/tmp/densekp_cpp_reference2/staged_lm_pipeline/frame_000513/cpp_fixture \
    --cpp-executable data_process/tmp/staged_lm_phaseA_build/multiframe_staged_lm_reference_main \
    --output-dir data_process/Vis/densekp_phase41_sequence_k3_vis \
    --detect-interval 3

'''
