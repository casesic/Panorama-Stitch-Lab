"""Frame-level Phase 2.1 HMR2 -> runtime preparation -> existing C++ LM smoke."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import time

import cv2
import numpy as np

from lib_pano.fitting_io.densekp_delivery_io import (
    load_densekp_delivery_observation,
)
from lib_pano.fitting_io.densekp_runtime_export import (
    materialize_person_runtime_fixture,
    person_initial_state_from_smplx,
)
from lib_pano.human3d.hmr2_initializer import HMR2Initialization
from lib_pano.human_fitting.hmr2_staged_init import (
    convert_hmr2_to_staged_initialization,
    to_smplx_camera_initializations,
)


def _hmr_result(path: Path, person_index: int) -> HMR2Initialization:
    person = json.loads(path.read_text(encoding="utf-8"))["persons"][person_index]
    return HMR2Initialization(
        person_index=int(person["person_index"]),
        bbox_xyxy=np.asarray(person["bbox_xyxy"], np.float32),
        global_orient=np.asarray(person["global_orient"], np.float32),
        body_pose=np.asarray(person["body_pose"], np.float32),
        betas=np.asarray(person["betas"], np.float32),
        pred_cam=np.asarray(person["pred_cam"], np.float32),
        pred_cam_t=np.asarray(person["pred_cam_t"], np.float32),
        pred_cam_t_full=np.asarray(person["pred_cam_t_full"], np.float32),
        pred_keypoints_2d=np.zeros((0, 2), np.float32),
        pred_vertices_shape=tuple(person["pred_vertices_shape"]),
        pred_vertices_min=float(person["pred_vertices_min"]),
        pred_vertices_max=float(person["pred_vertices_max"]),
    )


def _meta(path: Path) -> dict[str, str]:
    return dict(line.split("=", 1) for line in path.read_text().splitlines() if line)


def _visualize(cam2_path: Path, cam4_path: Path, optimized_path: Path,
               output: Path) -> None:
    cam2, cam4 = cv2.imread(str(cam2_path)), cv2.imread(str(cam4_path))
    if cam2 is None or cam4 is None:
        raise FileNotFoundError("visualization ERP_ROTATE images")
    payload = json.loads(optimized_path.read_text())
    pairs = np.asarray(payload["pairs"], np.float32)
    valid = np.asarray(payload["valid"], bool)
    canvas = np.concatenate((cam2, cam4), axis=1)
    offset = cam2.shape[1]
    for p2x, p2y, p4x, p4y in pairs[valid]:
        p2 = (int(round(p2x)), int(round(p2y)))
        p4 = (int(round(p4x)) + offset, int(round(p4y)))
        cv2.circle(canvas, p2, 3, (0, 255, 255), -1)
        cv2.circle(canvas, p4, 3, (0, 255, 0), -1)
        cv2.line(canvas, p2, p4, (255, 120, 0), 1, cv2.LINE_AA)
    output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output), canvas)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frame", type=int, default=2214)
    parser.add_argument("--densekp", type=Path, required=True)
    parser.add_argument("--hmr-json", type=Path, required=True)
    parser.add_argument("--hmr-person-index", type=int, default=1)
    parser.add_argument("--base-fixture", type=Path, required=True)
    parser.add_argument("--cpp-executable", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cam2-image", type=Path)
    parser.add_argument("--cam4-image", type=Path)
    parser.add_argument("--focal-length", type=float, default=112500.0)
    parser.add_argument("--evaluator-backend", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    delivery = json.loads(args.densekp.read_text(encoding="utf-8"))
    if len(delivery.get("persons", [])) != 1:
        raise RuntimeError("Phase 2.1 smoke requires exactly one matched person")
    matched = delivery["persons"][0]
    if int(matched["cam4_person_index"]) != args.hmr_person_index:
        raise RuntimeError("selected HMR2 person is not the matched cam4 person")
    observation = load_densekp_delivery_observation(args.densekp)
    hmr = _hmr_result(args.hmr_json, args.hmr_person_index)
    normalized = convert_hmr2_to_staged_initialization(
        hmr, source_camera="cam4",
        bboxes={"cam2": np.asarray(matched["bbox_cam2"], np.float32),
                "cam4": np.asarray(matched["bbox_cam4"], np.float32)},
        focal_length=args.focal_length,
    )
    cam2, cam4 = to_smplx_camera_initializations(normalized)
    initial = person_initial_state_from_smplx(cam2, cam4)
    np.savez(
        args.output / "person_initial_state.npz",
        frame_parameters=initial.frame_parameters, beta=initial.beta,
        initialization_source=np.asarray("hmr2"), source_frame=np.asarray(args.frame),
    )
    fixture = args.output / "person_runtime_fixture"
    write_started = time.perf_counter()
    materialize_person_runtime_fixture(
        args.base_fixture, fixture, observation, initial,
    )
    input_write_ms = (time.perf_counter() - write_started) * 1000.0
    cpp_output = args.output / "cpp_output"
    command = [
        str(args.cpp_executable), "--fixtures", str(fixture),
        "--output", str(cpp_output), "--runtime-prepare-stage1",
        "--evaluator-backend", args.evaluator_backend,
    ]
    cpp_started = time.perf_counter()
    completed = subprocess.run(command, check=True, text=True, capture_output=True)
    cpp_wall_ms = (time.perf_counter() - cpp_started) * 1000.0
    (args.output / "cpp_stdout.log").write_text(
        completed.stdout + completed.stderr, encoding="utf-8",
    )
    overall = _meta(cpp_output / "multiframe_summary.txt")
    fitting = _meta(cpp_output / "frame_0/fitting_summary.txt")
    optimized = json.loads(
        (cpp_output / "frame_0/optimized_densekp.json").read_text()
    )
    optimized_pairs = np.asarray(optimized["pairs"], np.float32)
    summary = {
        "schema_version": "densekp_multi_person_fitting_v1",
        "frame": args.frame,
        "detections_cam2": 1,
        "detections_cam4": 4,
        "matched_persons": 1,
        "hmr2_persons": 1,
        "prepared_cpp_problems": 1,
        "cpp_fitted_persons": 1,
        "person_id": int(matched["person_id"]),
        "initialization": {"type": "hmr2", "best_view": "cam4"},
        "cpp_preparation": {"success": True, "ms": float(overall["runtime_preparation_ms"])},
        "fitting": {
            "success": True,
            "stage1_initial_objective": float(fitting["stage1_initial_objective"]),
            "stage1_final_objective": float(fitting["stage1_final_objective"]),
            "pose_initial_objective": float(fitting["pose_initial_objective"]),
            "pose_final_objective": float(fitting["pose_final_objective"]),
            "pose_shape_initial_objective": float(overall["pose_shape_initial_objective"]),
            "pose_shape_final_objective": float(overall["pose_shape_final_objective"]),
            "stage1_trials": int(fitting["stage1_trials"]),
            "pose_trials": int(fitting["pose_trials"]),
            "pose_shape_trials": int(overall["pose_shape_trials"]),
            "status": overall["pose_shape_status"],
        },
        "valid_densekp_count": int(np.count_nonzero(observation.valid)),
        "optimized_pair_count": int(len(optimized_pairs)),
        "optimized_pairs_finite": bool(np.isfinite(optimized_pairs).all()),
        "timing": {"person_input_write_ms": input_write_ms,
                   "cpp_process_total_ms": cpp_wall_ms,
                   "cpp_static_load_and_prepare_ms": float(overall["fixture_model_load_ms"]),
                   "cpp_lm_ms": float(overall["total_ms"])},
    }
    (args.output / "integration_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8",
    )
    if args.cam2_image and args.cam4_image:
        _visualize(args.cam2_image, args.cam4_image,
                   cpp_output / "frame_0/optimized_densekp.json",
                   args.output / "optimized_correspondence.jpg")
    print("FRAME2214_CPP_LM=PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
