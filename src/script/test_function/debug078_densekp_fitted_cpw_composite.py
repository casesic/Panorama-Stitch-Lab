"""DenseKP delivery → C++ staged LM → optimized DenseKP → debug077 CPW."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import time

import config_script  # noqa: F401
import cv2
import numpy as np

from lib_pano.fitting_io.densekp_delivery_io import (
    load_densekp_delivery_observation,
)
from lib_pano.fitting_io.densekp_runtime_export import (
    materialize_runtime_fixture,
    write_runtime_observation,
)
from script.test_function.debug077_fitted_cpw_composite import run_cpw_direction


ROOT = Path(config_script.PROJECT_ROOT)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--frame", type=int, required=True)
    parser.add_argument("--densekp_json", type=Path, required=True)
    parser.add_argument("--base_cpp_fixture", type=Path, required=True)
    parser.add_argument("--cpp_executable", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--camera_group", choices=("old", "new"), default="old")
    parser.add_argument(
        "--direction",
        choices=("cam4_to_cam2", "cam2_to_cam4"),
        default="cam4_to_cam2",
    )
    parser.add_argument("--reference_corr_npz", type=Path)
    parser.add_argument(
        "--densekp_profile_json", type=Path,
        help="Optional delivery profile to merge into the end-to-end profile.",
    )
    parser.add_argument("--cpw_warp_cache_dir", type=Path)
    parser.add_argument(
        "--cpw_warp_cache_mode",
        choices=("off", "read_write", "refresh", "read_only"),
        default="read_write",
    )
    return parser.parse_args()


def run(command: list[str]) -> None:
    print("[RUN]", " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def validate_optimized(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("coord_space") != "erp_rotate":
        raise RuntimeError("optimized DenseKP is not ERP_ROTATE")
    pairs = np.asarray(data["pairs"], dtype=np.float32)
    valid = np.asarray(data["valid"], dtype=bool)
    if pairs.shape != (138, 4) or valid.shape != (138,):
        raise RuntimeError(
            f"invalid optimized DenseKP shape: pairs={pairs.shape}, valid={valid.shape}"
        )
    if not np.isfinite(pairs).all():
        raise RuntimeError("optimized DenseKP contains NaN/Inf")
    if not (
        ((pairs[:, (0, 2)] >= 0) & (pairs[:, (0, 2)] < 5760)).all()
        and ((pairs[:, (1, 3)] >= 0) & (pairs[:, (1, 3)] < 2880)).all()
    ):
        raise RuntimeError("optimized DenseKP is outside ERP_ROTATE bounds")
    return pairs[:, :2], pairs[:, 2:], valid


def compare_composites(actual_path: Path, reference_path: Path) -> None:
    actual = cv2.imread(str(actual_path), cv2.IMREAD_COLOR)
    reference = cv2.imread(str(reference_path), cv2.IMREAD_COLOR)
    if actual is None or reference is None or actual.shape != reference.shape:
        raise RuntimeError("debug077 composite comparison image mismatch")
    difference = cv2.absdiff(actual, reference)
    mean_abs = float(difference.mean())
    p95 = float(np.percentile(difference, 95))
    changed_percent = float(
        100.0 * np.mean(np.any(difference > 10, axis=2))
    )
    print(
        "[COMPARE debug077 CPW] "
        f"mean_abs={mean_abs:.6g} p95={p95:.6g} "
        f"changed_gt10={changed_percent:.6g}%"
    )
    # CPW changes are localized around the warped overlap. This gate catches a
    # coordinate-system or direction error without requiring pixel identity
    # between two independently optimized correspondence sets.
    if p95 > 2.0 or changed_percent > 2.0:
        raise RuntimeError("new CPW is not visually consistent with debug077")


def main() -> int:
    args = parse_args()
    runtime_profile = (
        json.loads(args.densekp_profile_json.read_text(encoding="utf-8"))
        if args.densekp_profile_json is not None else {}
    )
    frame_dir = args.output / f"frame_{args.frame:04d}"
    frame_dir.mkdir(parents=True, exist_ok=True)
    observation = load_densekp_delivery_observation(
        args.densekp_json, camera_group=args.camera_group
    )
    observation_json = frame_dir / "densekp_observation.json"
    write_runtime_observation(observation_json, observation)

    temporary = tempfile.TemporaryDirectory(prefix=f"debug078_{args.frame}_")
    scratch = Path(temporary.name)
    fixture = scratch / "runtime_fixture"
    exporter_start = time.perf_counter()
    materialize_runtime_fixture(
        args.base_cpp_fixture, fixture, observation,
    )
    exporter_ms = (time.perf_counter() - exporter_start) * 1000.0
    cpp_output = scratch / "cpp_output"
    lm_start = time.perf_counter()
    run([
        str(args.cpp_executable),
        "--fixture", str(fixture),
        "--output", str(cpp_output),
        "--optimized-densekp-only",
    ])
    lm_ms = (time.perf_counter() - lm_start) * 1000.0
    optimized_path = cpp_output / "optimized_densekp.json"
    final_path = frame_dir / "optimized_densekp.json"
    shutil.copyfile(optimized_path, final_path)
    points2, points4, valid = validate_optimized(final_path)

    data_dir = scratch / "cpw_data"
    corr_dir = data_dir / "corr_npz"
    corr_dir.mkdir(parents=True, exist_ok=True)
    corr_path = corr_dir / f"{args.frame:05d}.npz"
    np.savez_compressed(
        corr_path,
        frame_idx=np.int32(args.frame),
        point_ids=np.arange(138, dtype=np.int32),
        points_cam2=points2,
        points_cam4=points4,
        valid=valid,
        valid_cam2=valid,
        valid_cam4=valid,
    )
    reference_corr = None
    if args.reference_corr_npz:
        reference = np.load(args.reference_corr_npz)
        error = max(
            float(np.max(np.abs(points2 - reference["points_cam2"]))),
            float(np.max(np.abs(points4 - reference["points_cam4"]))),
        )
        print(f"[COMPARE debug077 points] max_abs={error:.6g}px")
        reference_data = scratch / "debug077_reference_data" / "corr_npz"
        reference_data.mkdir(parents=True, exist_ok=True)
        reference_corr = reference_data / f"{args.frame:05d}.npz"
        shutil.copyfile(args.reference_corr_npz, reference_corr)

    src, target = args.direction.split("_to_")
    image_dir = ROOT / "data_process/Output" / args.dataset / "image/equirect_rotate"
    cpw_cache_dir = args.cpw_warp_cache_dir or args.output / ".cpw_warp_cache"
    cpw_start = time.perf_counter()
    run_cpw_direction(
        [args.frame], data_dir, scratch / "cpw", image_dir, src, target,
        args.direction, profile_timing=runtime_profile,
        warp_map_cache_dir=cpw_cache_dir,
        warp_map_cache_mode=args.cpw_warp_cache_mode,
        camera_group=args.camera_group,
    )
    cpw_ms = (time.perf_counter() - cpw_start) * 1000.0
    cpw_source = (
        scratch / "cpw" / args.direction / "composite_target_priority"
        / f"{args.frame:05d}.jpg"
    )
    if not cpw_source.exists() or cv2.imread(str(cpw_source)) is None:
        raise RuntimeError("CPW did not produce a valid composite")
    cpw_result = frame_dir / "cpw_result.jpg"
    shutil.copyfile(cpw_source, cpw_result)
    if reference_corr is not None:
        reference_output = scratch / "debug077_reference"
        run_cpw_direction(
            [args.frame], reference_corr.parent.parent, reference_output,
            image_dir, src, target, args.direction, warp_map_cache_mode="off",
            camera_group=args.camera_group,
        )
        reference_composite = (
            reference_output / args.direction / "composite_target_priority"
            / f"{args.frame:05d}.jpg"
        )
        compare_composites(cpw_result, reference_composite)
    runtime_profile["input_exporter_ms"] = exporter_ms
    runtime_profile["lm_ms"] = lm_ms
    runtime_profile.setdefault("erp_remap_ms", 0.0)
    runtime_profile.setdefault("ort_init_ms", 0.0)
    runtime_profile.setdefault("detector_ms", 0.0)
    runtime_profile.setdefault("densekp_inference_ms", 0.0)
    runtime_profile.setdefault("postprocess_ms", 0.0)
    runtime_profile.setdefault("cpw_ms", cpw_ms)
    (frame_dir / "densekp_runtime_profile.json").write_text(
        json.dumps(runtime_profile, indent=2, sort_keys=True), encoding="utf-8"
    )
    temporary.cleanup()
    print(
        f"[TIMING] input_exporter_ms={exporter_ms:.3f} "
        f"cpp_lm_ms={lm_ms:.3f} cpw_ms={cpw_ms:.3f}"
    )
    print("DENSEKP_END_TO_END_CPW_READY=PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
