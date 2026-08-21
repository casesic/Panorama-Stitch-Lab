"""Main DenseKP delivery pipeline — fisheye pair → DenseKP → matching → JSON."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import numpy as np

from .cache import CacheManager
from .camera.fisheye_equirect import (
    DEFAULT_ERP_WIDTH,
    DEFAULT_ERP_HEIGHT,
    R_EXTRINSIC_CAM2,
    R_EXTRINSIC_CAM4,
    build_equirect_lut,
    build_erp_valid_mask_from_lut_and_lens,
    build_fisheye_lens_mask,
    convert_fisheye_pair_to_erp,
    select_camera_params,
)
from .camera.erp_rotate import (
    DEFAULT_ERP_TO_ERP_ROTATE_R,
    rotate_erp_points,
)
from .camera.fisheye_points import erp_points_to_fisheye_points
from .image_io import read_image
from .person_match import match_persons
from .runners.base import DenseKPResult
from .schema import build_empty_json, build_matched_json, write_json
from .vis import save_visualizations

logger = logging.getLogger(__name__)


def run_pipeline(args, runtime=None) -> None:
    runtime_start = time.perf_counter()
    profile: dict[str, float | int | bool | str] = {}
    logging.basicConfig(level=logging.DEBUG if getattr(args, 'verbose', False) else logging.INFO,
                        format='%(levelname)s:%(name)s: %(message)s')
    out_dir = Path(args.out_dir)
    backend = args.backend
    coord_space = getattr(args, 'output_coord_space', 'erp')
    erp_w = args.erp_width
    erp_h = args.erp_height
    fisheye_cam2_path = str(Path(args.fisheye_cam2).resolve())
    fisheye_cam4_path = str(Path(args.fisheye_cam4).resolve())

    cache = CacheManager(
        out_dir, cache_mode=args.cache_mode,
        calibration_cache_dir=getattr(args, "calibration_cache_dir", None),
    )

    result_dir = out_dir / "result"
    result_dir.mkdir(parents=True, exist_ok=True)

    runner = runtime if runtime is not None else _build_runner(args)
    runner_profile_before = dict(getattr(runner, "profile_timing", {}))

    try:
        (erp_cam2, erp_cam4, fisheye_cam2, fisheye_cam4,
         luts, calibration) = _get_or_build_erp(args, cache, profile)
    except Exception as e:
        logger.error("Fisheye to ERP failed: %s", e)
        _write_err_json(result_dir, "fisheye_to_erp_failed", backend, coord_space, erp_w, erp_h, str(e))
        return

    try:
        mask_cam2, mask_cam4 = _get_or_build_valid_masks(
            args, cache, erp_cam2, erp_cam4,
            fisheye_cam2, fisheye_cam4, luts, calibration, profile,
        )
    except Exception as e:
        logger.warning("Valid mask build failed (non-fatal): %s", e)
        mask_cam2, mask_cam4 = None, None

    try:
        inference_start = time.perf_counter()
        result_cam2, result_cam4 = _run_densekp(
            args, cache, runner, erp_cam2, erp_cam4
        )
        profile["densekp_runner_total_ms"] = (
            time.perf_counter() - inference_start
        ) * 1000.0
    except Exception as e:
        logger.error("DenseKP inference failed: %s", e)
        _write_err_json(result_dir, "densekp_failed", backend, coord_space, erp_w, erp_h, str(e))
        return

    n2 = len(result_cam2.persons)
    n4 = len(result_cam4.persons)

    if n2 == 0 and n4 == 0:
        status = "no_detection_both"
    elif n2 == 0:
        status = "no_detection_cam2"
    elif n4 == 0:
        status = "no_detection_cam4"
    else:
        status = "ok"

    if status != "ok":
        err_json = build_empty_json(status, erp_w, erp_h, backend, coord_space)
        write_json(err_json, result_dir / "densekp_pairs.json")
        save_visualizations(out_dir, args.vis_option, result_cam2, result_cam4,
                            erp_cam2, erp_cam4)
        return

    postprocess_start = time.perf_counter()
    try:
        match_data = match_persons(
            result_cam2, result_cam4,
            erp_mask_cam2=mask_cam2,
            erp_mask_cam4=mask_cam4,
            person_idx_cam2=args.person_idx_cam2,
            person_idx_cam4=args.person_idx_cam4,
        )
    except Exception as e:
        logger.error("Person matching failed: %s", e)
        _write_err_json(result_dir, "densekp_failed", backend, coord_space, erp_w, erp_h, f"matching failed: {e}")
        return

    kps_cam2 = match_data["kps_cam2"]
    kps_cam4 = match_data["kps_cam4"]
    keep = match_data["keep"]

    if mask_cam2 is None or mask_cam4 is None:
        keep = np.ones(len(kps_cam2), dtype=bool)
        h2, w2 = erp_cam2.shape[:2]
        h4, w4 = erp_cam4.shape[:2]
        for i in range(len(kps_cam2)):
            x2, y2 = kps_cam2[i, 0], kps_cam2[i, 1]
            x4, y4 = kps_cam4[i, 0], kps_cam4[i, 1]
            if not (np.isfinite(x2) and np.isfinite(y2) and 0 <= x2 < w2 and 0 <= y2 < h2):
                keep[i] = False
            if not (np.isfinite(x4) and np.isfinite(y4) and 0 <= x4 < w4 and 0 <= y4 < h4):
                keep[i] = False
        match_data["keep"] = keep
        match_data["valid_count"] = int(keep.sum())
    profile["delivery_postprocess_ms"] = (
        time.perf_counter() - postprocess_start
    ) * 1000.0

    if coord_space == "erp":
        _write_erp_json(result_dir, backend, coord_space, args, erp_w, erp_h,
                        fisheye_cam2_path, fisheye_cam4_path, match_data, kps_cam2, kps_cam4, keep)
        _handle_save_option(args, out_dir, result_cam2, result_cam4, match_data, erp_cam2, erp_cam4)
        save_visualizations(out_dir, args.vis_option, result_cam2, result_cam4,
                            erp_cam2, erp_cam4, match_data, coord_space=coord_space)

    elif coord_space == "erp_rotate":
        coordinate_start = time.perf_counter()
        kps_cam2_rot = rotate_erp_points(kps_cam2, erp_w, erp_h)
        kps_cam4_rot = rotate_erp_points(kps_cam4, erp_w, erp_h)
        profile["coordinate_transform_ms"] = (
            time.perf_counter() - coordinate_start
        ) * 1000.0

        _write_erp_json(result_dir, backend, coord_space, args, erp_w, erp_h,
                        fisheye_cam2_path, fisheye_cam4_path, match_data,
                        kps_cam2_rot, kps_cam4_rot, keep,
                        json_name="densekp_pairs_erp_rotate.json",
                        erp_rotate_matrix=DEFAULT_ERP_TO_ERP_ROTATE_R,
                        profile=profile)
        _handle_save_option(args, out_dir, result_cam2, result_cam4, match_data,
                            erp_cam2, erp_cam4)
        save_visualizations(out_dir, args.vis_option, result_cam2, result_cam4,
                            erp_cam2, erp_cam4, match_data, coord_space="erp",
                            det_image_cam2=erp_cam2, det_image_cam4=erp_cam4)

    elif coord_space == "fisheye":
        lut2, lut4 = luts
        fish2 = read_image(args.fisheye_cam2)
        fish4 = read_image(args.fisheye_cam4)
        h2, w2 = fish2.shape[:2]
        h4, w4 = fish4.shape[:2]

        kps_cam2_fish = erp_points_to_fisheye_points(kps_cam2, lut2[0], lut2[1])
        kps_cam4_fish = erp_points_to_fisheye_points(kps_cam4, lut4[0], lut4[1])

        keep_fish = keep.copy()
        finite2 = np.isfinite(kps_cam2_fish).all(axis=1)
        finite4 = np.isfinite(kps_cam4_fish).all(axis=1)
        keep_fish = keep_fish & finite2 & finite4

        fisheye_meta = {
            "image_size_cam2": [int(w2), int(h2)],
            "image_size_cam4": [int(w4), int(h4)],
        }

        _write_fisheye_json(result_dir, backend, coord_space, args, fisheye_meta,
                            fisheye_cam2_path, fisheye_cam4_path,
                            match_data, kps_cam2_fish, kps_cam4_fish, keep_fish,
                            json_name="densekp_pairs_fisheye.json")
        _handle_save_option(args, out_dir, result_cam2, result_cam4, match_data,
                            erp_cam2, erp_cam4)
        match_data_fish = dict(match_data)
        match_data_fish["keep"] = keep_fish
        match_data_fish["valid_count"] = int(keep_fish.sum())
        save_visualizations(out_dir, args.vis_option, result_cam2, result_cam4,
                            erp_cam2, erp_cam4, match_data, coord_space="erp",
                            det_image_cam2=erp_cam2, det_image_cam4=erp_cam4)
        save_visualizations(out_dir, args.vis_option, result_cam2, result_cam4,
                            fish2, fish4, match_data_fish, coord_space=coord_space,
                            kps_cam2=kps_cam2_fish, kps_cam4=kps_cam4_fish,
                            det_image_cam2=erp_cam2, det_image_cam4=erp_cam4)

    elif coord_space == "both":
        _write_erp_json(result_dir, backend, "erp", args, erp_w, erp_h,
                        fisheye_cam2_path, fisheye_cam4_path, match_data, kps_cam2, kps_cam4, keep,
                        json_name="densekp_pairs_erp.json", profile=profile)

        coordinate_start = time.perf_counter()
        kps_cam2_rot = rotate_erp_points(kps_cam2, erp_w, erp_h)
        kps_cam4_rot = rotate_erp_points(kps_cam4, erp_w, erp_h)
        profile["coordinate_transform_ms"] = (
            time.perf_counter() - coordinate_start
        ) * 1000.0

        _write_erp_json(result_dir, backend, "erp_rotate", args, erp_w, erp_h,
                        fisheye_cam2_path, fisheye_cam4_path, match_data,
                        kps_cam2_rot, kps_cam4_rot, keep,
                        json_name="densekp_pairs_erp_rotate.json",
                        erp_rotate_matrix=DEFAULT_ERP_TO_ERP_ROTATE_R,
                        profile=profile)

        _handle_save_option(args, out_dir, result_cam2, result_cam4, match_data,
                            erp_cam2, erp_cam4)
        save_visualizations(out_dir, args.vis_option, result_cam2, result_cam4,
                            erp_cam2, erp_cam4, match_data, coord_space="erp",
                            det_image_cam2=erp_cam2, det_image_cam4=erp_cam4)

    if getattr(args, "profile_runtime", False):
        runner_profile_total = getattr(runner, "profile_timing", {})
        runner_profile = {
            name: float(value) - float(runner_profile_before.get(name, 0.0))
            for name, value in runner_profile_total.items()
        }
        profile.update(runner_profile)
        profile["detector_ms"] = float(
            runner_profile.get("detector_inference_ms", 0.0)
        )
        profile["postprocess_ms"] = float(
            profile.get("delivery_postprocess_ms", 0.0)
            + runner_profile.get("densekp_preprocess_ms", 0.0)
            + runner_profile.get("densekp_postprocess_ms", 0.0)
        )
        profile["session_creation_count"] = int(
            getattr(runner, "session_creation_count", 0)
        )
        profile["detector_initialization_count"] = int(
            getattr(runner, "detector_initialization_count", 0)
        )
        persistent = runtime is not None
        profile["persistent_runtime"] = persistent
        profile["runtime_startup_ms"] = float(
            getattr(runtime, "startup_ms", 0.0) if persistent else
            runner_profile.get("ort_init_ms", 0.0)
            + runner_profile.get("detector_init_ms", 0.0)
        )
        profile["runtime_load_count"] = int(
            getattr(runtime, "load_count", 1) if persistent else 1
        )
        profile["frame_level_runner_creation"] = not persistent
        profile["model_weights_reloaded_per_process"] = not persistent
        profile["lm_ms"] = 0.0
        profile["cpw_ms"] = 0.0
        profile["runtime_total_ms"] = (
            time.perf_counter() - runtime_start
        ) * 1000.0
        (result_dir / "densekp_runtime_profile.json").write_text(
            json.dumps(profile, indent=2, sort_keys=True), encoding="utf-8"
        )
        inference_profile = {
            "preprocess_ms": float(
                runner_profile.get("densekp_preprocess_ms", 0.0)
            ),
            "detector_ms": float(
                runner_profile.get("detector_inference_ms", 0.0)
            ),
            "bbox_postprocess_ms": float(
                runner_profile.get("bbox_postprocess_ms", 0.0)
            ),
            "densekp_model_ms": float(
                runner_profile.get("densekp_inference_ms", 0.0)
            ),
            "decode_ms": float(
                runner_profile.get("densekp_postprocess_ms", 0.0)
            ),
            "coordinate_ms": float(
                profile.get("coordinate_transform_ms", 0.0)
            ),
        }
        (result_dir / "densekp_inference_profile.json").write_text(
            json.dumps(inference_profile, indent=2, sort_keys=True),
            encoding="utf-8",
        )


def _err_json_name(coord_space: str) -> str:
    if coord_space == "erp_rotate":
        return "densekp_pairs_erp_rotate.json"
    if coord_space == "fisheye":
        return "densekp_pairs_fisheye.json"
    return "densekp_pairs.json"


def _write_err_json(result_dir, status, backend, coord_space, erp_w, erp_h, error_msg):
    err_json = build_empty_json(status, erp_w, erp_h, backend, coord_space,
                                extra={"error_message": error_msg})
    write_json(err_json, result_dir / _err_json_name(coord_space))


def _write_erp_json(result_dir, backend, coord_space, args, erp_w, erp_h,
                    fisheye_cam2_path, fisheye_cam4_path, match_data,
                    kps_cam2, kps_cam4, keep,
                    json_name="densekp_pairs.json",
                    erp_rotate_matrix=None, profile=None):
    start = time.perf_counter()
    json_data = build_matched_json(
        backend=backend,
        image_width=erp_w, image_height=erp_h,
        camera=args.camera,
        fisheye_cam2=fisheye_cam2_path,
        fisheye_cam4=fisheye_cam4_path,
        coord_space=coord_space,
        match_result=match_data,
        kps_cam2=kps_cam2, kps_cam4=kps_cam4, keep=keep,
        erp_rotate_matrix=erp_rotate_matrix,
    )
    write_json(json_data, result_dir / json_name)
    if profile is not None:
        profile["json_export_ms"] = float(profile.get("json_export_ms", 0.0)) + (
            time.perf_counter() - start
        ) * 1000.0
    logger.info("Wrote %s: valid=%d/%d", json_name, int(keep.sum()), len(keep))


def _write_fisheye_json(result_dir, backend, coord_space, args, fisheye_meta,
                        fisheye_cam2_path, fisheye_cam4_path, match_data,
                        kps_cam2, kps_cam4, keep, json_name="densekp_pairs_fisheye.json"):
    json_data = build_matched_json(
        backend=backend,
        image_width=int(fisheye_meta["image_size_cam2"][0]),
        image_height=int(fisheye_meta["image_size_cam2"][1]),
        camera=args.camera,
        fisheye_cam2=fisheye_cam2_path,
        fisheye_cam4=fisheye_cam4_path,
        coord_space=coord_space,
        match_result=match_data,
        kps_cam2=kps_cam2, kps_cam4=kps_cam4, keep=keep,
        fisheye_meta=fisheye_meta,
    )
    write_json(json_data, result_dir / json_name)
    logger.info("Wrote %s: valid=%d/%d (fisheye)", json_name, int(keep.sum()), len(keep))


def _build_runner(args):
    from .runtime import DenseKPRuntime

    return DenseKPRuntime.from_args(args)._runner


def _get_or_build_erp(args, cache, profile):
    start = time.perf_counter()
    fisheye2 = read_image(args.fisheye_cam2)
    fisheye4 = read_image(args.fisheye_cam4)
    profile["image_load_ms"] = (time.perf_counter() - start) * 1000.0
    ds2, ds4 = select_camera_params(args.camera)
    r2 = R_EXTRINSIC_CAM2
    r4 = R_EXTRINSIC_CAM4
    calibration = {
        "schema_version": 1,
        "camera_group": args.camera,
        "erp_width": int(args.erp_width),
        "erp_height": int(args.erp_height),
        "fisheye_cam2_size": [int(fisheye2.shape[1]), int(fisheye2.shape[0])],
        "fisheye_cam4_size": [int(fisheye4.shape[1]), int(fisheye4.shape[0])],
        "cam2_params": {key: float(value) for key, value in sorted(ds2.items())},
        "cam4_params": {key: float(value) for key, value in sorted(ds4.items())},
        "cam2_extrinsic": r2.tolist(),
        "cam4_extrinsic": r4.tolist(),
    }
    start = time.perf_counter()
    luts = cache.read_calibration_luts(calibration)
    profile["lut_cache_load_ms"] = (time.perf_counter() - start) * 1000.0
    if luts is None:
        if cache.cache_mode == "read_only":
            raise RuntimeError("calibration LUT cache is missing")
        lut_start = time.perf_counter()
        logger.info("Building cam2 calibration LUT...")
        lut2 = build_equirect_lut(
            ds2, args.erp_width, args.erp_height,
            R_extrinsic=r2, verbose=args.verbose,
        )
        logger.info("Building cam4 calibration LUT...")
        lut4 = build_equirect_lut(
            ds4, args.erp_width, args.erp_height,
            R_extrinsic=r4, verbose=args.verbose,
        )
        luts = (lut2, lut4)
        cache.write_calibration_luts(calibration, luts)
        profile["lut_build_ms"] = (time.perf_counter() - lut_start) * 1000.0
    else:
        profile["lut_build_ms"] = 0.0

    start = time.perf_counter()
    erp_cam2, erp_cam4 = convert_fisheye_pair_to_erp(
        fisheye2, fisheye4, luts[0], luts[1], args.erp_width, args.erp_height,
    )
    profile["erp_remap_ms"] = (time.perf_counter() - start) * 1000.0
    return erp_cam2, erp_cam4, fisheye2, fisheye4, luts, calibration


def _get_or_build_valid_masks(
    args, cache, erp_cam2, erp_cam4, fisheye_cam2, fisheye_cam4, luts,
    calibration, profile,
):
    start = time.perf_counter()
    cached = cache.read_calibration_valid_masks(calibration)
    profile["mask_cache_load_ms"] = (time.perf_counter() - start) * 1000.0
    if cached is not None:
        profile["mask_build_ms"] = 0.0
        return cached
    if cache.cache_mode == "read_only":
        raise RuntimeError("calibration valid-mask cache is missing")

    start = time.perf_counter()
    lut2, lut4 = luts
    lens2 = build_fisheye_lens_mask(fisheye_cam2)
    lens4 = build_fisheye_lens_mask(fisheye_cam4)
    mask_cam2 = build_erp_valid_mask_from_lut_and_lens(
        lut2, lens2, (erp_cam2.shape[1], erp_cam2.shape[0]),
    )
    mask_cam4 = build_erp_valid_mask_from_lut_and_lens(
        lut4, lens4, (erp_cam4.shape[1], erp_cam4.shape[0]),
    )
    mask_cam2 = mask_cam2 > 0
    mask_cam4 = mask_cam4 > 0

    cache.write_calibration_valid_masks(calibration, (mask_cam2, mask_cam4))
    profile["mask_build_ms"] = (time.perf_counter() - start) * 1000.0

    return mask_cam2, mask_cam4


def _run_densekp(args, cache, runner, erp_cam2, erp_cam4):
    result_cam2 = _run_densekp_one_camera(runner, erp_cam2, "cam2")
    result_cam4 = _run_densekp_one_camera(runner, erp_cam4, "cam4")
    return result_cam2, result_cam4


def _run_densekp_one_camera(runner, erp_img, camera_name):
    h, w = erp_img.shape[:2]
    image_path = f"erp_{camera_name}"
    result = runner.run(erp_img, image_path, camera_name)
    return result


def _handle_save_option(args, out_dir, result_cam2, result_cam4, match_data, erp_cam2, erp_cam4):
    save_opt = args.save_option
    if save_opt == "none":
        return

    if save_opt in ("npz", "all"):
        _save_npz(out_dir, "cam2", result_cam2, erp_cam2.shape[:2])
        _save_npz(out_dir, "cam4", result_cam4, erp_cam4.shape[:2])

    if save_opt in ("debug", "all"):
        debug_dir = out_dir / "debug"
        debug_dir.mkdir(parents=True, exist_ok=True)
        args_data = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
        with open(debug_dir / "args.json", "w", encoding="utf-8") as f:
            json.dump(args_data, f, ensure_ascii=False, indent=2, default=str)
        with open(debug_dir / "match_candidates.json", "w", encoding="utf-8") as f:
            json.dump({
                "cam2_person_count": len(result_cam2.persons),
                "cam4_person_count": len(result_cam4.persons),
                "selected_match": {
                    "cam2_person_index": match_data["cam2_person_index"],
                    "cam4_person_index": match_data["cam4_person_index"],
                    "valid_count": match_data["valid_count"],
                },
            }, f, ensure_ascii=False, indent=2)
        with open(debug_dir / "summary.json", "w", encoding="utf-8") as f:
            json.dump({
                "backend": args.backend,
                "valid_count": int(match_data["valid_count"]),
                "total_points": int(match_data["total_points"]),
            }, f, ensure_ascii=False, indent=2)


def _save_npz(out_dir, camera_name, result, image_shape):
    npz_dir = out_dir / camera_name
    npz_dir.mkdir(parents=True, exist_ok=True)
    for p in result.persons:
        path = npz_dir / f"person_{p.person_index:03d}.npz"
        np.savez_compressed(str(path),
                            keypoints_orig_xy=p.keypoints.astype(np.float32),
                            bbox_xyxy=np.array(p.bbox, dtype=np.float32),
                            keypoint_value=(p.raw_scores if p.raw_scores is not None
                                            else np.ones(len(p.keypoints), dtype=np.float32)),
                            image_size=np.array(image_shape, dtype=np.int32))
