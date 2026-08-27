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
from .image_io import read_image, read_image_pair_profiled
from .overlap_roi import (
    build_compact_detection_rois, build_detection_rois,
    crop_erp_roi, deduplicate_periodic_persons, merge_roi_results,
    restore_result_to_full_erp,
)
from .person_match import match_all_persons
from .profiling import write_runtime_profiles
from .runners.base import DenseKPResult
from .schema import build_empty_json, build_matched_json, write_json
from .vis import save_visualizations

logger = logging.getLogger(__name__)
# The compact-strips candidate is retained for repeatable attribution, but the
# fixed-batch YOLOX benchmark makes the two-component layout the faster policy.
DETECTION_ROI_STRATEGY = "component_bbox"


def _build_roi_geometry(mask_cam2, mask_cam4):
    strict, detection, component_rois = build_detection_rois(mask_cam2, mask_cam4)
    compact_rois = (
        build_compact_detection_rois(detection, component_rois)
        if DETECTION_ROI_STRATEGY == "compact_strips" else component_rois
    )
    return strict, detection, component_rois, compact_rois


def load_erp_pair(config, runtime=None):
    """Load one fisheye pair and produce normal-ERP images without inference."""
    cache = CacheManager(
        config.out_dir,
        cache_mode=config.cache_mode,
        calibration_cache_dir=config.calibration_cache_dir,
    )
    profile: dict[str, float | int | bool | str] = {}
    erp_cam2, erp_cam4, *_ = _get_or_build_erp(
        config, cache, profile, runtime,
    )
    return erp_cam2, erp_cam4


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
         luts, calibration) = _get_or_build_erp(
            args, cache, profile, runtime,
        )
    except Exception as e:
        logger.error("Fisheye to ERP failed: %s", e)
        _write_err_json(result_dir, "fisheye_to_erp_failed", backend, coord_space, erp_w, erp_h, str(e))
        return

    try:
        mask_cam2, mask_cam4 = _get_or_build_valid_masks(
            args, cache, erp_cam2, erp_cam4,
            fisheye_cam2, fisheye_cam4, luts, calibration, profile,
            runtime,
        )
    except Exception as e:
        logger.error("Valid mask build failed: %s", e)
        _write_err_json(result_dir, "valid_mask_failed", backend, coord_space,
                        erp_w, erp_h, str(e))
        return

    roi_start = time.perf_counter()
    roi_geometry = (runtime.get_static_geometry(
                        calibration, f"overlap_roi_{DETECTION_ROI_STRATEGY}")
                    if runtime is not None else None)
    if roi_geometry is None:
        roi_geometry = _build_roi_geometry(mask_cam2, mask_cam4)
        if runtime is not None:
            runtime.put_static_geometry(calibration, f"overlap_roi_{DETECTION_ROI_STRATEGY}", roi_geometry)
        profile["overlap_roi_memory_hit_count"] = 0
    else:
        profile["overlap_roi_memory_hit_count"] = 1
    strict_overlap, detection_mask, component_rois, detection_rois = roi_geometry
    profile["overlap_roi_build_ms"] = (time.perf_counter() - roi_start) * 1000.0
    profile["full_erp_pixels"] = int(erp_w * erp_h)
    profile["detection_roi_pixels"] = int(sum(r.pixel_count for r in detection_rois))
    profile["detection_roi_ratio"] = float(profile["detection_roi_pixels"] / max(1, erp_w * erp_h))
    profile["detection_roi_count"] = len(detection_rois)
    profile["roi_count"] = len(detection_rois)
    profile["roi_source_pixels_total"] = profile["detection_roi_pixels"]
    profile["roi_source_pixel_ratio"] = profile["detection_roi_ratio"]
    profile["roi_mask_pixels_total"] = int(detection_mask.sum())
    profile["roi_fill_ratio"] = float(detection_mask.sum() / max(1, profile["detection_roi_pixels"]))
    profile["detector_call_count"] = 2 * len(detection_rois)
    profile["roi_specification"] = []
    for index, roi in enumerate(detection_rois):
        mask_pixels = int(detection_mask[
            roi.y_start:roi.y_start + roi.height
        ][:, roi.x_indices()].sum())
        profile["roi_specification"].append({
            "roi_index": index, "x_start": roi.x_start,
            "x_segments": ([[roi.x_start, roi.erp_width],
                            [0, roi.x_start + roi.width - roi.erp_width]]
                           if roi.wraps_seam else
                           [[roi.x_start, roi.x_start + roi.width]]),
            "y0": roi.y_start, "y1": roi.y_start + roi.height,
            "width": roi.width, "height": roi.height,
            "roi_pixels": roi.pixel_count,
            "mask_pixels_inside_roi": mask_pixels,
            "fill_ratio": mask_pixels / max(1, roi.pixel_count),
            "aspect_ratio": roi.width / max(1, roi.height),
        })

    try:
        inference_start = time.perf_counter()
        result_cam2, result_cam4 = _run_densekp(
            runner, erp_cam2, erp_cam4, detection_rois
        )
        profile["densekp_runner_total_ms"] = (
            time.perf_counter() - inference_start
        ) * 1000.0
        profile["densekp_person_count_before_dedup"] = int(
            getattr(result_cam2, "person_count_before_dedup", len(result_cam2.persons))
            + getattr(result_cam4, "person_count_before_dedup", len(result_cam4.persons))
        )
        profile["densekp_person_count_after_dedup"] = int(
            len(result_cam2.persons) + len(result_cam4.persons)
        )
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
        write_json(err_json, result_dir / _err_json_name(coord_space))
        save_visualizations(out_dir, args.vis_option, result_cam2, result_cam4,
                            erp_cam2, erp_cam4)
        write_runtime_profiles(
            args, result_dir, profile, runner, runner_profile_before,
            runtime, runtime_start,
        )
        return

    postprocess_start = time.perf_counter()
    try:
        match_data = match_all_persons(
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

    profile["delivery_postprocess_ms"] = (
        time.perf_counter() - postprocess_start
    ) * 1000.0

    if coord_space == "erp":
        _write_erp_json(result_dir, backend, coord_space, args, erp_w, erp_h,
                        fisheye_cam2_path, fisheye_cam4_path, match_data)
        _handle_save_option(args, out_dir, result_cam2, result_cam4, match_data, erp_cam2, erp_cam4)
        save_visualizations(out_dir, args.vis_option, result_cam2, result_cam4,
                            erp_cam2, erp_cam4, match_data, coord_space=coord_space,
                            detection_rois=detection_rois,
                            strict_overlap_mask=strict_overlap,
                            detection_mask=detection_mask)

    elif coord_space == "erp_rotate":
        coordinate_start = time.perf_counter()
        transformed = _transform_matches(match_data, lambda p: rotate_erp_points(p, erp_w, erp_h))
        profile["coordinate_transform_ms"] = (
            time.perf_counter() - coordinate_start
        ) * 1000.0

        _write_erp_json(result_dir, backend, coord_space, args, erp_w, erp_h,
                        fisheye_cam2_path, fisheye_cam4_path, transformed,
                        json_name="densekp_pairs_erp_rotate.json",
                        erp_rotate_matrix=DEFAULT_ERP_TO_ERP_ROTATE_R,
                        profile=profile)
        _handle_save_option(args, out_dir, result_cam2, result_cam4, match_data,
                            erp_cam2, erp_cam4)
        save_visualizations(out_dir, args.vis_option, result_cam2, result_cam4,
                            erp_cam2, erp_cam4, match_data, coord_space="erp",
                            det_image_cam2=erp_cam2, det_image_cam4=erp_cam4,
                            detection_rois=detection_rois,
                            strict_overlap_mask=strict_overlap,
                            detection_mask=detection_mask)

    elif coord_space == "fisheye":
        lut2, lut4 = luts
        fish2 = read_image(args.fisheye_cam2)
        fish4 = read_image(args.fisheye_cam4)
        h2, w2 = fish2.shape[:2]
        h4, w4 = fish4.shape[:2]

        transformed = _transform_matches_two(
            match_data,
            lambda p: erp_points_to_fisheye_points(p, lut2[0], lut2[1]),
            lambda p: erp_points_to_fisheye_points(p, lut4[0], lut4[1]),
            range_sizes=((w2, h2), (w4, h4)),
        )

        fisheye_meta = {
            "image_size_cam2": [int(w2), int(h2)],
            "image_size_cam4": [int(w4), int(h4)],
        }

        _write_fisheye_json(result_dir, backend, coord_space, args, fisheye_meta,
                            fisheye_cam2_path, fisheye_cam4_path,
                            transformed,
                            json_name="densekp_pairs_fisheye.json")
        _handle_save_option(args, out_dir, result_cam2, result_cam4, match_data,
                            erp_cam2, erp_cam4)
        save_visualizations(out_dir, args.vis_option, result_cam2, result_cam4,
                            erp_cam2, erp_cam4, match_data, coord_space="erp",
                            det_image_cam2=erp_cam2, det_image_cam4=erp_cam4,
                            detection_rois=detection_rois,
                            strict_overlap_mask=strict_overlap,
                            detection_mask=detection_mask)
        save_visualizations(out_dir, args.vis_option, result_cam2, result_cam4,
                            fish2, fish4, transformed, coord_space=coord_space,
                            det_image_cam2=erp_cam2, det_image_cam4=erp_cam4)

    elif coord_space == "both":
        _write_erp_json(result_dir, backend, "erp", args, erp_w, erp_h,
                        fisheye_cam2_path, fisheye_cam4_path, match_data,
                        json_name="densekp_pairs_erp.json", profile=profile)

        coordinate_start = time.perf_counter()
        transformed = _transform_matches(match_data, lambda p: rotate_erp_points(p, erp_w, erp_h))
        profile["coordinate_transform_ms"] = (
            time.perf_counter() - coordinate_start
        ) * 1000.0

        _write_erp_json(result_dir, backend, "erp_rotate", args, erp_w, erp_h,
                        fisheye_cam2_path, fisheye_cam4_path, transformed,
                        json_name="densekp_pairs_erp_rotate.json",
                        erp_rotate_matrix=DEFAULT_ERP_TO_ERP_ROTATE_R,
                        profile=profile)

        _handle_save_option(args, out_dir, result_cam2, result_cam4, match_data,
                            erp_cam2, erp_cam4)
        save_visualizations(out_dir, args.vis_option, result_cam2, result_cam4,
                            erp_cam2, erp_cam4, match_data, coord_space="erp",
                            det_image_cam2=erp_cam2, det_image_cam4=erp_cam4,
                            detection_rois=detection_rois,
                            strict_overlap_mask=strict_overlap,
                            detection_mask=detection_mask)

    write_runtime_profiles(
        args, result_dir, profile, runner, runner_profile_before,
        runtime, runtime_start,
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
        erp_rotate_matrix=erp_rotate_matrix,
    )
    write_json(json_data, result_dir / json_name)
    if profile is not None:
        profile["json_export_ms"] = float(profile.get("json_export_ms", 0.0)) + (
            time.perf_counter() - start
        ) * 1000.0
    logger.info("Wrote %s: matched_persons=%d", json_name,
                len(match_data.get("matches", [])))


def _write_fisheye_json(result_dir, backend, coord_space, args, fisheye_meta,
                        fisheye_cam2_path, fisheye_cam4_path, match_data,
                        json_name="densekp_pairs_fisheye.json"):
    json_data = build_matched_json(
        backend=backend,
        image_width=int(fisheye_meta["image_size_cam2"][0]),
        image_height=int(fisheye_meta["image_size_cam2"][1]),
        camera=args.camera,
        fisheye_cam2=fisheye_cam2_path,
        fisheye_cam4=fisheye_cam4_path,
        coord_space=coord_space,
        match_result=match_data,
        fisheye_meta=fisheye_meta,
    )
    write_json(json_data, result_dir / json_name)
    logger.info("Wrote %s: matched_persons=%d (fisheye)", json_name,
                len(match_data.get("matches", [])))


def _transform_matches_two(match_data, transform2, transform4, range_sizes=None):
    transformed = dict(match_data)
    transformed["matches"] = []
    for match in match_data.get("matches", []):
        item = dict(match)
        item["kps_cam2"] = transform2(match["kps_cam2"])
        item["kps_cam4"] = transform4(match["kps_cam4"])
        keep = np.asarray(match["keep"], dtype=bool).copy()
        keep &= np.isfinite(item["kps_cam2"]).all(axis=1)
        keep &= np.isfinite(item["kps_cam4"]).all(axis=1)
        if range_sizes is not None:
            for key, (width, height) in zip(("kps_cam2", "kps_cam4"), range_sizes):
                points = item[key]
                keep &= ((points[:, 0] >= 0) & (points[:, 0] < width)
                         & (points[:, 1] >= 0) & (points[:, 1] < height))
        item["keep"] = keep
        item["valid_count"] = int(keep.sum())
        transformed["matches"].append(item)
    return transformed


def _transform_matches(match_data, transform):
    return _transform_matches_two(match_data, transform, transform)


def _finite_matrix_for_json(matrix):
    array = np.asarray(matrix)
    return [[(float(value) if np.isfinite(value) else None) for value in row]
            for row in array]


def _build_runner(args):
    from .runtime import DenseKPRuntime

    return DenseKPRuntime.from_args(args)._runner


def _get_or_build_erp(args, cache, profile, runtime=None):
    fisheye2, fisheye4, image_io_profile = read_image_pair_profiled(
        args.fisheye_cam2, args.fisheye_cam4,
        executor=(runtime.image_io_executor if runtime is not None else None),
    )
    profile["image_io_profile"] = image_io_profile
    profile["cam2_read_ms"] = image_io_profile["cam2_read_ms"]
    profile["cam4_read_ms"] = image_io_profile["cam4_read_ms"]
    profile["image_load_ms"] = image_io_profile["total_read_ms"]
    profile["image_disk_read_ms"] = image_io_profile["disk_read_ms"]
    profile["image_decode_ms"] = image_io_profile["decode_ms"]
    profile["image_copy_ms"] = image_io_profile["copy_ms"]
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
    if runtime is not None:
        calibration = runtime.intern_calibration(calibration)
    start = time.perf_counter()
    luts = (
        runtime.get_static_geometry(calibration, "luts")
        if runtime is not None else None
    )
    if luts is not None:
        runtime.static_cache_counts["lut_memory_hit_count"] += 1
        profile["lut_memory_hit_count"] = 1
        profile["lut_disk_load_count"] = 0
    else:
        luts = cache.read_calibration_luts(calibration)
        if luts is not None and runtime is not None:
            runtime.static_cache_counts["lut_disk_load_count"] += 1
            profile["lut_disk_load_count"] = 1
            profile["lut_memory_hit_count"] = 0
            runtime.put_static_geometry(calibration, "luts", luts)
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
        if runtime is not None:
            runtime.put_static_geometry(calibration, "luts", luts)
        profile["lut_build_ms"] = (time.perf_counter() - lut_start) * 1000.0
    else:
        profile["lut_build_ms"] = 0.0

    start = time.perf_counter()
    requested_remap_backend = getattr(args, "erp_remap_backend", "cuda")
    actual_remap_backend = "cpu"
    remapper = None
    temporary_remapper = False
    if requested_remap_backend == "cuda":
        try:
            from .camera.cuda_remap import CudaErpRemapper

            remapper = (
                runtime.get_static_geometry(calibration, "cuda_erp_remapper")
                if runtime is not None else None
            )
            if remapper is None:
                library_path = getattr(args, "erp_remap_cuda_library", None)
                if not library_path:
                    raise FileNotFoundError("--erp-remap-cuda-library is missing")
                remapper = CudaErpRemapper(
                    library_path,
                    fisheye2.shape[:2],
                    (args.erp_height, args.erp_width),
                    luts,
                )
                if runtime is not None:
                    runtime.put_static_geometry(
                        calibration, "cuda_erp_remapper", remapper,
                    )
                else:
                    temporary_remapper = True
            erp_cam2, erp_cam4 = remapper.remap_pair(fisheye2, fisheye4)
            actual_remap_backend = "cuda"
        except Exception as error:
            logger.warning("CUDA ERP remap unavailable; using CPU: %s", error)
    if actual_remap_backend == "cpu":
        erp_cam2, erp_cam4 = convert_fisheye_pair_to_erp(
            fisheye2, fisheye4, luts[0], luts[1],
            args.erp_width, args.erp_height,
        )
    if temporary_remapper:
        remapper.close()
    profile["erp_remap_ms"] = (time.perf_counter() - start) * 1000.0
    profile["erp_remap_backend"] = actual_remap_backend
    return erp_cam2, erp_cam4, fisheye2, fisheye4, luts, calibration


def _get_or_build_valid_masks(
    args, cache, erp_cam2, erp_cam4, fisheye_cam2, fisheye_cam4, luts,
    calibration, profile, runtime=None,
):
    start = time.perf_counter()
    cached = (
        runtime.get_static_geometry(calibration, "valid_masks")
        if runtime is not None else None
    )
    if cached is not None:
        runtime.static_cache_counts["valid_mask_memory_hit_count"] += 1
        profile["valid_mask_memory_hit_count"] = 1
        profile["valid_mask_disk_load_count"] = 0
    else:
        cached = cache.read_calibration_valid_masks(calibration)
        if cached is not None and runtime is not None:
            runtime.static_cache_counts["valid_mask_disk_load_count"] += 1
            profile["valid_mask_disk_load_count"] = 1
            profile["valid_mask_memory_hit_count"] = 0
            runtime.put_static_geometry(calibration, "valid_masks", cached)
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
    if runtime is not None:
        runtime.put_static_geometry(
            calibration, "valid_masks", (mask_cam2, mask_cam4),
        )
    profile["mask_build_ms"] = (time.perf_counter() - start) * 1000.0

    return mask_cam2, mask_cam4


def _run_densekp(runner, erp_cam2, erp_cam4, detection_rois):
    if not detection_rois:
        return (
            DenseKPResult("erp_cam2_overlap_roi", erp_cam2.shape[1], erp_cam2.shape[0], [], "cam2"),
            DenseKPResult("erp_cam4_overlap_roi", erp_cam4.shape[1], erp_cam4.shape[0], [], "cam4"),
        )
    outputs=[]
    for camera,image in (("cam2",erp_cam2),("cam4",erp_cam4)):
        restored=[]
        for roi_index,roi in enumerate(detection_rois):
            crop=crop_erp_roi(image,roi)
            restored.append(restore_result_to_full_erp(
                _run_densekp_one_camera(runner,crop,f"{camera}_compact_{roi_index}"),roi,
            ))
        merged=merge_roi_results(restored,image.shape[1],image.shape[0],camera)
        before=sum(len(r.persons) for r in restored)
        if DETECTION_ROI_STRATEGY == "compact_strips":
            merged.persons=deduplicate_periodic_persons(merged.persons,image.shape[1])
        setattr(merged,"person_count_before_dedup",before)
        outputs.append(merged)
    return tuple(outputs)


def run_densekp_erp_window(
    runner,
    frames: list[tuple[int, np.ndarray, np.ndarray]],
    detection_rois,
    densekp_batch_size: int,
) -> tuple[dict[int, tuple[DenseKPResult, DenseKPResult]], dict[str, float]]:
    """Experimental delivery batch over preloaded ERP frames.

    Detection retains the fixed-B1 production model; only DenseKP person crops
    are combined across frame/camera/ROI.  Metadata is restored by explicit
    list position rather than inference result ordering.
    """
    timing = {"roi_extraction_ms": 0.0, "metadata_restoration_ms": 0.0}
    if not detection_rois:
        return {
            frame: (
                DenseKPResult("erp_cam2_overlap_roi", cam2.shape[1], cam2.shape[0], [], "cam2"),
                DenseKPResult("erp_cam4_overlap_roi", cam4.shape[1], cam4.shape[0], [], "cam4"),
            ) for frame, cam2, cam4 in frames
        }, timing

    items = []
    metadata = []
    started = time.perf_counter()
    for frame, cam2, cam4 in frames:
        for camera, image in (("cam2", cam2), ("cam4", cam4)):
            for roi_index, roi in enumerate(detection_rois):
                items.append((
                    crop_erp_roi(image, roi),
                    f"{frame}_{camera}_compact_{roi_index}",
                    f"{camera}_compact_{roi_index}",
                ))
                metadata.append((frame, camera, roi_index, roi, image.shape))
    timing["roi_extraction_ms"] = (time.perf_counter() - started) * 1000.0

    results = runner.run_batch(items, densekp_batch_size=densekp_batch_size)
    started = time.perf_counter()
    grouped: dict[tuple[int, str], list[DenseKPResult]] = {}
    shapes: dict[tuple[int, str], tuple[int, ...]] = {}
    for result, (frame, camera, _, roi, shape) in zip(results, metadata):
        grouped.setdefault((frame, camera), []).append(
            restore_result_to_full_erp(result, roi))
        shapes[(frame, camera)] = shape

    output = {}
    for frame, _, _ in frames:
        camera_results = []
        for camera in ("cam2", "cam4"):
            shape = shapes[(frame, camera)]
            restored = grouped[(frame, camera)]
            merged = merge_roi_results(restored, shape[1], shape[0], camera)
            before = sum(len(result.persons) for result in restored)
            if DETECTION_ROI_STRATEGY == "compact_strips":
                merged.persons = deduplicate_periodic_persons(
                    merged.persons, shape[1])
            setattr(merged, "person_count_before_dedup", before)
            camera_results.append(merged)
        output[frame] = tuple(camera_results)
    timing["metadata_restoration_ms"] = (
        time.perf_counter() - started) * 1000.0
    return output, timing


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
                "candidate_indices_cam2": match_data["candidate_indices_cam2"],
                "candidate_indices_cam4": match_data["candidate_indices_cam4"],
                "pairwise_cost_matrix_deg": _finite_matrix_for_json(match_data["pairwise_cost_matrix_deg"]),
                "pairwise_median_angular_error_deg": _finite_matrix_for_json(match_data["pairwise_median_angular_error_deg"]),
                "pairwise_eligible": match_data["pairwise_eligible"].tolist(),
                "pairwise_valid_counts": match_data["pairwise_valid_counts"].tolist(),
                "matches": [{k: v for k, v in match.items()
                             if k not in ("kps_cam2", "kps_cam4", "keep")}
                            for match in match_data["matches"]],
                "unmatched_cam2": match_data["unmatched_cam2"],
                "unmatched_cam4": match_data["unmatched_cam4"],
            }, f, ensure_ascii=False, indent=2)
        with open(debug_dir / "summary.json", "w", encoding="utf-8") as f:
            json.dump({
                "backend": args.backend,
                "matched_person_count": len(match_data["matches"]),
                "valid_count": sum(int(m["valid_count"]) for m in match_data["matches"]),
                "total_points": sum(int(m["total_points"]) for m in match_data["matches"]),
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
