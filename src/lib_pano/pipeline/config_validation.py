from __future__ import annotations

from typing import Any


FieldSpec = str | tuple[str, ...]


REQUIRED_FIELDS_BY_STEP: dict[str, tuple[FieldSpec, ...]] = {
    "020": ("reviewed", "fisheye_dir"),
    "021": ("reviewed", "fisheye_dir", "equirect_rotate_dir", "camera"),
    "035": ("reviewed", "equirect_rotate_dir", "click_points"),
    "040": ("reviewed", "equirect_rotate_dir", "bbox_dir"),
    "041": ("reviewed", "bbox_dir", "equirect_rotate_dir"),
    "025": ("reviewed", ("crop_bbox_dir", "bbox_dir"), "equirect_rotate_dir", "equirect_crop_rotate_dir"),
    "026": ("reviewed", "equirect_crop_rotate_dir", "rtm_erp_dir", "device"),
    "027": ("reviewed", "equirect_crop_rotate_dir", ("rtmw_erp_dir", "rtm_erp_dir"), "pose_config", "pose_checkpoint"),
    "024": ("reviewed", "4d_humans_dir", "hmr_cache_dir"),
    "029": ("reviewed", "rtm_erp_dir", "hmr_cache_dir", "fitting_dir", "equirect_crop_rotate_dir"),
    "030": ("fitting_dir", "rtm_erp_dir", "equirect_rotate_dir", "vis_dir"),
    "031": (("fitting_dual_dir", "fitting_dir"), "equirect_rotate_dir", "flow_dir"),
    "038": (("sam3_model_dir", "sam_model"), "equirect_crop_rotate_dir", "seg_dir", "reviewed"),
    "043": ("equirect_crop_rotate_dir", "seg_dir"),
    "037": ("reviewed", "seg_dir", "flow_dir", "equirect_rotate_dir", "equirect_crop_rotate_dir", "composite_dir", "blend_direction"),
}


OPTIONAL_FIELDS_BY_STEP: dict[str, tuple[str, ...]] = {
    "020": ("video_path", "dataset_root", "output_root"),
    "021": (),
    "035": (),
    "040": ("click_points",),
    "041": (
        "bbox_smooth_dir",
        "image_width",
        "image_height",
        "bbox_smooth_window",
        "bbox_smooth_max_gap",
        "bbox_smooth_max_expand_ratio",
        "bbox_smooth_save_vis",
    ),
    "025": ("invalid_detections", "padding_left", "padding_right", "padding_top", "padding_bottom"),
    "026": (),
    "027": (
        "device",
        "invalid_detections",
        "rtmw_verbose_warnings",
        "rtmw_disable_mmpose_vis",
        "rtmw_kpt_thr",
    ),
    "024": (),
    "029": (
        "rtm_invalid_json",
        "hmr_invalid_json",
        "invalid_detections",
        "max_segment_len",
    ),
    "030": ("mesh", "scale"),
    "031": ("vis_dir",),
    "038": ("sam3_text", "sam3_threshold", "sam3_mask_threshold", "sam3_limit"),
    "043": (
        "mask_fusion_input_seg_dir",
        "mask_fusion_output_dir",
        "mask_fusion_bbox_dir",
        "mask_fusion_start_frame",
        "mask_fusion_end_frame",
        "mask_fusion_method",
        "mask_fusion_max_width",
        "mask_fusion_max_height",
        "mask_fusion_min_matches",
        "mask_fusion_min_inliers",
        "mask_fusion_max_mad",
        "image_width",
        "image_height",
        "mask_fusion_min_component_area",
        "mask_fusion_close_kernel",
        "mask_fusion_save_vis",
    ),
    "037": (),
}


def get_required_fields_by_step(step_id: str) -> tuple[FieldSpec, ...]:
    return REQUIRED_FIELDS_BY_STEP.get(step_id, ())


def get_optional_fields_by_step(step_id: str) -> tuple[str, ...]:
    return OPTIONAL_FIELDS_BY_STEP.get(step_id, ())


def _has_value(config: dict[str, Any], field_name: str) -> bool:
    value = config.get(field_name)
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    return True


def format_missing_field_error(dataset_name: str, step_id: str, field_name: str, fallback: str | None = None) -> str:
    if fallback is None:
        return f"dataset={dataset_name} step={step_id} missing required field '{field_name}'"
    return (
        f"dataset={dataset_name} step={step_id} missing required field '{field_name}' "
        f"(fallback '{fallback}' was also not provided)"
    )


def maybe_get_with_fallback(
    config: dict[str, Any],
    primary: str,
    fallback: str | None = None,
    default: Any = None,
) -> Any:
    if _has_value(config, primary):
        return config[primary]
    if fallback is not None and _has_value(config, fallback):
        return config[fallback]
    return default


def validate_dataset_config(dataset_name: str, dataset_config: dict[str, Any], step_ids: list[str]) -> list[str]:
    errors: list[str] = []
    for step_id in step_ids:
        for field_spec in get_required_fields_by_step(step_id):
            if isinstance(field_spec, tuple):
                if any(_has_value(dataset_config, field_name) for field_name in field_spec):
                    continue
                primary = field_spec[0]
                fallback = " or ".join(field_spec[1:]) if len(field_spec) > 1 else None
                errors.append(format_missing_field_error(dataset_name, step_id, primary, fallback))
            elif not _has_value(dataset_config, field_spec):
                errors.append(format_missing_field_error(dataset_name, step_id, field_spec))
    return errors
