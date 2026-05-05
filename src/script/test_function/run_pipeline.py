from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable


StepBuilder = Callable[[str, dict[str, Any]], list[str]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch run selected pipeline steps for multiple datasets."
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to pipeline config json.",
    )
    parser.add_argument(
        "--steps",
        type=str,
        default=None,
        help="Comma-separated step ids, e.g. 020,021,035,040,041,025,026,024,029,030,031,038,037",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        default=None,
        help="Optional comma-separated dataset names. If omitted, process all datasets in config.",
    )
    parser.add_argument(
        "--list_steps",
        action="store_true",
        help="List supported step ids and scripts, then exit.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print commands without executing them.",
    )
    return parser.parse_args()


def load_config(config_path: Path) -> dict[str, dict[str, Any]]:
    with config_path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    if not isinstance(data, dict):
        raise ValueError("Config json must contain a top-level object mapping dataset names to configs.")

    normalized: dict[str, dict[str, Any]] = {}
    for dataset_name, dataset_config in data.items():
        if not isinstance(dataset_name, str) or not dataset_name:
            raise ValueError("Each dataset name in config must be a non-empty string.")
        if not isinstance(dataset_config, dict):
            raise ValueError(f"Config for dataset '{dataset_name}' must be a json object.")
        normalized[dataset_name] = dataset_config
    return normalized


def _require_field(dataset_name: str, dataset_config: dict[str, Any], field_name: str) -> str:
    value = dataset_config.get(field_name)
    if value is None:
        raise KeyError(f"Dataset '{dataset_name}' is missing required config field '{field_name}'.")
    if isinstance(value, str) and value.strip():
        return value
    raise ValueError(f"Dataset '{dataset_name}' field '{field_name}' must be a non-empty string.")


def _require_any_field(dataset_name: str, dataset_config: dict[str, Any], *field_names: str) -> str:
    for field_name in field_names:
        value = dataset_config.get(field_name)
        if isinstance(value, str) and value.strip():
            return value
        if value is not None:
            raise ValueError(f"Dataset '{dataset_name}' field '{field_name}' must be a non-empty string.")
    joined = ", ".join(f"'{field_name}'" for field_name in field_names)
    raise KeyError(f"Dataset '{dataset_name}' is missing required config field; expected one of: {joined}.")


def _get_field_with_default(dataset_config: dict[str, Any], field_name: str, default: Any) -> Any:
    return dataset_config.get(field_name, default)


def _append_optional_arg(command: list[str], dataset_config: dict[str, Any], field_name: str, arg_name: str | None = None) -> None:
    value = dataset_config.get(field_name)
    if value is None:
        return
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Optional field '{field_name}' must be a non-empty string when provided.")
    command.extend([f"--{arg_name or field_name}", value])


def _append_bool_flag(command: list[str], dataset_config: dict[str, Any], field_name: str, flag_name: str) -> None:
    if bool(_get_field_with_default(dataset_config, field_name, False)):
        command.append(f"--{flag_name}")


def _default_bbox_smooth_dir(bbox_dir: str) -> str:
    bbox_path = Path(bbox_dir)
    return str(bbox_path.parent / f"{bbox_path.name}_smooth")


def build_args_020(dataset_name: str, dataset_config: dict[str, Any]) -> list[str]:
    return [
        "--reviewed",
        _require_field(dataset_name, dataset_config, "reviewed"),
        "--output_dir",
        _require_field(dataset_name, dataset_config, "fisheye_dir"),
    ]


def build_args_021(dataset_name: str, dataset_config: dict[str, Any]) -> list[str]:
    return [
        "--reviewed",
        _require_field(dataset_name, dataset_config, "reviewed"),
        "--fisheye_dir",
        _require_field(dataset_name, dataset_config, "fisheye_dir"),
        "--output_dir",
        _require_field(dataset_name, dataset_config, "equirect_rotate_dir"),
        "--camera",
        _require_field(dataset_name, dataset_config, "camera"),
        "--apply_extrinsic",
    ]


def build_args_035(dataset_name: str, dataset_config: dict[str, Any]) -> list[str]:
    return [
        "--reviewed",
        _require_field(dataset_name, dataset_config, "reviewed"),
        "--fisheye_dir",
        _require_field(dataset_name, dataset_config, "equirect_rotate_dir"),
        "--output",
        _require_field(dataset_name, dataset_config, "click_points"),
    ]


def build_args_040(dataset_name: str, dataset_config: dict[str, Any]) -> list[str]:
    cmd = [
        "--reviewed",
        _require_field(dataset_name, dataset_config, "reviewed"),
        "--input_dir",
        _require_field(dataset_name, dataset_config, "equirect_rotate_dir"),
        "--output_dir",
        _require_field(dataset_name, dataset_config, "bbox_dir"),
    ]
    _append_optional_arg(cmd, dataset_config, "click_points")
    return cmd


def build_args_041(dataset_name: str, dataset_config: dict[str, Any]) -> list[str]:
    bbox_dir = _require_field(dataset_name, dataset_config, "bbox_dir")
    bbox_smooth_dir = _get_field_with_default(dataset_config, "bbox_smooth_dir", _default_bbox_smooth_dir(bbox_dir))
    cmd = [
        "--reviewed",
        _require_field(dataset_name, dataset_config, "reviewed"),
        "--input_dir",
        bbox_dir,
        "--output_dir",
        bbox_smooth_dir,
        "--image_dir",
        _require_field(dataset_name, dataset_config, "equirect_rotate_dir"),
        "--image_width",
        str(_get_field_with_default(dataset_config, "image_width", 5760)),
        "--image_height",
        str(_get_field_with_default(dataset_config, "image_height", 2880)),
        "--window",
        str(_get_field_with_default(dataset_config, "bbox_smooth_window", 2)),
        "--max_gap",
        str(_get_field_with_default(dataset_config, "bbox_smooth_max_gap", 2)),
        "--max_expand_ratio",
        str(_get_field_with_default(dataset_config, "bbox_smooth_max_expand_ratio", 3)),
    ]
    if bool(_get_field_with_default(dataset_config, "bbox_smooth_save_vis", True)):
        cmd.append("--save_vis")
    return cmd


def build_args_025(dataset_name: str, dataset_config: dict[str, Any]) -> list[str]:
    bbox_dir = _require_any_field(dataset_name, dataset_config, "crop_bbox_dir", "bbox_dir")
    cmd = [
        "--reviewed",
        _require_field(dataset_name, dataset_config, "reviewed"),
        "--bbox_dir",
        bbox_dir,
        "--equirect_dir",
        _require_field(dataset_name, dataset_config, "equirect_rotate_dir"),
        "--output_dir",
        _require_field(dataset_name, dataset_config, "equirect_crop_rotate_dir"),
        "--padding_left",
        str(_get_field_with_default(dataset_config, "padding_left", 250)),
        "--padding_right",
        str(_get_field_with_default(dataset_config, "padding_right", 250)),
        "--padding_top",
        str(_get_field_with_default(dataset_config, "padding_top", 150)),
        "--padding_bottom",
        str(_get_field_with_default(dataset_config, "padding_bottom", 150)),
    ]
    _append_optional_arg(cmd, dataset_config, "invalid_detections", "invalid_json")
    return cmd


def build_args_026(dataset_name: str, dataset_config: dict[str, Any]) -> list[str]:
    return [
        "--reviewed",
        _require_field(dataset_name, dataset_config, "reviewed"),
        "--crop_dir",
        _require_field(dataset_name, dataset_config, "equirect_crop_rotate_dir"),
        "--output_dir",
        _require_field(dataset_name, dataset_config, "rtm_erp_dir"),
        "--device",
        _require_any_field(dataset_name, dataset_config, "device"),
    ]


def build_args_027(dataset_name: str, dataset_config: dict[str, Any]) -> list[str]:
    # rtm_erp_dir is kept as a fallback for older configs that predate rtmw_erp_dir.
    cmd = [
        "--reviewed",
        _require_field(dataset_name, dataset_config, "reviewed"),
        "--crop_dir",
        _require_field(dataset_name, dataset_config, "equirect_crop_rotate_dir"),
        "--output_dir",
        _require_any_field(dataset_name, dataset_config, "rtmw_erp_dir", "rtm_erp_dir"),
        "--pose_config",
        _require_field(dataset_name, dataset_config, "pose_config"),
        "--pose_checkpoint",
        _require_field(dataset_name, dataset_config, "pose_checkpoint"),
        "--device",
        str(_get_field_with_default(dataset_config, "device", "cuda:0")),
    ]
    _append_optional_arg(cmd, dataset_config, "invalid_detections", "invalid_json")
    _append_bool_flag(cmd, dataset_config, "rtmw_verbose_warnings", "verbose_warnings")
    _append_bool_flag(cmd, dataset_config, "rtmw_disable_mmpose_vis", "disable_mmpose_vis")
    cmd.extend(["--kpt_thr", str(_get_field_with_default(dataset_config, "rtmw_kpt_thr", 0.3))])
    return cmd


def build_args_024(dataset_name: str, dataset_config: dict[str, Any]) -> list[str]:
    humans_dir = _require_field(dataset_name, dataset_config, "4d_humans_dir")
    return [
        "--reviewed",
        _require_field(dataset_name, dataset_config, "reviewed"),
        "--4d_humans_dir",
        humans_dir,
        "--output_dir",
        _require_field(dataset_name, dataset_config, "hmr_cache_dir"),
        "--invalid_json",
        str(Path(humans_dir).parent / "invalid.json"),
    ]


def build_args_029(dataset_name: str, dataset_config: dict[str, Any]) -> list[str]:
    cmd = [
        "--reviewed",
        _require_field(dataset_name, dataset_config, "reviewed"),
        "--rtm_dir",
        _require_field(dataset_name, dataset_config, "rtm_erp_dir"),
        "--hmr_dir",
        _require_field(dataset_name, dataset_config, "hmr_cache_dir"),
        "--output_dir",
        _require_field(dataset_name, dataset_config, "fitting_dir"),
        "--bbox_dir",
        _require_field(dataset_name, dataset_config, "equirect_crop_rotate_dir"),
    ]
    _append_optional_arg(cmd, dataset_config, "rtm_invalid_json")
    _append_optional_arg(cmd, dataset_config, "seg_invalid_json")
    _append_optional_arg(cmd, dataset_config, "hmr_invalid_json")
    if "rtm_invalid_json" not in dataset_config:
        _append_optional_arg(cmd, dataset_config, "invalid_detections", "rtm_invalid_json")

    seg_dir = _require_field(dataset_name, dataset_config, "seg_dir")
    cmd.extend(["--seg_dir", str(Path(seg_dir) / "equirect_rotate_crop")])

    if not bool(_get_field_with_default(dataset_config, "use_silhouette_loss", True)):
        cmd.append("--no_silhouette_loss")

    cmd.extend(["--max_segment_len", str(_get_field_with_default(dataset_config, "max_segment_len", 10))])
    return cmd


def build_args_030(dataset_name: str, dataset_config: dict[str, Any]) -> list[str]:
    cmd = [
        "--result_dir",
        _require_field(dataset_name, dataset_config, "fitting_dir"),
        "--rtm_dir",
        _require_field(dataset_name, dataset_config, "rtm_erp_dir"),
        "--image_dir",
        _require_field(dataset_name, dataset_config, "equirect_rotate_dir"),
        "--output_dir",
        _require_field(dataset_name, dataset_config, "vis_dir"),
        "--scale",
        str(_get_field_with_default(dataset_config, "scale", 0.5)),
    ]
    if bool(_get_field_with_default(dataset_config, "mesh", False)):
        cmd.append("--mesh")
    return cmd


def build_args_031(dataset_name: str, dataset_config: dict[str, Any]) -> list[str]:
    cmd = [
        "--fitting_dir",
        _require_any_field(dataset_name, dataset_config, "fitting_dual_dir", "fitting_dir"),
        "--image_dir",
        _require_field(dataset_name, dataset_config, "equirect_rotate_dir"),
        "--output_dir",
        _require_field(dataset_name, dataset_config, "flow_dir"),
    ]

    vis_dir = dataset_config.get("vis_dir")
    if vis_dir:
        invalid_json_path = Path(vis_dir) / "invalid.json"
        if invalid_json_path.exists():
            cmd.extend(["--invalid_json", str(invalid_json_path)])
    return cmd


def build_args_038(dataset_name: str, dataset_config: dict[str, Any]) -> list[str]:
    return [
        "--model_dir",
        _require_any_field(dataset_name, dataset_config, "sam3_model_dir", "sam_model"),
        "--input_dir",
        _require_field(dataset_name, dataset_config, "equirect_crop_rotate_dir"),
        "--output_dir",
        str(Path(_require_field(dataset_name, dataset_config, "seg_dir")) / "equirect_crop_rotate_sam3"),
        "--reviewed",
        _require_field(dataset_name, dataset_config, "reviewed"),
        "--text",
        str(_get_field_with_default(dataset_config, "sam3_text", "person")),
        "--threshold",
        str(_get_field_with_default(dataset_config, "sam3_threshold", 0.5)),
        "--mask_threshold",
        str(_get_field_with_default(dataset_config, "sam3_mask_threshold", 0.5)),
        "--limit",
        str(_get_field_with_default(dataset_config, "sam3_limit", 0)),
    ]


def build_args_037(dataset_name: str, dataset_config: dict[str, Any]) -> list[str]:
    return [
        "--reviewed",
        _require_field(dataset_name, dataset_config, "reviewed"),
        "--seg_dir",
        str(Path(_require_field(dataset_name, dataset_config, "seg_dir")) / "equirect_crop_rotate_sam3"),
        "--flow_dir",
        _require_field(dataset_name, dataset_config, "flow_dir"),
        "--image_dir",
        _require_field(dataset_name, dataset_config, "equirect_rotate_dir"),
        "--bbox_dir",
        _require_field(dataset_name, dataset_config, "equirect_crop_rotate_dir"),
        "--output_dir",
        _require_field(dataset_name, dataset_config, "composite_dir"),
        "--blend_direction",
        _require_field(dataset_name, dataset_config, "blend_direction"),
    ]


STEP_SPECS: dict[str, dict[str, Any]] = {
    "020": {"script": "debug020_run_export.py", "builder": build_args_020, "description": "Export reviewed fisheye frames"},
    "021": {"script": "debug021_run_fisheye_to_equirect.py", "builder": build_args_021, "description": "Rotate fisheye frames to ERP"},
    "035": {"script": "debug035_annotate_click_points.py", "builder": build_args_035, "description": "Annotate click points"},
    "040": {"script": "debug040_detect_grounding_dino.py", "builder": build_args_040, "description": "Detect ERP bboxes with GroundingDINO"},
    "041": {"script": "debug041_smooth_bboxes.py", "builder": build_args_041, "description": "Smooth bbox sequences"},
    "025": {"script": "debug025_crop_equirect_rotate_from_bboxes.py", "builder": build_args_025, "description": "Crop rotated ERP images from bboxes"},
    "026": {"script": "debug026_run_rtmlib_erp.py", "builder": build_args_026, "description": "Run RTMLib on ERP crops"},
    "027": {"script": "debug027_run_rtmw_mmpose.py", "builder": build_args_027, "description": "Run RTMW MMPose on ERP crops"},
    "024": {"script": "debug024_run_hmr_convert.py", "builder": build_args_024, "description": "Convert 4D Humans outputs"},
    "029": {"script": "debug029_run_dual_erp_fitting.py", "builder": build_args_029, "description": "Run dual ERP SMPL-X fitting"},
    "030": {"script": "debug030_vis_dual_erp.py", "builder": build_args_030, "description": "Visualize dual ERP fitting"},
    "031": {"script": "debug031_batch_semantic_flow.py", "builder": build_args_031, "description": "Generate semantic flow"},
    "038": {"script": "debug038_seg_sam3.py", "builder": build_args_038, "description": "Segment ERP crops with SAM3"},
    "037": {"script": "debug037_composite_human.py", "builder": build_args_037, "description": "Composite humans with semantic flow"},
}
SCRIPT_BY_STEP = {step: str(spec["script"]) for step, spec in STEP_SPECS.items()}
SUPPORTED_STEPS = tuple(STEP_SPECS.keys())


def parse_steps(raw_steps: str) -> list[str]:
    raw_items = [step.strip() for step in raw_steps.split(",") if step.strip()]
    steps = [step.zfill(3) if step.isdigit() else step for step in raw_items]
    if not steps:
        raise ValueError("--steps must contain at least one step.")

    invalid_steps = [step for step in steps if step not in STEP_SPECS]
    if invalid_steps:
        supported = ", ".join(SUPPORTED_STEPS)
        invalid = ", ".join(invalid_steps)
        raise ValueError(f"Unsupported step(s): {invalid}. Supported steps: {supported}")
    return steps


def select_datasets(all_configs: dict[str, dict[str, Any]], raw_datasets: str | None) -> list[str]:
    if raw_datasets is None:
        return list(all_configs.keys())

    datasets = [name.strip() for name in raw_datasets.split(",") if name.strip()]
    if not datasets:
        raise ValueError("--datasets was provided but no valid dataset names were found.")

    missing = [name for name in datasets if name not in all_configs]
    if missing:
        missing_str = ", ".join(missing)
        raise ValueError(f"Dataset(s) not found in config: {missing_str}")
    return datasets


def build_step_args(step: str, dataset_name: str, dataset_config: dict[str, Any]) -> list[str]:
    try:
        builder: StepBuilder = STEP_SPECS[step]["builder"]
    except KeyError as exc:
        raise ValueError(f"Unsupported step: {step}") from exc
    return builder(dataset_name, dataset_config)


def list_steps() -> None:
    for step, spec in STEP_SPECS.items():
        print(f"{step}  {spec['script']}  - {spec['description']}")


def run_step_for_dataset(
    step: str,
    dataset_name: str,
    dataset_config: dict[str, Any],
    script_dir: Path,
    project_root: Path,
    dry_run: bool = False,
) -> int:
    script_name = SCRIPT_BY_STEP[step]
    script_path = script_dir / script_name
    if not script_path.exists():
        print(f"[ERROR] dataset={dataset_name} step={step} script not found: {script_path}")
        return 1

    try:
        step_args = build_step_args(step, dataset_name, dataset_config)
    except (KeyError, ValueError) as exc:
        print(f"[ERROR] dataset={dataset_name} step={step} failed to build command: {exc}")
        return 1

    command = [sys.executable, str(script_path), *step_args]
    prefix = "[DRY-RUN]" if dry_run else "[RUN]"
    print(f"{prefix} dataset={dataset_name} step={step}")
    print(" ".join(command), flush=True)
    if dry_run:
        return 0

    try:
        completed = subprocess.run(command, cwd=project_root, check=False)
    except Exception as exc:  # pragma: no cover
        print(f"[ERROR] dataset={dataset_name} step={step} execution raised exception: {exc}")
        return 1

    if completed.returncode != 0:
        print(
            f"[ERROR] dataset={dataset_name} step={step} exited with return code {completed.returncode}",
            flush=True,
        )
    else:
        print(f"[OK] dataset={dataset_name} step={step}", flush=True)
    return completed.returncode


def main() -> None:
    args = parse_args()
    if args.list_steps:
        list_steps()
        return
    if args.config is None:
        raise ValueError("--config is required unless --list_steps is used.")
    if args.steps is None:
        raise ValueError("--steps is required unless --list_steps is used.")

    config_path = Path(args.config).expanduser().resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    dataset_configs = load_config(config_path)
    steps = parse_steps(args.steps)
    datasets = select_datasets(dataset_configs, args.datasets)
    script_dir = Path(__file__).resolve().parent
    project_root = Path(__file__).resolve().parents[3]

    failures: list[tuple[str, str, int]] = []
    for step in steps:
        for dataset_name in datasets:
            return_code = run_step_for_dataset(
                step=step,
                dataset_name=dataset_name,
                dataset_config=dataset_configs[dataset_name],
                script_dir=script_dir,
                project_root=project_root,
                dry_run=args.dry_run,
            )
            if return_code != 0:
                failures.append((dataset_name, step, return_code))

    if failures:
        print("\n[SUMMARY] Some commands failed:")
        for dataset_name, step, return_code in failures:
            print(f"  dataset={dataset_name} step={step} returncode={return_code}")
    else:
        print("\n[SUMMARY] All commands completed successfully.")


if __name__ == "__main__":
    main()


r'''
python src/script/test_function/run_pipeline.py \
  --config data_process/Output/config.json \
  --steps 020,021,035,040,041,025,026,024,029,030,031,038,037

python src/script/test_function/run_pipeline.py `
  --config data_process/Output/config.json `
  --steps 029 `
  --datasets dataset1
'''
