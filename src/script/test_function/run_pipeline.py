from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


SCRIPT_BY_STEP: dict[str, str] = {
    "020": "debug020_run_export.py",
    "021": "debug021_run_fisheye_to_equirect.py",
    "022": "debug022_detect_fisheye_bbox.py",
    "023": "debug023_run_erp_fitting.py",
    "024": "debug024_run_hmr_convert.py",
    "025": "debug025_crop_equirect_for_4dhumans.py",
    "026": "debug026_run_rtmlib_erp.py",
    "027": "debug027_segment_human_erp.py",
    "034": "debug034_generate_review.py",
    "035": "debug035_annotate_click_points.py",
    "036": "debug036_detect_fisheye_bbox_direct.py",
}

SUPPORTED_STEPS = tuple(SCRIPT_BY_STEP.keys())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch run selected debug020-027/debug034-debug036 pipeline steps for multiple datasets."
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to pipeline config json.",
    )
    parser.add_argument(
        "--steps",
        type=str,
        required=True,
        help="Comma-separated step ids, e.g. 034,020,021,035,022,036,025,026,027",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        default=None,
        help="Optional comma-separated dataset names. If omitted, process all datasets in config.",
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


def parse_steps(raw_steps: str) -> list[str]:
    steps = [step.strip() for step in raw_steps.split(",") if step.strip()]
    if not steps:
        raise ValueError("--steps must contain at least one step.")

    invalid_steps = [step for step in steps if step not in SCRIPT_BY_STEP]
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


def _require_field(dataset_name: str, dataset_config: dict[str, Any], field_name: str) -> str:
    value = dataset_config.get(field_name)
    if value is None:
        raise KeyError(f"Dataset '{dataset_name}' is missing required config field '{field_name}'.")
    if isinstance(value, str) and value.strip():
        return value
    raise ValueError(f"Dataset '{dataset_name}' field '{field_name}' must be a non-empty string.")


def _default_seg_path_from_reviewed(reviewed: str) -> str:
    return str(Path(reviewed).with_name("seg.json"))


def build_step_args(step: str, dataset_name: str, dataset_config: dict[str, Any]) -> list[str]:
    reviewed = _require_field(dataset_name, dataset_config, "reviewed")

    if step == "034":
        return [
            "--seg",
            _default_seg_path_from_reviewed(reviewed),
            "--output",
            reviewed,
        ]
    if step == "020":
        return [
            "--reviewed",
            reviewed,
            "--output_dir",
            _require_field(dataset_name, dataset_config, "fisheye_dir"),
        ]
    if step == "021":
        return [
            "--reviewed",
            reviewed,
            "--fisheye_dir",
            _require_field(dataset_name, dataset_config, "fisheye_dir"),
            "--output_dir",
            _require_field(dataset_name, dataset_config, "equirect_dir"),
            "--camera",
            _require_field(dataset_name, dataset_config, "camera"),
        ]
    if step == "022":
        cmd = [
            "--reviewed",
            reviewed,
            "--input_dir",
            _require_field(dataset_name, dataset_config, "fisheye_dir"),
            "--output_dir",
            _require_field(dataset_name, dataset_config, "rtm_fisheye_dir"),
            "--camera_group",
            _require_field(dataset_name, dataset_config, "camera"),
            "--device",
            _require_field(dataset_name, dataset_config, "device"),
        ]
        _append_optional_arg(cmd, dataset_config, "click_points")
        return cmd
    if step == "035":
        return [
            "--reviewed",
            reviewed,
            "--fisheye_dir",
            _require_field(dataset_name, dataset_config, "fisheye_dir"),
            "--output",
            _require_field(dataset_name, dataset_config, "click_points"),
        ]
    if step == "036":
        cmd = [
            "--reviewed",
            reviewed,
            "--input_dir",
            _require_field(dataset_name, dataset_config, "fisheye_dir"),
            "--output_dir",
            _require_field(dataset_name, dataset_config, "rtm_fisheye_dir"),
            "--device",
            _require_field(dataset_name, dataset_config, "device"),
        ]
        _append_optional_arg(cmd, dataset_config, "click_points")
        return cmd
    if step == "023":
        cmd = [
            "--reviewed",
            reviewed,
            "--rtm_dir",
            _require_field(dataset_name, dataset_config, "rtm_erp_dir"),
            "--hmr_dir",
            _require_field(dataset_name, dataset_config, "hmr_dir"),
            "--output_dir",
            _require_field(dataset_name, dataset_config, "fitting_dir"),
        ]
        _append_optional_arg(cmd, dataset_config, "invalid_detections")
        return cmd
    if step == "024":
        return [
            "--reviewed",
            reviewed,
            "--hmr_dir",
            _require_field(dataset_name, dataset_config, "hmr_dir"),
            "--output_dir",
            _require_field(dataset_name, dataset_config, "hmr_cache_dir"),
        ]
    if step == "025":
        cmd = [
            "--reviewed",
            reviewed,
            "--rtm_dir",
            _require_field(dataset_name, dataset_config, "rtm_fisheye_dir"),
            "--equirect_dir",
            _require_field(dataset_name, dataset_config, "equirect_dir"),
            "--output_dir",
            _require_field(dataset_name, dataset_config, "equirect_crop_dir"),
            "--camera",
            _require_field(dataset_name, dataset_config, "camera"),
        ]
        _append_optional_arg(cmd, dataset_config, "invalid_detections")
        return cmd
    if step == "026":
        return [
            "--reviewed",
            reviewed,
            "--crop_dir",
            _require_field(dataset_name, dataset_config, "equirect_crop_dir"),
            "--output_dir",
            _require_field(dataset_name, dataset_config, "rtm_erp_dir"),
            "--device",
            _require_field(dataset_name, dataset_config, "device"),
        ]
    if step == "027":
        cmd = [
            "--reviewed",
            reviewed,
            "--crop_dir",
            _require_field(dataset_name, dataset_config, "equirect_crop_dir"),
            "--rtm_dir",
            _require_field(dataset_name, dataset_config, "rtm_erp_dir"),
            "--output_dir",
            _require_field(dataset_name, dataset_config, "seg_dir"),
        ]
        _append_optional_arg(cmd, dataset_config, "sam_model")
        return cmd

    raise ValueError(f"Unsupported step: {step}")


def _append_optional_arg(command: list[str], dataset_config: dict[str, Any], field_name: str) -> None:
    value = dataset_config.get(field_name)
    if value is None:
        return
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Optional field '{field_name}' must be a non-empty string when provided.")
    command.extend([f"--{field_name}", value])


def run_step_for_dataset(
    step: str,
    dataset_name: str,
    dataset_config: dict[str, Any],
    script_dir: Path,
    project_root: Path,
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
    print(f"[RUN] dataset={dataset_name} step={step}")
    print(" ".join(command), flush=True)

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
