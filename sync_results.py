from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any


REMOTE_OUTPUT_ROOT = "/home/yuyue/Projects/pano-stitch/tools/4D-Humans/demo_out"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sync remote 4D-Humans results from demo_out back to local output directories."
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to pipeline config json.",
    )
    parser.add_argument(
        "--host",
        type=str,
        required=True,
        help="SSH host, e.g. yuyue@192.168.x.x",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=49000,
        help="SSH port for rsync, default is 49000.",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        default=None,
        help="Optional comma-separated dataset names. If omitted, sync all datasets.",
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


def require_field(dataset_name: str, dataset_config: dict[str, Any], field_name: str) -> str:
    value = dataset_config.get(field_name)
    if value is None:
        raise KeyError(f"Dataset '{dataset_name}' is missing required config field '{field_name}'.")
    if isinstance(value, str) and value.strip():
        return value
    raise ValueError(f"Dataset '{dataset_name}' field '{field_name}' must be a non-empty string.")


def ensure_trailing_slash(path_str: str) -> str:
    return path_str if path_str.endswith(("/", "\\")) else f"{path_str}/"


def sync_dataset(dataset_name: str, dataset_config: dict[str, Any], host: str, port: int) -> int:
    remote_input_subdir = require_field(dataset_name, dataset_config, "remote_input_subdir")
    local_output_dir = Path(require_field(dataset_name, dataset_config, "local_output_dir")).expanduser()

    local_output_dir.mkdir(parents=True, exist_ok=True)

    remote_source = f"{host}:{REMOTE_OUTPUT_ROOT}/{remote_input_subdir}/"
    command = [
        "rsync",
        "-avz",
        "--progress",
        "-e",
        f"ssh -p {port}",
        remote_source,
        ensure_trailing_slash(str(local_output_dir)),
    ]

    print(f"[RUN] dataset={dataset_name}")
    print(" ".join(command), flush=True)

    try:
        completed = subprocess.run(command, check=False)
    except Exception as exc:  # pragma: no cover
        print(f"[ERROR] dataset={dataset_name} sync raised exception: {exc}")
        return 1

    if completed.returncode != 0:
        print(f"[ERROR] dataset={dataset_name} sync failed with return code {completed.returncode}")
    else:
        print(f"[OK] dataset={dataset_name} sync finished")

    return completed.returncode


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).expanduser().resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    dataset_configs = load_config(config_path)
    datasets = select_datasets(dataset_configs, args.datasets)

    failures: list[tuple[str, int]] = []
    for dataset_name in datasets:
        try:
            return_code = sync_dataset(dataset_name, dataset_configs[dataset_name], args.host, args.port)
        except (KeyError, ValueError) as exc:
            print(f"[ERROR] dataset={dataset_name} invalid config: {exc}")
            return_code = 1

        if return_code != 0:
            failures.append((dataset_name, return_code))

    if failures:
        print("\n[SUMMARY] Some syncs failed:")
        for dataset_name, return_code in failures:
            print(f"  dataset={dataset_name} returncode={return_code}")
    else:
        print("\n[SUMMARY] All syncs completed successfully.")


if __name__ == "__main__":
    main()
