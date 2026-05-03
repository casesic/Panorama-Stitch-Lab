from __future__ import annotations

import argparse
import importlib
import json
import pickle
import shutil
import sys
from pathlib import Path

import torch
from tqdm import tqdm

import config_script
from batch_utils import load_reviewed_entries


def _ensure_paths() -> Path:
    """Ensure the SingleFitting package is importable."""
    repo_root = Path(__file__).resolve().parents[3]
    sf_root = repo_root / "tools" / "SingleFitting"
    if str(sf_root) not in sys.path:
        sys.path.insert(0, str(sf_root))
    return repo_root


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for batch HMR conversion."""
    parser = argparse.ArgumentParser(
        description="Convert 4D-Humans HMR pkls into SMPL_Param_Seq caches for reviewed frames."
    )
    parser.add_argument(
        "--reviewed",
        type=str,
        required=True,
        help="Path to reviewed.json.",
    )
    parser.add_argument(
        "--4d_humans_dir",
        dest="four_d_humans_dir",
        type=str,
        required=True,
        help="4D-Humans pkl root directory, e.g. data_process/Output/20260311_riding/hmr.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output root directory, e.g. data_process/Output/20260311_riding/single_frame_cache.",
    )
    parser.add_argument(
        "--invalid_json",
        type=str,
        default=None,
        help="Optional path to invalid.json containing invalid frames to skip.",
    )
    parser.add_argument(
        "--verbose_warnings",
        action="store_true",
        help="Print per-frame warnings while processing.",
    )
    return parser.parse_args()


def _build_smpl_init_from_single_frame_pkl(pkl_path: Path):
    """Convert one single-frame HMR pkl dict into a SMPL_Param_Seq instance."""
    smpl_mod = importlib.import_module("lib_dynamic_hvip.fitting.smpl_seq_forward_module")
    SMPLFitter = smpl_mod.SMPL_Seq_Forward_Module

    with pkl_path.open("rb") as file:
        src = pickle.load(file)

    if isinstance(src, SMPLFitter.SMPL_Param_Seq):
        print("no change needed for smpl_init")
        return src
    if not isinstance(src, dict):
        raise TypeError(f"Unsupported HMR pkl object type: {type(src)}")

    num_batch = int(src["num_batch"])
    dim_shape = int(src["beta"].shape[-1])
    gender = str(src.get("gender", "neutral"))

    smpl_init = SMPLFitter.SMPL_Param_Seq(dim_shape=dim_shape, gender=gender, num_batch=num_batch)
    smpl_init.orient = src["orient"].clone()
    smpl_init.transl = src["transl"].clone()
    smpl_init.pose = src["pose"].clone()
    smpl_init.beta = src["beta"].clone()
    smpl_init.scale = src["scale"].clone()
    smpl_init.valid = src.get("valid", torch.ones(num_batch, dtype=torch.bool)).clone()
    return smpl_init


def _extract_frame_indices(obj) -> set[int]:
    frames: set[int] = set()
    if isinstance(obj, int):
        frames.add(int(obj))
    elif isinstance(obj, list):
        for item in obj:
            frames.update(_extract_frame_indices(item))
    elif isinstance(obj, dict):
        if "frame_idx" in obj:
            frames.update(_extract_frame_indices(obj["frame_idx"]))
        if "frame_idxs" in obj:
            frames.update(_extract_frame_indices(obj["frame_idxs"]))
        if "frames" in obj:
            frames.update(_extract_frame_indices(obj["frames"]))
        if "invalid_frames" in obj:
            frames.update(_extract_frame_indices(obj["invalid_frames"]))
    return frames


def load_invalid_frames(invalid_json_path: Path, verbose_warnings: bool) -> set[int]:
    """Load invalid frame indices from invalid.json."""
    if not invalid_json_path.exists():
        if verbose_warnings:
            print(f"Warning: invalid.json not found, no invalid frames loaded: {invalid_json_path}")
        return set()

    with invalid_json_path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    return _extract_frame_indices(data)


def _process_camera_frame(hmr_pkl_path: Path, output_path: Path, verbose_warnings: bool) -> str:
    """Convert one camera HMR pkl into an SMPL_Param_Seq cache file."""
    if not hmr_pkl_path.exists():
        if verbose_warnings:
            print(f"Warning: hmr pkl not found, skipping: {hmr_pkl_path}")
        return "missing"

    smpl_init = _build_smpl_init_from_single_frame_pkl(hmr_pkl_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as file:
        pickle.dump(smpl_init, file)
    return "ok"


def main() -> None:
    """Batch-convert HMR pkls for reviewed equirect frames."""
    _ensure_paths()
    args = parse_args()

    reviewed_path = Path(args.reviewed)
    four_d_humans_dir = Path(args.four_d_humans_dir)
    output_dir = Path(args.output_dir)
    invalid_json_path = Path(args.invalid_json) if args.invalid_json else four_d_humans_dir.parent / "invalid.json"
    invalid_frames = load_invalid_frames(invalid_json_path, args.verbose_warnings)
    reviewed_pairs = load_reviewed_entries(reviewed_path)
    total_steps = len(reviewed_pairs) * 2
    stats = {
        "cam2": {"ok": 0, "missing": 0, "skipped_invalid": 0},
        "cam4": {"ok": 0, "missing": 0, "skipped_invalid": 0},
    }

    with tqdm(total=total_steps, desc="Converting HMR", unit="frame") as progress:
        for _, frame_idx in reviewed_pairs:
            for camera, suffix in (("cam2", "c2"), ("cam4", "c4")):
                if frame_idx in invalid_frames:
                    stats[camera]["skipped_invalid"] += 1
                    progress.update(1)
                    continue

                status = _process_camera_frame(
                    hmr_pkl_path=four_d_humans_dir / camera / f"hmr_s{frame_idx}_pose.pkl",
                    output_path=output_dir / camera / f"hmr_s{frame_idx:05d}_{suffix}.pkl",
                    verbose_warnings=args.verbose_warnings,
                )
                stats[camera][status] += 1
                progress.update(1)

    print("Summary:")
    print(f"  invalid_json: {invalid_json_path}")
    print(f"  invalid_frames: {len(invalid_frames)}")
    for camera in ("cam2", "cam4"):
        camera_stats = stats[camera]
        print(
            f"  {camera}: ok={camera_stats['ok']} "
            f"missing={camera_stats['missing']} "
            f"skipped_invalid={camera_stats['skipped_invalid']}"
        )

    if invalid_json_path.exists():
        output_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(invalid_json_path, output_dir / "invalid.json")
    else:
        print(f"Warning: invalid.json not found, skipping copy: {invalid_json_path}")


if __name__ == "__main__":
    main()

r'''
python src\script\test_function\debug024_run_hmr_convert.py `
  --reviewed data_process\Input\20260320_riding\dataset1\review.json `
  --4d_humans_dir data_process\Output\20260320_riding_dataset1\4dhumans\equirect_rotate_crop `
  --output_dir data_process\Output\20260320_riding_dataset1\hmr_cache `
  --invalid_json data_process\Output\20260320_riding_dataset1\4dhumans\invalid.json

python src\script\test_function\debug024_run_hmr_convert.py `
  --reviewed data_process\Input\20260320_riding\dataset1\review.json `
  --4d_humans_dir data_process\Output\20260320_riding_dataset1\4dhumans\equirect_rotate_crop `
  --output_dir data_process\Output\20260320_riding_dataset1\hmr_cache `
  --invalid_json data_process\Output\20260320_riding_dataset1\4dhumans\invalid.json `
  --verbose_warnings
'''
