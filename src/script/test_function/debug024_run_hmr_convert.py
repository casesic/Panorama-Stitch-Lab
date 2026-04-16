from __future__ import annotations

import argparse
import importlib
import pickle
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
        "--hmr_dir",
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
    return parser.parse_args()


def _build_smpl_init_from_single_frame_pkl(pkl_path: Path):
    """Convert one single-frame HMR pkl dict into a SMPL_Param_Seq instance."""
    smpl_mod = importlib.import_module("lib_dynamic_hvip.fitting.smpl_seq_forward_module")
    SMPLFitter = smpl_mod.SMPL_Seq_Forward_Module

    with pkl_path.open("rb") as file:
        src = pickle.load(file)

    if isinstance(src, SMPLFitter.SMPL_Param_Seq):
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


def _process_camera_frame(hmr_pkl_path: Path, output_path: Path) -> None:
    """Convert one camera HMR pkl into an SMPL_Param_Seq cache file."""
    if not hmr_pkl_path.exists():
        print(f"Warning: hmr pkl not found, skipping: {hmr_pkl_path}")
        return

    smpl_init = _build_smpl_init_from_single_frame_pkl(hmr_pkl_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as file:
        pickle.dump(smpl_init, file)


def main() -> None:
    """Batch-convert HMR pkls for reviewed equirect frames."""
    _ensure_paths()
    args = parse_args()

    reviewed_path = Path(args.reviewed)
    hmr_dir = Path(args.hmr_dir)
    output_dir = Path(args.output_dir)
    reviewed_pairs = load_reviewed_entries(reviewed_path)
    total_steps = len(reviewed_pairs) * 2

    with tqdm(total=total_steps, desc="Converting HMR", unit="frame") as progress:
        for _, frame_idx in reviewed_pairs:
            _process_camera_frame(
                hmr_pkl_path=hmr_dir / "cam2" / f"hmr_s{frame_idx}_pose.pkl",
                output_path=output_dir / "cam2" / f"hmr_s{frame_idx:05d}_c2.pkl",
            )
            progress.update(1)

            _process_camera_frame(
                hmr_pkl_path=hmr_dir / "cam4" / f"hmr_s{frame_idx}_pose.pkl",
                output_path=output_dir / "cam4" / f"hmr_s{frame_idx:05d}_c4.pkl",
            )
            progress.update(1)


if __name__ == "__main__":
    main()
