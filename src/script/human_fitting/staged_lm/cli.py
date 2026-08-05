"""Command-line parser for the staged LM production compatibility entry."""

from __future__ import annotations

import argparse


def build_argument_parser() -> argparse.ArgumentParser:
    """Construct the single-path production staged-LM parser."""
    parser = argparse.ArgumentParser(description="Staged LM with IRLS Charbonnier")
    parser.add_argument("--frames", type=str, required=True)
    parser.add_argument("--hmr_dir", type=str, required=True)
    parser.add_argument("--sam3d_dense_dir", type=str, required=True)
    parser.add_argument("--bbox_dir", type=str, required=True)
    parser.add_argument("--image_width", type=int, default=5760)
    parser.add_argument("--image_height", type=int, default=2880)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--device", type=str, choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--validate_device_contract", action="store_true", default=False)
    parser.add_argument(
        "--reproj_objective",
        choices=["raw_l2", "charbonnier_irls"],
        default="charbonnier_irls",
    )
    parser.add_argument(
        "--stop_after_stage",
        choices=["stage1", "stage2_pose", "stage2"],
        default="stage2",
    )
    parser.add_argument("--profile_timing", action="store_true", default=False)
    parser.add_argument(
        "--export_debug029_format",
        action="store_true",
        default=False,
        help=(
            "Additionally export cam2/cam4 seg_<start>_<end>_seq.pkl files "
            "compatible with debug029/debug077. Frames must be contiguous."
        ),
    )
    return parser


def parse_args() -> argparse.Namespace:
    """Parse production staged-LM arguments."""
    return build_argument_parser().parse_args()
