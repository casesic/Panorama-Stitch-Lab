"""Shared CLI and numerical helpers for staged-LM validation."""

from __future__ import annotations

import argparse

import torch


def add_data_arguments(parser: argparse.ArgumentParser) -> None:
    for name in ("frames", "hmr_dir", "sam3d_dense_dir", "bbox_dir"):
        parser.add_argument(f"--{name}", required=True)
    parser.add_argument("--image_width", type=int, default=5760)
    parser.add_argument("--image_height", type=int, default=2880)


def add_profile_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--profile", action="store_true")


def relative_error(actual: float, expected: float) -> float:
    return abs(actual - expected) / max(abs(expected), 1e-16)


def metric_close(actual: float, expected: float) -> bool:
    return abs(actual - expected) <= 1e-3 or relative_error(actual, expected) <= 1e-5


def synchronize(reference: torch.Tensor | torch.device) -> None:
    device = reference.device if isinstance(reference, torch.Tensor) else reference
    if device.type == "cuda":
        torch.cuda.synchronize(device)
