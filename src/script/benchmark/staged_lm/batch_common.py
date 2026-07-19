"""Shared padded layouts, timing, error, and reduction helpers for bake-offs."""

from __future__ import annotations

from dataclasses import dataclass
import math
import statistics
import time
from typing import Any, Callable, Sequence

import torch


@dataclass
class PaddedGroupLayout:
    frame_positions: torch.Tensor
    camera_indices: torch.Tensor
    point_counts: torch.Tensor
    valid_mask: torch.Tensor
    targets: torch.Tensor
    nmax: int
    group_count: int
    frame_count: int
    padding_ratio: float
    padded_bytes: int


def tensor_bytes(tensor: torch.Tensor) -> int:
    return tensor.numel() * tensor.element_size()


def build_padded_group_layout(groups: Sequence[Any]) -> PaddedGroupLayout:
    if not groups:
        raise ValueError("EMPTY_PADDED_GROUP_LAYOUT")
    reference = groups[0].targets
    device, dtype = reference.device, reference.dtype
    counts = [int(group.targets.shape[0]) for group in groups]
    nmax, group_count = max(counts), len(groups)
    frame_count = max(int(group.frame_position) for group in groups) + 1
    mask = torch.zeros(group_count, nmax, dtype=torch.bool, device=device)
    targets = torch.zeros(group_count, nmax, 2, dtype=dtype, device=device)
    for index, (group, count) in enumerate(zip(groups, counts)):
        mask[index, :count] = True
        targets[index, :count] = group.targets
    frames = torch.tensor(
        [int(group.frame_position) for group in groups], dtype=torch.long, device=device,
    )
    cameras = torch.tensor(
        [0 if group.camera == "cam2" else 1 for group in groups],
        dtype=torch.long, device=device,
    )
    count_tensor = torch.tensor(counts, dtype=torch.long, device=device)
    total_slots = group_count * nmax
    padded_bytes = sum(tensor_bytes(value) for value in (mask, targets, frames, cameras, count_tensor))
    return PaddedGroupLayout(
        frames, cameras, count_tensor, mask, targets, nmax, group_count,
        frame_count, (total_slots - sum(counts)) / max(total_slots, 1), padded_bytes,
    )


def benchmark_callable(
    fn: Callable[[], Any], reference_tensor: torch.Tensor, warmup: int,
    repeats: int, *, device_timing: bool = True,
) -> dict[str, float | int]:
    for _ in range(warmup):
        fn()
    if reference_tensor.is_cuda:
        torch.cuda.synchronize(reference_tensor.device)
    samples: list[float] = []
    for _ in range(repeats):
        if reference_tensor.is_cuda and device_timing:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record(); fn(); end.record(); end.synchronize()
            elapsed = float(start.elapsed_time(end))
        else:
            start_time = time.perf_counter(); fn()
            if reference_tensor.is_cuda:
                torch.cuda.synchronize(reference_tensor.device)
            elapsed = (time.perf_counter() - start_time) * 1000.0
        samples.append(elapsed)
    ordered = sorted(samples)
    return {
        "p50": statistics.median(samples),
        "p90": ordered[min(len(ordered) - 1, math.ceil(0.9 * len(ordered)) - 1)],
        "mean": statistics.mean(samples), "min": min(samples), "max": max(samples),
        "std": statistics.pstdev(samples), "calls": repeats,
    }


def numerical_error(candidate: torch.Tensor, reference: torch.Tensor) -> dict[str, float]:
    difference = candidate - reference
    denominator = max(float(reference.norm()), 1e-30)
    return {
        "max_abs": float(difference.abs().max()) if difference.numel() else 0.0,
        "mean_abs": float(difference.abs().mean()) if difference.numel() else 0.0,
        "relative_frobenius": float(difference.norm()) / denominator,
    }


def environment_record(device: torch.device, dtype: torch.dtype) -> dict[str, Any]:
    cuda = device.type == "cuda"
    return {
        "torch_version": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device) if cuda else "CPU",
        "compute_capability": list(torch.cuda.get_device_capability(device)) if cuda else None,
        "cpu_thread_count": torch.get_num_threads(),
        "interop_thread_count": torch.get_num_interop_threads(),
        "dtype": str(dtype),
        "cuda_matmul_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_tf32": torch.backends.cudnn.allow_tf32,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
    }


class TF32Disabled:
    def __enter__(self):
        self.matmul = torch.backends.cuda.matmul.allow_tf32
        self.cudnn = torch.backends.cudnn.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        return self

    def __exit__(self, *_):
        torch.backends.cuda.matmul.allow_tf32 = self.matmul
        torch.backends.cudnn.allow_tf32 = self.cudnn


def active_indices(camera_index: int, *, device: torch.device) -> torch.Tensor:
    camera_start = 0 if camera_index == 0 else 6
    return torch.cat((
        torch.arange(camera_start, camera_start + 6, device=device),
        torch.arange(12, 75, device=device),
    ))


def reduce_active_pose_blocks(
    h_active: torch.Tensor, g_active: torch.Tensor, layout: PaddedGroupLayout,
    *, python_scatter: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    dtype, device = h_active.dtype, h_active.device
    h = torch.zeros(layout.frame_count, 75, 75, dtype=dtype, device=device)
    g = torch.zeros(layout.frame_count, 75, dtype=dtype, device=device)
    # G is at most 16 in this bake-off. This loop only scatters small matrices;
    # all point-wise derivative work remains batched.
    for group in range(layout.group_count):
        frame = int(layout.frame_positions[group])
        camera = int(layout.camera_indices[group])
        indices = active_indices(camera, device=device)
        h[frame][indices[:, None], indices[None, :]] += h_active[group]
        g[frame][indices] += g_active[group]
    return h, g
