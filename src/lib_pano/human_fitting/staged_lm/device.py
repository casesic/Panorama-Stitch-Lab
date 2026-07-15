"""Explicit execution-device contract for the staged SMPL-X fitter."""

from __future__ import annotations

import re
from dataclasses import fields, is_dataclass
from typing import Any, Iterable

import torch

from .types import FrozenFrameParameters


_SMPLX_TENSOR_ATTRIBUTES = (
    "orient", "transl", "body_pose", "left_hand_pose",
    "right_hand_pose", "beta", "scale",
)


def resolve_execution_device(requested: str) -> torch.device:
    """解析设备，支持 auto/cpu/cuda/cuda:N"""
    requested = str(requested).strip().lower()
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 直接丢给 PyTorch 解析，格式不对或设备不存在 PyTorch 会自己抛错
    return torch.device(requested)


def _move_float_tensor(
    tensor: torch.Tensor, device: torch.device, dtype: torch.dtype,
) -> torch.Tensor:
    requires_grad = tensor.requires_grad
    target_dtype = dtype if tensor.is_floating_point() else tensor.dtype
    moved = tensor.detach().to(device=device, dtype=target_dtype)
    if requires_grad and moved.is_floating_point():
        moved.requires_grad_(True)
    return moved


def move_smplx_param_sequence(
    sequence: Any, device: torch.device, dtype: torch.dtype = torch.float32,
) -> Any:
    """Move tensor attributes of SMPLX_Param or SMPLX_Param_Seq in place."""
    if sequence is None:
        return None
    for name in _SMPLX_TENSOR_ATTRIBUTES:
        value = getattr(sequence, name, None)
        if isinstance(value, torch.Tensor):
            setattr(sequence, name, _move_float_tensor(value, device, dtype))
    return sequence


def move_frozen_frame_parameters(
    frozen: FrozenFrameParameters,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> FrozenFrameParameters:
    """Return frozen hand/scale tensors on the requested device."""
    return FrozenFrameParameters(**{
        field.name: _move_float_tensor(getattr(frozen, field.name), device, dtype)
        for field in fields(FrozenFrameParameters)
    })


def deterministic_beta_latent_from_anchor(
    beta_anchor: torch.Tensor, beta_abs_limit: float,
) -> torch.Tensor:
    """Compute the shared latent with one CPU-float32 transcendental contract.

    CPU and CUDA ``atanh`` kernels may legitimately differ by one or two ULPs.
    Computing this derived *initial value* on CPU before copying it to the
    execution device makes independently built CPU/CUDA contexts bitwise
    comparable without changing the optimisation parameterisation.
    """
    anchor_cpu = beta_anchor.detach().to(device="cpu", dtype=torch.float32)
    latent_cpu = torch.atanh(anchor_cpu / float(beta_abs_limit))
    return latent_cpu.to(device=beta_anchor.device, dtype=beta_anchor.dtype)


def describe_execution_environment(device: torch.device) -> dict[str, Any]:
    """Return a JSON-safe description of the resolved execution environment."""
    cuda = device.type == "cuda"
    return {
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda or "N/A",
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
        "cuda_device_name": torch.cuda.get_device_name(device) if cuda else "N/A",
        "cuda_compute_capability": list(torch.cuda.get_device_capability(device)) if cuda else "N/A",
        "model_dtype": "float32",
    }


def print_execution_environment(
    requested: str, device: torch.device, environment: dict[str, Any],
) -> None:
    print(
        "[DEVICE] "
        f"requested={requested} resolved={device} "
        f"torch_version={environment['torch_version']} "
        f"torch_cuda_version={environment['torch_cuda_version']} "
        f"cuda_available={environment['cuda_available']} "
        f"cuda_device_count={environment['cuda_device_count']} "
        f"cuda_device_name={environment['cuda_device_name']} "
        f"compute_capability={environment['cuda_compute_capability']} "
        "model_dtype=float32"
    )


def _check_tensor(
    name: str, tensor: torch.Tensor, device: torch.device,
    *, dtype: torch.dtype | None = None,
) -> None:
    if tensor.device != device or (dtype is not None and tensor.dtype != dtype):
        raise RuntimeError(
            "DEVICE_CONTRACT_VIOLATION: "
            f"tensor={name} expected_device={device} actual_device={tensor.device} "
            f"expected_dtype={dtype} actual_dtype={tensor.dtype}"
        )


def validate_context_device(context: Any) -> None:
    """Validate all long-lived context tensors; never repairs mismatches."""
    device, dtype = context.device, context.dtype
    for name in ("v_template", "shapedirs", "expr_dirs", "J_regressor",
                 "posedirs", "lbs_weights", "parents"):
        tensor = getattr(context.model, name)
        _check_tensor(f"model.{name}", tensor, device,
                      dtype=None if name == "parents" else dtype)
    _check_tensor("beta_anchor", context.beta_anchor, device, dtype=dtype)
    _check_tensor("beta_latent0", context.beta_latent0, device, dtype=dtype)
    for index, tensor in enumerate(context.body_pose_anchors):
        _check_tensor(f"body_pose_anchors[{index}]", tensor, device, dtype=dtype)
    for index, frozen in enumerate(context.frozen_params_by_frame):
        for field in fields(FrozenFrameParameters):
            _check_tensor(
                f"frozen[{index}].{field.name}", getattr(frozen, field.name), device, dtype=dtype,
            )
    observation_lists = (
        ("cam2_vertex_ids", context.cam2_vertex_ids_by_frame, torch.long),
        ("cam2_targets", context.cam2_targets_by_frame, dtype),
        ("cam4_vertex_ids", context.cam4_vertex_ids_by_frame, torch.long),
        ("cam4_targets", context.cam4_targets_by_frame, dtype),
    )
    for name, tensors, expected_dtype in observation_lists:
        for index, tensor in enumerate(tensors):
            _check_tensor(f"{name}[{index}]", tensor, device, dtype=expected_dtype)
    for camera, sequence in (("cam2", context.cam2_smplx_parameters),
                             ("cam4", context.cam4_smplx_parameters)):
        if sequence is None:
            continue
        for name in _SMPLX_TENSOR_ATTRIBUTES:
            tensor = getattr(sequence, name, None)
            if isinstance(tensor, torch.Tensor):
                _check_tensor(f"{camera}_smplx.{name}", tensor, device, dtype=dtype)
    _check_tensor("fitter.R_cam4", context.fitter.R_cam4, device, dtype=dtype)


def validate_stage_initial_parameters(
    stage: Any, context: Any, *, include_cache: bool = False,
) -> None:
    """Validate FP32 optimisation vectors and optionally analytic cache tensors."""
    for label, parameters in (
        ("lm", stage.initial_lm_parameters),
        ("lbfgs", getattr(stage, "initial_lbfgs_parameters", None)),
    ):
        if parameters is not None:
            _check_tensor(f"{stage.name}.{label}_initial", parameters, context.device,
                          dtype=torch.float32)
    if include_cache and stage.analytic_cache is not None:
        for name, tensor in iter_named_tensors(stage.analytic_cache):
            _check_tensor(f"{stage.name}.cache.{name}", tensor, context.device)


def iter_named_tensors(value: Any, prefix: str = "") -> Iterable[tuple[str, torch.Tensor]]:
    """Recursively enumerate tensors in dataclasses/lists/dicts for debug audits."""
    if isinstance(value, torch.Tensor):
        yield prefix or "tensor", value
    elif is_dataclass(value):
        for field in fields(value):
            yield from iter_named_tensors(getattr(value, field.name), f"{prefix}.{field.name}".strip("."))
    elif isinstance(value, dict):
        for key, item in value.items():
            yield from iter_named_tensors(item, f"{prefix}.{key}".strip("."))
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            yield from iter_named_tensors(item, f"{prefix}[{index}]")
