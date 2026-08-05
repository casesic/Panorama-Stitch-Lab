"""CPU/CUDA execution-device contract tests for staged LM."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch

try:
    from ._bootstrap import ensure_test_paths
except ImportError:
    from _bootstrap import ensure_test_paths
ensure_test_paths(__file__)

from lib_pano.human_fitting.smplx_sequence import (  # noqa: E402
    SMPLX_Seq_Forward_Module,
)

from lib_pano.human_fitting.staged_lm.data_loading import (  # noqa: E402
    to_tensor_vertex_ids_and_targets,
)
from lib_pano.human_fitting.staged_lm.device import (  # noqa: E402
    deterministic_beta_latent_from_anchor,
    move_frozen_frame_parameters,
    move_smplx_param_sequence,
    resolve_execution_device,
    validate_context_device,
    validate_stage_initial_parameters,
)
from lib_pano.human_fitting.staged_lm.lm.equations import _validate_normal_equations  # noqa: E402
from lib_pano.human_fitting.staged_lm.parameter_layout import (  # noqa: E402
    build_stage1_initial_parameters,
)
from lib_pano.human_fitting.staged_lm.stage_handoff import handoff_s1_to_pose  # noqa: E402
from lib_pano.human_fitting.staged_lm.types import (  # noqa: E402
    FrameValidPoints,
    FrozenFrameParameters,
    NormalEquations,
    ValidPointSet,
)


def _raises(marker: str, fn) -> None:
    try:
        fn()
    except RuntimeError as exc:
        assert marker in str(exc), str(exc)
    else:
        raise AssertionError(f"Expected {marker}")


def test_device_resolution() -> None:
    assert resolve_execution_device("cpu") == torch.device("cpu")
    with mock.patch("torch.cuda.is_available", return_value=False):
        _raises("CUDA_REQUESTED_BUT_UNAVAILABLE", lambda: resolve_execution_device("cuda"))
    with mock.patch("torch.cuda.is_available", return_value=True):
        assert resolve_execution_device("cuda") == torch.device("cuda:0")
    _raises("INVALID_EXECUTION_DEVICE", lambda: resolve_execution_device("auto"))
    _raises("INVALID_EXECUTION_DEVICE", lambda: resolve_execution_device("cuda:0"))
    _raises("INVALID_EXECUTION_DEVICE", lambda: resolve_execution_device("gpu"))


def test_parameter_and_frozen_moves() -> None:
    classes = (
        SMPLX_Seq_Forward_Module.SMPLX_Param(),
        SMPLX_Seq_Forward_Module.SMPLX_Param_Seq(num_batch=2),
    )
    for sequence in classes:
        sequence.orient.requires_grad_(True)
        original_shape = sequence.orient.shape
        assert move_smplx_param_sequence(sequence, torch.device("cpu")) is sequence
        assert sequence.orient.shape == original_shape
        assert sequence.orient.device.type == "cpu"
        assert sequence.orient.dtype == torch.float32
        assert sequence.orient.requires_grad
        assert sequence.gender == "neutral"
    assert move_smplx_param_sequence(None, torch.device("cpu")) is None

    frozen = FrozenFrameParameters(*[torch.ones(1, dtype=torch.float64) for _ in range(6)])
    moved = move_frozen_frame_parameters(frozen, torch.device("cpu"))
    for value in vars(moved).values():
        assert value.device.type == "cpu" and value.dtype == torch.float32

    anchor = torch.tensor([0.1, -0.3, 1.7], dtype=torch.float32)
    expected = torch.atanh(anchor / 3.0)
    actual = deterministic_beta_latent_from_anchor(anchor, 3.0)
    assert torch.equal(actual, expected)


def test_observation_and_parameter_layout_device() -> None:
    valid = FrameValidPoints(
        cam2=ValidPointSet(torch.tensor([3, 1]), torch.randn(2, 2)),
        cam4=ValidPointSet(torch.tensor([4]), torch.randn(1, 2)),
    )
    c2_ids, c2_targets, c4_ids, c4_targets = to_tensor_vertex_ids_and_targets(
        [valid], device=torch.device("cpu"), target_dtype=torch.float32,
    )
    assert c2_ids[0].dtype == c4_ids[0].dtype == torch.long
    assert c2_targets[0].dtype == c4_targets[0].dtype == torch.float32
    assert c2_ids[0].device.type == c2_targets[0].device.type == "cpu"

    cam2 = SMPLX_Seq_Forward_Module.SMPLX_Param_Seq(num_batch=1)
    cam4 = SMPLX_Seq_Forward_Module.SMPLX_Param_Seq(num_batch=1)
    stage1 = build_stage1_initial_parameters(cam2, cam4, 1)
    pose = handoff_s1_to_pose(stage1, [torch.zeros(63)], 1, 63)
    assert stage1.device.type == pose.device.type == "cpu"
    assert stage1.dtype == pose.dtype == torch.float32


def _fake_context(target_dtype: torch.dtype = torch.float32):
    f32 = lambda *shape: torch.zeros(*shape, dtype=torch.float32)
    model = SimpleNamespace(
        v_template=f32(1, 2, 3), shapedirs=f32(2, 3, 1), expr_dirs=f32(2, 3, 1),
        J_regressor=f32(1, 2), posedirs=f32(9, 6), lbs_weights=f32(2, 1),
        parents=torch.zeros(1, dtype=torch.long),
    )
    frozen = FrozenFrameParameters(*[f32(1) for _ in range(6)])
    seq = SMPLX_Seq_Forward_Module.SMPLX_Param_Seq(num_batch=1)
    return SimpleNamespace(
        device=torch.device("cpu"), dtype=torch.float32, model=model,
        beta_anchor=f32(1), beta_latent0=f32(1), body_pose_anchors=[f32(3)],
        frozen_params_by_frame=[frozen],
        cam2_vertex_ids_by_frame=[torch.tensor([0], dtype=torch.long)],
        cam2_targets_by_frame=[torch.zeros(1, 2, dtype=target_dtype)],
        cam4_vertex_ids_by_frame=[torch.tensor([1], dtype=torch.long)],
        cam4_targets_by_frame=[f32(1, 2)],
        cam2_smplx_parameters=seq, cam4_smplx_parameters=seq,
        fitter=SimpleNamespace(R_cam4=f32(3, 3)),
    )


def test_context_and_stage_validation() -> None:
    validate_context_device(_fake_context())
    _raises("DEVICE_CONTRACT_VIOLATION", lambda: validate_context_device(_fake_context(torch.float64)))
    context = _fake_context()
    stage = SimpleNamespace(
        name="TEST", initial_lm_parameters=torch.zeros(3),
        analytic_cache=None,
    )
    validate_stage_initial_parameters(stage, context)
    stage.initial_lm_parameters = torch.zeros(3, dtype=torch.float64)
    _raises("DEVICE_CONTRACT_VIOLATION", lambda: validate_stage_initial_parameters(stage, context))


def test_normal_equations_device_and_dtype() -> None:
    parameters = torch.zeros(4, dtype=torch.float32)
    equations = NormalEquations(
        residual=torch.zeros(2), hessian=torch.eye(4, dtype=torch.float64),
        gradient=torch.zeros(4, dtype=torch.float64),
    )
    _validate_normal_equations(equations, parameters)
    bad = NormalEquations(
        residual=torch.zeros(2), hessian=torch.eye(4), gradient=torch.zeros(4),
    )
    _raises("DEVICE_CONTRACT_VIOLATION", lambda: _validate_normal_equations(bad, parameters))
    # Output serialization boundaries are explicitly allowed to move tensors.
    assert parameters.detach().cpu().numpy().shape == (4,)


def test_cuda_branch_when_available() -> None:
    if not torch.cuda.is_available():
        print("test_cuda_branch_when_available: SKIP (CUDA unavailable)")
        return
    device = resolve_execution_device("cuda")
    seq = SMPLX_Seq_Forward_Module.SMPLX_Param_Seq(num_batch=1)
    move_smplx_param_sequence(seq, device)
    assert seq.orient.is_cuda and seq.orient.dtype == torch.float32
    anchor_cpu = torch.tensor([0.1, -0.3, 1.7], dtype=torch.float32)
    latent_cuda = deterministic_beta_latent_from_anchor(anchor_cpu.to(device), 3.0)
    assert torch.equal(latent_cuda.cpu(), torch.atanh(anchor_cpu / 3.0))


if __name__ == "__main__":
    test_device_resolution()
    test_parameter_and_frozen_moves()
    test_observation_and_parameter_layout_device()
    test_context_and_stage_validation()
    test_normal_equations_device_and_dtype()
    test_cuda_branch_when_available()
    print("All staged-LM device-contract tests PASSED")
