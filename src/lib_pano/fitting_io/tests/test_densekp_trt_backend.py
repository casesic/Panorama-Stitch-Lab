from __future__ import annotations

from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from densekp_delivery.runtime import DenseKPRuntime
from densekp_delivery.cli import parse_args
from densekp_delivery.runners.ort_runner import ORTRunner
from densekp_delivery.runners.trt_runner import (
    TensorRTRunner,
    validate_engine_contract,
)


def _runtime_args(backend: str, **overrides):
    values = dict(
        backend=backend,
        checkpoint=None,
        onnx_path="densekp.onnx",
        trt_engine_path="densekp.plan",
        device="cuda",
        detector_backend="detector",
        detector_threshold=0.25,
        batch_size=1,
    )
    values.update(overrides)
    return Namespace(**values)


def test_runtime_selects_ort_backend():
    runtime = DenseKPRuntime.from_args(_runtime_args("ort"))
    try:
        assert isinstance(runtime._runner, ORTRunner)
        assert not isinstance(runtime._runner, TensorRTRunner)
    finally:
        runtime.close()


def test_runtime_selects_trt_backend():
    runtime = DenseKPRuntime.from_args(_runtime_args("trt"))
    try:
        assert isinstance(runtime._runner, TensorRTRunner)
        assert runtime._runner.engine_path == "densekp.plan"
    finally:
        runtime.close()


def test_cli_accepts_explicit_trt_backend_and_engine():
    args = parse_args([
        "--backend", "trt", "--trt_engine_path", "densekp.plan",
        "--fisheye_cam2", "cam2.jpg", "--fisheye_cam4", "cam4.jpg",
        "--out_dir", "output", "--dry_run",
    ])
    assert args.backend == "trt"
    assert args.trt_engine_path == "densekp.plan"


def test_missing_trt_engine_is_a_clear_error(tmp_path: Path):
    runner = TensorRTRunner(str(tmp_path / "missing.plan"))
    with pytest.raises(FileNotFoundError, match=r"backend=trt engine missing"):
        runner.load()


class _FakeEngine:
    def __init__(self, output_shape=(1, 138, 3)):
        self.num_io_tensors = 2
        self._output_shape = output_shape

    def get_tensor_name(self, index):
        return ("img", "pred_keypoints")[index]

    def get_tensor_shape(self, name):
        return (1, 3, 256, 256) if name == "img" else self._output_shape

    def get_tensor_dtype(self, _name):
        return "fp32"

    def get_tensor_mode(self, name):
        return "input" if name == "img" else "output"


_FAKE_TRT = SimpleNamespace(
    float32="fp32", TensorIOMode=SimpleNamespace(INPUT="input", OUTPUT="output"))


def test_engine_contract_validation_accepts_fixed_b1_fp32_io():
    assert validate_engine_contract(_FakeEngine(), _FAKE_TRT) == (
        "img", "pred_keypoints")


def test_engine_contract_validation_rejects_wrong_output_shape():
    with pytest.raises(RuntimeError, match="engine contract mismatch"):
        validate_engine_contract(_FakeEngine((1, 137, 3)), _FAKE_TRT)


def _run_shared_delivery(runner):
    runner._lazy_init = lambda: None
    runner._infer_densekp = lambda tensor: np.zeros((tensor.shape[0], 138, 3), np.float32)
    image = np.zeros((320, 480, 3), np.uint8)
    return runner.run_bboxes([(image, "frame.jpg", "cam2", [[80, 40, 280, 300]])])[0]


def test_ort_and_trt_share_preprocess_decode_and_affine_restore():
    ort_result = _run_shared_delivery(ORTRunner("densekp.onnx"))
    trt_result = _run_shared_delivery(TensorRTRunner("densekp.plan"))
    np.testing.assert_array_equal(
        ort_result.persons[0].keypoints, trt_result.persons[0].keypoints)
    np.testing.assert_array_equal(
        ort_result.persons[0].raw_scores, trt_result.persons[0].raw_scores)
    assert ort_result.persons[0].source == "ort"
    assert trt_result.persons[0].source == "trt"


def test_loaded_trt_runner_reuses_engine_context():
    runner = TensorRTRunner("densekp.plan")
    runner._context = object()
    runner._detector = object()
    runner.engine_deserialization_count = 1
    assert runner.load() is runner
    assert runner.load() is runner
    assert runner.engine_deserialization_count == 1
