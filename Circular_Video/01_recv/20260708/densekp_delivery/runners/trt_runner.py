"""TensorRT FP16 DenseKP inference with shared ORT-runner preprocessing."""

from __future__ import annotations

from pathlib import Path
import time
from typing import Any

import numpy as np

from .ort_runner import ORTRunner


EXPECTED_INPUT = ("img", (1, 3, 256, 256), "float32")
EXPECTED_OUTPUT = ("pred_keypoints", (1, 138, 3), "float32")


def validate_engine_contract(engine: Any, trt: Any) -> tuple[str, str]:
    """Validate the accepted fixed-B1 TensorRT engine I/O contract."""
    inputs = []
    outputs = []
    for index in range(int(engine.num_io_tensors)):
        name = engine.get_tensor_name(index)
        record = (name, tuple(int(value) for value in engine.get_tensor_shape(name)),
                  engine.get_tensor_dtype(name))
        if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
            inputs.append(record)
        else:
            outputs.append(record)
    expected_input = (EXPECTED_INPUT[0], EXPECTED_INPUT[1], trt.float32)
    expected_output = (EXPECTED_OUTPUT[0], EXPECTED_OUTPUT[1], trt.float32)
    if inputs != [expected_input] or outputs != [expected_output]:
        raise RuntimeError(
            "TensorRT DenseKP engine contract mismatch: "
            f"expected input={expected_input} output={expected_output}; "
            f"actual inputs={inputs} outputs={outputs}"
        )
    return inputs[0][0], outputs[0][0]


class TensorRTRunner(ORTRunner):
    """Fixed-B1 TRT engine; detector and all geometry are inherited/shared."""

    backend_name = "trt"

    def __init__(
        self,
        engine_path: str,
        device: str = "cuda",
        detector_backend: str = "rtmlib_yolox",
        detector_threshold: float = 0.25,
        batch_size: int = 1,
    ):
        if not engine_path:
            raise ValueError(
                "DenseKP backend=trt requires an explicit engine path")
        super().__init__(
            onnx_path="", device=device, detector_backend=detector_backend,
            detector_threshold=detector_threshold, batch_size=batch_size,
        )
        self.engine_path = str(engine_path)
        self._trt_runtime = None
        self._engine = None
        self._context = None
        self._torch = None
        self._stream = None
        self._input_buffer = None
        self._output_buffer = None
        self._trt_input_name = None
        self._trt_output_name = None
        self.profile_timing["trt_init_ms"] = 0.0
        self.profile_timing["densekp_trt_call_count"] = 0.0
        self.profile_timing["densekp_trt_batch_item_count"] = 0.0
        self.engine_deserialization_count = 0

    def _lazy_init(self) -> None:
        if self._context is not None:
            return
        path = Path(self.engine_path)
        if not path.is_file():
            raise FileNotFoundError(
                f"DenseKP backend=trt engine missing: {path}"
            )
        if self.device != "cuda" and not self.device.startswith("cuda:"):
            raise RuntimeError(
                f"DenseKP backend=trt requires CUDA device, got {self.device!r}"
            )
        if int(self.batch_size) != 1:
            raise RuntimeError(
                "DenseKP backend=trt fixed-B1 engine requires batch_size=1, "
                f"got {self.batch_size}"
            )
        try:
            import tensorrt as trt
            import torch
            from detectors import build_person_detector
        except Exception as error:
            raise RuntimeError(
                f"DenseKP backend=trt dependency initialization failed "
                f"for engine={path}: {error}"
            ) from error

        started = time.perf_counter()
        try:
            logger = trt.Logger(trt.Logger.WARNING)
            self._trt_runtime = trt.Runtime(logger)
            self._engine = self._trt_runtime.deserialize_cuda_engine(
                path.read_bytes())
            if self._engine is None:
                raise RuntimeError("deserialize_cuda_engine returned None")
            self.engine_deserialization_count += 1
            self._trt_input_name, self._trt_output_name = (
                validate_engine_contract(self._engine, trt))
            self._context = self._engine.create_execution_context()
            if self._context is None:
                raise RuntimeError("create_execution_context returned None")
            self._torch = torch
            self._stream = torch.cuda.Stream(device=self.device)
            with torch.cuda.device(self.device):
                self._input_buffer = torch.empty(
                    EXPECTED_INPUT[1], dtype=torch.float32, device=self.device)
                self._output_buffer = torch.empty(
                    EXPECTED_OUTPUT[1], dtype=torch.float32, device=self.device)
            self._context.set_tensor_address(
                self._trt_input_name, self._input_buffer.data_ptr())
            self._context.set_tensor_address(
                self._trt_output_name, self._output_buffer.data_ptr())
        except Exception as error:
            self.close()
            raise RuntimeError(
                f"DenseKP backend=trt initialization failed engine={path}; "
                f"expected input={EXPECTED_INPUT} output={EXPECTED_OUTPUT}: {error}"
            ) from error
        self.profile_timing["trt_init_ms"] += (
            time.perf_counter() - started) * 1000.0
        self.session_creation_count += 1

        started = time.perf_counter()
        self._detector = build_person_detector(
            backend=self.detector_backend, threshold=self.detector_threshold,
            device=self.device)
        self.profile_timing["detector_init_ms"] += (
            time.perf_counter() - started) * 1000.0
        self.detector_initialization_count += 1

    @property
    def loaded(self) -> bool:
        return self._context is not None and self._detector is not None

    @property
    def execution_providers(self) -> list[str]:
        return ["TensorRTExecutionProvider"] if self._context is not None else []

    def _infer_densekp(self, tensor: np.ndarray) -> np.ndarray:
        if tensor.shape != EXPECTED_INPUT[1] or tensor.dtype != np.float32:
            raise RuntimeError(
                f"DenseKP backend=trt input contract mismatch: "
                f"expected shape={EXPECTED_INPUT[1]} dtype=float32, "
                f"got shape={tensor.shape} dtype={tensor.dtype}"
            )
        try:
            source = self._torch.from_numpy(np.ascontiguousarray(tensor))
            with self._torch.cuda.stream(self._stream):
                self._input_buffer.copy_(source, non_blocking=False)
                ok = self._context.execute_async_v3(self._stream.cuda_stream)
                if not ok:
                    raise RuntimeError("execute_async_v3 returned false")
            self._stream.synchronize()
            return self._output_buffer.cpu().numpy().copy()
        except Exception as error:
            raise RuntimeError(
                f"DenseKP backend=trt execute failed engine={self.engine_path}; "
                f"expected input={EXPECTED_INPUT} output={EXPECTED_OUTPUT}: {error}"
            ) from error

    def _record_backend_call(self, item_count: int) -> None:
        self.profile_timing["densekp_backend_call_count"] += 1
        self.profile_timing["densekp_backend_batch_item_count"] += item_count
        self.profile_timing["densekp_trt_call_count"] += 1
        self.profile_timing["densekp_trt_batch_item_count"] += item_count

    def close(self) -> None:
        self._detector = None
        self._context = None
        self._engine = None
        self._trt_runtime = None
        self._stream = None
        self._input_buffer = None
        self._output_buffer = None
        self._torch = None
