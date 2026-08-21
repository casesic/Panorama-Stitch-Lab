"""Long-lived DenseKP inference runtime shared by multiple frame pairs."""

from __future__ import annotations

import time
from typing import Any


class DenseKPRuntime:
    """Own one model/session and detector for the lifetime of a frame stream.

    Constructing the object is cheap.  ``load`` performs model/session and
    detector initialization once; subsequent ``infer`` calls reuse them.
    """

    def __init__(self, runner: Any):
        self._runner = runner
        self.startup_ms = 0.0
        self.load_count = 0
        self.closed = False

    @classmethod
    def from_args(cls, args: Any) -> "DenseKPRuntime":
        if args.backend == "pytorch":
            from .runners.pytorch_runner import PyTorchRunner

            runner = PyTorchRunner(
                checkpoint=args.checkpoint,
                device=args.device,
                detector_backend=args.detector_backend,
                detector_threshold=args.detector_threshold,
                batch_size=getattr(args, "batch_size", 1),
            )
        elif args.backend == "ort":
            from .runners.ort_runner import ORTRunner

            runner = ORTRunner(
                onnx_path=args.onnx_path,
                device=args.device,
                detector_backend=args.detector_backend,
                detector_threshold=args.detector_threshold,
                batch_size=getattr(args, "batch_size", 1),
            )
        else:
            raise ValueError(f"Unknown backend: {args.backend}")
        return cls(runner)

    def load(self) -> "DenseKPRuntime":
        if self.closed:
            raise RuntimeError("DenseKPRuntime is closed")
        if self.loaded:
            return self
        start = time.perf_counter()
        self._runner.load()
        self.startup_ms += (time.perf_counter() - start) * 1000.0
        self.load_count += 1
        return self

    def infer(self, image_bgr, image_path: str, camera_name: str = ""):
        self.load()
        return self._runner.run(image_bgr, image_path, camera_name)

    def process(self, args: Any) -> None:
        """Run one delivery frame while retaining this runtime for the next."""
        from .pipeline import run_pipeline

        self.load()
        run_pipeline(args, runtime=self)

    # Runner-compatible entry point used by the existing delivery pipeline.
    def run(self, image_bgr, image_path: str, camera_name: str = ""):
        return self.infer(image_bgr, image_path, camera_name)

    def close(self) -> None:
        if self.closed:
            return
        self._runner.close()
        self.closed = True

    @property
    def loaded(self) -> bool:
        return bool(getattr(self._runner, "loaded", False))

    @property
    def profile_timing(self) -> dict[str, float]:
        return getattr(self._runner, "profile_timing", {})

    @property
    def session_creation_count(self) -> int:
        return int(getattr(self._runner, "session_creation_count", 0))

    @property
    def detector_initialization_count(self) -> int:
        return int(getattr(self._runner, "detector_initialization_count", 0))

    def __enter__(self) -> "DenseKPRuntime":
        return self.load()

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()
