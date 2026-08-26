"""Long-lived DenseKP inference runtime shared by multiple frame pairs."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
import time
from typing import Any


class DenseKPRuntime:
    """Own one model/session and detector for the lifetime of a frame stream.

    Constructing the object is cheap.  ``load`` performs model/session and
    detector initialization once; subsequent ``infer`` calls reuse them.
    """

    def __init__(self, runner: Any):
        # Model resources live for the whole stream.
        self._runner = runner
        self.startup_ms = 0.0
        self.load_count = 0
        self.closed = False
        # Image decode workers and calibration-derived GPU/CPU geometry are
        # shared by frames; rebuilding either per frame dominates runtime.
        self.image_io_executor = ThreadPoolExecutor(
            max_workers=2, thread_name_prefix="densekp-image-io",
        )
        self._static_geometry: dict[str, dict[str, Any]] = {}
        self.static_cache_counts = {
            "lut_disk_load_count": 0,
            "lut_memory_hit_count": 0,
            "valid_mask_disk_load_count": 0,
            "valid_mask_memory_hit_count": 0,
        }

    @staticmethod
    def _calibration_key(specification: dict[str, Any]) -> str:
        return json.dumps(specification, sort_keys=True, separators=(",", ":"))

    def intern_calibration(self, specification: dict[str, Any]) -> dict[str, Any]:
        """Return the one resident calibration object for this specification."""
        key = self._calibration_key(specification)
        entry = self._static_geometry.setdefault(key, {})
        return entry.setdefault("calibration", specification)

    def get_static_geometry(
        self, specification: dict[str, Any], name: str,
    ) -> Any | None:
        entry = self._static_geometry.get(self._calibration_key(specification))
        return None if entry is None else entry.get(name)

    def put_static_geometry(
        self, specification: dict[str, Any], name: str, value: Any,
    ) -> None:
        key = self._calibration_key(specification)
        self._static_geometry.setdefault(key, {"calibration": specification})[
            name
        ] = value

    @property
    def static_geometry_entry_count(self) -> int:
        return len(self._static_geometry)

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

    @classmethod
    def from_config(cls, config: Any) -> "DenseKPRuntime":
        """Construct from the public configuration interface."""
        return cls.from_args(config)

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
        self.image_io_executor.shutdown()
        for entry in self._static_geometry.values():
            for value in entry.values():
                close = getattr(value, "close", None)
                if close is not None:
                    close()
        self._static_geometry.clear()
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

    @property
    def execution_providers(self) -> list[str]:
        return list(getattr(self._runner, "execution_providers", []))

    def __enter__(self) -> "DenseKPRuntime":
        return self.load()

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()
