"""Public configuration objects for DenseKP delivery."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class DenseKPConfig:
    """Configuration consumed directly by :class:`DenseKPRuntime`.

    The CLI converts argparse values into this object.  Production callers
    construct it directly and therefore do not depend on private CLI parsing.
    """

    backend: str
    fisheye_cam2: str
    fisheye_cam4: str
    out_dir: str
    checkpoint: str | None = None
    onnx_path: str | None = None
    trt_engine_path: str | None = None
    camera: str = "old"
    erp_width: int = 5760
    erp_height: int = 2880
    erp_remap_backend: str = "cuda"
    erp_remap_cuda_library: str | None = None
    cache_mode: str = "read_write"
    calibration_cache_dir: str | None = None
    detector_backend: str = "rtmlib_yolox"
    detector_threshold: float = 0.25
    person_idx_cam2: int | None = None
    person_idx_cam4: int | None = None
    output_coord_space: str = "erp"
    vis_option: str = "none"
    save_option: str = "none"
    device: str = "cuda"
    batch_size: int = 1
    verbose: bool = False
    profile_runtime: bool = False

    @classmethod
    def from_namespace(cls, args: Any) -> "DenseKPConfig":
        return cls(**{
            name: getattr(args, name)
            for name in cls.__dataclass_fields__
        })

    def for_frame(
        self, cam2: str | Path, cam4: str | Path, output: str | Path,
    ) -> "DenseKPConfig":
        """Return per-frame paths while retaining the shared runtime setup."""
        return replace(
            self,
            fisheye_cam2=str(cam2),
            fisheye_cam4=str(cam4),
            out_dir=str(output),
        )
