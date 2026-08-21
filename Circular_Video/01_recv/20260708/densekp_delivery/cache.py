"""Runtime cache for immutable camera-calibration products."""

from __future__ import annotations

import json
import hashlib
import logging
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)

VALID_CACHE_MODES = ("read_write", "refresh", "read_only", "off")


class CacheManager:
    def __init__(
        self,
        out_dir: str | Path,
        cache_mode: str = "read_write",
        calibration_cache_dir: str | Path | None = None,
    ):
        if cache_mode not in VALID_CACHE_MODES:
            raise ValueError(f"cache_mode must be one of {VALID_CACHE_MODES}, got {cache_mode!r}")
        self.out_dir = Path(out_dir)
        self.cache_mode = cache_mode
        self.calibration_cache_dir = Path(
            calibration_cache_dir
            if calibration_cache_dir is not None
            else Path.home() / ".cache/pano_stitch/densekp_delivery"
        )

    def calibration_directory(self, specification: dict) -> Path:
        payload = json.dumps(specification, sort_keys=True, separators=(",", ":"))
        digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:20]
        return self.calibration_cache_dir / digest

    def read_calibration_luts(
        self, specification: dict,
    ) -> tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]] | None:
        if not self.should_read():
            return None
        directory = self.calibration_directory(specification)
        metadata = directory / "calibration.json"
        paths = [
            directory / "cam2_map_x.npy", directory / "cam2_map_y.npy",
            directory / "cam4_map_x.npy", directory / "cam4_map_y.npy",
        ]
        if not metadata.is_file() or not all(path.is_file() for path in paths):
            return None
        if json.loads(metadata.read_text(encoding="utf-8")) != specification:
            return None
        arrays = [np.load(path, mmap_mode="r") for path in paths]
        logger.info("Using calibration LUT cache: %s", directory)
        return (arrays[0], arrays[1]), (arrays[2], arrays[3])

    def write_calibration_luts(
        self,
        specification: dict,
        luts: tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]],
    ) -> None:
        if not self.should_write():
            return
        directory = self.calibration_directory(specification)
        directory.mkdir(parents=True, exist_ok=True)
        names = ("cam2_map_x.npy", "cam2_map_y.npy", "cam4_map_x.npy", "cam4_map_y.npy")
        arrays = (*luts[0], *luts[1])
        for name, array in zip(names, arrays):
            np.save(directory / name, np.asarray(array, dtype=np.float32))
        (directory / "calibration.json").write_text(
            json.dumps(specification, sort_keys=True, indent=2), encoding="utf-8"
        )
        logger.info("Wrote calibration LUT cache: %s", directory)

    def read_calibration_valid_masks(
        self, specification: dict,
    ) -> tuple[np.ndarray, np.ndarray] | None:
        if not self.should_read():
            return None
        directory = self.calibration_directory(specification)
        paths = directory / "cam2_valid.png", directory / "cam4_valid.png"
        masks = [cv2.imread(str(path), cv2.IMREAD_GRAYSCALE) for path in paths]
        if any(mask is None for mask in masks):
            return None
        logger.info("Using calibration valid-mask cache: %s", directory)
        return masks[0] > 0, masks[1] > 0

    def write_calibration_valid_masks(
        self, specification: dict, masks: tuple[np.ndarray, np.ndarray],
    ) -> None:
        if not self.should_write():
            return
        directory = self.calibration_directory(specification)
        directory.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(directory / "cam2_valid.png"), masks[0].astype(np.uint8) * 255)
        cv2.imwrite(str(directory / "cam4_valid.png"), masks[1].astype(np.uint8) * 255)

    def should_read(self) -> bool:
        return self.cache_mode in ("read_write", "read_only")

    def should_write(self) -> bool:
        return self.cache_mode in ("read_write", "refresh")
