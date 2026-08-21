"""Self-test DenseKP delivery ERP/fisheye normalization to ERP_ROTATE."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "src"))

from lib_pano.camera.ds_camera_params import DS_PARAMS, project
from lib_pano.fitting_io.densekp_delivery_io import (
    DELIVERY_EXTRINSIC,
    erp_to_erp_rotate,
    fisheye_to_erp,
)


DELIVERY_ROTATE = (
    ROOT / "Circular_Video/01_recv/20260708/densekp_delivery/camera/erp_rotate.py"
)


def circular_x_error(left: np.ndarray, right: np.ndarray, width: int) -> np.ndarray:
    difference = np.abs(left - right)
    return np.minimum(difference, width - difference)


def main() -> int:
    width, height = 5760, 2880
    points = np.array(
        [[0.0, 1440.0], [100.25, 800.5], [2880.0, 1440.0],
         [5700.5, 2100.25]],
        dtype=np.float32,
    )
    spec = importlib.util.spec_from_file_location("delivery_erp_rotate", DELIVERY_ROTATE)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load densekp_delivery ERP rotation reference")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    reference = module.rotate_erp_points(points, width, height)
    actual = erp_to_erp_rotate(points, width, height)
    if not np.array_equal(actual, reference):
        raise AssertionError(f"ERP rotation mismatch: {np.max(np.abs(actual-reference))}")

    # Build valid fisheye samples with the same delivery forward convention,
    # then verify that the analytic loader returns the original ERP rays.
    for camera in ("cam2", "cam4"):
        center_x = 2880.0 if camera == "cam2" else 0.0
        erp_points = np.array(
            [[center_x, 1200.0], [(center_x + 300.0) % width, 1500.0],
             [(center_x - 300.0) % width, 1300.0]],
            dtype=np.float64,
        )
        lon = erp_points[:, 0] / width * (2.0 * np.pi) - np.pi
        lat = erp_points[:, 1] / height * np.pi - np.pi / 2.0
        rays = np.stack(
            [np.cos(lat) * np.sin(lon), np.sin(lat),
             np.cos(lat) * np.cos(lon)],
            axis=1,
        )
        camera_rays = rays @ DELIVERY_EXTRINSIC[camera].T
        pixels, valid = project(camera_rays, DS_PARAMS)
        if not valid.all():
            raise AssertionError(f"test rays are outside {camera} fisheye domain")
        recovered = fisheye_to_erp(pixels, camera, width, height, "old")
        x_error = circular_x_error(recovered[:, 0], erp_points[:, 0], width)
        y_error = np.abs(recovered[:, 1] - erp_points[:, 1])
        max_error = float(max(x_error.max(), y_error.max()))
        if max_error > 1e-3:
            raise AssertionError(f"{camera} fisheye roundtrip error={max_error}px")

    print("DENSEKP_DELIVERY_COORDINATES_READY=PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
