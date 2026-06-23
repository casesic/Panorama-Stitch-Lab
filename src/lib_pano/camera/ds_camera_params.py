"""Double Sphere 鱼眼相机参数集和投影函数，独立于 tools/fisheye_extractor。"""

from __future__ import annotations

from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# DS camera parameter sets
# ---------------------------------------------------------------------------
DS_PARAMS: dict[str, float] = {
    "xi": 0.02541467,
    "alpha": 0.58707823,
    "fx": 819.659819122500,
    "fy": 819.389361330000,
    "cx": 1455.78851018250,
    "cy": 1435.04276513250,
}

DS_PARAMS_CAM2: dict[str, float] = {
    "xi": -0.12464126,
    "alpha": 0.56540615,
    "fx": 681.49808088,
    "fy": 682.11759351,
    "cx": 1445.73197064,
    "cy": 1444.25576211,
}

DS_PARAMS_CAM4: dict[str, float] = {
    "xi": -0.15331803,
    "alpha": 0.55745717,
    "fx": 656.78496421,
    "fy": 655.70192322,
    "cx": 1440.93901552,
    "cy": 1443.33089179,
}


# ---------------------------------------------------------------------------
# DS projection (copied from tools/fisheye_extractor/ds_model.py)
# ---------------------------------------------------------------------------
def _validate_params(params: dict[str, Any]) -> tuple[float, float, float, float, float, float]:
    required_keys = ("xi", "alpha", "fx", "fy", "cx", "cy")
    missing = [key for key in required_keys if key not in params]
    if missing:
        raise KeyError(f"Missing DS parameters: {missing}")
    return (
        float(params["xi"]),
        float(params["alpha"]),
        float(params["fx"]),
        float(params["fy"]),
        float(params["cx"]),
        float(params["cy"]),
    )


def _as_points_3d(points_3d: np.ndarray) -> tuple[np.ndarray, bool]:
    points = np.asarray(points_3d, dtype=np.float64)
    if points.ndim == 1:
        if points.shape[0] != 3:
            raise ValueError(f"Expected shape (3,), got {points.shape}")
        return points.reshape(1, 3), True
    if points.ndim == 2 and points.shape[1] == 3:
        return points, False
    raise ValueError(f"Expected shape (N, 3) or (3,), got {points.shape}")


def _finalize_output(values: np.ndarray, is_single: bool) -> np.ndarray:
    return values[0] if is_single else values


def project(
    points_3d: np.ndarray,
    params: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    """Project 3D camera-frame points onto fisheye pixels with the Double Sphere model."""
    xi, alpha, fx, fy, cx, cy = _validate_params(params)
    points, is_single = _as_points_3d(points_3d)

    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]

    d1 = np.sqrt(np.maximum(x * x + y * y + z * z, 0.0))
    z_shifted = xi * d1 + z
    d2 = np.sqrt(np.maximum(x * x + y * y + z_shifted * z_shifted, 0.0))

    if alpha <= 0.5:
        w1 = alpha / (1.0 - alpha)
    else:
        w1 = (1.0 - alpha) / alpha
    w2 = (w1 + xi) / np.sqrt(np.maximum(2.0 * w1 * xi + xi * xi + 1.0, 0.0))
    valid_mask = z_shifted > (-w2 * d1)

    denom = alpha * d2 + (1.0 - alpha) * z_shifted
    safe_denom = np.where(np.abs(denom) > 0.0, denom, 1.0)

    u = fx * x / safe_denom + cx
    v = fy * y / safe_denom + cy

    pixels = np.stack([u, v], axis=1).astype(np.float32)
    valid_mask = valid_mask.astype(bool)
    return _finalize_output(pixels, is_single), _finalize_output(valid_mask, is_single)
