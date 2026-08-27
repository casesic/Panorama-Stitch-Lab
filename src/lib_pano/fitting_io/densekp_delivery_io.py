"""Load ``densekp_delivery`` pair JSON into full ERP_ROTATE coordinates."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np

from lib_pano.camera.ds_camera_params import (
    DS_PARAMS,
    DS_PARAMS_CAM2,
    DS_PARAMS_CAM4,
)


ERP_TO_ERP_ROTATE = np.array(
    [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]],
    dtype=np.float64,
)
DELIVERY_ROTATION_MATRIX = np.eye(3, dtype=np.float64)
DELIVERY_EXTRINSIC = {
    "cam2": np.eye(3, dtype=np.float64),
    "cam4": np.diag([-1.0, 1.0, -1.0]).astype(np.float64),
}


@dataclass(frozen=True)
class DenseKPDeliveryObservation:
    points_cam2: np.ndarray
    points_cam4: np.ndarray
    point_ids: np.ndarray
    valid: np.ndarray
    coord_space: str = "erp_rotate"


def _erp_to_ray(points: np.ndarray, width: int, height: int) -> np.ndarray:
    lon = points[:, 0] / width * (2.0 * np.pi) - np.pi
    lat = points[:, 1] / height * np.pi - np.pi / 2.0
    cos_lat = np.cos(lat)
    return np.stack(
        [cos_lat * np.sin(lon), np.sin(lat), cos_lat * np.cos(lon)], axis=1
    )


def _ray_to_erp(rays: np.ndarray, width: int, height: int) -> np.ndarray:
    rays = rays / np.linalg.norm(rays, axis=1, keepdims=True)
    lon = np.arctan2(rays[:, 0], rays[:, 2])
    lat = np.arcsin(np.clip(rays[:, 1], -1.0, 1.0))
    x = ((lon + np.pi) / (2.0 * np.pi) * width) % width
    y = np.clip((lat + np.pi / 2.0) / np.pi * height, 0.0, height - 1.0)
    return np.stack([x, y], axis=1).astype(np.float32)


def erp_to_erp_rotate(
    points: np.ndarray, width: int, height: int
) -> np.ndarray:
    rays = _erp_to_ray(np.asarray(points, dtype=np.float64), width, height)
    return _ray_to_erp(rays @ ERP_TO_ERP_ROTATE.T, width, height)


def _unproject_double_sphere(
    points: np.ndarray, params: dict[str, float]
) -> np.ndarray:
    xi, alpha = float(params["xi"]), float(params["alpha"])
    mx = (points[:, 0] - float(params["cx"])) / float(params["fx"])
    my = (points[:, 1] - float(params["cy"])) / float(params["fy"])
    r2 = mx * mx + my * my
    sqrt_term = 1.0 - (2.0 * alpha - 1.0) * r2
    if np.any(sqrt_term < 0.0):
        raise ValueError("fisheye DenseKP point is outside the Double Sphere domain")
    mz = (1.0 - alpha * alpha * r2) / (
        alpha * np.sqrt(sqrt_term) + 1.0 - alpha
    )
    inner = mz * mz + (1.0 - xi * xi) * r2
    factor = (mz * xi + np.sqrt(np.maximum(inner, 0.0))) / (mz * mz + r2)
    return np.stack([factor * mx, factor * my, factor * mz - xi], axis=1)


def fisheye_to_erp(
    points: np.ndarray,
    camera: str,
    width: int,
    height: int,
    camera_group: str,
) -> np.ndarray:
    if camera not in DELIVERY_EXTRINSIC:
        raise ValueError(f"unsupported camera: {camera}")
    params = (
        DS_PARAMS_CAM2 if camera == "cam2" else DS_PARAMS_CAM4
    ) if camera_group == "new" else DS_PARAMS
    camera_rays = _unproject_double_sphere(
        np.asarray(points, dtype=np.float64), params
    )
    world_rays = (
        camera_rays
        @ DELIVERY_ROTATION_MATRIX
        @ DELIVERY_EXTRINSIC[camera]
    )
    return _ray_to_erp(world_rays, width, height)


def _observation_from_payload(
    data: dict,
    *,
    image_width: int,
    image_height: int,
    camera_group: str,
) -> DenseKPDeliveryObservation:
    """Normalize one person payload from v1 or v2 to ERP_ROTATE."""
    pairs = np.asarray(data["pairs"], dtype=np.float32)
    if pairs.ndim != 2 or pairs.shape[1] != 4:
        raise ValueError(f"invalid DenseKP pairs shape: {pairs.shape}")
    point_ids = np.asarray(data.get("point_ids", np.arange(len(pairs))), dtype=np.int32)
    valid = np.asarray(data.get("valid", np.ones(len(pairs))), dtype=bool)
    if point_ids.shape != (len(pairs),) or valid.shape != (len(pairs),):
        raise ValueError("DenseKP point_ids/valid length mismatch")
    points2, points4 = pairs[:, :2], pairs[:, 2:]
    coord_space = data.get("coord_space", "erp")
    if coord_space == "erp":
        points2 = erp_to_erp_rotate(points2, image_width, image_height)
        points4 = erp_to_erp_rotate(points4, image_width, image_height)
    elif coord_space == "fisheye":
        points2 = erp_to_erp_rotate(fisheye_to_erp(points2, "cam2", image_width, image_height, camera_group), image_width, image_height)
        points4 = erp_to_erp_rotate(fisheye_to_erp(points4, "cam4", image_width, image_height, camera_group), image_width, image_height)
    elif coord_space != "erp_rotate":
        raise ValueError(f"unsupported DenseKP coordinate space: {coord_space}")
    finite = np.isfinite(points2).all(axis=1) & np.isfinite(points4).all(axis=1)
    in_range = ((points2[:,0]>=0)&(points2[:,0]<image_width)&(points2[:,1]>=0)&(points2[:,1]<image_height)&(points4[:,0]>=0)&(points4[:,0]<image_width)&(points4[:,1]>=0)&(points4[:,1]<image_height))
    return DenseKPDeliveryObservation(np.ascontiguousarray(points2,np.float32),np.ascontiguousarray(points4,np.float32),point_ids,valid&finite&in_range)


def load_densekp_delivery_observations(
    path: str | Path,
    *,
    image_width: int = 5760,
    image_height: int = 2880,
    camera_group: str = "old",
) -> list[DenseKPDeliveryObservation]:
    """Load every matched person; this is the canonical v2 API."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if data.get("status", "matched") != "matched":
        raise ValueError(f"DenseKP delivery is not matched: {data.get('status')}")
    schema = data.get("schema_version", "densekp_pairs_v1")
    payloads = data.get("persons", []) if schema == "densekp_pairs_v2" else [data]
    return [_observation_from_payload(
        {**person, "coord_space": data.get("coord_space", "erp")},
        image_width=image_width, image_height=image_height,
        camera_group=camera_group,
    ) for person in payloads]


def load_densekp_delivery_observation(
    path: str | Path, *, image_width: int = 5760, image_height: int = 2880,
    camera_group: str = "old",
) -> DenseKPDeliveryObservation:
    """Compatibility API: accept v1 or exactly one v2 person, never choose person 0."""
    observations = load_densekp_delivery_observations(
        path, image_width=image_width, image_height=image_height,
        camera_group=camera_group,
    )
    if len(observations) != 1:
        raise ValueError(
            "single-person staged-LM requires exactly one matched DenseKP person; "
            f"delivery contains {len(observations)}. Select/track one person explicitly."
        )
    return observations[0]
