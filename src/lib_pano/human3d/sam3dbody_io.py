"""
SAM 3D Body reconstruction cache IO (npz read/write).

No torch dependency. Reads/writes the npz format produced by
debug066_batch_export_sam3dbody_recon.py.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class Sam3DBodyRecon:
    """Single-camera SAM 3D Body reconstruction at one frame."""

    image_path: str
    cam: str
    frame_idx: int
    image_shape: tuple[int, int, int]
    bbox: np.ndarray
    focal_length: float
    pred_vertices: np.ndarray
    pred_cam_t: np.ndarray
    vertices_cam: np.ndarray
    vertices_2d: np.ndarray
    valid_z: np.ndarray
    inside: np.ndarray
    faces: np.ndarray | None = None
    pred_keypoints_2d: np.ndarray | None = None


def load_sam3dbody_recon(npz_path: str | Path) -> Sam3DBodyRecon:
    with np.load(npz_path, allow_pickle=True) as data:
        image_path = str(data["image_path"])
        cam = str(data["cam"])
        frame_idx = int(data["frame_idx"])
        image_shape = tuple(int(x) for x in data["image_shape"])

        bbox = data["bbox"].astype(np.float32)
        focal_length = float(data["focal_length"])
        pred_vertices = data["pred_vertices"].astype(np.float32)
        pred_cam_t = data["pred_cam_t"].astype(np.float32)
        vertices_cam = data["vertices_cam"].astype(np.float32)
        vertices_2d = data["vertices_2d"].astype(np.float32)
        valid_z = data["valid_z"].astype(bool)
        inside = data["inside"].astype(bool)

        faces = None
        if "faces" in data.files:
            faces = data["faces"].astype(np.int64)

        pred_keypoints_2d = None
        if "pred_keypoints_2d" in data.files:
            pred_keypoints_2d = data["pred_keypoints_2d"].astype(np.float32)

    return Sam3DBodyRecon(
        image_path=image_path,
        cam=cam,
        frame_idx=frame_idx,
        image_shape=(int(image_shape[0]), int(image_shape[1]), 3),
        bbox=bbox,
        focal_length=focal_length,
        pred_vertices=pred_vertices,
        pred_cam_t=pred_cam_t,
        vertices_cam=vertices_cam,
        vertices_2d=vertices_2d,
        valid_z=valid_z,
        inside=inside,
        faces=faces,
        pred_keypoints_2d=pred_keypoints_2d,
    )


def save_sam3dbody_recon(npz_path: str | Path, recon: Sam3DBodyRecon) -> None:
    Path(npz_path).parent.mkdir(parents=True, exist_ok=True)
    kwargs: dict = dict(
        image_path=str(recon.image_path),
        cam=recon.cam,
        frame_idx=recon.frame_idx,
        image_shape=np.array(recon.image_shape, dtype=np.int32),
        bbox=recon.bbox.astype(np.float32),
        focal_length=np.array(recon.focal_length, dtype=np.float32),
        pred_vertices=recon.pred_vertices.astype(np.float32),
        pred_cam_t=recon.pred_cam_t.astype(np.float32),
        vertices_cam=recon.vertices_cam.astype(np.float32),
        vertices_2d=recon.vertices_2d.astype(np.float32),
        valid_z=recon.valid_z.astype(bool),
        inside=recon.inside.astype(bool),
    )
    if recon.faces is not None:
        kwargs["faces"] = recon.faces.astype(np.int64)
    if recon.pred_keypoints_2d is not None:
        kwargs["pred_keypoints_2d"] = recon.pred_keypoints_2d.astype(np.float32)
    np.savez_compressed(npz_path, **kwargs)
