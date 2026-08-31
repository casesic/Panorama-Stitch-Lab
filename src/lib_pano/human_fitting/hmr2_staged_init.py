"""Explicit HMR2-to-staged-LM initialization conversion helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping
import importlib

import numpy as np

from lib_pano.human3d.hmr2_initializer import HMR2Initialization


def rotation_matrices_to_axis_angle(rotations: np.ndarray) -> np.ndarray:
    """Use the same OpenCV Rodrigues conversion as the proven local demo."""
    import cv2

    shape = np.asarray(rotations).shape[:-2]
    values = np.asarray(rotations, dtype=np.float64).reshape(-1, 3, 3)
    converted = [cv2.Rodrigues(value)[0].reshape(3) for value in values]
    return np.asarray(converted, dtype=np.float32).reshape(*shape, 3)


def axis_angle_to_rotation_matrices(axis_angles: np.ndarray) -> np.ndarray:
    import cv2

    shape = np.asarray(axis_angles).shape[:-1]
    values = np.asarray(axis_angles, dtype=np.float64).reshape(-1, 3)
    converted = [cv2.Rodrigues(value)[0] for value in values]
    return np.asarray(converted, dtype=np.float32).reshape(*shape, 3, 3)


def cam_crop_to_full_numpy(
    pred_cam: np.ndarray,
    bbox_xyxy: np.ndarray,
    image_size: tuple[int, int],
    focal_length: float,
) -> np.ndarray:
    """Numerical equivalent of ``hmr2.utils.renderer.cam_crop_to_full``."""
    bbox = np.asarray(bbox_xyxy, dtype=np.float64)
    cam = np.asarray(pred_cam, dtype=np.float64)
    center = (bbox[:2] + bbox[2:]) / 2.0
    width, height = image_size
    # ViTDetDataset expands to the configured 192:256 bbox aspect ratio.
    raw_width, raw_height = bbox[2:] - bbox[:2]
    box_size = max(raw_width, raw_height * 192.0 / 256.0,
                   raw_height, raw_width * 256.0 / 192.0)
    denominator = box_size * cam[0] + 1e-9
    return np.asarray((
        2.0 * (center[0] - width / 2.0) / denominator + cam[1],
        2.0 * (center[1] - height / 2.0) / denominator + cam[2],
        2.0 * focal_length / denominator,
    ), dtype=np.float32)


@dataclass(frozen=True)
class HMR2StagedInitialization:
    """Person-specific values required to materialize a T=1 fixture."""

    body_pose_smpl69: np.ndarray
    beta: np.ndarray
    global_orient: np.ndarray
    translation_cam2: np.ndarray
    translation_cam4: np.ndarray
    scale: np.ndarray
    source_camera: str


# Translation magnitude for cold starts.  The staged-LM ERP projection is
# directional (project_unified_erp normalizes the point), so the translation
# must stay within ~a few body-size units.  Phase-A bake-off (2213/2214/2219,
# unchanged C++ LM): the objective basin is flat for radius in [0.5, 7] and
# collapses again toward radius >= 10; 2.5 sits mid-basin for every case.
SPHERICAL_TRANSLATION_RADIUS = 2.5
MINIMUM_POINTS_FOR_SPHERICAL_TRANSLATION = 4


def erp_pixel_direction(points: np.ndarray, image_width: int,
                        image_height: int) -> np.ndarray:
    """ERP_ROTATE pixel -> unit ray in the staged-LM model world frame.

    Exact inverse of the C++ ``project_unified_erp`` convention (including its
    fixed rotation), so observation-derived directions live in the same space
    as the model vertices.
    """
    points = np.asarray(points, dtype=np.float64)
    longitude = (points[..., 0] / image_width - 0.5) * 2.0 * np.pi
    latitude = (0.5 - points[..., 1] / image_height) * np.pi
    return np.stack([
        np.cos(latitude) * np.sin(longitude),
        -np.cos(latitude) * np.cos(longitude),
        -np.sin(latitude),
    ], axis=-1)


def project_model_points(points: np.ndarray, image_width: int,
                         image_height: int) -> np.ndarray:
    """staged-LM model-world 3D points -> ERP_ROTATE pixels (C++ convention)."""
    points = np.asarray(points, dtype=np.float64)
    camera = np.stack([points[:, 0], points[:, 2], -points[:, 1]], axis=-1)
    longitude = np.arctan2(camera[:, 0], camera[:, 2])
    latitude = np.arcsin(np.clip(
        -camera[:, 1] / np.linalg.norm(camera, axis=1), -1.0, 1.0))
    return np.stack([
        (longitude / (2.0 * np.pi) + 0.5) * image_width,
        (0.5 - latitude / np.pi) * image_height,
    ], axis=-1)


def spherical_translation_initialization(
    points_cam2: np.ndarray, points_cam4: np.ndarray, valid: np.ndarray,
    image_width: int = 5760, image_height: int = 2880,
    radius: float = SPHERICAL_TRANSLATION_RADIUS,
    bbox_cam2: np.ndarray | None = None, bbox_cam4: np.ndarray | None = None,
    bbox_cam2_wraps_seam: bool = False, bbox_cam4_wraps_seam: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Cold-start translation in staged-LM directional/spherical semantics.

    The HMR2 ``cam_crop_to_full`` translation is pinhole-camera
    ``(u_px, v_px, depth)``; injecting its depth component (~hundreds) here
    collapses every vertex direction to a single point under the directional
    ERP projection.  Instead, place the body at an observation-derived
    direction with a magnitude inside the verified objective basin:

    A. ``valid`` DenseKP points >= 4: spherical mean direction of the
       observation points (per camera).
    B. fewer valid points: center of the matched person's ERP bbox, rotated
       from its normal-ERP space into the staged-LM ERP_ROTATE frame via the
       established ``DEFAULT_ERP_TO_ERP_ROTATE_R``.  Seam-wrapping bboxes use
       a minimal circular arc for the center.
    C. no usable bbox either: ``ValueError`` — an unknown cold-start
       translation is never silently accepted; the HMR2 pinhole translation
       must not fall back into the spherical state under any circumstance.

    Returns ``(translation_cam2, translation_cam4)``.
    """
    translations = []
    for camera, points, bbox, wraps in (
            ("cam2", np.asarray(points_cam2, np.float64),
             bbox_cam2, bbox_cam2_wraps_seam),
            ("cam4", np.asarray(points_cam4, np.float64),
             bbox_cam4, bbox_cam4_wraps_seam)):
        valid_mask = np.asarray(valid, bool)
        chosen = points[valid_mask]
        if chosen.shape[0] >= MINIMUM_POINTS_FOR_SPHERICAL_TRANSLATION:
            direction = erp_pixel_direction(chosen, image_width, image_height)
            direction = direction.mean(axis=0)
        elif bbox is not None and np.isfinite(bbox).all():
            center = _seam_safe_bbox_center(bbox, wraps, image_width)
            # bbox is normal ERP; staged-LM consumes ERP_ROTATE directions.
            rotated = _rotate_erp_pixel(center=(float(center[0]), float(center[1])),
                                        image_width=image_width,
                                        image_height=image_height)
            direction = erp_pixel_direction(np.asarray([[rotated[0], rotated[1]]]),
                                            image_width, image_height)
            direction = direction[0]
        else:
            raise ValueError(
                f"{camera}: no valid DenseKP points and no usable bbox; "
                "refusing to fall back to the pinhole-semantics HMR2 translation")
        norm = float(np.linalg.norm(direction))
        if not np.isfinite(norm) or norm == 0.0:
            raise ValueError(f"{camera}: degenerate spherical direction")
        translations.append(radius * direction / norm)
    return translations[0], translations[1]


def _rotate_erp_pixel(center: tuple[float, float], image_width: int,
                      image_height: int) -> tuple[float, float]:
    """Normal-ERP pixel -> ERP_ROTATE pixel via the established rotation."""
    from densekp_delivery.camera.erp_rotate import rotate_erp_points

    rotated = rotate_erp_points(
        np.asarray([[center[0], center[1]]], np.float64),
        image_width, image_height)
    return float(rotated[0, 0]), float(rotated[0, 1])


def _seam_safe_bbox_center(bbox: np.ndarray, wraps: bool,
                           image_width: int) -> tuple[float, float]:
    """Center of a normal-ERP xyxy bbox; minimal arc when it wraps the seam."""
    x1, y1, x2, y2 = (float(v) for v in bbox)
    if not wraps:
        return (x1 + x2) / 2.0, (y1 + y2) / 2.0
    width = float(image_width)
    x1m, x2m = x1 % width, x2 % width
    if x2m >= x1m:                     # wrapping already resolved upstream
        return (x1m + x2m) / 2.0, (y1 + y2) / 2.0
    arc = (x1m + (x2m + width - x1m) / 2.0) % width
    return arc, (y1 + y2) / 2.0


def to_smplx_camera_initializations(
    initialization: HMR2StagedInitialization,
) -> tuple[object, object]:
    """Reuse the established SMPL_Param_Seq -> SMPL-X conversion verbatim."""
    import torch
    from lib_pano.human_fitting import init_smplx_from_smpl54_seq

    smpl_module = importlib.import_module(
        "lib_dynamic_hvip.fitting.smpl_seq_forward_module"
    )
    parameter_type = smpl_module.SMPL_Seq_Forward_Module.SMPL_Param_Seq

    def build(translation: np.ndarray):
        value = parameter_type(dim_shape=10, gender="neutral", num_batch=1)
        value.orient[0] = torch.from_numpy(initialization.global_orient)
        value.transl[0] = torch.from_numpy(translation)
        value.pose[0] = torch.from_numpy(initialization.body_pose_smpl69)
        value.beta = torch.from_numpy(initialization.beta.copy())
        value.scale = torch.from_numpy(initialization.scale.copy())
        value.valid = torch.ones(1, dtype=torch.bool)
        return init_smplx_from_smpl54_seq(value, [0])

    return build(initialization.translation_cam2), build(initialization.translation_cam4)


def convert_hmr2_to_staged_initialization(
    initialization: HMR2Initialization,
    *,
    source_camera: str,
    bboxes: Mapping[str, np.ndarray],
    image_size: tuple[int, int] = (5760, 2880),
    focal_length: float,
) -> HMR2StagedInitialization:
    """Convert HMR2 values while keeping camera semantics explicit.

    The same predicted crop camera is evaluated with each matched view's bbox.
    This provides the two full-image translations required by Stage1 without a
    second HMR2 forward.  The actual SMPL->SMPL-X conversion remains the proven
    ``init_smplx_from_smpl54_seq`` path.  This helper intentionally does not
    guess the SMPL-23 to SMPL-X-21 joint mapping.
    """
    if source_camera not in {"cam2", "cam4"}:
        raise ValueError(f"unsupported source camera: {source_camera}")
    orient = rotation_matrices_to_axis_angle(initialization.global_orient).reshape(3)
    pose69 = rotation_matrices_to_axis_angle(initialization.body_pose).reshape(69)
    if initialization.betas.shape != (10,):
        raise ValueError("HMR2 beta must have shape [10]")
    translations = {
        camera: cam_crop_to_full_numpy(
            initialization.pred_cam, np.asarray(bboxes[camera]), image_size,
            focal_length,
        ) for camera in ("cam2", "cam4")
    }
    return HMR2StagedInitialization(
        body_pose_smpl69=pose69,
        beta=initialization.betas.copy(),
        global_orient=orient,
        translation_cam2=translations["cam2"],
        translation_cam4=translations["cam4"],
        scale=np.asarray([initialization.pred_cam[0]], dtype=np.float32),
        source_camera=source_camera,
    )
