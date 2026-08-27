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
