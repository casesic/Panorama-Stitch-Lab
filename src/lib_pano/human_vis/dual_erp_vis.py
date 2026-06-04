from __future__ import annotations

from typing import Callable, Optional

import numpy as np
import torch

from lib_pano.human_fitting.projection import project_cam2 as _project_cam2
from lib_pano.human_vis.draw_2d import draw_pts, draw_skel
from lib_pano.human_vis.mesh_render import render_mesh
from lib_pano.human_vis.smplx_forward_vis import _load_pickle, normalize_keypoints_format

# SMPL-X joint index constants
SMPLX_LHAND = list(range(25, 40))
SMPLX_RHAND = list(range(40, 55))
SMPLX_FACE = [55, 56, 57, 58, 59]

SMPLX_BODY_PAIRS = [
    (0, 3), (3, 6), (6, 9), (9, 12), (12, 15),
    (9, 13), (13, 16), (16, 18), (18, 20),
    (9, 14), (14, 17), (17, 19), (19, 21),
    (0, 1), (1, 4), (4, 7), (7, 10),
    (0, 2), (2, 5), (5, 8), (8, 11),
]

COCO_PAIRS = [
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
]

SMPLX_BODY_22 = {
    "Pelvis": 0, "L_Hip": 1, "R_Hip": 2, "Spine1": 3,
    "L_Knee": 4, "R_Knee": 5, "Spine2": 6,
    "L_Ankle": 7, "R_Ankle": 8, "Spine3": 9,
    "L_Foot": 10, "R_Foot": 11, "Neck": 12,
    "L_Collar": 13, "R_Collar": 14, "Head": 15,
    "L_Shoulder": 16, "R_Shoulder": 17,
    "L_Elbow": 18, "R_Elbow": 19,
    "L_Wrist": 20, "R_Wrist": 21,
    "Nose": 55, "L_Eye": 57, "R_Eye": 56,
    "L_Ear": 59, "R_Ear": 58,
}

# Rotation constants (kept here for mesh rendering; match lib_pano.human_fitting.projection)
_R_CAM2_NP = np.array([1, 0, 0, 0, 0, 1, 0, -1, 0], dtype=np.float32).reshape(3, 3)


def project_cam2(points_3d: torch.Tensor, W: int, H: int) -> torch.Tensor:
    return _project_cam2(points_3d, W, H)


def project_cam4(points_3d: torch.Tensor, W: int, H: int) -> torch.Tensor:
    """cam4 visualization projection: uses the same rotation as cam2 (R_CAM4_PROJ = R_CAM2)."""
    return _project_cam2(points_3d, W, H)


def load_rtm_keypoints(rtm_pkl_path) -> tuple[Optional[np.ndarray], dict]:
    import numpy as np
    from pathlib import Path

    rtm_pkl_path = Path(rtm_pkl_path)
    if not rtm_pkl_path.exists():
        print(f"[warn] RTM pkl not found: {rtm_pkl_path}")
        return None, {}

    det_raw = _load_pickle(rtm_pkl_path)
    if not hasattr(det_raw, "array_keypoints"):
        print(f"[warn] Unsupported RTM result format: {rtm_pkl_path}")
        return None, {}

    kp_all = np.asarray(det_raw.array_keypoints)
    if kp_all.ndim != 3 or kp_all.shape[0] == 0:
        return None, normalize_keypoints_format(getattr(det_raw, "keypoints_format", {}))

    kp133 = kp_all[0].copy()
    kp_fmt = normalize_keypoints_format(getattr(det_raw, "keypoints_format", {}))
    return kp133, kp_fmt


def project_points_erp(
    points_3d: np.ndarray,
    width: int,
    height: int,
    project_fn: Callable[[torch.Tensor, int, int], torch.Tensor],
) -> np.ndarray:
    points_t = torch.from_numpy(np.asarray(points_3d, dtype=np.float32))
    points_2d = project_fn(points_t, width, height)
    return points_2d.detach().cpu().numpy()


def visualize_frame(
    img: np.ndarray,
    smplx_param,
    kp133: Optional[np.ndarray],
    kp_fmt: dict,
    forwarder,
    image_width: int,
    image_height: int,
    project_fn: Callable[[torch.Tensor, int, int], torch.Tensor],
    draw_mesh: bool = False,
    thresh: float = 0.1,
    mesh_alpha: float = 0.6,
) -> tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    vis = img.copy()

    verts, joints_all, joints_body = forwarder.forward_param_to_numpy(smplx_param)
    verts = np.asarray(verts, dtype=np.float32)
    joints_all = np.asarray(joints_all, dtype=np.float32)
    joints_body = np.asarray(joints_body, dtype=np.float32)

    faces = None
    if draw_mesh:
        faces = forwarder.get_model_faces()
        verts_proj = verts @ _R_CAM2_NP.T
        vis = render_mesh(
            vis,
            verts_proj,
            faces,
            width=image_width,
            height=image_height,
            alpha=mesh_alpha,
            use_shader=True,
        )

    kp_names = set(normalize_keypoints_format(kp_fmt).keys())
    matched_idx = [idx for name, idx in SMPLX_BODY_22.items() if name in kp_names and idx < 22]
    all_body_2d = project_points_erp(joints_body, image_width, image_height, project_fn)
    if matched_idx:
        matched_pts = joints_all[matched_idx]
        matched_2d = project_points_erp(matched_pts, image_width, image_height, project_fn)
        filtered_pairs = [(i, j) for i, j in SMPLX_BODY_PAIRS if i in matched_idx and j in matched_idx]
        draw_skel(vis, all_body_2d, filtered_pairs, color=(0, 100, 255), thick=2)
        draw_pts(vis, matched_2d, r=5, color=(0, 140, 255))
    elif not kp_names:
        draw_skel(vis, all_body_2d, SMPLX_BODY_PAIRS, color=(0, 100, 255), thick=2)
        draw_pts(vis, all_body_2d, r=5, color=(0, 140, 255))

    if not kp_names or {"Nose", "L_Eye", "R_Eye", "L_Ear", "R_Ear"} & kp_names:
        draw_pts(
            vis,
            project_points_erp(joints_all[SMPLX_FACE], image_width, image_height, project_fn),
            r=4,
            color=(0, 255, 180),
        )

    draw_pts(
        vis,
        project_points_erp(joints_all[SMPLX_LHAND], image_width, image_height, project_fn),
        r=4,
        color=(255, 80, 0),
    )
    draw_pts(
        vis,
        project_points_erp(joints_all[SMPLX_RHAND], image_width, image_height, project_fn),
        r=4,
        color=(200, 0, 255),
    )
    return vis, verts, faces
