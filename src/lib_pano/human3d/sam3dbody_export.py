"""
Export SAM 3D Body reconstruction data to standard observation dicts.

No torch, no model loading. Reads Sam3DBodyRecon and outputs
JSON-serializable dicts for downstream use.
"""

from __future__ import annotations

import numpy as np

from lib_pano.human3d.sam3dbody_io import Sam3DBodyRecon
from lib_pano.human3d.crop_upright_coord import upright_points_to_equirect_rotate


def sam3dbody_recon_to_observation_dict(
    recon: Sam3DBodyRecon,
    bbox_full: list[int] | tuple[int, int, int, int] | None = None,
    restore_to_full: bool = False,
    max_vertices: int | None = None,
) -> dict:
    """Convert a single-camera SAM 3D Body reconstruction to an observation dict.

    Args:
        recon: loaded Sam3DBodyRecon.
        bbox_full: crop bbox [x1,y1,x2,y2] in equirect_rotate coords.
            Required if restore_to_full=True.
        restore_to_full: if True, restore vertices_2d and keypoints_2d
            from crop_upright to equirect_rotate coordinates.
        max_vertices: if set, only export up to this many valid inside vertices.
    """
    valid = recon.valid_z & recon.inside
    valid_ids = np.where(valid)[0]

    if max_vertices is not None and len(valid_ids) > max_vertices:
        valid_ids = valid_ids[:max_vertices]

    space = "equirect_rotate" if restore_to_full else "crop_upright"

    verts = recon.vertices_2d[valid_ids].copy()
    if restore_to_full:
        if bbox_full is None:
            raise ValueError("bbox_full is required when restore_to_full=True")
        verts = upright_points_to_equirect_rotate(verts, list(bbox_full), recon.frame_idx)

    verts_list = [[float(v[0]), float(v[1])] for v in verts]

    # keypoints
    kp_raw = None
    if recon.pred_keypoints_2d is not None:
        kp_raw = recon.pred_keypoints_2d.copy()
    if kp_raw is not None:
        if kp_raw.shape[1] == 2:
            kp_raw = np.concatenate([kp_raw, np.ones((kp_raw.shape[0], 1), dtype=np.float32)], axis=1)
        if restore_to_full:
            kp_xy = kp_raw[:, :2]
            kp_xy_restored = upright_points_to_equirect_rotate(kp_xy, list(bbox_full), recon.frame_idx)
            kp_raw = np.concatenate([kp_xy_restored, kp_raw[:, 2:3]], axis=1)
        keypoints_2d = [[float(k[0]), float(k[1]), float(k[2])] for k in kp_raw]
    else:
        keypoints_2d = None

    bbox_crop = [float(x) for x in recon.bbox]
    bbox_full_export = [float(x) for x in bbox_full] if bbox_full is not None else None

    return {
        "source": "sam3dbody",
        "frame_idx": int(recon.frame_idx),
        "cam": recon.cam,
        "image_path": recon.image_path,
        "image_shape": [int(recon.image_shape[0]), int(recon.image_shape[1]), int(recon.image_shape[2])],
        "space": space,
        "bbox_crop": bbox_crop,
        "bbox_full": bbox_full_export,
        "focal_length": float(recon.focal_length),
        "num_vertices": int(recon.vertices_2d.shape[0]),
        "num_valid_z": int(recon.valid_z.sum()),
        "num_inside": int(recon.inside.sum()),
        "vertices_2d": verts_list,
        "valid_vertex_ids": [int(x) for x in valid_ids],
        "keypoints_2d": keypoints_2d,
    }


def build_sam3dbody_pair_summary_dict(
    frame_idx: int,
    src_cam: str,
    dst_cam: str,
    src_pts: np.ndarray,
    dst_pts: np.ndarray,
    vertex_ids: np.ndarray,
) -> dict:
    """Build a lightweight pair summary dict for cam4->cam2 correspondences."""
    disp = src_pts - dst_pts
    disp_norm = np.linalg.norm(disp, axis=1)

    n_show = min(5, len(vertex_ids))
    first_ids = [int(x) for x in vertex_ids[:n_show]]

    return {
        "source": "sam3dbody",
        "frame_idx": int(frame_idx),
        "src_cam": src_cam,
        "dst_cam": dst_cam,
        "num_pairs": int(len(src_pts)),
        "src_min": [float(src_pts[:, 0].min()), float(src_pts[:, 1].min())],
        "src_max": [float(src_pts[:, 0].max()), float(src_pts[:, 1].max())],
        "dst_min": [float(dst_pts[:, 0].min()), float(dst_pts[:, 1].min())],
        "dst_max": [float(dst_pts[:, 0].max()), float(dst_pts[:, 1].max())],
        "disp_mean": float(disp_norm.mean()),
        "disp_median": float(np.median(disp_norm)),
        "disp_max": float(disp_norm.max()),
        "first_vertex_ids": first_ids,
    }
