from pathlib import Path

import numpy as np
import torch

from lib_dynamic_hvip.smpl.smpl_lib import get_smplx_batch, forward_smplx_tensor_batch
from single_fitting.debug_shared import ds_project_np, erp_project_np, load_pickle, erp_project_consistent


def project_verts_to_rotated_erp(
    verts_world: np.ndarray,
    rotation_matrix: np.ndarray,
    erp_h: int,
    erp_w: int,
) -> np.ndarray:
    verts_rotated = verts_world @ rotation_matrix
    pts_2d = erp_project_consistent(verts_rotated, erp_w, erp_h)
    return pts_2d


def compute_semantic_flow(
    verts_world_A: np.ndarray,
    verts_world_B: np.ndarray,
    faces: np.ndarray,
    erp_h: int,
    erp_w: int,
    rotation_matrix: np.ndarray,
    roi_bbox: tuple,
    visible_mask_B: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    pts_2d_A = project_verts_to_rotated_erp(verts_world_A, rotation_matrix, erp_h, erp_w)
    pts_2d_B = project_verts_to_rotated_erp(verts_world_B, rotation_matrix, erp_h, erp_w)
    depths_A = np.linalg.norm(verts_world_A, axis=1)
    # 稠密化
    _, face_id_map, bary_map = rasterize_mesh_from_2d(pts_2d_A, depths_A, faces, erp_h, erp_w)

    x1, y1, x2, y2 = roi_bbox
    face_id_roi = face_id_map[y1:y2, x1:x2]  # (roi_h, roi_w)
    bary_roi = bary_map[y1:y2, x1:x2]        # (roi_h, roi_w, 3)

    roi_h = y2 - y1
    roi_w = x2 - x1
    flow_roi = np.zeros((roi_h, roi_w, 2), dtype=np.float64)
    confidence_roi = np.zeros((roi_h, roi_w), dtype=np.uint8)

    valid = face_id_roi != -1  # find pixels that fall on  human body splice
    if not np.any(valid):
        return flow_roi, confidence_roi

    face_ids = face_id_roi[valid]  # (N, 0)  N effective pixels 
    v_ids = faces[face_ids]   # (N, 3) N pixel corresponds to a face, and each face has 3 vertex indices
    bary = bary_roi[valid]

    pts_B = (
        bary[:, [0]] * pts_2d_B[v_ids[:, 0]]
        + bary[:, [1]] * pts_2d_B[v_ids[:, 1]]
        + bary[:, [2]] * pts_2d_B[v_ids[:, 2]]
    )  # (N, 2)
    pts_A = (
        bary[:, [0]] * pts_2d_A[v_ids[:, 0]]
        + bary[:, [1]] * pts_2d_A[v_ids[:, 1]]
        + bary[:, [2]] * pts_2d_A[v_ids[:, 2]]
    )  # (N, 2)
    flow = pts_B - pts_A  # (N, 2)
    flow_roi[valid] = flow  # (roi_h, roi_w, 2)

    vis_faces = visible_mask_B[v_ids]
    all_visible = np.all(vis_faces, axis=1)  # (N, )
    confidence = np.ones(face_ids.shape[0], dtype=np.uint8)  # (N, )
    confidence[all_visible] = 2
    confidence_roi[valid] = confidence  # (roi_h, roi_w)

    return flow_roi, confidence_roi


def rasterize_mesh_from_2d(
    pts_2d: np.ndarray,
    depths: np.ndarray,
    faces: np.ndarray,
    image_h: int,
    image_w: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    depth_map = np.full((image_h, image_w), np.inf, dtype=np.float64)
    face_id_map = np.full((image_h, image_w), -1, dtype=np.int32)
    bary_map = np.zeros((image_h, image_w, 3), dtype=np.float64)

    for face_idx, face in enumerate(faces):
        tri_2d = pts_2d[face]
        tri_depth = depths[face]

        if np.ptp(tri_2d[:, 0]) > image_w // 2:
            continue

        min_u = max(0, int(np.ceil(np.min(tri_2d[:, 0]))))
        max_u = min(image_w - 1, int(np.floor(np.max(tri_2d[:, 0]))))
        min_v = max(0, int(np.ceil(np.min(tri_2d[:, 1]))))
        max_v = min(image_h - 1, int(np.floor(np.max(tri_2d[:, 1]))))
        if min_u > max_u or min_v > max_v:
            continue

        a = tri_2d[0]
        b = tri_2d[1]
        c = tri_2d[2]
        v0 = b - a
        v1 = c - a

        d00 = float(np.dot(v0, v0))
        d01 = float(np.dot(v0, v1))
        d11 = float(np.dot(v1, v1))
        denom = d00 * d11 - d01 * d01
        if abs(denom) < 1e-12:
            continue
        inv_denom = 1.0 / denom

        cols = np.arange(min_u, max_u + 1, dtype=np.float64)
        rows = np.arange(min_v, max_v + 1, dtype=np.float64)
        grid_u, grid_v = np.meshgrid(cols, rows)
        p = np.stack([grid_u, grid_v], axis=-1)
        v2 = p - a

        d20 = v2[..., 0] * v0[0] + v2[..., 1] * v0[1]
        d21 = v2[..., 0] * v1[0] + v2[..., 1] * v1[1]
        lambda2 = (d11 * d20 - d01 * d21) * inv_denom
        lambda3 = (d00 * d21 - d01 * d20) * inv_denom
        lambda1 = 1.0 - lambda2 - lambda3

        valid = (lambda1 >= 0.0) & (lambda2 >= 0.0) & (lambda3 >= 0.0)
        if not np.any(valid):
            continue

        d_interp = lambda1 * tri_depth[0] + lambda2 * tri_depth[1] + lambda3 * tri_depth[2]
        patch_depth = depth_map[min_v : max_v + 1, min_u : max_u + 1]
        update_mask = valid & (d_interp < patch_depth)
        if not np.any(update_mask):
            continue

        patch_depth[update_mask] = d_interp[update_mask]
        face_id_patch = face_id_map[min_v : max_v + 1, min_u : max_u + 1]
        face_id_patch[update_mask] = face_idx
        bary_patch = bary_map[min_v : max_v + 1, min_u : max_u + 1]
        bary_patch[update_mask, 0] = lambda1[update_mask]
        bary_patch[update_mask, 1] = lambda2[update_mask]
        bary_patch[update_mask, 2] = lambda3[update_mask]

    return depth_map, face_id_map, bary_map


def rasterize_mesh_fisheye(
    verts_cam: np.ndarray,
    faces: np.ndarray,
    ds_params: dict,
    image_h: int,
    image_w: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pts_2d = ds_project_np(verts_cam, **ds_params)
    depths = np.linalg.norm(verts_cam, axis=1)
    return rasterize_mesh_from_2d(pts_2d, depths, faces, image_h, image_w)


def get_visible_vertices(
    verts_cam: np.ndarray,
    faces: np.ndarray,
    ds_params: dict,
    image_h: int,
    image_w: int,
    depth_tol: float = 0.01,
) -> tuple[np.ndarray, np.ndarray]:
    depth_map, _, _ = rasterize_mesh_fisheye(
        verts_cam,
        faces,
        ds_params,
        image_h,
        image_w,
    )
    pts_2d = ds_project_np(verts_cam, **ds_params)
    vert_depth = np.linalg.norm(verts_cam, axis=1)
    px = np.rint(pts_2d).astype(np.int32)

    in_bounds = (
        (px[:, 0] >= 0)
        & (px[:, 0] < image_w)
        & (px[:, 1] >= 0)
        & (px[:, 1] < image_h)
    )

    visible_mask = np.zeros(verts_cam.shape[0], dtype=bool)
    if not np.any(in_bounds):
        return pts_2d, visible_mask

    valid_rows = px[in_bounds, 1]
    valid_cols = px[in_bounds, 0]
    sampled_depth = depth_map[valid_rows, valid_cols]
    covered = np.isfinite(sampled_depth)
    depth_match = np.zeros_like(covered, dtype=bool)
    depth_match[covered] = (vert_depth[in_bounds][covered] - sampled_depth[covered]) < depth_tol

    visible_mask[in_bounds] = covered & depth_match
    return pts_2d, visible_mask


def load_smplx_faces_and_verts(
    pkl_path: str,
    gender: str = "neutral",
) -> tuple[np.ndarray, np.ndarray]:
    smplx_param = load_pickle(Path(pkl_path))
    model = get_smplx_batch(gender=gender)
    faces = model.faces.numpy() if hasattr(model.faces, "numpy") else np.asarray(model.faces)
    faces = np.asarray(faces, dtype=int)

    n_frame = smplx_param.num_batch
    with torch.no_grad():
        verts_t, _, _ = forward_smplx_tensor_batch(
            smplx_model=model,
            global_orient=smplx_param.orient,
            body_pose=smplx_param.body_pose,
            left_hand_pose=smplx_param.left_hand_pose,
            right_hand_pose=smplx_param.right_hand_pose,
            betas=smplx_param.beta.unsqueeze(0).expand(n_frame, -1),
            transl=smplx_param.transl,
            scale=smplx_param.scale.expand(n_frame),
        )

    verts = verts_t.detach().cpu().numpy()
    return verts, faces
