import sys
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from flow.visibility import (
    get_visible_vertices,
    load_smplx_faces_and_verts,
    rasterize_mesh_fisheye,
)
from single_fitting.debug_shared import DS_PARAMS


pkl_path = "tools\\SingleFitting\\outputs\\fisheye_smplx_cam2\\fisheye_cam2_s0_i0_0_seq.pkl"
image_h, image_w = 2880, 2880

print(f"Loading SMPL-X sequence from: {pkl_path}")
verts, faces = load_smplx_faces_and_verts(pkl_path)
print(f"verts.shape: {verts.shape}")
print(f"faces.shape: {faces.shape}")

verts_cam = verts[0]
print(f"Using frame 0, verts_cam.shape: {verts_cam.shape}")

depth_map, face_id_map, bary_map = rasterize_mesh_fisheye(
    verts_cam,
    faces,
    DS_PARAMS,
    image_h,
    image_w,
)
valid_depth = np.isfinite(depth_map)
valid_depth_count = int(valid_depth.sum())
valid_depth_values = depth_map[valid_depth]
face_id_valid = face_id_map[face_id_map >= 0]

print(f"depth_map.shape: {depth_map.shape}")
print(f"valid depth pixels: {valid_depth_count}")
print(f"face_id_map max: {int(face_id_valid.max()) if face_id_valid.size > 0 else -1}")
if valid_depth_values.size > 0:
    print(
        "depth_map valid range: "
        f"min={float(valid_depth_values.min()):.6f}, max={float(valid_depth_values.max()):.6f}"
    )
else:
    print("depth_map valid range: no valid pixels")
print(f"bary_map.shape: {bary_map.shape}")

pts_2d, visible_mask = get_visible_vertices(
    verts_cam,
    faces,
    DS_PARAMS,
    image_h,
    image_w,
)
visible_pts = pts_2d[visible_mask]

print(f"pts_2d.shape: {pts_2d.shape}")
print(f"visible vertices: {int(visible_mask.sum())}")
print(f"visible ratio: {float(visible_mask.mean()):.6f}")
if visible_pts.shape[0] > 0:
    print(
        "visible 2D range: "
        f"u_min={float(visible_pts[:, 0].min()):.3f}, "
        f"u_max={float(visible_pts[:, 0].max()):.3f}, "
        f"v_min={float(visible_pts[:, 1].min()):.3f}, "
        f"v_max={float(visible_pts[:, 1].max()):.3f}"
    )
else:
    print("visible 2D range: no visible vertices")

canvas = np.zeros((image_h, image_w, 3), dtype=np.uint8)
pts_px = np.rint(pts_2d).astype(np.int32)
in_bounds = (
    (pts_px[:, 0] >= 0)
    & (pts_px[:, 0] < image_w)
    & (pts_px[:, 1] >= 0)
    & (pts_px[:, 1] < image_h)
)

invisible_mask = (~visible_mask) & in_bounds
for u, v in pts_px[invisible_mask]:
    cv2.circle(canvas, (int(u), int(v)), 1, (0, 0, 255), -1)

visible_in_bounds = visible_mask & in_bounds
for u, v in pts_px[visible_in_bounds]:
    cv2.circle(canvas, (int(u), int(v)), 2, (0, 255, 0), -1)

out_path = "tools\\SingleFitting\\outputs\\fisheye_smplx_cam2\\vis\\frame_0_visibility.png"
cv2.imwrite(out_path, canvas)
print(f"Saved visualization to: {out_path}")
