import os
import sys
import json
from pathlib import Path

os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')

import numpy as np
import cv2

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from single_fitting.debug_shared import R_CAM4, DS_PARAMS
from flow.visibility import (
    load_smplx_faces_and_verts,
    get_visible_vertices,
    compute_semantic_flow,
    project_verts_to_rotated_erp,
)

PKL_CAM2 = r"data_process\Output\20260311_riding\fisheye_smplx\cam2\fisheye_cam2_s0_i0_29_seq.pkl"
PKL_CAM4 = r"data_process\Output\20260311_riding\fisheye_smplx\cam4\fisheye_cam4_s0_i22_39_seq.pkl"
PKL_INDEX_START_CAM2 = 490
PKL_INDEX_START_CAM4 = 512
STITCH_DIR = r"data_process/Output/20260311_riding/stitch_batch"
OUTPUT_DIR = r"data_process/Output/20260311_riding/batch_flow"
INDEX_START = 513
NUM_FRAMES = 5
ROTATION_MATRIX = np.array([1, 0, 0, 0, 0, 1, 0, -1, 0], dtype=np.float64).reshape(3, 3)
SAVE_VIS = True
WARP_DIRECTION = "cam2_to_cam4"  # 可选 "cam4_to_cam2" 或 "cam2_to_cam4"


def load_frame_data(frame_idx, stitch_dir):
    frame_dir = Path(stitch_dir) / f"frame_{frame_idx:05d}"
    cam2_path = frame_dir / f"rotated_cam2_frame_{frame_idx:05d}.png"
    cam4_path = frame_dir / f"rotated_cam4_frame_{frame_idx:05d}.png"

    img_cam2 = cv2.imread(str(cam2_path), cv2.IMREAD_COLOR) if cam2_path.exists() else None
    img_cam4 = cv2.imread(str(cam4_path), cv2.IMREAD_COLOR) if cam4_path.exists() else None
    return img_cam2, img_cam4


def compute_human_bbox(pts_list, erp_h, erp_w, padding=30):
    valid_pts = []
    for pts in pts_list:
        if pts is None or len(pts) == 0:
            continue
        pts = np.asarray(pts, dtype=np.float64)
        finite_mask = np.isfinite(pts).all(axis=1)
        if np.any(finite_mask):
            valid_pts.append(pts[finite_mask])

    if not valid_pts:
        return (0, 0, erp_w, erp_h)

    all_pts = np.concatenate(valid_pts, axis=0)
    x1 = max(0, int(np.floor(all_pts[:, 0].min())) - padding)
    y1 = max(0, int(np.floor(all_pts[:, 1].min())) - padding)
    x2 = min(erp_w, int(np.ceil(all_pts[:, 0].max())) + padding)
    y2 = min(erp_h, int(np.ceil(all_pts[:, 1].max())) + padding)
    return (x1, y1, x2, y2)


def visualize_mesh_on_erp(img, pts_2d, color=(0, 255, 0), radius=2):
    canvas = img.copy()
    if pts_2d is None or len(pts_2d) == 0:
        return canvas

    h, w = canvas.shape[:2]
    pts_px = np.rint(np.asarray(pts_2d, dtype=np.float64)).astype(np.int32)
    for u, v in pts_px:
        if 0 <= u < w and 0 <= v < h:
            cv2.circle(canvas, (u, v), radius, color, -1)
    return canvas


def visualize_flow(flow_roi, confidence_roi):
    flow_mag = np.linalg.norm(flow_roi, axis=2)
    flow_angle = np.arctan2(flow_roi[..., 1], flow_roi[..., 0])

    hsv = np.zeros((flow_roi.shape[0], flow_roi.shape[1], 3), dtype=np.uint8)
    hsv[..., 0] = ((flow_angle + np.pi) / (2 * np.pi) * 179).astype(np.uint8)
    hsv[..., 1] = 255
    max_mag = max(float(flow_mag.max()), 1e-6)
    hsv[..., 2] = (np.clip(flow_mag / max_mag, 0.0, 1.0) * 255).astype(np.uint8)
    flow_color = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    flow_color[confidence_roi == 0] = 0

    flow_arrows = np.zeros((flow_roi.shape[0], flow_roi.shape[1], 3), dtype=np.uint8)
    step = 20
    for r in range(0, flow_roi.shape[0], step):
        for c in range(0, flow_roi.shape[1], step):
            conf = int(confidence_roi[r, c])
            if conf == 0:
                continue
            dx, dy = flow_roi[r, c]
            color = (0, 255, 0) if conf == 2 else (255, 0, 0)
            cv2.arrowedLine(
                flow_arrows,
                (c, r),
                (int(round(c + dx)), int(round(r + dy))),
                color,
                1,
                tipLength=0.3,
            )

    return flow_color, flow_arrows


def process_frame(frame_idx, verts_cam2_all, verts_cam4_all, faces, stitch_dir, output_dir, save_vis):
    img_cam2, img_cam4 = load_frame_data(frame_idx, stitch_dir)

    pkl_i_cam2 = frame_idx - PKL_INDEX_START_CAM2
    pkl_i_cam4 = frame_idx - PKL_INDEX_START_CAM4
    has_cam2 = img_cam2 is not None and 0 <= pkl_i_cam2 < len(verts_cam2_all)
    has_cam4 = img_cam4 is not None and 0 <= pkl_i_cam4 < len(verts_cam4_all)

    if has_cam2 and has_cam4:
        flag = "ok"
    elif has_cam2:
        flag = "missing_cam4"
    elif has_cam4:
        flag = "missing_cam2"
    else:
        flag = "missing_both"

    if flag != "ok":
        return {"frame_idx": frame_idx, "valid": False, "reason": flag}

    verts_c2 = verts_cam2_all[pkl_i_cam2]
    verts_c4 = verts_cam4_all[pkl_i_cam4]
    if WARP_DIRECTION == "cam4_to_cam2":
        verts_world_A = verts_c2
        verts_world_B = verts_c4 @ R_CAM4
        _, visible_mask_B = get_visible_vertices(verts_c4, faces, DS_PARAMS, 2880, 2880)
    elif WARP_DIRECTION == "cam2_to_cam4":
        verts_world_A = verts_c4 @ R_CAM4
        verts_world_B = verts_c2
        _, visible_mask_B = get_visible_vertices(verts_c2, faces, DS_PARAMS, 2880, 2880)
    else:
        raise ValueError(f"Unsupported WARP_DIRECTION: {WARP_DIRECTION}")

    erp_h, erp_w = img_cam2.shape[:2]
    pts_2d_A = project_verts_to_rotated_erp(verts_world_A, ROTATION_MATRIX, erp_h, erp_w)
    pts_2d_B = project_verts_to_rotated_erp(verts_world_B, ROTATION_MATRIX, erp_h, erp_w)

    human_bbox = compute_human_bbox([pts_2d_A, pts_2d_B], erp_h, erp_w)
    flow_roi, confidence_roi = compute_semantic_flow(
        verts_world_A=verts_world_A,
        verts_world_B=verts_world_B,
        faces=faces,
        erp_h=erp_h,
        erp_w=erp_w,
        rotation_matrix=ROTATION_MATRIX,
        roi_bbox=human_bbox,
        visible_mask_B=visible_mask_B,
    )

    frame_output_dir = Path(output_dir) / f"frame_{frame_idx:05d}"
    frame_output_dir.mkdir(parents=True, exist_ok=True)
    np.save(frame_output_dir / "semantic_flow.npy", flow_roi)
    np.save(frame_output_dir / "semantic_flow_confidence.npy", confidence_roi)
    np.save(frame_output_dir / "human_bbox.npy", np.asarray(human_bbox, dtype=np.int32))

    if save_vis:
        if WARP_DIRECTION == "cam4_to_cam2":
            mesh_cam2 = visualize_mesh_on_erp(img_cam2, pts_2d_A)
            mesh_cam4 = visualize_mesh_on_erp(img_cam4, pts_2d_B)
        elif WARP_DIRECTION == "cam2_to_cam4":
            mesh_cam4 = visualize_mesh_on_erp(img_cam4, pts_2d_A)
            mesh_cam2 = visualize_mesh_on_erp(img_cam2, pts_2d_B)
        else:
            raise ValueError(f"Unsupported WARP_DIRECTION: {WARP_DIRECTION}")
        flow_color, flow_arrows = visualize_flow(flow_roi, confidence_roi)
        cv2.imwrite(str(frame_output_dir / "mesh_on_erp_cam2.jpg"), mesh_cam2, [cv2.IMWRITE_JPEG_QUALITY, 95])
        cv2.imwrite(str(frame_output_dir / "mesh_on_erp_cam4.jpg"), mesh_cam4, [cv2.IMWRITE_JPEG_QUALITY, 95])
        cv2.imwrite(str(frame_output_dir / "flow_color.png"), flow_color)
        cv2.imwrite(str(frame_output_dir / "flow_arrows.png"), flow_arrows)

    return {
        "frame_idx": frame_idx,
        "valid": True,
        "reason": "ok",
        "human_bbox": list(human_bbox),
        "flow_pixels": int((confidence_roi > 0).sum()),
    }


def main():
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    verts_cam2_all, faces = load_smplx_faces_and_verts(PKL_CAM2)
    verts_cam4_all, faces_cam4 = load_smplx_faces_and_verts(PKL_CAM4)
    if not np.array_equal(faces, faces_cam4):
        raise ValueError("SMPL-X faces mismatch between cam2 and cam4 PKL inputs")

    meta = []
    for frame_idx in range(INDEX_START, INDEX_START + NUM_FRAMES):
        result = process_frame(
            frame_idx=frame_idx,
            verts_cam2_all=verts_cam2_all,
            verts_cam4_all=verts_cam4_all,
            faces=faces,
            stitch_dir=STITCH_DIR,
            output_dir=OUTPUT_DIR,
            save_vis=SAVE_VIS,
        )
        meta.append(result)
        print(f"[frame {frame_idx:05d}] {result['reason']}")

    meta_path = output_dir / "flow_meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    valid_count = sum(1 for item in meta if item["valid"])
    missing_cam2 = sum(1 for item in meta if item["reason"] == "missing_cam2")
    missing_cam4 = sum(1 for item in meta if item["reason"] == "missing_cam4")
    missing_both = sum(1 for item in meta if item["reason"] == "missing_both")

    print(f"valid frames: {valid_count}")
    print(f"missing_cam2: {missing_cam2}")
    print(f"missing_cam4: {missing_cam4}")
    print(f"missing_both: {missing_both}")
    print(f"saved meta to: {meta_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
