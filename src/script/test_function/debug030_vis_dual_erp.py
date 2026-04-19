from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np
import torch

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")


def _ensure_paths() -> Path:
    repo_root = Path(__file__).resolve().parents[3]
    sf_root = repo_root / "tools" / "SingleFitting"
    if str(sf_root) not in sys.path:
        sys.path.insert(0, str(sf_root))
    return repo_root


_ensure_paths()

from single_fitting.debug_shared import LHAND_IDX, RHAND_IDX, load_pickle, normalize_keypoints_format
from single_fitting.debug_vis_shared import draw_pts, draw_skel
from single_fitting.debug_vis_erp_shared import render_mesh
from single_fitting.fitting import ERPFitter, unpack_smplx_result


R_CAM2 = torch.tensor([1, 0, 0, 0, 0, 1, 0, -1, 0], dtype=torch.float32).reshape(3, 3)
R_CAM4 = torch.tensor([-1, 0, 0, 0, 1, 0, 0, 0, -1], dtype=torch.float32).reshape(3, 3)
R_CAM4_PROJ = R_CAM2@R_CAM4  # M1 @ R_cam4，用于cam4经纬图投影 single_erp 不用乘R_cam4  dual_erp需要乘

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


def _project_with_rotation(points_3d: torch.Tensor, R: torch.Tensor, W: int, H: int) -> torch.Tensor:
    pts = points_3d @ R.to(points_3d.device, dtype=points_3d.dtype).T
    x, y, z = pts[..., 0], pts[..., 1], pts[..., 2]
    r = torch.norm(pts, dim=-1).clamp_min(1e-6)
    lon = torch.atan2(x, z)
    lat = torch.asin((-y / r).clamp(-1 + 1e-6, 1 - 1e-6))
    u = (lon / (2.0 * math.pi) + 0.5) * W
    v = (0.5 - lat / math.pi) * H
    return torch.stack([u, v], dim=-1)


def project_cam2(points_3d: torch.Tensor, W: int, H: int) -> torch.Tensor:
    return _project_with_rotation(points_3d, R_CAM2, W, H)


def project_cam4(points_3d: torch.Tensor, W: int, H: int) -> torch.Tensor:
    return _project_with_rotation(points_3d, R_CAM4_PROJ, W, H)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize dual-camera ERP fitting results from debug029 outputs.")
    parser.add_argument("--result_dir", type=Path, required=True, help="debug029 output root containing cam2/ and cam4/")
    parser.add_argument("--segment", type=str, required=True, help="Segment range in start_end format, e.g. 513_519")
    parser.add_argument("--rtm_dir", type=Path, required=True, help="RTM pkl root containing cam2/ and cam4/")
    parser.add_argument("--image_dir", type=Path, required=True, help="Equirect image root containing cam2/ and cam4/")
    parser.add_argument("--output_dir", type=Path, default=Path("outputs/erp_vis_dual"), help="Output directory")
    parser.add_argument("--mesh", action="store_true", help="Render mesh overlay")
    parser.add_argument("--mesh_alpha", type=float, default=0.6, help="Mesh alpha")
    parser.add_argument("--conf_thresh", type=float, default=0.1, help="Keypoint confidence threshold")
    parser.add_argument("--save_video", action="store_true", help="Save mp4 video")
    parser.add_argument("--fps", type=float, default=30.0, help="Video fps")
    parser.add_argument("--image_width", type=int, default=5760, help="ERP image width")
    parser.add_argument("--image_height", type=int, default=2880, help="ERP image height")
    parser.add_argument("--scale", type=float, default=0.5, help="Output image scale ratio")
    return parser.parse_args()


def parse_segment_range(segment: str) -> tuple[int, int]:
    parts = segment.split("_")
    if len(parts) != 2:
        raise ValueError(f"segment must match start_end: {segment}")
    start_frame, end_frame = int(parts[0]), int(parts[1])
    if start_frame > end_frame:
        raise ValueError(f"segment start must be <= end: {segment}")
    return start_frame, end_frame


def load_rtm_keypoints(rtm_pkl_path: Path) -> tuple[Optional[np.ndarray], dict]:
    if not rtm_pkl_path.exists():
        print(f"[warn] RTM pkl not found: {rtm_pkl_path}")
        return None, {}

    det_raw = load_pickle(rtm_pkl_path)
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
    fitter: ERPFitter,
    image_width: int,
    image_height: int,
    project_fn: Callable[[torch.Tensor, int, int], torch.Tensor],
    draw_mesh: bool = False,
    thresh: float = 0.1,
    mesh_alpha: float = 0.6,
) -> tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    vis = img.copy()
    if kp133 is not None:
        body17 = kp133[:17]
        draw_skel(vis, body17[:, :2], COCO_PAIRS, conf=body17[:, 2], thresh=thresh, color=(0, 200, 0))
        draw_pts(vis, body17[:, :2], conf=body17[:, 2], thresh=thresh, r=6, color=(0, 230, 0))
        if kp133.shape[0] >= 133:
            draw_pts(vis, kp133[LHAND_IDX, :2], conf=kp133[LHAND_IDX, 2], thresh=thresh, r=4, color=(0, 220, 255))
            draw_pts(vis, kp133[RHAND_IDX, :2], conf=kp133[RHAND_IDX, 2], thresh=thresh, r=4, color=(255, 220, 0))

    verts, joints_all, joints_body = fitter.forward_param_to_numpy(smplx_param)
    verts = np.asarray(verts, dtype=np.float32)
    joints_all = np.asarray(joints_all, dtype=np.float32)
    joints_body = np.asarray(joints_body, dtype=np.float32)

    faces = None
    if draw_mesh:
        faces = fitter.get_model_faces()
        if project_fn is project_cam2:
            verts_proj = verts @ R_CAM2.detach().cpu().numpy().T
        else:
            verts_proj = verts @ R_CAM4_PROJ.detach().cpu().numpy().T
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

    if {"Nose", "L_Eye", "R_Eye", "L_Ear", "R_Ear"} & kp_names:
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


def main() -> int:
    args = parse_args()
    start_frame, end_frame = parse_segment_range(args.segment)
    frame_indices = list(range(start_frame, end_frame + 1))

    result_pkl_cam2 = args.result_dir / "cam2" / f"seg_{args.segment}_seq.pkl"
    result_pkl_cam4 = args.result_dir / "cam4" / f"seg_{args.segment}_seq.pkl"
    result_cam2 = load_pickle(result_pkl_cam2)
    result_cam4 = load_pickle(result_pkl_cam4)
    params_cam2 = unpack_smplx_result(result_cam2)
    params_cam4 = unpack_smplx_result(result_cam4)

    total = min(len(frame_indices), len(params_cam2), len(params_cam4))
    if total < len(frame_indices):
        print(
            f"[warn] Param count mismatch with frame count; using the first {total} frames."
        )

    fitter = ERPFitter(gender="neutral")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    video_writer = None
    for idx in range(total):
        frame_idx = frame_indices[idx]

        image_path_c2 = args.image_dir / "cam2" / f"{frame_idx:05d}.jpg"
        image_path_c4 = args.image_dir / "cam4" / f"{frame_idx:05d}.jpg"
        img_c2 = cv2.imread(str(image_path_c2))
        img_c4 = cv2.imread(str(image_path_c4))
        if img_c2 is None:
            print(f"[warn] Failed to read image: {image_path_c2}")
            continue
        if img_c4 is None:
            print(f"[warn] Failed to read image: {image_path_c4}")
            continue

        rtm_pkl_path_c2 = args.rtm_dir / "cam2" / f"rtm_s{frame_idx:05d}_c2.pkl"
        rtm_pkl_path_c4 = args.rtm_dir / "cam4" / f"rtm_s{frame_idx:05d}_c4.pkl"
        kp133_c2, kp_fmt_c2 = load_rtm_keypoints(rtm_pkl_path_c2)
        kp133_c4, kp_fmt_c4 = load_rtm_keypoints(rtm_pkl_path_c4)

        vis_c2, _, _ = visualize_frame(
            img=img_c2,
            smplx_param=params_cam2[idx],
            kp133=kp133_c2,
            kp_fmt=kp_fmt_c2,
            fitter=fitter,
            image_width=args.image_width,
            image_height=args.image_height,
            project_fn=project_cam2,
            draw_mesh=args.mesh,
            thresh=args.conf_thresh,
            mesh_alpha=args.mesh_alpha,
        )
        vis_c4, _, _ = visualize_frame(
            img=img_c4,
            smplx_param=params_cam4[idx],
            kp133=kp133_c4,
            kp_fmt=kp_fmt_c4,
            fitter=fitter,
            image_width=args.image_width,
            image_height=args.image_height,
            project_fn=project_cam4,
            draw_mesh=args.mesh,
            thresh=args.conf_thresh,
            mesh_alpha=args.mesh_alpha,
        )

        vis_dual = np.concatenate([vis_c2, vis_c4], axis=0)
        if args.scale != 1.0:
            vis_dual = cv2.resize(vis_dual, None, fx=args.scale, fy=args.scale, interpolation=cv2.INTER_AREA)

        out_path = args.output_dir / f"vis_dual_{frame_idx:05d}.jpg"
        cv2.imwrite(str(out_path), vis_dual, [cv2.IMWRITE_JPEG_QUALITY, 92])
        print(f"[{idx + 1}/{total}] -> {out_path}")

        if args.save_video:
            if video_writer is None:
                h, w = vis_dual.shape[:2]
                video_writer = cv2.VideoWriter(
                    str(args.output_dir / "vis_dual.mp4"),
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    args.fps,
                    (w, h),
                )
            video_writer.write(vis_dual)

    if video_writer is not None:
        video_writer.release()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

r'''
python src\script\test_function\debug030_vis_dual_erp.py `
  --result_dir data_process\Output\20260311_riding\fitting\dual_cam `
  --segment 513_518 `
  --rtm_dir data_process\Output\20260311_riding\rtm_cache\equirect `
  --image_dir data_process\Output\20260311_riding\image\equirect `
  --output_dir data_process\Output\20260311_riding\vis\dual_erp `
  --mesh  --scale 0.5
'''
