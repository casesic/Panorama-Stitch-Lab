from __future__ import annotations

import argparse
import json
import os
import pickle
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np

from .io_vr_minimal import get_sync_frame_window, load_vr_camera_intrinsic

REPO_ROOT = Path(__file__).resolve().parents[1]


def _draw_points(
    image: np.ndarray,
    points_xyc: np.ndarray,
    color: Tuple[int, int, int],
    radius: int = 3,
    conf_thr: float = 0.3,
) -> None:
    h, w = image.shape[:2]
    for x, y, c in points_xyc:
        if c < conf_thr:
            continue
        px, py = int(round(float(x))), int(round(float(y)))
        if 0 <= px < w and 0 <= py < h:
            cv2.circle(image, (px, py), radius, color, -1, lineType=cv2.LINE_AA)


def _project_points_camera_to_2d(points_3d: np.ndarray, k: np.ndarray) -> np.ndarray:
    """
    points_3d: (J, 3) in camera coordinates
    returns: (J, 3) -> (x, y, conf), conf=1 for z>0 else 0
    """
    proj = points_3d @ k.T
    z = proj[:, 2:3]
    z_safe = np.clip(z, 1e-6, None)
    xy = proj[:, :2] / z_safe
    conf = (z[:, 0] > 1e-6).astype(np.float32)
    out = np.zeros((points_3d.shape[0], 3), dtype=np.float32)
    out[:, :2] = xy.astype(np.float32)
    out[:, 2] = conf
    return out


def _build_joint_index_pairs(
    keypoints_format: Dict[str, int],
    smpl_all_54: Dict[str, int],
) -> List[Tuple[int, int]]:
    pairs: List[Tuple[int, int]] = []
    for name, idx_kp in keypoints_format.items():
        if name in smpl_all_54:
            pairs.append((int(idx_kp), int(smpl_all_54[name])))
    return pairs


def _load_pickle(path: Path):
    with path.open("rb") as f:
        return pickle.load(f)


def _render_smpl_mesh_on_image(
    image_bgr: np.ndarray,
    verts: np.ndarray,
    faces: np.ndarray,
    camera_k: np.ndarray,
    alpha_value: float,
) -> np.ndarray:
    """
    Render SMPL mesh in camera coordinates and alpha-blend onto image.
    """
    # Must set before importing OpenGL stack in headless environment.
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

    import trimesh  # pylint: disable=import-outside-toplevel
    import pyrender  # pylint: disable=import-outside-toplevel
    from lib_dynamic_hvip.render.camera import (  # pylint: disable=import-outside-toplevel
        get_renderer_camera_pose_from_RT,
    )

    h, w = image_bgr.shape[:2]

    mesh_tri = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    mesh_py = pyrender.Mesh.from_trimesh(mesh_tri, smooth=True)

    scene = pyrender.Scene(bg_color=np.zeros(3), ambient_light=np.array([0.25, 0.25, 0.25]))
    scene.add(mesh_py)

    camera = pyrender.IntrinsicsCamera(
        fx=float(camera_k[0, 0]),
        fy=float(camera_k[1, 1]),
        cx=float(camera_k[0, 2]),
        cy=float(camera_k[1, 2]),
        znear=0.05,
        zfar=15000.0,
    )
    cam_pose = get_renderer_camera_pose_from_RT(np.eye(4, dtype=np.float32))
    scene.add(camera, pose=cam_pose)

    light_front = pyrender.DirectionalLight(color=np.ones(3), intensity=4.0)
    light_side = pyrender.DirectionalLight(color=np.ones(3), intensity=3.0)
    scene.add(light_front, pose=cam_pose)
    side_pose = cam_pose.copy()
    side_pose[:3, 3] += np.array([1.0, 1.0, 1.0], dtype=np.float32)
    scene.add(light_side, pose=side_pose)

    renderer = pyrender.OffscreenRenderer(viewport_width=w, viewport_height=h)
    color, depth = renderer.render(scene, flags=pyrender.RenderFlags.NONE)
    renderer.delete()

    color_bgr = cv2.cvtColor(color, cv2.COLOR_RGB2BGR)
    alpha = np.expand_dims((depth > 5e-4).astype(np.float32), axis=2) * float(alpha_value)
    blended = color_bgr * alpha + image_bgr * (1.0 - alpha)
    return blended.astype(np.uint8)


def _resolve_meta_path(meta_path: Path) -> Path:
    """
    Resolve meta path from common working directories:
    1) as given
    2) relative to SingleFitting root
    3) relative to SingleFitting parent (Repo root)
    """
    candidates = []
    if meta_path.is_absolute():
        candidates.append(meta_path)
    else:
        candidates.append(Path.cwd() / meta_path)
        candidates.append(REPO_ROOT / meta_path)
        candidates.append(REPO_ROOT.parent / meta_path)

    for candidate in candidates:
        if candidate.exists():
            return candidate

    searched = "\n".join(str(p) for p in candidates)
    raise FileNotFoundError(
        f"meta file not found: {meta_path}\nSearched:\n{searched}\n"
        "Hint: run fitting first, or pass absolute --meta path.",
    )


def _resolve_ref_path(path_from_meta: Path, meta_path: Path, field_name: str) -> Path:
    """
    Resolve path stored inside meta.json.
    The path may be absolute, or relative to a different working directory
    where fitting was executed.
    """
    candidates = []
    if path_from_meta.is_absolute():
        candidates.append(path_from_meta)
    else:
        candidates.append(meta_path.parent / path_from_meta)
        candidates.append(Path.cwd() / path_from_meta)
        candidates.append(REPO_ROOT / path_from_meta)
        candidates.append(REPO_ROOT.parent / path_from_meta)

    for candidate in candidates:
        if candidate.exists():
            return candidate

    searched = "\n".join(str(p) for p in candidates)
    raise FileNotFoundError(
        f"{field_name} file not found: {path_from_meta}\nSearched:\n{searched}",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize SingleFitting result by overlaying RTM 2D and projected SMPL joints."
    )
    parser.add_argument(
        "--meta",
        type=Path,
        required=True,
        help="Path to singlefitting_*_meta.json generated by run_single_fitting.py",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output mp4 path. Default: same folder as meta.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=-1,
        help="Limit number of frames to render. -1 means full sequence.",
    )
    parser.add_argument(
        "--point-thr",
        type=float,
        default=0.3,
        help="Confidence threshold for point rendering.",
    )
    parser.add_argument(
        "--vr-data-path",
        type=Path,
        default=Path("/mnt/ssd2/zl/Data_Process/vrCollect/sync2/"),
        help="Root folder that contains S{sequence_id}/data/video.mp4",
    )
    parser.add_argument(
        "--render-mesh",
        action="store_true",
        help="Render and overlay SMPL mesh on each frame.",
    )
    parser.add_argument(
        "--mesh-alpha",
        type=float,
        default=0.7,
        help="Mesh overlay alpha in [0, 1].",
    )
    return parser.parse_args()


def run() -> Path:
    args = parse_args()
    meta_path = _resolve_meta_path(args.meta)

    import torch  # pylint: disable=import-outside-toplevel
    from lib_dynamic_hvip.constants_skeleton import (  # pylint: disable=import-outside-toplevel
        SMPL_ALL_54,
    )
    from lib_dynamic_hvip.fitting.smpl_seq_forward_module import (  # pylint: disable=import-outside-toplevel
        SMPL_Seq_Forward_Module as SMPLFitter,
    )

    with meta_path.open("r", encoding="utf-8") as f:
        meta = json.load(f)

    seq_path = _resolve_ref_path(Path(meta["output_seq"]), meta_path, "output_seq")
    rtm_cache_path = _resolve_ref_path(Path(meta["rtm_cache"]), meta_path, "rtm_cache")
    sequence_id = int(meta["sequence_id"])
    camera_id = int(meta["camera_id"])
    index_start = int(meta["index_start"])
    index_end = int(meta["index_end"])

    smpl_seq = _load_pickle(seq_path)
    det_results = _load_pickle(rtm_cache_path)

    if not hasattr(det_results, "get_result_by_frame_index"):
        raise TypeError(f"Unexpected detection cache type: {type(det_results)}")

    k, _ = load_vr_camera_intrinsic(args.vr_data_path)
    frame_sync_start, _ = get_sync_frame_window(args.vr_data_path, sequence_id, fps=30)
    video_path = args.vr_data_path / f"S{sequence_id}" / "data" / "video.mp4"
    if not video_path.exists():
        raise FileNotFoundError(video_path)

    fitter = SMPLFitter()
    with torch.no_grad():
        verts_t, joints_54_t = fitter.forward_smpl_param_batch(smpl_seq)
    verts_batch = verts_t.detach().cpu().numpy()
    joints_54 = joints_54_t.detach().cpu().numpy()
    faces_smpl = np.asarray(fitter.smpl_models["neutral"].faces, dtype=np.int32)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    first_raw_idx = frame_sync_start + index_start
    cap.set(cv2.CAP_PROP_POS_FRAMES, first_raw_idx)

    ret, frame0 = cap.read()
    if not ret:
        raise RuntimeError(f"Cannot read frame at raw index {first_raw_idx} from {video_path}")

    h, w = frame0.shape[:2]
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0

    if args.output is None:
        output_path = meta_path.parent / (
            f"singlefitting_s{sequence_id}_c{camera_id}_vis.mp4"
        )
    else:
        output_path = args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)

    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(fps),
        (w, h),
    )

    pair_idx = _build_joint_index_pairs(det_results.keypoints_format, SMPL_ALL_54)
    if len(pair_idx) == 0:
        raise RuntimeError("No shared joints between detection keypoints and SMPL_ALL_54.")

    # Put back the first frame we already consumed.
    cap.set(cv2.CAP_PROP_POS_FRAMES, first_raw_idx)

    n_seq = joints_54.shape[0]
    if args.max_frames > 0:
        n_seq = min(n_seq, args.max_frames)

    for seq_idx in range(n_seq):
        ok, frame = cap.read()
        if not ok:
            break

        frame_idx = index_start + seq_idx
        if frame_idx >= index_end:
            break

        det = det_results.get_result_by_frame_index(frame_idx)
        kp_det = np.asarray(det.keypoints, dtype=np.float32)
        kp_smpl_all = _project_points_camera_to_2d(joints_54[seq_idx], k)

        kp_smpl_mapped = np.zeros((len(pair_idx), 3), dtype=np.float32)
        kp_det_mapped = np.zeros((len(pair_idx), 3), dtype=np.float32)
        for i, (idx_det, idx_smpl) in enumerate(pair_idx):
            kp_smpl_mapped[i] = kp_smpl_all[idx_smpl]
            kp_det_mapped[i] = kp_det[idx_det]

        vis = frame.copy()
        if args.render_mesh:
            vis = _render_smpl_mesh_on_image(
                image_bgr=vis,
                verts=verts_batch[seq_idx],
                faces=faces_smpl,
                camera_k=k,
                alpha_value=args.mesh_alpha,
            )

        _draw_points(vis, kp_det_mapped, color=(0, 255, 0), radius=3, conf_thr=args.point_thr)
        _draw_points(vis, kp_smpl_mapped, color=(0, 0, 255), radius=3, conf_thr=0.01)

        cv2.putText(
            vis,
            f"S{sequence_id} C{camera_id} frame={frame_idx}",
            (20, 36),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            vis,
            "Green: RTM 2D   Red: SMPL proj   Mesh: ON" if args.render_mesh else "Green: RTM 2D   Red: SMPL proj",
            (20, 72),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        writer.write(vis)

    writer.release()
    cap.release()
    print(f"Saved visualization video: {output_path}")
    return output_path


def main() -> int:
    run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
