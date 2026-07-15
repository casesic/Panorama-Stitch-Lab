"""Visualize staged-LM JSON/NPZ outputs without rerunning optimization.

The fitting context is rebuilt from the saved JSON configuration solely to
recover HMR-derived frozen hand poses/scales, pose/beta anchors, and model
metadata. LM and L-BFGS are never invoked by this script.

All visualization modes write directly into the same ``--output_dir``. File
name prefixes distinguish normal, stage-comparison, and solver-comparison
outputs; the script does not create a separate subdirectory for each mode.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from script.human_fitting.staged_lm.app import build_context
from lib_pano.human_fitting.staged_lm.config import STAGE1_PER_FRAME_DIM
from lib_pano.human_fitting.staged_lm.parameter_layout import unpack_frame_parameters
from lib_pano.human_fitting.staged_lm.types import StageContext
from lib_pano.human_fitting.smplx_sequence import SMPLX_Seq_Forward_Module
from lib_pano.human_vis import (
    SMPLXForwarder,
    load_rtm_keypoints,
    project_cam2,
    project_cam4,
    visualize_frame,
)
from lib_pano.human_vis.dual_erp_vis import project_points_erp


STAGES = ("stage1", "pose_warmup", "joint_refinement")
SOLVERS = ("lm", "lbfgs")
PARAMETER_KEYS = {
    ("stage1", "lm"): "x_s1_lm",
    ("stage1", "lbfgs"): "x_s1_lbfgs",
    ("pose_warmup", "lm"): "x_stage2_pose_lm",
    ("pose_warmup", "lbfgs"): "x_stage2_pose_lbfgs",
    ("joint_refinement", "lm"): "x_stage2_joint_lm",
    ("joint_refinement", "lbfgs"): "x_stage2_joint_lbfgs",
}
BETA_KEYS = {
    ("stage1", "lm"): "beta_anchor",
    ("stage1", "lbfgs"): "beta_anchor",
    ("pose_warmup", "lm"): "beta_anchor",
    ("pose_warmup", "lbfgs"): "beta_anchor",
    ("joint_refinement", "lm"): "beta_lm",
    ("joint_refinement", "lbfgs"): "beta_lbfgs",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize staged-LM JSON/NPZ outputs")
    parser.add_argument("--result_dir", type=Path, required=True)
    parser.add_argument("--image_dir", type=Path, required=True)
    parser.add_argument("--stage", choices=STAGES, default="joint_refinement")
    parser.add_argument("--solver", choices=SOLVERS, default="lm")
    parser.add_argument(
        "--output_dir",
        type=Path,
        required=True,
        help="Shared output directory for normal and comparison visualizations",
    )
    parser.add_argument("--mesh", action="store_true")
    parser.add_argument("--scale", type=float, default=0.5)
    parser.add_argument("--save_video", action="store_true")
    parser.add_argument("--video_only", action="store_true")
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--rtm_dir", type=Path, default=None)
    parser.add_argument("--conf_thresh", type=float, default=0.1)
    parser.add_argument("--compare_stages", action="store_true")
    parser.add_argument("--compare_solvers", action="store_true")
    parser.add_argument("--draw_sam3d_dense", action="store_true")
    parser.add_argument("--dense_draw_stride", type=int, default=20)
    args = parser.parse_args()
    if args.compare_stages and args.compare_solvers:
        parser.error("--compare_stages and --compare_solvers are mutually exclusive")
    if args.scale <= 0:
        parser.error("--scale must be positive")
    if args.dense_draw_stride <= 0:
        parser.error("--dense_draw_stride must be positive")
    return args


def parse_frame_indices(frames: str) -> list[int]:
    """Parse the saved comma-separated frame list without assuming continuity."""
    try:
        indices = [int(token.strip()) for token in frames.split(",") if token.strip()]
    except ValueError as exc:
        raise ValueError(f"Invalid config.frames value: {frames!r}") from exc
    if not indices:
        raise ValueError("config.frames contains no frame indices")
    return indices


def load_staged_lm_outputs(result_dir: Path) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    json_path = result_dir / "staged_stage2b_beta.json"
    npz_path = result_dir / "staged_stage2b_beta.npz"
    if not json_path.is_file():
        raise FileNotFoundError(f"staged-LM JSON not found: {json_path}")
    if not npz_path.is_file():
        raise FileNotFoundError(f"staged-LM NPZ not found: {npz_path}")
    with json_path.open("r", encoding="utf-8") as file:
        metadata = json.load(file)
    with np.load(npz_path) as archive:
        arrays = {key: archive[key].copy() for key in archive.files}
    if not isinstance(metadata.get("config"), dict):
        raise ValueError(f"Missing JSON config object: {json_path}")
    return metadata, arrays


def build_context_namespace(config: dict[str, Any]) -> argparse.Namespace:
    """Convert the complete saved staged-LM configuration to a Namespace."""
    required = {
        "frames", "hmr_dir", "sam3d_dense_dir", "bbox_dir",
        "image_width", "image_height", "reproj_objective",
    }
    missing = sorted(required - config.keys())
    if missing:
        raise ValueError(f"staged-LM JSON config is missing required fields: {', '.join(missing)}")
    return argparse.Namespace(**config)


def expected_parameter_dimension(context: StageContext, stage: str) -> tuple[int, int]:
    if stage == "stage1":
        per_frame_dim = STAGE1_PER_FRAME_DIM
        expected = context.frame_count * per_frame_dim
    else:
        per_frame_dim = STAGE1_PER_FRAME_DIM + context.body_pose_dim
        expected = context.frame_count * per_frame_dim
        if stage == "joint_refinement":
            expected += context.beta_dim
    return expected, per_frame_dim


def validate_parameter_dimension(context: StageContext, parameters: np.ndarray, stage: str) -> int:
    expected, per_frame_dim = expected_parameter_dimension(context, stage)
    actual = int(np.asarray(parameters).size)
    if actual != expected:
        raise ValueError(
            "staged-LM parameter dimension mismatch: "
            f"stage={stage}, expected={expected}, actual={actual}, "
            f"frame_count={context.frame_count}, per_frame_dim={per_frame_dim}, "
            f"beta_dim={context.beta_dim}"
        )
    return per_frame_dim


def require_result_arrays(
    arrays: dict[str, np.ndarray], stage: str, solver: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the parameter vector and already-physical beta for a result."""
    parameter_key = PARAMETER_KEYS[(stage, solver)]
    if parameter_key not in arrays:
        if solver == "lbfgs":
            raise KeyError(
                f"Requested L-BFGS visualization, but {parameter_key} is absent. "
                "Run the joint-fk validation with --run_lbfgs_oracle."
            )
        raise KeyError(f"Requested visualization result is absent: {parameter_key}")
    beta_key = BETA_KEYS[(stage, solver)]
    if beta_key not in arrays:
        raise KeyError(f"Required physical beta array is absent: {beta_key}")
    # beta_lm/beta_lbfgs are exported physical beta values. Never tanh them here.
    return arrays[parameter_key], arrays[beta_key]


def _as_flat_tensor(value: Any, name: str, expected_size: int) -> torch.Tensor:
    tensor = torch.as_tensor(value).detach().clone().reshape(-1).to(dtype=torch.float32)
    if tensor.numel() != expected_size:
        raise ValueError(f"{name} shape mismatch: expected {expected_size} values, got {tuple(tensor.shape)}")
    return tensor


def build_visualization_params(
    context: StageContext,
    parameters: np.ndarray,
    physical_beta: np.ndarray,
    stage: str,
) -> tuple[list[Any], list[Any]]:
    """Return complete per-frame SMPL-X params for cam2 and cam4.

    The concrete list elements are ``SMPLX_Param`` objects because that is the
    interface consumed by ``SMPLXForwarder.forward_param_to_numpy()``.
    """
    if stage not in STAGES:
        raise ValueError(f"Unsupported stage: {stage}")
    per_frame_dim = validate_parameter_dimension(context, parameters, stage)
    parameter_tensor = torch.as_tensor(parameters).detach().reshape(-1).to(dtype=torch.float32)
    beta = _as_flat_tensor(physical_beta, "physical beta", context.beta_dim)
    include_body_pose = stage != "stage1"
    cam2_params: list[Any] = []
    cam4_params: list[Any] = []

    if len(context.frozen_params_by_frame) != context.frame_count:
        raise ValueError(
            f"frozen parameter frame count mismatch: expected {context.frame_count}, "
            f"got {len(context.frozen_params_by_frame)}"
        )
    for position in range(context.frame_count):
        body_pose_override = None if include_body_pose else context.body_pose_anchors[position].reshape(1, -1)
        unpacked = unpack_frame_parameters(
            parameter_tensor, position, per_frame_dim, include_body_pose,
            body_pose_override=body_pose_override,
        )
        frozen = context.frozen_params_by_frame[position]
        body_pose = _as_flat_tensor(unpacked.body_pose, "body pose", context.body_pose_dim)
        for camera, output in (("cam2", cam2_params), ("cam4", cam4_params)):
            param = SMPLX_Seq_Forward_Module.SMPLX_Param(gender="neutral")
            param.orient = _as_flat_tensor(getattr(unpacked, f"{camera}_orient"), f"{camera} orient", 3)
            param.transl = _as_flat_tensor(getattr(unpacked, f"{camera}_transl"), f"{camera} transl", 3)
            param.body_pose = body_pose.clone()
            param.left_hand_pose = _as_flat_tensor(
                getattr(frozen, f"{camera}_left_hand_pose"), f"{camera} left hand pose", 45,
            )
            param.right_hand_pose = _as_flat_tensor(
                getattr(frozen, f"{camera}_right_hand_pose"), f"{camera} right hand pose", 45,
            )
            param.beta = beta.clone()
            param.scale = _as_flat_tensor(getattr(frozen, f"{camera}_scale"), f"{camera} scale", 1)
            output.append(param)
    return cam2_params, cam4_params


def _load_optional_rtm(rtm_dir: Path | None, camera: str, frame_index: int) -> tuple[Any, dict]:
    if rtm_dir is None:
        return None, {}
    camera_number = "2" if camera == "cam2" else "4"
    return load_rtm_keypoints(rtm_dir / camera / f"rtm_s{frame_index:05d}_c{camera_number}.pkl")


def _draw_dense_diagnostic(
    image: np.ndarray,
    vertices: np.ndarray,
    vertex_ids: torch.Tensor,
    targets: torch.Tensor,
    project_fn: Any,
    width: int,
    height: int,
    stride: int,
) -> None:
    ids = vertex_ids.detach().cpu().numpy()[::stride].astype(np.int64)
    target_points = targets.detach().cpu().numpy()[::stride]
    projected = project_points_erp(vertices[ids], width, height, project_fn)
    for target, prediction in zip(target_points, projected):
        target_xy = tuple(np.rint(target).astype(int))
        prediction_xy = tuple(np.rint(prediction).astype(int))
        cv2.line(image, target_xy, prediction_xy, (0, 255, 255), 1, cv2.LINE_AA)
        cv2.circle(image, target_xy, 3, (0, 255, 0), -1, cv2.LINE_AA)
        cv2.circle(image, prediction_xy, 3, (0, 0, 255), -1, cv2.LINE_AA)


def _render_camera(
    image: np.ndarray,
    param: Any,
    camera: str,
    frame_position: int,
    context: StageContext,
    forwarder: SMPLXForwarder,
    args: argparse.Namespace,
) -> np.ndarray:
    frame_index = context.frame_indices[frame_position]
    keypoints, keypoint_format = _load_optional_rtm(args.rtm_dir, camera, frame_index)
    project_fn = project_cam2 if camera == "cam2" else project_cam4
    visualized, vertices, _ = visualize_frame(
        img=image, smplx_param=param, kp133=keypoints, kp_fmt=keypoint_format,
        forwarder=forwarder, image_width=context.image_width,
        image_height=context.image_height, project_fn=project_fn,
        draw_mesh=args.mesh, thresh=args.conf_thresh,
    )
    if args.draw_sam3d_dense:
        if camera == "cam2":
            vertex_ids = context.cam2_vertex_ids_by_frame[frame_position]
            targets = context.cam2_targets_by_frame[frame_position]
        else:
            vertex_ids = context.cam4_vertex_ids_by_frame[frame_position]
            targets = context.cam4_targets_by_frame[frame_position]
        _draw_dense_diagnostic(
            visualized, vertices, vertex_ids, targets, project_fn,
            context.image_width, context.image_height, args.dense_draw_stride,
        )
    return visualized


def _add_title(image: np.ndarray, title: str) -> np.ndarray:
    result = image.copy()
    font_scale = max(0.7, result.shape[1] / 2500.0)
    cv2.putText(result, title, (24, 48), cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                (0, 0, 0), 5, cv2.LINE_AA)
    cv2.putText(result, title, (24, 48), cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                (255, 255, 255), 2, cv2.LINE_AA)
    return result


def _read_images(image_dir: Path, frame_index: int) -> tuple[np.ndarray, np.ndarray]:
    paths = [image_dir / camera / f"{frame_index:05d}.jpg" for camera in ("cam2", "cam4")]
    images = [cv2.imread(str(path)) for path in paths]
    missing = [str(path) for path, image in zip(paths, images) if image is None]
    if missing:
        raise FileNotFoundError(f"Failed to read ERP image(s): {', '.join(missing)}")
    return images[0], images[1]


def _resize_output(image: np.ndarray, scale: float) -> np.ndarray:
    if scale == 1.0:
        return image
    return cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)


def _available_stage_params(
    arrays: dict[str, np.ndarray], context: StageContext, solver: str,
) -> list[tuple[str, list[Any], list[Any]]]:
    available = []
    for stage in STAGES:
        key = PARAMETER_KEYS[(stage, solver)]
        beta_key = BETA_KEYS[(stage, solver)]
        if key not in arrays or beta_key not in arrays:
            print(f"[warn] Skipping {stage}: required NPZ key {key if key not in arrays else beta_key} is absent")
            continue
        cam2, cam4 = build_visualization_params(context, arrays[key], arrays[beta_key], stage)
        available.append((stage, cam2, cam4))
    if not available:
        raise RuntimeError(f"No visualizable stage results are available for solver={solver}")
    return available


def _render_normal(
    context: StageContext, arrays: dict[str, np.ndarray], forwarder: SMPLXForwarder,
    args: argparse.Namespace, position: int, images: tuple[np.ndarray, np.ndarray],
) -> np.ndarray:
    parameters, beta = require_result_arrays(arrays, args.stage, args.solver)
    cam2_params, cam4_params = build_visualization_params(context, parameters, beta, args.stage)
    frame_index = context.frame_indices[position]
    cam2 = _render_camera(images[0], cam2_params[position], "cam2", position, context, forwarder, args)
    cam4 = _render_camera(images[1], cam4_params[position], "cam4", position, context, forwarder, args)
    title = f"Frame {frame_index} | {args.stage} | {args.solver.upper()}"
    return _add_title(np.concatenate([cam2, cam4], axis=0), title)


def _render_stage_comparison(
    context: StageContext, arrays: dict[str, np.ndarray], forwarder: SMPLXForwarder,
    args: argparse.Namespace, position: int, images: tuple[np.ndarray, np.ndarray],
) -> np.ndarray:
    columns = []
    for stage, cam2_params, cam4_params in _available_stage_params(arrays, context, args.solver):
        cam2 = _render_camera(images[0], cam2_params[position], "cam2", position, context, forwarder, args)
        cam4 = _render_camera(images[1], cam4_params[position], "cam4", position, context, forwarder, args)
        columns.append((_add_title(cam2, stage), _add_title(cam4, stage)))
    return np.concatenate([
        np.concatenate([column[0] for column in columns], axis=1),
        np.concatenate([column[1] for column in columns], axis=1),
    ], axis=0)


def _render_solver_comparison(
    context: StageContext, arrays: dict[str, np.ndarray], forwarder: SMPLXForwarder,
    args: argparse.Namespace, position: int, images: tuple[np.ndarray, np.ndarray],
) -> np.ndarray:
    solver_columns = []
    for solver in SOLVERS:
        parameters, beta = require_result_arrays(arrays, args.stage, solver)
        cam2_params, cam4_params = build_visualization_params(context, parameters, beta, args.stage)
        cam2 = _render_camera(images[0], cam2_params[position], "cam2", position, context, forwarder, args)
        cam4 = _render_camera(images[1], cam4_params[position], "cam4", position, context, forwarder, args)
        solver_columns.append((_add_title(cam2, solver.upper()), _add_title(cam4, solver.upper())))
    return np.concatenate([
        np.concatenate([column[0] for column in solver_columns], axis=1),
        np.concatenate([column[1] for column in solver_columns], axis=1),
    ], axis=0)


def main() -> int:
    args = parse_args()
    metadata, arrays = load_staged_lm_outputs(args.result_dir)
    config = metadata["config"]
    saved_frames = parse_frame_indices(str(config["frames"]))
    context = build_context(build_context_namespace(config))
    if context.frame_indices != saved_frames:
        raise RuntimeError(f"Rebuilt context frame mismatch: expected {saved_frames}, got {context.frame_indices}")
    print("[context] Recovered frozen hand poses/scales and anchors; no optimizer was run")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    forwarder = SMPLXForwarder(gender="neutral")
    video_writer = None
    for position, frame_index in enumerate(context.frame_indices):
        images = _read_images(args.image_dir, frame_index)
        if args.compare_stages:
            visualized = _render_stage_comparison(context, arrays, forwarder, args, position, images)
            filename = f"compare_stages_{args.solver}_{frame_index:05d}.jpg"
            video_name = f"compare_stages_{args.solver}.mp4"
        elif args.compare_solvers:
            visualized = _render_solver_comparison(context, arrays, forwarder, args, position, images)
            filename = f"compare_solvers_{args.stage}_{frame_index:05d}.jpg"
            video_name = f"compare_solvers_{args.stage}.mp4"
        else:
            visualized = _render_normal(context, arrays, forwarder, args, position, images)
            filename = f"vis_{args.stage}_{args.solver}_{frame_index:05d}.jpg"
            video_name = f"vis_{args.stage}_{args.solver}.mp4"
        visualized = _resize_output(visualized, args.scale)
        if not args.video_only:
            output_path = args.output_dir / filename
            if not cv2.imwrite(str(output_path), visualized, [cv2.IMWRITE_JPEG_QUALITY, 92]):
                raise RuntimeError(f"Failed to write image: {output_path}")
            print(f"[{position + 1}/{context.frame_count}] -> {output_path}")
        if args.save_video or args.video_only:
            if video_writer is None:
                height, width = visualized.shape[:2]
                video_writer = cv2.VideoWriter(
                    str(args.output_dir / video_name), cv2.VideoWriter_fourcc(*"mp4v"),
                    args.fps, (width, height),
                )
                if not video_writer.isOpened():
                    raise RuntimeError(f"Failed to create video: {args.output_dir / video_name}")
            video_writer.write(visualized)
    if video_writer is not None:
        video_writer.release()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

r'''
python src/script/human_fitting/staged_lm/visualize.py \
    --result_dir data_process/Vis/staged_lm/513_518 \
    --image_dir data_process/Output/20260311_riding/image/equirect_rotate \
    --output_dir data_process/Output/20260311_riding/vis/staged_lm_stage2_joint \
    --stage joint_refinement \
    --solver lm \
    --mesh \
    --scale 0.5
'''
