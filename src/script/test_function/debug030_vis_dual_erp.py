from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

import cv2
import numpy as np
import config_script

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from lib_pano.human_vis import (
    SMPLXForwarder,
    _load_pickle as load_pickle,
    load_rtm_keypoints,
    normalize_keypoints_format,
    project_cam2,
    project_cam4,
    unpack_smplx_result,
    visualize_frame,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize dual-camera ERP fitting results from debug029 outputs.")
    parser.add_argument("--result_dir", type=Path, required=True, help="debug029 output root containing cam2/ and cam4/")
    parser.add_argument("--segment", type=str, required=False, default=None, help="Segment range in start_end format, e.g. 513_519")
    parser.add_argument(
        "--rtm_dir",
        type=Path,
        default=None,
        help="Optional RTM pkl root; if omitted, RTM keypoints are not loaded.",
    )
    parser.add_argument("--image_dir", type=Path, required=True, help="Equirect image root containing cam2/ and cam4/")
    parser.add_argument("--output_dir", type=Path, default=Path("outputs/erp_vis_dual"), help="Output directory")
    parser.add_argument("--mesh", action="store_true", help="Render mesh overlay")
    parser.add_argument("--conf_thresh", type=float, default=0.1, help="Keypoint confidence threshold")
    parser.add_argument("--save_video", action="store_true", help="Save mp4 video")
    parser.add_argument("--video_only", action="store_true", help="Only save mp4 video, do not save per-frame jpg images")
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


def discover_segments(result_dir: Path) -> list[str]:
    cam2_dir = result_dir / "cam2"
    pattern = re.compile(r"^seg_(\d+)_(\d+)_seq\.pkl$")
    discovered: list[tuple[int, str]] = []
    for pkl_path in cam2_dir.glob("seg_*_seq.pkl"):
        match = pattern.match(pkl_path.name)
        if match is None:
            continue
        start_frame = int(match.group(1))
        segment = f"{match.group(1)}_{match.group(2)}"
        discovered.append((start_frame, segment))

    discovered.sort(key=lambda item: item[0])
    return [segment for _, segment in discovered]


def main() -> int:
    args = parse_args()
    segments = [args.segment] if args.segment is not None else discover_segments(args.result_dir)
    if not segments:
        print(f"[warn] No segments found under: {args.result_dir / 'cam2'}")
        return 0

    fitter = SMPLXForwarder(gender="neutral")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    invalid_cam2: list[int] = []
    invalid_cam4: list[int] = []
    missing_rtm_cam2: list[int] = []
    missing_rtm_cam4: list[int] = []
    video_writer = None
    for segment in segments:
        start_frame, end_frame = parse_segment_range(segment)
        frame_indices = list(range(start_frame, end_frame + 1))

        result_pkl_cam2 = args.result_dir / "cam2" / f"seg_{segment}_seq.pkl"
        result_pkl_cam4 = args.result_dir / "cam4" / f"seg_{segment}_seq.pkl"
        result_cam2 = load_pickle(result_pkl_cam2)
        result_cam4 = load_pickle(result_pkl_cam4)
        params_cam2 = unpack_smplx_result(result_cam2)
        params_cam4 = unpack_smplx_result(result_cam4)

        total = min(len(frame_indices), len(params_cam2), len(params_cam4))
        if total < len(frame_indices):
            print(
                f"[warn] Param count mismatch with frame count for segment {segment}; using the first {total} frames."
            )

        for idx in range(total):
            frame_idx = frame_indices[idx]

            image_path_c2 = args.image_dir / "cam2" / f"{frame_idx:05d}.jpg"
            image_path_c4 = args.image_dir / "cam4" / f"{frame_idx:05d}.jpg"
            img_c2 = cv2.imread(str(image_path_c2))
            img_c4 = cv2.imread(str(image_path_c4))

            if img_c2 is None:
                invalid_cam2.append(frame_idx)
            if img_c4 is None:
                invalid_cam4.append(frame_idx)

            if img_c2 is None:
                print(f"[warn] Failed to read image: {image_path_c2}")
                continue
            if img_c4 is None:
                print(f"[warn] Failed to read image: {image_path_c4}")
                continue

            if args.rtm_dir is None:
                kp133_c2, kp_fmt_c2 = None, {}
                kp133_c4, kp_fmt_c4 = None, {}
            else:
                rtm_pkl_path_c2 = args.rtm_dir / "cam2" / f"rtm_s{frame_idx:05d}_c2.pkl"
                rtm_pkl_path_c4 = args.rtm_dir / "cam4" / f"rtm_s{frame_idx:05d}_c4.pkl"
                if not rtm_pkl_path_c2.exists():
                    missing_rtm_cam2.append(frame_idx)
                if not rtm_pkl_path_c4.exists():
                    missing_rtm_cam4.append(frame_idx)
                kp133_c2, kp_fmt_c2 = load_rtm_keypoints(rtm_pkl_path_c2)
                kp133_c4, kp_fmt_c4 = load_rtm_keypoints(rtm_pkl_path_c4)

            vis_c2, _, _ = visualize_frame(
                img=img_c2,
                smplx_param=params_cam2[idx],
                kp133=kp133_c2,
                kp_fmt=kp_fmt_c2,
                forwarder=fitter,
                image_width=args.image_width,
                image_height=args.image_height,
                project_fn=project_cam2,
                draw_mesh=args.mesh,
                thresh=args.conf_thresh,
            )
            vis_c4, _, _ = visualize_frame(
                img=img_c4,
                smplx_param=params_cam4[idx],
                kp133=kp133_c4,
                kp_fmt=kp_fmt_c4,
                forwarder=fitter,
                image_width=args.image_width,
                image_height=args.image_height,
                project_fn=project_cam4,
                draw_mesh=args.mesh,
                thresh=args.conf_thresh,
            )

            vis_dual = np.concatenate([vis_c2, vis_c4], axis=0)
            if args.scale != 1.0:
                vis_dual = cv2.resize(vis_dual, None, fx=args.scale, fy=args.scale, interpolation=cv2.INTER_AREA)

            if not args.video_only:
                out_path = args.output_dir / f"vis_dual_{frame_idx:05d}.jpg"
                cv2.imwrite(str(out_path), vis_dual, [cv2.IMWRITE_JPEG_QUALITY, 92])
                print(f"[{segment} {idx + 1}/{total}] -> {out_path}")

            if args.save_video or args.video_only:
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

    invalid_json = {
        "cam2": {"equirect": sorted(set(invalid_cam2))},
        "cam4": {"equirect": sorted(set(invalid_cam4))},
        "missing_rtm": {
            "cam2": sorted(set(missing_rtm_cam2)),
            "cam4": sorted(set(missing_rtm_cam4)),
        },
    }
    with (args.output_dir / "invalid.json").open("w", encoding="utf-8") as f:
        json.dump(invalid_json, f, ensure_ascii=False, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

r'''
python src/script/test_function/debug030_vis_dual_erp.py \
  --result_dir data_process/Output/20260311_riding/fitting/dual_cam_sam3d_dense \
  --image_dir data_process/Output/20260311_riding/image/equirect_rotate \
  --output_dir data_process/Output/20260311_riding/vis/dual_erp_sam3d_dense \
  --mesh --scale 0.5
'''

r'''
python src/script/test_function/debug030_vis_dual_erp.py \
  --result_dir data_process/Vis/debug109_output/513_513 \
  --rtm_dir data_process/Output/20260311_riding/rtm_cache/equirect_rotate \
  --image_dir data_process/Output/20260311_riding/image/equirect_rotate \
  --output_dir data_process/Output/20260311_riding/vis/staged \
  --mesh  \
  --scale 0.5
'''
