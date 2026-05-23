from __future__ import annotations

import argparse
import pickle
import re
import sys
from pathlib import Path

import numpy as np


USED_JOINTS = {
    "Pelvis", "L_Hip", "R_Hip", "Spine1", "L_Knee", "R_Knee",
    "Spine2", "L_Ankle", "R_Ankle", "Spine3", "L_Foot", "R_Foot",
    "Neck", "L_Collar", "R_Collar", "Head", "L_Shoulder", "R_Shoulder",
    "L_Elbow", "R_Elbow", "L_Wrist", "R_Wrist",
    "Nose", "R_Eye", "L_Eye", "R_Ear", "L_Ear",
    "L_BigToe", "L_SmallToe", "L_Heel", "R_BigToe", "R_SmallToe", "R_Heel",
}


def _ensure_paths() -> Path:
    repo_root = Path(__file__).resolve().parents[3]
    sf_root = repo_root / "tools" / "SingleFitting"
    if str(sf_root) not in sys.path:
        sys.path.insert(0, str(sf_root))
    return repo_root


REPO_ROOT = _ensure_paths()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect RTM / fitting / HMR data for a single ERP frame.")
    parser.add_argument("--rtm_dir", type=Path, help="RTM pkl root containing cam2/ and cam4/")
    parser.add_argument("--fitting_dir", type=Path, help="Fitting pkl root containing cam2/ and cam4/")
    parser.add_argument("--hmr_dir", type=Path, help="HMR pkl root containing cam2/ and cam4/")
    parser.add_argument("--cam", type=str, required=True, choices=["cam2", "cam4"], help="Camera id")
    parser.add_argument("--frame", type=int, required=True, help="Frame index")
    parser.add_argument(
        "--mode",
        type=str,
        nargs="+",
        required=True,
        choices=["rtm", "fit", "hmr"],
        help="One or more inspection modes to run in order.",
    )
    return parser.parse_args()


def load_keypoints(pkl_path: Path) -> tuple[np.ndarray, dict]:
    with pkl_path.open("rb") as file:
        seq = pickle.load(file)

    if not hasattr(seq, "array_keypoints") or not hasattr(seq, "keypoints_format"):
        raise TypeError(f"Unsupported RTM result format: {type(seq)}")

    keypoints = np.asarray(seq.array_keypoints[0], dtype=np.float32)
    keypoints_format = dict(seq.keypoints_format)
    return keypoints, keypoints_format


def _validate_args(args: argparse.Namespace) -> None:
    if "rtm" in args.mode and args.rtm_dir is None:
        print("error: --rtm_dir is required when mode includes 'rtm'", file=sys.stderr)
        raise SystemExit(2)
    if "fit" in args.mode and args.fitting_dir is None:
        print("error: --fitting_dir is required when mode includes 'fit'", file=sys.stderr)
        raise SystemExit(2)
    if "hmr" in args.mode and args.hmr_dir is None:
        print("error: --hmr_dir is required when mode includes 'hmr'", file=sys.stderr)
        raise SystemExit(2)


def _build_smpl_init_from_single_frame_pkl(pkl_path: Path):
    import importlib

    import torch

    smpl_mod = importlib.import_module("lib_dynamic_hvip.fitting.smpl_seq_forward_module")
    SMPLFitter = smpl_mod.SMPL_Seq_Forward_Module

    with pkl_path.open("rb") as file:
        src = pickle.load(file)

    if isinstance(src, SMPLFitter.SMPL_Param_Seq):
        return src
    if not isinstance(src, dict):
        raise TypeError(f"Unsupported HMR pkl object type: {type(src)}")

    num_batch = int(src["num_batch"])
    dim_shape = int(src["beta"].shape[-1])
    gender = str(src.get("gender", "neutral"))

    smpl_init = SMPLFitter.SMPL_Param_Seq(dim_shape=dim_shape, gender=gender, num_batch=num_batch)
    smpl_init.orient = src["orient"].clone()
    smpl_init.transl = src["transl"].clone()
    smpl_init.pose = src["pose"].clone()
    smpl_init.beta = src["beta"].clone()
    smpl_init.scale = src["scale"].clone()
    smpl_init.valid = src.get("valid", torch.ones(num_batch, dtype=torch.bool)).clone()
    return smpl_init


def find_segment_pkl(cam_dir: Path, frame_idx: int) -> tuple[Path, int, int] | None:
    pattern = re.compile(r"^seg_(\d+)_(\d+)_seq\.pkl$")
    for pkl_path in sorted(cam_dir.glob("seg_*_seq.pkl")):
        match = pattern.match(pkl_path.name)
        if match is None:
            continue
        segment_start = int(match.group(1))
        segment_end = int(match.group(2))
        if segment_start <= frame_idx <= segment_end:
            return pkl_path, segment_start, segment_end
    return None


def _to_numpy(value) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    return np.asarray(value, dtype=np.float32)


def _format_values(value) -> str:
    arr = _to_numpy(value).reshape(-1)
    return "[" + ", ".join(f"{x:.6f}" for x in arr.tolist()) + "]"


def _format_shape(value) -> str:
    shape = list(_to_numpy(value).shape)
    return "[" + ",".join(str(x) for x in shape) + "]"


def _format_stats(value) -> str:
    arr = _to_numpy(value).reshape(-1)
    return f"min={arr.min():.6f}  max={arr.max():.6f}  mean={arr.mean():.6f}"


def _format_bool_values(value) -> str:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    arr = np.asarray(value, dtype=bool).reshape(-1)
    return "[" + ", ".join("True" if x else "False" for x in arr.tolist()) + "]"


def inspect_rtm(args: argparse.Namespace) -> None:
    cam_suffix = "c2" if args.cam == "cam2" else "c4"
    pkl_path = Path(args.rtm_dir) / args.cam / f"rtm_s{args.frame:05d}_{cam_suffix}.pkl"
    if not pkl_path.exists():
        raise FileNotFoundError(f"RTM pkl not found: {pkl_path}")

    keypoints, keypoints_format = load_keypoints(pkl_path)

    filtered = [
        (name, int(kp_idx), float(keypoints[kp_idx, 2]))
        for name, kp_idx in keypoints_format.items()
        if name in USED_JOINTS and int(kp_idx) < keypoints.shape[0]
    ]
    filtered.sort(key=lambda item: item[1])

    print(f"=== frame {args.frame:05d} {args.cam} keypoint confidence ===")
    for name, kp_idx, conf in filtered:
        if conf < 0.1:
            tag = "  VERY LOW"
        elif conf < 0.5:
            tag = "  LOW"
        else:
            tag = ""
        print(f"  {name:<21} [{kp_idx:2d}] conf={conf:.3f}{tag}")


def inspect_fit(args: argparse.Namespace) -> None:
    cam_dir = Path(args.fitting_dir) / args.cam
    found = find_segment_pkl(cam_dir, args.frame)
    if found is None:
        print(f"no segment found for frame {args.frame}")
        return

    pkl_path, segment_start, segment_end = found
    frame_offset = args.frame - segment_start

    with pkl_path.open("rb") as file:
        smplx_seq = pickle.load(file)

    transl = _to_numpy(smplx_seq.transl[frame_offset])
    orient = _to_numpy(smplx_seq.orient[frame_offset])
    body_pose = _to_numpy(smplx_seq.body_pose[frame_offset])
    left_hand_pose = _to_numpy(smplx_seq.left_hand_pose[frame_offset])
    right_hand_pose = _to_numpy(smplx_seq.right_hand_pose[frame_offset])
    beta = _to_numpy(smplx_seq.beta)
    scale = _to_numpy(smplx_seq.scale)


    print(f"=== frame {args.frame:05d} {args.cam} · Fitting (SMPLX) ===")
    print(f"  segment: seg_{segment_start}_{segment_end}  offset={frame_offset}")
    print()
    print(f"  transl          {_format_values(transl)}")
    print(f"  orient          {_format_values(orient)}")
    print(f"  pose       shape=[{body_pose.shape[0]}]   {_format_stats(body_pose)}")
    print(f"  beta            {_format_values(beta)}")
    print(f"  scale           {_format_values(scale)}")
    print(f"  left_hand_pose  shape=[{left_hand_pose.shape[0]}]   {_format_stats(left_hand_pose)}")
    print(f"  right_hand_pose shape=[{right_hand_pose.shape[0]}]   {_format_stats(right_hand_pose)}")


def inspect_hmr(args: argparse.Namespace) -> None:
    cam_suffix = "c2" if args.cam == "cam2" else "c4"
    pkl_path = Path(args.hmr_dir) / args.cam / f"hmr_s{args.frame:05d}_{cam_suffix}.pkl"
    if not pkl_path.exists():
        print("hmr pkl not found")
        return

    smpl_seq = _build_smpl_init_from_single_frame_pkl(pkl_path)

    print(f"=== frame {args.frame:05d} {args.cam} · HMR (SMPL) ===")
    print(f"  transl   shape={_format_shape(smpl_seq.transl)}   values={_format_values(smpl_seq.transl)}")
    print(f"  orient   shape={_format_shape(smpl_seq.orient)}   values={_format_values(smpl_seq.orient)}")
    print(f"  pose     shape={_format_shape(smpl_seq.pose)}  {_format_stats(smpl_seq.pose)}")
    print(f"  beta     shape={_format_shape(smpl_seq.beta)}  values={_format_values(smpl_seq.beta)}")
    print(f"  scale    shape={_format_shape(smpl_seq.scale)}   values={_format_values(smpl_seq.scale)}")
    print(f"  valid    {_format_bool_values(smpl_seq.valid)}")


def main() -> int:
    args = parse_args()
    _validate_args(args)

    handlers = {
        "rtm": inspect_rtm,
        "fit": inspect_fit,
        "hmr": inspect_hmr,
    }

    for index, mode in enumerate(args.mode):
        if index > 0:
            print()
        handlers[mode](args)

    return 0

if __name__ == "__main__":
    raise SystemExit(main())

r'''
python src\script\test_function\debug_inspect_kp_conf.py `
  --rtm_dir data_process\Output\20260311_riding\rtm_cache\equirect_rotate_rtmw `
  --fitting_dir data_process\Output\20260311_riding\fitting\dual_cam `
  --hmr_dir data_process\Output\20260311_riding\hmr_cache `
  --cam cam2 `
  --frame 513 `
  --mode rtm fit hmr
'''
