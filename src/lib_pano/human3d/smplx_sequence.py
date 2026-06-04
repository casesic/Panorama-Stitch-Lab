from __future__ import annotations

import pickle
import re
import sys
from pathlib import Path

import numpy as np


_SEGMENT_PATTERN = re.compile(r"^seg_(\d+)_(\d+)_seq\.pkl$")


def _ensure_numpy_pickle_compat() -> None:
    core_mod = getattr(np, "core", None)
    if core_mod is None:
        return
    sys.modules.setdefault("numpy._core", core_mod)
    for submodule in ("multiarray", "numeric", "umath", "_multiarray_umath"):
        src = f"numpy.core.{submodule}"
        dst = f"numpy._core.{submodule}"
        module = sys.modules.get(src)
        if module is None:
            try:
                module = __import__(src, fromlist=["*"])
            except Exception:
                continue
        sys.modules.setdefault(dst, module)


def _load_pickle(path: Path):
    _ensure_numpy_pickle_compat()
    with path.open("rb") as f:
        return pickle.load(f)


def discover_segment_pkls(
    fitting_dir: Path,
    *,
    camera_name: str = "cam2",
    pattern: str = "seg_*_*_seq.pkl",
) -> list[Path]:
    return sorted((fitting_dir / camera_name).glob(pattern))


def find_segment_for_frame(fitting_dir: Path, frame_idx: int) -> tuple[Path | None, Path | None, int | None]:
    for pkl_path_c2 in discover_segment_pkls(fitting_dir):
        parts = pkl_path_c2.stem.split("_")
        if len(parts) != 4:
            continue
        _, start_s, end_s, _ = parts
        start = int(start_s)
        end = int(end_s)
        if start <= frame_idx <= end:
            pkl_path_c4 = fitting_dir / "cam4" / pkl_path_c2.name
            return pkl_path_c2, pkl_path_c4, start
    return None, None, None


def build_segment_index(
    fitting_dir: Path,
    *,
    warn_missing_pair: bool = False,
) -> dict[int, tuple[Path, Path, int]]:
    cam4_dir = fitting_dir / "cam4"
    segment_index: dict[int, tuple[Path, Path, int]] = {}

    for pkl_path_c2 in discover_segment_pkls(fitting_dir):
        match = _SEGMENT_PATTERN.match(pkl_path_c2.name)
        if match is None:
            continue
        start = int(match.group(1))
        end = int(match.group(2))
        pkl_path_c4 = cam4_dir / pkl_path_c2.name
        if not pkl_path_c4.exists():
            if warn_missing_pair:
                print(f"Warning: missing cam4 segment for {pkl_path_c2.name}, skipping segment")
            continue
        for frame_idx in range(start, end + 1):
            segment_index[frame_idx] = (pkl_path_c2, pkl_path_c4, start)

    return segment_index


def get_frame_offset_in_segment(frame_idx: int, segment_start: int) -> int:
    return int(frame_idx) - int(segment_start)


def load_segment_vertices(
    pkl_path: Path,
    model,
    n_expand_beta: int | None = None,
) -> np.ndarray:
    import torch
    from lib_dynamic_hvip.smpl.smpl_lib import forward_smplx_tensor_batch

    smplx_param = _load_pickle(pkl_path)
    n_frame = n_expand_beta if n_expand_beta is not None else smplx_param.num_batch
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
    return verts_t.detach().cpu().numpy()


def get_frame_vertices(
    pkl_path: Path,
    frame_idx: int,
    segment_start: int,
    model,
    segment_cache: dict[Path, np.ndarray] | None = None,
    n_expand_beta: int | None = None,
) -> np.ndarray:
    cache = segment_cache if segment_cache is not None else {}
    if pkl_path not in cache:
        cache[pkl_path] = load_segment_vertices(pkl_path, model, n_expand_beta=n_expand_beta)

    frame_offset = get_frame_offset_in_segment(frame_idx, segment_start)
    vertices = cache[pkl_path]
    if frame_offset < 0 or frame_offset >= len(vertices):
        raise IndexError(f"Frame {frame_idx} out of range for segment {pkl_path.name}")
    return vertices[frame_offset]
