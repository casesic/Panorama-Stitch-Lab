"""Generate the independent Python SMPL-X oracle for a runtime C++ fixture."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from lib_pano.human_fitting.smplx_sequence import (
    SMPLX_Seq_Forward_Module,
    forward_smplx_tensor_batch,
)


def _read(path: Path, count: int) -> torch.Tensor:
    value = np.fromfile(path, dtype="<f4")
    if value.size != count:
        raise RuntimeError(f"{path}: expected {count}, got {value.size}")
    return torch.from_numpy(value.copy())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gender", default="neutral")
    args = parser.parse_args()
    fixture = args.fixture

    initial = _read(fixture / "stage1_initial_parameters.f32", 12)
    body_pose = _read(fixture / "model_body_pose_anchor.f32", 63).reshape(1, 63)
    beta = _read(fixture / "model_beta_anchor.f32", 10).reshape(1, 10)
    support_ids = torch.from_numpy(np.fromfile(
        fixture / "densekp_support_vertex_ids.i32", dtype="<i4",
    ).astype(np.int64))
    if support_ids.shape != (204,):
        raise RuntimeError("DenseKP support IDs must have shape [204]")

    module = SMPLX_Seq_Forward_Module(gender=args.gender)
    model = module._get_model().eval()
    zero = torch.zeros((1, 3), dtype=torch.float32)
    result: dict[str, np.ndarray] = {}
    with torch.no_grad():
        for camera in ("cam2", "cam4"):
            left = _read(fixture / f"model_{camera}_left_hand_pose.f32", 45).reshape(1, 45)
            right = _read(fixture / f"model_{camera}_right_hand_pose.f32", 45).reshape(1, 45)
            scale = _read(fixture / f"model_{camera}_scale.f32", 1)
            vertices, _, joints_body = forward_smplx_tensor_batch(
                model, zero, body_pose, left, right, beta, zero, scale,
            )
            selected = vertices[0].index_select(0, support_ids)
            center = joints_body[0, 0]
            result[f"{camera}_selected_vertices"] = selected.cpu().numpy()
            result[f"{camera}_support_local_vertices"] = (
                selected - center
            ).cpu().numpy()
            result[f"{camera}_rotation_center"] = center.cpu().numpy()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output, **result)
    print("PYTHON_STAGE1_GEOMETRY_ORACLE_READY=PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
