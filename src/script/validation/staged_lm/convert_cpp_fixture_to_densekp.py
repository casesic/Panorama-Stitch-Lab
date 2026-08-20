"""Convert a 204-support SAM3D C++ fixture to DenseKP observations.

The SMPL-X model arrays stay support-vertex arrays. Only Stage1 local points
and the value/Jacobian targets are replaced at the observation boundary.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from lib_pano.fitting_io.densekp_io import load_densekp_result
from lib_pano.human3d.crop_upright_coord import (
    resolve_dataset_rotate_rules,
    resolve_rotate_rule,
)
from lib_pano.pipeline.bbox_io import load_bbox_dict


def metadata(path: Path) -> dict[str, str]:
    return dict(line.split("=", 1) for line in path.read_text().splitlines()
                if line and not line.startswith("#"))


def write_metadata(path: Path, values: dict[str, str]) -> None:
    path.write_text("".join(f"{key}={value}\n" for key, value in sorted(values.items())))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--frame", type=int, required=True)
    parser.add_argument("--densekp-dir", type=Path, required=True)
    parser.add_argument("--bbox-dir", type=Path, required=True)
    parser.add_argument("--assets-dir", type=Path,
                        default=ROOT / "cpp/staged_lm/assets")
    parser.add_argument("--image-width", type=int, default=5760)
    parser.add_argument("--image-height", type=int, default=2880)
    args = parser.parse_args()

    ids = np.fromfile(args.assets_dir / "densekp_support_vertex_ids.i32", "<i4")
    regressor = np.fromfile(
        args.assets_dir / "densekp_regressor_138x204.f32", "<f4"
    ).reshape(138, 204)
    if ids.shape != (204,) or regressor.shape != (138, 204):
        raise RuntimeError("invalid fixed DenseKP assets")

    stage_meta_path = args.fixture / "stage1_fixture_meta.txt"
    model_meta_path = args.fixture / "model_fixture_meta.txt"
    stage_meta = metadata(stage_meta_path)
    model_meta = metadata(model_meta_path)
    rotate_rules = resolve_dataset_rotate_rules(
        ROOT / "configs/rotate_upright_rules.json", args.bbox_dir
    )
    rotate_rule = resolve_rotate_rule(args.frame, rotate_rules)

    for camera in ("cam2", "cam4"):
        support_count = int(model_meta[f"{camera}_count"])
        if support_count != 204 or int(stage_meta[f"{camera}_count"]) != 204:
            raise RuntimeError(
                f"{camera}: source fixture must contain all 204 support rows"
            )
        bboxes = load_bbox_dict(args.bbox_dir / f"{camera}_bboxes.json")
        observation = load_densekp_result(
            args.frame, args.densekp_dir, camera,
            args.image_width, args.image_height,
            crop_bbox=bboxes.get(str(args.frame)), rotate_rule=rotate_rule,
        )
        if observation is None:
            raise RuntimeError(f"missing DenseKP observation: {camera} frame={args.frame}")
        valid_ids = np.flatnonzero(observation.valid).astype("<i4")
        targets = np.ascontiguousarray(observation.keypoints_2d[valid_ids], dtype="<f4")
        local_path = args.fixture / f"stage1_{camera}_local_vertices.f32"
        support_local = np.fromfile(local_path, "<f4").reshape(204, 3)
        densekp_local = np.ascontiguousarray(regressor[valid_ids] @ support_local,
                                             dtype="<f4")
        densekp_local.tofile(local_path)
        targets.tofile(args.fixture / f"stage1_{camera}_targets.f32")
        targets.tofile(args.fixture / f"model_{camera}_targets.f32")
        valid_ids.tofile(args.fixture / f"model_{camera}_densekp_ids.i32")
        stage_meta[f"{camera}_count"] = str(valid_ids.size)
        stage_meta[f"{camera}_alpha_per_point"] = repr(1.0 / valid_ids.size)
        model_meta[f"{camera}_support_count"] = "204"
        model_meta[f"{camera}_observation_count"] = str(valid_ids.size)
        model_meta[f"{camera}_alpha"] = repr(1.0 / valid_ids.size)

    model_meta.update({
        "observation_mode": "densekp",
        "densekp_support_count": "204",
        "densekp_row_count": "138",
    })
    write_metadata(stage_meta_path, stage_meta)
    write_metadata(model_meta_path, model_meta)
    for name in ("densekp_support_vertex_ids.i32",
                 "densekp_regressor_138x204.f32"):
        shutil.copyfile(args.assets_dir / name, args.fixture / name)
    print("DENSEKP_CPP_FIXTURE_READY=PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
