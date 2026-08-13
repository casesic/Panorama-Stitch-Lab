"""Build an input-only frame-513 fixture for the LBS derivative kernels."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT / "data_process/Vis/cpp_reference_dump/lbs_direct_contraction/frame_000513/fixture"
POSE = ROOT / "data_process/Vis/cpp_reference_dump/pose/frame_000513"
SHAPE = ROOT / "data_process/Vis/cpp_reference_dump/pose_shape/frame_000513"


def read(path: Path) -> np.ndarray:
    return np.fromfile(path, "<f4")


def write_context(name: str, camera: str, q: int, combined: bool) -> int:
    fixture = SHAPE / "cpp_fixture"
    count = read(fixture / f"{camera}_selected_lbs_weights.f32").size // 55
    directory = OUT / name
    directory.mkdir(parents=True, exist_ok=True)
    weights = read(fixture / f"{camera}_selected_lbs_weights.f32")
    value = SHAPE / "cpp_output"
    transforms = read(value / f"{camera}_selected_transforms.f32")
    posed = read(value / f"{camera}_v_posed.f32")
    jac = SHAPE / "cpp_jacobian_output"
    relative76 = read(jac / f"{camera}_d_relative_transforms_combined.f32").reshape(55, 16, 76)
    posed76 = read(jac / f"{camera}_d_v_posed_combined.f32").reshape(count, 3, 76)
    relative = relative76 if combined else relative76[:, :, :66]
    posed_derivative = posed76 if combined else posed76[:, :, :66]
    weights.astype("<f4").tofile(directory / "weights.f32")
    relative.astype("<f4").tofile(directory / "relative_transform_derivatives.f32")
    transforms.astype("<f4").tofile(directory / "selected_transforms.f32")
    posed.astype("<f4").tofile(directory / "posed_vertices.f32")
    posed_derivative.astype("<f4").tofile(directory / "posed_vertex_derivatives.f32")
    return count


def main() -> None:
    contexts = {}
    definitions = (
        ("pose_cam2_q66", "cam2", 66, False),
        ("pose_cam4_q66", "cam4", 66, False),
        ("pose_shape_base_cam2_q66", "cam2", 66, False),
        ("pose_shape_base_cam4_q66", "cam4", 66, False),
        ("pose_shape_combined_cam2_q76", "cam2", 76, True),
        ("pose_shape_combined_cam4_q76", "cam4", 76, True),
    )
    for name, camera, q, combined in definitions:
        contexts[name] = {"point_count": write_context(name, camera, q, combined),
                          "parameter_count": q, "camera": camera}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "manifest.json").write_text(json.dumps({
        "schema_version": "lbs_direct_input_fixture_v1", "frame": 513,
        "joint_count": 55, "dtype": "float32", "contexts": contexts,
        "contains_expected_outputs": False, "REFERENCE_LEAKAGE": "NONE",
    }, indent=2) + "\n")
    print("LBS_DIRECT_FIXTURE_WRITTEN")


if __name__ == "__main__":
    main()
