from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

# Current semantic flow output format is .npy + flow_meta.json.
# NPZ helpers are compatibility helpers only, not the default output.

def save_semantic_flow_npz(path: Path, data: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **data)


def load_semantic_flow_npz(path: Path) -> dict[str, Any]:
    path = Path(path)
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def save_flow_meta_json(meta_path: Path, meta: list[dict[str, Any]]) -> None:
    meta_path = Path(meta_path)
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    with meta_path.open("w", encoding="utf-8") as file:
        json.dump(meta, file, ensure_ascii=False, indent=2)


def save_semantic_flow_frame_npy(
    frame_output_dir: Path,
    human_bbox: tuple[int, int, int, int],
    flow_results: dict[str, tuple[np.ndarray, np.ndarray] | None],
) -> None:
    frame_output_dir = Path(frame_output_dir)
    frame_output_dir.mkdir(parents=True, exist_ok=True)

    c2_to_c4 = flow_results.get("c2_to_c4")
    if c2_to_c4 is not None:
        flow_c2c4, conf_c2c4 = c2_to_c4
        np.save(frame_output_dir / "semantic_flow_c2_to_c4.npy", flow_c2c4)
        np.save(frame_output_dir / "semantic_flow_confidence_c2_to_c4.npy", conf_c2c4)

    c4_to_c2 = flow_results.get("c4_to_c2")
    if c4_to_c2 is not None:
        flow_c4c2, conf_c4c2 = c4_to_c2
        np.save(frame_output_dir / "semantic_flow_c4_to_c2.npy", flow_c4c2)
        np.save(frame_output_dir / "semantic_flow_confidence_c4_to_c2.npy", conf_c4c2)

    np.save(frame_output_dir / "human_bbox.npy", np.asarray(human_bbox, dtype=np.int32))


def save_flow_visualization_paths(frame_output_dir: Path) -> dict[str, Path]:
    frame_output_dir = Path(frame_output_dir)
    return {
        "mesh_cam2": frame_output_dir / "mesh_on_erp_cam2.jpg",
        "mesh_cam4": frame_output_dir / "mesh_on_erp_cam4.jpg",
        "flow_color_c2_to_c4": frame_output_dir / "flow_color_c2_to_c4.png",
        "flow_arrows_c2_to_c4": frame_output_dir / "flow_arrows_c2_to_c4.png",
        "flow_color_c4_to_c2": frame_output_dir / "flow_color_c4_to_c2.png",
        "flow_arrows_c4_to_c2": frame_output_dir / "flow_arrows_c4_to_c2.png",
    }


def list_semantic_flow_files(flow_dir: Path) -> list[Path]:
    flow_dir = Path(flow_dir)
    return sorted(flow_dir.glob("frame_*/semantic_flow*.npy"))
