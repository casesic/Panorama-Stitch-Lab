#!/usr/bin/env python3
"""Generate deterministic region-stratified SMPL-X sampling assets."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from lib_pano.human3d.mesh_surface_sampling import save_vertex_ids
from lib_pano.human3d.smplx_region_sampling import (
    REGION_WEIGHTS,
    sample_smplx_vertices_by_region,
)
from lib_pano.pipeline.paths import ensure_single_fitting_path

ensure_single_fitting_path(PROJECT_ROOT)

COUNTS = (50, 100, 138, 200)
SEED = 0


def load_template() -> tuple[np.ndarray, np.ndarray]:
    from lib_pano.human_fitting.smplx_sequence import get_smplx_batch

    model = get_smplx_batch(gender="neutral")
    vertices = model.v_template.detach().cpu().numpy().squeeze().astype(np.float64)
    weights = model.lbs_weights.detach().cpu().numpy().astype(np.float64)
    return vertices, weights


def render_coverage(
    vertices: np.ndarray,
    samples: dict[str, np.ndarray],
    title: str,
) -> np.ndarray:
    size, title_height, margin = 480, 48, 20
    views = (("front", 0, 1), ("side", 2, 1), ("top", 0, 2))
    canvas = np.full((size + title_height, size * 3, 3), 255, np.uint8)
    cv2.putText(canvas, title, (14, 32), cv2.FONT_HERSHEY_SIMPLEX,
                0.75, (0, 0, 0), 2, cv2.LINE_AA)
    palette = [
        (230, 25, 75), (60, 180, 75), (255, 225, 25), (0, 130, 200),
        (245, 130, 48), (145, 30, 180), (70, 240, 240), (240, 50, 230),
        (210, 245, 60), (250, 190, 212), (0, 128, 128), (220, 190, 255),
        (170, 110, 40), (128, 128, 128),
    ]
    for view_index, (name, horizontal, vertical) in enumerate(views):
        panel = canvas[title_height:, view_index * size:(view_index + 1) * size]
        projected = vertices[:, [horizontal, vertical]]
        minimum = projected.min(axis=0)
        span = np.maximum(projected.max(axis=0) - minimum, 1e-9)

        def pixels(ids: np.ndarray) -> np.ndarray:
            normalized = (projected[ids] - minimum) / span
            return np.stack([
                margin + normalized[:, 0] * (size - 2 * margin),
                margin + (1.0 - normalized[:, 1]) * (size - 2 * margin),
            ], axis=1)

        for x, y in pixels(np.arange(vertices.shape[0])):
            cv2.circle(panel, (round(x), round(y)), 1, (210, 210, 210), -1)
        for color, (_, ids) in zip(palette, samples.items()):
            for x, y in pixels(ids):
                cv2.circle(panel, (round(x), round(y)), 4, color, -1,
                           cv2.LINE_AA)
        cv2.putText(panel, name, (margin, margin + 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 2)
    return canvas


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    asset_dir = SRC_ROOT / "lib_pano/human3d/assets"
    vis_dir = PROJECT_ROOT / "data_process/Vis/sam3d_surface_point_sampling_bakeoff/coverage"
    asset_dir.mkdir(parents=True, exist_ok=True)
    vis_dir.mkdir(parents=True, exist_ok=True)
    vertices, weights = load_template()
    metadata = {
        "version": 1, "model": "SMPL-X neutral 10475",
        "method": "dominant-LBS-region extrema plus deterministic region-local FPS",
        "seed": SEED, "base_region_weights": REGION_WEIGHTS, "assets": {},
    }
    for count in COUNTS:
        ids, samples, quotas = sample_smplx_vertices_by_region(
            vertices, weights, count, SEED)
        repeat, _, _ = sample_smplx_vertices_by_region(vertices, weights, count, SEED)
        if not np.array_equal(ids, repeat):
            raise RuntimeError(f"N={count} is not deterministic")
        path = asset_dir / f"smplx_surface_ids_region_fps_{count}.npy"
        if path.exists() and not args.overwrite:
            raise FileExistsError(f"{path} exists; pass --overwrite")
        save_vertex_ids(path, ids, total_vertices=10475)
        image = render_coverage(
            vertices, samples, f"SMPL-X region sampling N={count}, seed={SEED}")
        image_path = vis_dir / f"smplx_surface_ids_region_fps_{count}.jpg"
        cv2.imwrite(str(image_path), image, [cv2.IMWRITE_JPEG_QUALITY, 95])
        metadata["assets"][f"S{count}"] = {
            "path": str(path.relative_to(PROJECT_ROOT)),
            "count": count, "quotas": quotas,
            "region_counts": {name: int(value.size) for name, value in samples.items()},
            "coverage": str(image_path.relative_to(PROJECT_ROOT)),
        }
    metadata_path = asset_dir / "smplx_surface_ids_region_fps_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    print(metadata_path)


if __name__ == "__main__":
    main()
