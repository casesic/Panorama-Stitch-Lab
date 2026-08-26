"""Visualize observed and optimized DenseKP pairs on normal-ERP images."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
DELIVERY_ROOT = ROOT / "Circular_Video/01_recv/20260708"
for path in (ROOT / "src", DELIVERY_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from densekp_delivery.camera.erp_rotate import rotate_erp_points  # noqa: E402
from densekp_delivery.config import DenseKPConfig  # noqa: E402
from densekp_delivery.pipeline import load_erp_pair  # noqa: E402


def _load_pairs(path: Path) -> tuple[np.ndarray, np.ndarray]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("coord_space") != "erp_rotate":
        raise ValueError(f"expected erp_rotate points: {path}")
    pairs = np.asarray(payload["pairs"], dtype=np.float32)
    valid = np.asarray(payload["valid"], dtype=bool)
    if pairs.shape != (138, 4) or valid.shape != (138,):
        raise ValueError(f"invalid DenseKP shape in {path}")
    return pairs, valid


def _to_normal_erp(points: np.ndarray, width: int, height: int) -> np.ndarray:
    # rotate_erp_points applies ray_out = R @ ray_in; R.T is the inverse.
    from densekp_delivery.camera.erp_rotate import DEFAULT_ERP_TO_ERP_ROTATE_R

    return rotate_erp_points(
        points, width, height, R=DEFAULT_ERP_TO_ERP_ROTATE_R.T,
    )


def _draw(
    image: np.ndarray,
    observed: np.ndarray,
    optimized: np.ndarray,
    valid: np.ndarray,
    camera: str,
) -> np.ndarray:
    canvas = image.copy()
    height, width = canvas.shape[:2]
    for source, target, keep in zip(observed, optimized, valid):
        if not keep or not np.isfinite(source).all() or not np.isfinite(target).all():
            continue
        sx, sy = np.rint(source).astype(int)
        tx, ty = np.rint(target).astype(int)
        if not (0 <= sx < width and 0 <= sy < height):
            continue
        if not (0 <= tx < width and 0 <= ty < height):
            continue
        cv2.line(canvas, (sx, sy), (tx, ty), (0, 255, 255), 2, cv2.LINE_AA)
        cv2.circle(canvas, (sx, sy), 7, (255, 80, 0), -1, cv2.LINE_AA)
        cv2.circle(canvas, (tx, ty), 7, (0, 255, 0), -1, cv2.LINE_AA)
    cv2.putText(
        canvas,
        f"{camera}: observed=blue, optimized=green, valid={int(valid.sum())}/138",
        (30, 65), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (255, 255, 255), 3,
        cv2.LINE_AA,
    )
    return canvas


def _resize(image: np.ndarray, width: int = 1440) -> np.ndarray:
    scale = width / image.shape[1]
    return cv2.resize(
        image, (width, int(round(image.shape[0] * scale))),
        interpolation=cv2.INTER_AREA,
    )


def visualize_optimized_densekp(
    run_directory: Path,
    fisheye_root: Path,
    calibration_cache_dir: Path,
    erp_remap_backend: str = "cuda",
    erp_remap_cuda_library: Path | None = None,
) -> Path:
    report = json.loads(
        (run_directory / "densekp_cuda_staged_lm_report.json").read_text(
            encoding="utf-8"
        )
    )
    mapping = report["cpp_output_frame_mapping"]
    output = run_directory / "visualization"
    output.mkdir(parents=True, exist_ok=True)
    scratch = ROOT / "data_process/tmp/densekp_visualization"
    scratch.mkdir(parents=True, exist_ok=True)

    for cpp_frame, source_frame in mapping.items():
        frame = int(source_frame)
        observed, observed_valid = _load_pairs(
            run_directory / "observations" / f"frame_{frame:06d}.json"
        )
        optimized, optimized_valid = _load_pairs(
            run_directory / "cpp_lm" / cpp_frame / "optimized_densekp.json"
        )
        valid = observed_valid & optimized_valid
        config = DenseKPConfig(
            backend="ort",
            onnx_path="visualization_only",
            fisheye_cam2=str(fisheye_root / "cam2" / f"{frame:05d}.jpg"),
            fisheye_cam4=str(fisheye_root / "cam4" / f"{frame:05d}.jpg"),
            out_dir=str(scratch / f"frame_{frame:06d}"),
            calibration_cache_dir=str(calibration_cache_dir),
            cache_mode="read_only",
            erp_remap_backend=erp_remap_backend,
            erp_remap_cuda_library=(
                str(erp_remap_cuda_library) if erp_remap_cuda_library else None
            ),
        )
        image2, image4 = load_erp_pair(config)
        height, width = image2.shape[:2]
        observed2 = _to_normal_erp(observed[:, :2], width, height)
        observed4 = _to_normal_erp(observed[:, 2:], width, height)
        optimized2 = _to_normal_erp(optimized[:, :2], width, height)
        optimized4 = _to_normal_erp(optimized[:, 2:], width, height)
        vis2 = _resize(_draw(image2, observed2, optimized2, valid, "cam2"))
        vis4 = _resize(_draw(image4, observed4, optimized4, valid, "cam4"))
        pair = np.concatenate((vis2, vis4), axis=1)
        for name, image in (
            (f"frame_{frame:06d}_cam2.jpg", vis2),
            (f"frame_{frame:06d}_cam4.jpg", vis4),
            (f"frame_{frame:06d}_pair.jpg", pair),
        ):
            if not cv2.imwrite(str(output / name), image):
                raise RuntimeError(f"failed to write {output / name}")
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-directory", type=Path, required=True)
    parser.add_argument("--fisheye-root", type=Path, required=True)
    parser.add_argument("--calibration-cache-dir", type=Path, required=True)
    parser.add_argument("--erp-remap-backend", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--erp-remap-cuda-library", type=Path)
    args = parser.parse_args()
    output = visualize_optimized_densekp(
        args.run_directory, args.fisheye_root, args.calibration_cache_dir,
        args.erp_remap_backend, args.erp_remap_cuda_library,
    )
    print(f"DENSEKP_STAGED_LM_VISUALIZATION_READY=PASS output={output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
