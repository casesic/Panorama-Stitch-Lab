"""Compare raw DenseKP CPW with C++ staged-LM fitted DenseKP CPW."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import shutil
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from script.test_function.debug077_fitted_cpw_composite import run_cpw_direction


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--runtime-output", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--camera-group", choices=("old", "new"), required=True)
    parser.add_argument("--warp-cache-dir", type=Path, required=True)
    return parser.parse_args()


def _load_pairs(path: Path) -> tuple[np.ndarray, np.ndarray]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("coord_space") != "erp_rotate":
        raise RuntimeError(f"unexpected coordinate space: {path}")
    pairs = np.asarray(payload["pairs"], dtype=np.float32)
    valid = np.asarray(payload["valid"], dtype=bool)
    if pairs.shape != (138, 4) or valid.shape != (138,):
        raise RuntimeError(f"invalid DenseKP observation: {path}")
    return pairs, valid


def _write_corr(path: Path, frame: int, pairs: np.ndarray, valid: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        frame_idx=np.int32(frame),
        point_ids=np.arange(138, dtype=np.int32),
        points_cam2=pairs[:, :2],
        points_cam4=pairs[:, 2:],
        valid=valid,
        valid_cam2=valid,
        valid_cam4=valid,
    )


def _cpw_statuses(path: Path) -> dict[int, str]:
    with path.open(newline="", encoding="utf-8") as stream:
        return {int(row["frame"]): row["status"] for row in csv.DictReader(stream)}


def main() -> int:
    args = parse_args()
    report = json.loads(
        (args.runtime_output / "densekp_cuda_staged_lm_report.json").read_text(
            encoding="utf-8"
        )
    )
    mapping = {
        int(frame): int(name.removeprefix("frame_"))
        for name, frame in report["cpp_output_frame_mapping"].items()
    }
    raw_data = args.output / "raw" / "data"
    fitted_data = args.output / "fitted" / "data"
    rows: list[dict[str, float | int | str]] = []
    for frame in report["valid_frames"]:
        frame = int(frame)
        raw, raw_valid = _load_pairs(
            args.runtime_output / "observations" / f"frame_{frame:06d}.json"
        )
        fitted, fitted_valid = _load_pairs(
            args.runtime_output / "cpp_lm" / f"frame_{mapping[frame]}"
            / "optimized_densekp.json"
        )
        valid = raw_valid & fitted_valid
        error = np.linalg.norm(
            fitted.reshape(138, 2, 2) - raw.reshape(138, 2, 2), axis=2,
        )[valid].reshape(-1)
        rows.append({
            "frame": frame,
            "valid_points": int(valid.sum()),
            "fit_to_observation_rms_px": float(np.sqrt(np.mean(error ** 2))),
            "fit_to_observation_p95_px": float(np.percentile(error, 95)),
        })
        _write_corr(
            raw_data / "corr_npz" / f"{frame:05d}.npz", frame, raw, valid,
        )
        _write_corr(
            fitted_data / "corr_npz" / f"{frame:05d}.npz", frame, fitted, valid,
        )

    frames = [int(row["frame"]) for row in rows]
    image_dir = ROOT / "data_process/Output" / args.dataset / "image/equirect_rotate"
    directions = ("cam4_to_cam2", "cam2_to_cam4")
    for label, data in (("raw", raw_data), ("fitted", fitted_data)):
        for direction in directions:
            src, target = direction.split("_to_")
            run_cpw_direction(
                frames, data, args.output / label / "cpw", image_dir, src, target,
                direction,
                warp_map_cache_dir=args.warp_cache_dir / label,
                warp_map_cache_mode="read_write",
                camera_group=args.camera_group,
            )
            direction_output = args.output / label / "cpw" / direction
            for unnecessary in ("overlay", "warp", "composite_target_priority"):
                shutil.rmtree(direction_output / unnecessary, ignore_errors=True)

    statuses = {
        label: {
            direction: _cpw_statuses(
                args.output / label / "cpw" / direction / "cpw_summary.csv"
            )
            for direction in directions
        }
        for label in ("raw", "fitted")
    }
    for label in ("raw", "fitted"):
        for direction in directions:
            source = args.output / label / "cpw" / direction / "cpw_summary.csv"
            destination = args.output / label / "cpw" / f"{direction}_summary.csv"
            source.replace(destination)
    for row in rows:
        frame = int(row["frame"])
        for direction in directions:
            row[f"raw_{direction}_status"] = statuses["raw"][direction][frame]
            row[f"fitted_{direction}_status"] = statuses["fitted"][direction][frame]

    summary = {
        "dataset": args.dataset,
        "directions": list(directions),
        "coordinate_space": "erp_rotate",
        "frame_count": len(rows),
        "cpw_ok_counts": {
            label: {
                direction: sum(statuses[label][direction][frame] == "ok" for frame in frames)
                for direction in directions
            }
            for label in ("raw", "fitted")
        },
        "frames": rows,
    }
    (args.output / "comparison_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
