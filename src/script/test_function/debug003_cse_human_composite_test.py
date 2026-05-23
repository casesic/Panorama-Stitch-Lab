#!/usr/bin/env python3
"""
CSE-only human composite test for cam2 <-> cam4.

This script uses DensePose CSE outputs from debug002.
It does not use SAM3 masks.

match_mode:
  - vertex_average: old sanity-check mode. Same vertex_id pixels in source are averaged.
  - embedding_nn: target pixels match source pixels by CSE embedding nearest neighbor,
                  restricted to the same SMPL vertex_id.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def list_image_stems(image_dir: Path) -> List[str]:
    return sorted(p.stem for p in image_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def find_image(image_dir: Path, stem: str) -> Path:
    for ext in [".jpg", ".jpeg", ".png", ".bmp"]:
        p = image_dir / f"{stem}{ext}"
        if p.exists():
            return p
    raise FileNotFoundError(f"No image found for {image_dir}/{stem}.*")


def read_npz(npz_dir: Path, stem: str) -> Dict[str, np.ndarray]:
    path = npz_dir / f"{stem}.npz"
    data = np.load(path)

    item = {
        "vertex_id_map": data["vertex_id_map"].astype(np.int32),
        "cse_valid_mask": data["cse_valid_mask"].astype(bool),
    }

    sparse_keys = ["ys", "xs", "vertex_ids", "embeddings"]
    if all(k in data.files for k in sparse_keys):
        item["ys"] = data["ys"].astype(np.int32)
        item["xs"] = data["xs"].astype(np.int32)
        item["vertex_ids"] = data["vertex_ids"].astype(np.int32)
        item["embeddings"] = data["embeddings"].astype(np.float32)

    return item


def get_sparse_arrays(cse: Dict[str, np.ndarray], image_shape: Tuple[int, int]):
    required = ["ys", "xs", "vertex_ids", "embeddings"]
    missing = [k for k in required if k not in cse]
    if missing:
        raise KeyError(
            "Missing sparse CSE fields in npz: "
            + ", ".join(missing)
            + ". Please rerun debug002 with sparse embedding export."
        )

    H, W = image_shape
    ys = cse["ys"]
    xs = cse["xs"]
    vertex_ids = cse["vertex_ids"]
    embeddings = cse["embeddings"]

    valid = (
        (vertex_ids >= 0)
        & (ys >= 0)
        & (ys < H)
        & (xs >= 0)
        & (xs < W)
    )
    return ys[valid], xs[valid], vertex_ids[valid].astype(np.int64), embeddings[valid]


def vertex_mean_colors(src_img: np.ndarray, src_vertex: np.ndarray, src_valid: np.ndarray):
    valid = src_valid & (src_vertex >= 0)
    vids = src_vertex[valid].astype(np.int64)
    colors = src_img[valid].astype(np.float64)

    if len(vids) == 0:
        return np.empty((0,), dtype=np.int64), np.empty((0, 3), dtype=np.uint8), 0

    unique_vids, inverse = np.unique(vids, return_inverse=True)
    color_sums = np.zeros((len(unique_vids), 3), dtype=np.float64)
    counts = np.bincount(inverse).astype(np.float64)
    np.add.at(color_sums, inverse, colors)
    mean_colors = np.clip(color_sums / counts[:, None], 0, 255).astype(np.uint8)
    return unique_vids, mean_colors, int(len(vids))


def warp_by_vertex_average(
    src_img: np.ndarray,
    src_cse: Dict[str, np.ndarray],
    tgt_img: np.ndarray,
    tgt_cse: Dict[str, np.ndarray],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, float]]:
    src_vertex = src_cse["vertex_id_map"]
    src_mask = src_cse["cse_valid_mask"]
    tgt_vertex = tgt_cse["vertex_id_map"]
    tgt_mask = tgt_cse["cse_valid_mask"]

    src_valid = src_mask & (src_vertex >= 0)
    tgt_valid = tgt_mask & (tgt_vertex >= 0)

    unique_vids, mean_colors, source_valid_pixels = vertex_mean_colors(src_img, src_vertex, src_valid)

    warp = np.zeros_like(tgt_img)
    overlay = tgt_img.copy()
    matched_mask = np.zeros(tgt_vertex.shape, dtype=bool)

    target_valid_pixels = int(tgt_valid.sum())
    if len(unique_vids) > 0 and target_valid_pixels > 0:
        target_vids = tgt_vertex[tgt_valid].astype(np.int64)
        idx = np.searchsorted(unique_vids, target_vids)
        in_range = idx < len(unique_vids)
        found = np.zeros_like(target_vids, dtype=bool)
        found[in_range] = unique_vids[idx[in_range]] == target_vids[in_range]

        target_coords = np.where(tgt_valid)
        yy = target_coords[0][found]
        xx = target_coords[1][found]
        colors = mean_colors[idx[found]]

        warp[yy, xx] = colors
        overlay[yy, xx] = colors
        matched_mask[yy, xx] = True

    source_unique_vertices = int(len(unique_vids))
    target_unique_vertices = int(len(np.unique(tgt_vertex[tgt_valid]))) if target_valid_pixels > 0 else 0
    common_vertices = (
        int(len(np.intersect1d(unique_vids, np.unique(tgt_vertex[tgt_valid]))))
        if source_unique_vertices > 0 and target_unique_vertices > 0
        else 0
    )

    matched_target_pixels = int(matched_mask.sum())
    stats = {
        "match_mode": "vertex_average",
        "source_valid_pixels": int(source_valid_pixels),
        "target_valid_pixels": int(target_valid_pixels),
        "matched_target_pixels": int(matched_target_pixels),
        "matched_ratio": float(matched_target_pixels / target_valid_pixels) if target_valid_pixels else 0.0,
        "source_unique_vertices": int(source_unique_vertices),
        "target_unique_vertices": int(target_unique_vertices),
        "common_vertices": int(common_vertices),
    }
    return warp, overlay, matched_mask, stats


def build_vertex_groups(vertex_ids: np.ndarray) -> Dict[int, np.ndarray]:
    groups: Dict[int, List[int]] = {}
    for i, v in enumerate(vertex_ids.tolist()):
        groups.setdefault(int(v), []).append(i)
    return {v: np.asarray(idx, dtype=np.int64) for v, idx in groups.items()}


def nearest_source_indices(
    src_emb: np.ndarray,
    tgt_emb: np.ndarray,
    max_pairs_per_chunk: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    For each target embedding, find nearest source embedding in the same group.
    Returns:
      nearest_local_indices: index into src_emb
      nearest_distances: L2 distance
    """
    ns = len(src_emb)
    nt = len(tgt_emb)
    if ns == 0 or nt == 0:
        return np.empty((0,), dtype=np.int64), np.empty((0,), dtype=np.float32)

    src_emb = np.ascontiguousarray(src_emb.astype(np.float32, copy=False))
    tgt_emb = np.ascontiguousarray(tgt_emb.astype(np.float32, copy=False))

    if ns == 1:
        dist = np.linalg.norm(tgt_emb - src_emb[0:1], axis=1).astype(np.float32)
        return np.zeros((nt,), dtype=np.int64), dist

    chunk = max(1, int(max_pairs_per_chunk // ns))
    nearest = np.empty((nt,), dtype=np.int64)
    distances = np.empty((nt,), dtype=np.float32)

    src_sq = np.sum(src_emb * src_emb, axis=1)[None, :]

    for start in range(0, nt, chunk):
        end = min(nt, start + chunk)
        q = tgt_emb[start:end]
        q_sq = np.sum(q * q, axis=1)[:, None]
        dist2 = q_sq + src_sq - 2.0 * (q @ src_emb.T)
        dist2 = np.maximum(dist2, 0.0)
        idx = np.argmin(dist2, axis=1)
        nearest[start:end] = idx
        distances[start:end] = np.sqrt(dist2[np.arange(end - start), idx]).astype(np.float32)

    return nearest, distances


def warp_by_embedding_nn_same_vertex(
    src_img: np.ndarray,
    src_cse: Dict[str, np.ndarray],
    tgt_img: np.ndarray,
    tgt_cse: Dict[str, np.ndarray],
    max_pairs_per_chunk: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, float]]:
    sy, sx, svid, semb = get_sparse_arrays(src_cse, src_img.shape[:2])
    ty, tx, tvid, temb = get_sparse_arrays(tgt_cse, tgt_img.shape[:2])

    warp = np.zeros_like(tgt_img)
    overlay = tgt_img.copy()
    matched_mask = np.zeros(tgt_img.shape[:2], dtype=bool)

    src_groups = build_vertex_groups(svid)
    tgt_groups = build_vertex_groups(tvid)
    common_vertices = sorted(set(src_groups.keys()) & set(tgt_groups.keys()))

    all_distances = []
    matched_target_pixels = 0

    for vid in common_vertices:
        sidx = src_groups[vid]
        tidx = tgt_groups[vid]

        nearest_local, distances = nearest_source_indices(
            src_emb=semb[sidx],
            tgt_emb=temb[tidx],
            max_pairs_per_chunk=max_pairs_per_chunk,
        )
        matched_sidx = sidx[nearest_local]

        yy = ty[tidx]
        xx = tx[tidx]
        colors = src_img[sy[matched_sidx], sx[matched_sidx]]

        warp[yy, xx] = colors
        overlay[yy, xx] = colors
        matched_mask[yy, xx] = True

        matched_target_pixels += len(tidx)
        if len(distances) > 0:
            all_distances.append(distances)

    source_valid_pixels = int(len(svid))
    target_valid_pixels = int(len(tvid))
    source_unique_vertices = int(len(src_groups))
    target_unique_vertices = int(len(tgt_groups))

    if all_distances:
        d = np.concatenate(all_distances)
        mean_dist = float(np.mean(d))
        median_dist = float(np.median(d))
    else:
        mean_dist = -1.0
        median_dist = -1.0

    stats = {
        "match_mode": "embedding_nn_same_vertex",
        "source_valid_pixels": int(source_valid_pixels),
        "target_valid_pixels": int(target_valid_pixels),
        "matched_target_pixels": int(matched_target_pixels),
        "matched_ratio": float(matched_target_pixels / target_valid_pixels) if target_valid_pixels else 0.0,
        "source_unique_vertices": int(source_unique_vertices),
        "target_unique_vertices": int(target_unique_vertices),
        "common_vertices": int(len(common_vertices)),
        "mean_embedding_l2": mean_dist,
        "median_embedding_l2": median_dist,
    }
    return warp, overlay, matched_mask, stats


def mask_overlay(img: np.ndarray, mask: np.ndarray, color: Tuple[int, int, int]) -> np.ndarray:
    out = img.copy()
    if mask.any():
        color_arr = np.array(color, dtype=np.uint8)
        out[mask] = (0.55 * out[mask] + 0.45 * color_arr).astype(np.uint8)
    return out


def to_three_channel(mask: np.ndarray) -> np.ndarray:
    m = mask.astype(np.uint8) * 255
    return cv2.cvtColor(m, cv2.COLOR_GRAY2BGR)


def add_label(img: np.ndarray, text: str) -> np.ndarray:
    out = img.copy()
    cv2.rectangle(out, (0, 0), (min(out.shape[1], 520), 32), (0, 0, 0), -1)
    cv2.putText(out, text, (8, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return out


def resize_to_height(img: np.ndarray, target_h: int) -> np.ndarray:
    h, w = img.shape[:2]
    if h == target_h:
        return img
    target_w = max(1, int(round(w * target_h / h)))
    return cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_AREA)


def make_debug_panel(
    src_img: np.ndarray,
    tgt_img: np.ndarray,
    src_valid: np.ndarray,
    tgt_valid: np.ndarray,
    warp: np.ndarray,
    overlay: np.ndarray,
    matched_mask: np.ndarray,
    direction: str,
    match_mode: str,
) -> np.ndarray:
    warp_label = "CSE embedding-NN warp" if match_mode == "embedding_nn" else "CSE vertex-average warp"
    panels = [
        add_label(src_img, f"source {direction.split('_to_')[0]}"),
        add_label(tgt_img, f"target {direction.split('_to_')[1]}"),
        add_label(mask_overlay(src_img, src_valid, (0, 255, 0)), "source CSE valid"),
        add_label(mask_overlay(tgt_img, tgt_valid, (0, 255, 255)), "target CSE valid"),
        add_label(warp, warp_label),
        add_label(overlay, "overlay on target"),
        add_label(to_three_channel(matched_mask), "matched target mask"),
    ]

    target_h = min(p.shape[0] for p in panels)
    panels = [resize_to_height(p, target_h) for p in panels]
    return np.concatenate(panels, axis=1)


def ensure_dirs(root: Path, direction: str) -> Dict[str, Path]:
    paths = {}
    for name in ["warp", "valid_mask", "overlay", "debug_panel", "stats"]:
        p = root / direction / name
        p.mkdir(parents=True, exist_ok=True)
        paths[name] = p
    return paths


def save_direction_result(
    output_root: Path,
    direction: str,
    stem: str,
    src_img: np.ndarray,
    src_cse: Dict[str, np.ndarray],
    tgt_img: np.ndarray,
    tgt_cse: Dict[str, np.ndarray],
    match_mode: str,
    max_pairs_per_chunk: int,
) -> Dict[str, float]:
    paths = ensure_dirs(output_root, direction)

    if match_mode == "vertex_average":
        warp, overlay, matched_mask, stats = warp_by_vertex_average(src_img, src_cse, tgt_img, tgt_cse)
    elif match_mode == "embedding_nn":
        warp, overlay, matched_mask, stats = warp_by_embedding_nn_same_vertex(
            src_img, src_cse, tgt_img, tgt_cse, max_pairs_per_chunk
        )
    else:
        raise ValueError(f"Unknown match_mode: {match_mode}")

    panel = make_debug_panel(
        src_img=src_img,
        tgt_img=tgt_img,
        src_valid=src_cse["cse_valid_mask"],
        tgt_valid=tgt_cse["cse_valid_mask"],
        warp=warp,
        overlay=overlay,
        matched_mask=matched_mask,
        direction=direction,
        match_mode=match_mode,
    )

    cv2.imwrite(str(paths["warp"] / f"{stem}.jpg"), warp)
    cv2.imwrite(str(paths["valid_mask"] / f"{stem}.png"), matched_mask.astype(np.uint8) * 255)
    cv2.imwrite(str(paths["overlay"] / f"{stem}.jpg"), overlay)
    cv2.imwrite(str(paths["debug_panel"] / f"{stem}.jpg"), panel)

    stats = {"frame": stem, "direction": direction, **stats}
    with open(paths["stats"] / f"{stem}.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    return stats


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_root", required=True)
    parser.add_argument("--cse_npz_root", required=True)
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--cams", nargs=2, default=["cam2", "cam4"])
    parser.add_argument("--max_frames", type=int, default=None)
    parser.add_argument(
        "--match_mode",
        choices=["vertex_average", "embedding_nn"],
        default="embedding_nn",
        help="vertex_average is the old sanity-check mode; embedding_nn uses sparse CSE embeddings.",
    )
    parser.add_argument(
        "--max_pairs_per_chunk",
        type=int,
        default=2_000_000,
        help="Memory cap for per-vertex embedding distance chunks.",
    )
    args = parser.parse_args()

    image_root = Path(args.image_root)
    cse_npz_root = Path(args.cse_npz_root)
    output_root = Path(args.output_root)
    cam_a, cam_b = args.cams

    stems_a = set(list_image_stems(image_root / cam_a))
    stems_b = set(list_image_stems(image_root / cam_b))
    npz_a = {p.stem for p in (cse_npz_root / cam_a).glob("*.npz")}
    npz_b = {p.stem for p in (cse_npz_root / cam_b).glob("*.npz")}
    stems = sorted(stems_a & stems_b & npz_a & npz_b)
    if args.max_frames is not None:
        stems = stems[: args.max_frames]

    print(f"[INFO] frames: {len(stems)}")
    print(f"[INFO] match_mode: {args.match_mode}")
    all_stats = []

    for i, stem in enumerate(stems, 1):
        img_a = cv2.imread(str(find_image(image_root / cam_a, stem)), cv2.IMREAD_COLOR)
        img_b = cv2.imread(str(find_image(image_root / cam_b, stem)), cv2.IMREAD_COLOR)
        if img_a is None or img_b is None:
            print(f"[WARN] failed to read images for frame {stem}")
            continue

        cse_a = read_npz(cse_npz_root / cam_a, stem)
        cse_b = read_npz(cse_npz_root / cam_b, stem)

        if img_a.shape[:2] != cse_a["vertex_id_map"].shape or img_b.shape[:2] != cse_b["vertex_id_map"].shape:
            print(f"[WARN] shape mismatch for frame {stem}, skip")
            continue

        try:
            stats_ab = save_direction_result(
                output_root=output_root,
                direction=f"{cam_a}_to_{cam_b}",
                stem=stem,
                src_img=img_a,
                src_cse=cse_a,
                tgt_img=img_b,
                tgt_cse=cse_b,
                match_mode=args.match_mode,
                max_pairs_per_chunk=args.max_pairs_per_chunk,
            )
            stats_ba = save_direction_result(
                output_root=output_root,
                direction=f"{cam_b}_to_{cam_a}",
                stem=stem,
                src_img=img_b,
                src_cse=cse_b,
                tgt_img=img_a,
                tgt_cse=cse_a,
                match_mode=args.match_mode,
                max_pairs_per_chunk=args.max_pairs_per_chunk,
            )
        except Exception as exc:
            print(f"[ERROR] {stem}: {exc}")
            all_stats.append({"frame": stem, "status": "error", "error": str(exc)})
            continue

        all_stats.extend([stats_ab, stats_ba])
        print(
            f"[{i:04d}/{len(stems):04d}] {stem} "
            f"{cam_a}->{cam_b} ratio={stats_ab['matched_ratio']:.3f}, "
            f"{cam_b}->{cam_a} ratio={stats_ba['matched_ratio']:.3f}"
        )

    output_root.mkdir(parents=True, exist_ok=True)
    with open(output_root / "summary.json", "w", encoding="utf-8") as f:
        json.dump(all_stats, f, ensure_ascii=False, indent=2)
    print(f"[DONE] saved to {output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


r'''
# New precise mode: embedding NN within same vertex_id.
python src/script/test_function/debug003_cse_human_composite_test.py   --image_root data_process/Output/20260311_riding/image/equirect_crop_rotate_upright   --cse_npz_root data_process/Output/20260311_riding/densepose/cse_npz   --output_root data_process/Output/20260311_riding/densepose/cse_human_composite_test_embedding_nn   --cams cam2 cam4   --match_mode embedding_nn   --max_frames 1

# Old sanity-check mode.
python src/script/test_function/debug003_cse_human_composite_test.py   --image_root data_process/Output/20260311_riding/image/equirect_crop_rotate_upright   --cse_npz_root data_process/Output/20260311_riding/densepose/cse_npz   --output_root data_process/Output/20260311_riding/densepose/cse_human_composite_test_vertex_average   --cams cam2 cam4   --match_mode vertex_average   --max_frames 10
'''
