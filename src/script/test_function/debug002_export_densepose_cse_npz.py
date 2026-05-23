"""
Export DensePose CSE dump results to per-frame full-image vertex maps and sparse
CSE embeddings.

Input:
  - apply_net.py dump output, e.g. densepose/cse_dump/cam2.pkl
  - corresponding image directory, e.g. image/equirect_crop_rotate_upright/cam2

Output per frame:
  - vertex_id_map: H x W int32, invalid pixels are -1
  - cse_valid_mask: H x W uint8, valid pixels are 1
  - bbox_xyxy: selected instance bbox
  - det_score: selected instance score
  - ys / xs: full-image coordinates of valid CSE pixels
  - vertex_ids: nearest SMPL vertex id for each valid CSE pixel
  - embeddings: resized CSE embedding for each valid CSE pixel, shape N x D

Optional visualization:
  - overlay_dir/00513.jpg: original image + CSE valid mask overlay

This script uses CSE's original embedding-to-vertex conversion logic, not dp_vertex
visualization images.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from detectron2.config import get_cfg


def add_densepose_project_to_path(project_dir: str) -> None:
    """Make `import densepose` work when running from the pano-stitch repo root."""
    path = Path(project_dir)
    if not path.is_absolute():
        path = Path.cwd() / path
    if path.exists():
        sys.path.insert(0, str(path))


def load_torch_dump(path: Path):
    try:
        return torch.load(str(path), map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(str(path), map_location="cpu")


def tensor_to_numpy(x: Any) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def setup_densepose_cfg(cfg_path: Path, model_path: Optional[Path] = None):
    from densepose import add_densepose_config

    cfg = get_cfg()
    add_densepose_config(cfg)
    cfg.merge_from_file(str(cfg_path))
    if model_path:
        cfg.MODEL.WEIGHTS = str(model_path)
    cfg.freeze()
    return cfg


def build_mesh_embeddings(cfg, device: torch.device, default_class: int):
    from densepose.data.utils import get_class_to_mesh_name_mapping
    from densepose.modeling import build_densepose_embedder

    class_to_mesh_name = get_class_to_mesh_name_mapping(cfg)
    if default_class not in class_to_mesh_name:
        raise KeyError(
            f"default_class={default_class} not in class_to_mesh_name={class_to_mesh_name}"
        )

    mesh_name = class_to_mesh_name[default_class]
    embedder = build_densepose_embedder(cfg)
    if hasattr(embedder, "has_embeddings") and not embedder.has_embeddings(mesh_name):
        raise RuntimeError(f"DensePose embedder has no embeddings for mesh: {mesh_name}")

    mesh_vertex_embeddings = embedder(mesh_name).to(device)
    return mesh_name, mesh_vertex_embeddings


def choose_instance(scores: Optional[np.ndarray], min_score: float) -> Optional[int]:
    if scores is None or len(scores) == 0:
        return None
    best_idx = int(np.argmax(scores))
    if float(scores[best_idx]) < min_score:
        return None
    return best_idx


def xyxy_to_int_xywh(box_xyxy: np.ndarray) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = box_xyxy.astype(float).tolist()
    x = int(x1)
    y = int(y1)
    w = int(x2 - x1)
    h = int(y2 - y1)
    return x, y, max(1, w), max(1, h)


def paste_local_vertex_map(
    full_vertex: np.ndarray,
    full_mask: np.ndarray,
    local_vertices: np.ndarray,
    local_mask: np.ndarray,
    x: int,
    y: int,
) -> None:
    H, W = full_vertex.shape
    h, w = local_mask.shape

    dst_x0 = max(0, x)
    dst_y0 = max(0, y)
    dst_x1 = min(W, x + w)
    dst_y1 = min(H, y + h)

    if dst_x1 <= dst_x0 or dst_y1 <= dst_y0:
        return

    src_x0 = dst_x0 - x
    src_y0 = dst_y0 - y
    src_x1 = src_x0 + (dst_x1 - dst_x0)
    src_y1 = src_y0 + (dst_y1 - dst_y0)

    v_patch = local_vertices[src_y0:src_y1, src_x0:src_x1]
    m_patch = local_mask[src_y0:src_y1, src_x0:src_x1]

    region_v = full_vertex[dst_y0:dst_y1, dst_x0:dst_x1]
    region_m = full_mask[dst_y0:dst_y1, dst_x0:dst_x1]
    region_v[m_patch] = v_patch[m_patch]
    region_m[m_patch] = 1


def save_cse_overlay(
    image: np.ndarray,
    cse_valid_mask: np.ndarray,
    bbox_xyxy: np.ndarray,
    out_path: Path,
    alpha: float = 0.45,
) -> None:
    """Save one simple CSE-mask overlay image for quick inspection."""
    out_path.parent.mkdir(parents=True, exist_ok=True)

    mask = cse_valid_mask > 0
    overlay = image.copy()

    if np.any(mask):
        color = np.zeros_like(image, dtype=np.uint8)
        color[:, :] = (0, 255, 0)
        overlay[mask] = cv2.addWeighted(overlay[mask], 1.0 - alpha, color[mask], alpha, 0)

    if bbox_xyxy is not None and bbox_xyxy[0] >= 0:
        x1, y1, x2, y2 = [int(round(v)) for v in bbox_xyxy.tolist()]
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 255), 2)

    cv2.imwrite(str(out_path), overlay)


def resize_embedding_to_bbox(E: torch.Tensor, h: int, w: int) -> torch.Tensor:
    """
    Resize one selected DensePose CSE embedding tensor to bbox resolution.

    Expected input from DensePose CSE is usually [1, C, h0, w0]. This returns
    [C, h, w].
    """
    if E.dim() == 4:
        E4 = E
    elif E.dim() == 3:
        E4 = E.unsqueeze(0)
    else:
        raise ValueError(f"Unexpected embedding shape: {tuple(E.shape)}")

    E_resized = F.interpolate(E4, size=(h, w), mode="bilinear", align_corners=False)
    return E_resized[0]


def make_sparse_cse_arrays(
    local_vertices: np.ndarray,
    local_mask: np.ndarray,
    embedding_chw: torch.Tensor,
    x: int,
    y: int,
    H: int,
    W: int,
    embedding_dtype: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert local bbox CSE outputs to sparse full-image coordinates.

    Returns:
      ys: N int32
      xs: N int32
      vertex_ids: N int32
      embeddings: N x D, float16 or float32
    """
    local_ys, local_xs = np.nonzero(local_mask)
    if len(local_ys) == 0:
        return empty_sparse_arrays(embedding_chw.shape[0], embedding_dtype)

    full_ys = local_ys + y
    full_xs = local_xs + x
    inside = (full_ys >= 0) & (full_ys < H) & (full_xs >= 0) & (full_xs < W)

    if not np.any(inside):
        return empty_sparse_arrays(embedding_chw.shape[0], embedding_dtype)

    local_ys = local_ys[inside]
    local_xs = local_xs[inside]
    full_ys = full_ys[inside].astype(np.int32)
    full_xs = full_xs[inside].astype(np.int32)

    vertex_ids = local_vertices[local_ys, local_xs].astype(np.int32)
    embeddings = embedding_chw[:, local_ys, local_xs].permute(1, 0).detach().cpu().numpy()

    if embedding_dtype == "float16":
        embeddings = embeddings.astype(np.float16)
    elif embedding_dtype == "float32":
        embeddings = embeddings.astype(np.float32)
    else:
        raise ValueError(f"Unsupported embedding_dtype: {embedding_dtype}")

    return full_ys, full_xs, vertex_ids, embeddings


def empty_sparse_arrays(embedding_dim: int, embedding_dtype: str):
    dtype = np.float16 if embedding_dtype == "float16" else np.float32
    return (
        np.empty((0,), dtype=np.int32),
        np.empty((0,), dtype=np.int32),
        np.empty((0,), dtype=np.int32),
        np.empty((0, int(embedding_dim)), dtype=dtype),
    )


def export_one_entry(
    entry: Dict[str, Any],
    image_dir: Path,
    output_dir: Path,
    min_score: float,
    mesh_vertex_embeddings: torch.Tensor,
    device: torch.device,
    overlay_dir: Optional[Path] = None,
    embedding_dtype: str = "float32",
) -> Dict[str, Any]:
    from densepose.modeling.cse.utils import get_closest_vertices_mask_from_ES

    image_name = Path(entry["file_name"]).name
    stem = Path(image_name).stem
    image_path = image_dir / image_name
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Failed to read image: {image_path}")

    H, W = image.shape[:2]
    vertex_id_map = np.full((H, W), -1, dtype=np.int32)
    cse_valid_mask = np.zeros((H, W), dtype=np.uint8)

    scores = tensor_to_numpy(entry.get("scores")) if "scores" in entry else None
    boxes_xyxy = tensor_to_numpy(entry.get("pred_boxes_XYXY")) if "pred_boxes_XYXY" in entry else None
    dpout = entry.get("pred_densepose")

    selected_idx = choose_instance(scores, min_score)
    det_score = -1.0
    bbox_xyxy = np.array([-1, -1, -1, -1], dtype=np.float32)
    status = "no_instance"

    ys, xs, vertex_ids, embeddings = empty_sparse_arrays(0, embedding_dtype)

    if selected_idx is not None and boxes_xyxy is not None and dpout is not None:
        det_score = float(scores[selected_idx])
        bbox_xyxy = boxes_xyxy[selected_idx].astype(np.float32)
        x, y, w, h = xyxy_to_int_xywh(bbox_xyxy)

        # Same core operation as DensePoseOutputsVertexVisualizer:
        # closest CSE embedding vertex + foreground mask inside the bbox.
        E = dpout.embedding[[selected_idx]].to(device)
        print(f"Embedding shape: {tuple(E.shape)}, dtype: {E.dtype}")
        S = dpout.coarse_segm[[selected_idx]].to(device)
        with torch.no_grad():
            closest_vertices, mask = get_closest_vertices_mask_from_ES(
                E, S, h, w, mesh_vertex_embeddings, device
            )
            embedding_chw = resize_embedding_to_bbox(E, h, w)
            print(f"embedding_chw shape: {tuple(embedding_chw.shape)}, dtype: {embedding_chw.dtype}")

        local_vertices = closest_vertices.detach().cpu().numpy().astype(np.int32)
        local_mask = mask.detach().cpu().numpy().astype(bool)
        print(f"local_vertices shape: {tuple(local_vertices.shape)}")
        print(f"cse_valid_mask shape: {tuple(cse_valid_mask.shape)}")
        paste_local_vertex_map(vertex_id_map, cse_valid_mask, local_vertices, local_mask, x, y)

        ys, xs, vertex_ids, embeddings = make_sparse_cse_arrays(
            local_vertices=local_vertices,
            local_mask=local_mask,
            embedding_chw=embedding_chw,
            x=x,
            y=y,
            H=H,
            W=W,
            embedding_dtype=embedding_dtype,
        )
        status = "ok"

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{stem}.npz"
    np.savez_compressed(
        out_path,
        vertex_id_map=vertex_id_map,
        cse_valid_mask=cse_valid_mask,
        bbox_xyxy=bbox_xyxy,
        det_score=np.float32(det_score),
        ys=ys,
        xs=xs,
        vertex_ids=vertex_ids,
        embeddings=embeddings,
    )

    if overlay_dir is not None:
        save_cse_overlay(
            image=image,
            cse_valid_mask=cse_valid_mask,
            bbox_xyxy=bbox_xyxy,
            out_path=overlay_dir / f"{stem}.jpg",
        )

    valid_pixels = int(cse_valid_mask.sum())
    unique_vertices = int(len(np.unique(vertex_id_map[vertex_id_map >= 0])))
    return {
        "frame": stem,
        "status": status,
        "det_score": det_score,
        "valid_pixels": valid_pixels,
        "unique_vertices": unique_vertices,
        "sparse_points": int(len(ys)),
        "embedding_dim": int(embeddings.shape[1]) if embeddings.ndim == 2 else 0,
        "embedding_dtype": str(embeddings.dtype),
        "output": str(out_path),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dump_pkl", required=True)
    parser.add_argument("--image_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--overlay_dir", default=None, help="Optional directory to save CSE mask overlay jpgs.")
    parser.add_argument("--min_score", type=float, default=0.8)
    parser.add_argument("--embedding_dtype", choices=["float16", "float32"], default="float32")
    parser.add_argument(
        "--cfg",
        default="tools/detectron2/projects/DensePose/configs/cse/densepose_rcnn_R_50_FPN_s1x.yaml",
    )
    parser.add_argument(
        "--model_path",
        default="tools/detectron2/projects/DensePose/models/cse/densepose_rcnn_R_50_FPN_s1x.pkl",
    )
    parser.add_argument(
        "--densepose_project_dir",
        default="tools/detectron2/projects/DensePose",
        help="Path containing the densepose Python package.",
    )
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--default_class", type=int, default=0)
    args = parser.parse_args()

    add_densepose_project_to_path(args.densepose_project_dir)

    dump_pkl = Path(args.dump_pkl)
    image_dir = Path(args.image_dir)
    output_dir = Path(args.output_dir)
    overlay_dir = Path(args.overlay_dir) if args.overlay_dir else None
    cfg_path = Path(args.cfg)
    model_path = Path(args.model_path) if args.model_path else None

    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    cfg = setup_densepose_cfg(cfg_path, model_path)
    mesh_name, mesh_vertex_embeddings = build_mesh_embeddings(cfg, device, args.default_class)

    results = load_torch_dump(dump_pkl)
    results = sorted(results, key=lambda e: Path(e["file_name"]).name)

    print(f"[INFO] dump: {dump_pkl}")
    print(f"[INFO] image_dir: {image_dir}")
    print(f"[INFO] output_dir: {output_dir}")
    if overlay_dir is not None:
        print(f"[INFO] overlay_dir: {overlay_dir}")
    print(f"[INFO] entries: {len(results)}")
    print(f"[INFO] device: {device}, mesh: {mesh_name}")
    print(f"[INFO] embedding_dtype: {args.embedding_dtype}")

    summaries = []
    for i, entry in enumerate(results, 1):
        if i==1:
            try:
                summary = export_one_entry(
                    entry=entry,
                    image_dir=image_dir,
                    output_dir=output_dir,
                    min_score=args.min_score,
                    mesh_vertex_embeddings=mesh_vertex_embeddings,
                    device=device,
                    overlay_dir=overlay_dir,
                    embedding_dtype=args.embedding_dtype,
                )
                summaries.append(summary)
                print(
                    f"[{i:04d}/{len(results):04d}] {summary['frame']} "
                    f"{summary['status']} valid={summary['valid_pixels']} "
                    f"unique_v={summary['unique_vertices']} sparse={summary['sparse_points']} "
                    f"emb_dim={summary['embedding_dim']} score={summary['det_score']:.3f}"
                )
            except Exception as exc:
                frame = Path(entry.get("file_name", f"entry_{i}")).stem
                summaries.append({"frame": frame, "status": "error", "error": str(exc)})
                print(f"[ERROR] {frame}: {exc}")

    summary_path = output_dir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summaries, f, ensure_ascii=False, indent=2)
    print(f"[DONE] summary saved to {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


r'''
python src/script/test_function/debug002_export_densepose_cse_npz.py \
  --dump_pkl data_process/Output/20260311_riding/densepose/cse_dump/cam4.pkl \
  --image_dir data_process/Output/20260311_riding/image/equirect_crop_rotate_upright/cam4 \
  --output_dir data_process/Output/20260311_riding/densepose/cse_npz/cam4 \
  --overlay_dir data_process/Output/20260311_riding/densepose/cse_npz_vis/cam4/overlay \
  --min_score 0.8 \
  --embedding_dtype float32
'''

r'''
python src/script/test_function/debug002_export_densepose_cse_npz.py \
  --dump_pkl data_process/Output/20260311_riding/densepose/cse_dump/cam2.pkl \
  --image_dir data_process/Output/20260311_riding/image/equirect_crop_rotate_upright/cam2 \
  --output_dir data_process/Output/20260311_riding/densepose/cse_npz/cam2 \
  --overlay_dir data_process/Output/20260311_riding/densepose/cse_npz_vis/cam2/overlay \
  --min_score 0.8 \
  --embedding_dtype float32
'''
