#!/usr/bin/env python3
"""生成固定、可复用的 SMPL-X surface vertex 采样 asset。

读取项目当前使用的 neutral SMPL-X template 顶点（与 fitter 一致的 `get_smplx_batch`），
复用 `lib_pano.human3d.mesh_surface_sampling` 的 FPS 采样，保存固定 vertex id 到
`src/lib_pano/human3d/assets/`，并用 cv2 输出一张三视图覆盖检查图。

本脚本不做拟合、不读数据集帧、不复制 FPS 算法，也不硬编码 SMPL-X 模型路径。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

THIS_FILE = Path(__file__).resolve()
SRC_ROOT = THIS_FILE.parents[2]
PROJECT_ROOT = THIS_FILE.parents[3]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from lib_pano.pipeline.paths import ensure_single_fitting_path

ensure_single_fitting_path(PROJECT_ROOT)

from lib_pano.human3d.mesh_surface_sampling import (
    farthest_point_sample_vertices,
    save_vertex_ids,
    validate_vertex_ids,
)

SMPLX_NUM_VERTS = 10475
SEED = 0


def load_template_vertices() -> np.ndarray:
    """读取与 fitter 一致的 neutral SMPL-X template 顶点，返回 [10475, 3] 数组。

    复用项目的 `get_smplx_batch`（即 visibility / fitter 使用的同一加载方式），取标准
    `smplx.SMPLXLayer.v_template`。顶点数非 10475 或含非有限值时抛出清晰错误。
    """
    # 延迟导入：避免在不需要模型的场景（仅做可视化/语法检查）触发 torch 初始化
    from lib_pano.human_fitting.smplx_sequence import get_smplx_batch

    model = get_smplx_batch(gender="neutral")
    v_template = getattr(model, "v_template", None)
    if v_template is None:
        raise RuntimeError("SMPL-X 模型缺少 v_template 属性，无法读取 template 顶点")

    if hasattr(v_template, "detach"):
        v_template = v_template.detach().cpu().numpy()
    vertices = np.squeeze(np.asarray(v_template))

    if vertices.shape != (SMPLX_NUM_VERTS, 3):
        raise ValueError(
            f"SMPL-X template 顶点形状异常，期望 ({SMPLX_NUM_VERTS}, 3)，实际为 {vertices.shape}"
        )
    if not np.isfinite(vertices).all():
        raise ValueError("SMPL-X template 顶点含非有限值（nan/inf）")

    return vertices.astype(np.float64)


def render_sampling_overview(
    all_vertices: np.ndarray,
    sample_ids: np.ndarray,
    num_points: int,
) -> np.ndarray:
    """生成三视图（front/side/top）采样覆盖检查图。

    背景为全部 template 顶点（灰色小点），前景为采样顶点（红色稍大点）；每个视图按各自
    min/max 归一化并保留约 20px 边距。不依赖 mesh faces 或 3D renderer。
    """
    view_size = 480
    margin = 20
    title_h = 50
    views = (("front", (0, 1)), ("side", (2, 1)), ("top", (0, 2)))

    canvas = np.full((title_h + view_size, view_size * len(views), 3), 255, np.uint8)
    cv2.putText(
        canvas,
        f"SMPL-X FPS surface sampling: {num_points} / {SMPLX_NUM_VERTS}",
        (15, 34),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 0, 0),
        2,
        cv2.LINE_AA,
    )

    sample = all_vertices[sample_ids]
    inner = view_size - 2 * margin

    for view_idx, (name, (ax_h, ax_v)) in enumerate(views):
        x0 = view_idx * view_size
        sub = canvas[title_h:title_h + view_size, x0:x0 + view_size]

        bg = all_vertices[:, (ax_h, ax_v)]
        fg = sample[:, (ax_h, ax_v)]
        mn = bg.min(axis=0)
        span = np.maximum(bg.max(axis=0) - mn, 1e-6)

        def to_px(pts: np.ndarray) -> np.ndarray:
            norm = (pts - mn) / span
            px = margin + norm[:, 0] * inner
            py = margin + (1.0 - norm[:, 1]) * inner  # 翻转竖轴，使视图朝上
            return np.stack([px, py], axis=1)

        for x, y in to_px(bg):
            cv2.circle(sub, (int(round(x)), int(round(y))), 1, (180, 180, 180), -1, cv2.LINE_AA)
        for x, y in to_px(fg):
            cv2.circle(sub, (int(round(x)), int(round(y))), 3, (0, 0, 255), -1, cv2.LINE_AA)

        cv2.putText(sub, name, (margin, margin + 6), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 0, 0), 2, cv2.LINE_AA)

    return canvas


def main() -> None:
    parser = argparse.ArgumentParser(
        description="生成固定的 SMPL-X FPS surface vertex 采样 asset 与覆盖检查图。"
    )
    parser.add_argument(
        "--num-points",
        type=int,
        default=1024,
        help="固定采样的 SMPL-X vertex 数量，默认 1024。",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="允许覆盖已经存在的 .npy asset。",
    )
    args = parser.parse_args()

    asset_dir = SRC_ROOT / "lib_pano" / "human3d" / "assets"
    output_path = asset_dir / f"smplx_surface_ids_fps_{args.num_points}.npy"
    vis_dir = PROJECT_ROOT / "data_process" / "Vis" / "debug076_smplx_surface_sampling"
    vis_path = vis_dir / f"smplx_surface_ids_fps_{args.num_points}.jpg"

    template_vertices = load_template_vertices()
    print(f"SMPL-X template vertices: {template_vertices.shape}")
    print("Sampling method: FPS")
    print(f"Seed: {SEED}")
    print(f"Requested points: {args.num_points}")

    if output_path.exists() and not args.overwrite:
        existing = np.load(output_path)
        vertex_ids = validate_vertex_ids(
            existing,
            total_vertices=SMPLX_NUM_VERTS,
            expected_count=args.num_points,
        )
        print(f"Loaded existing asset: {output_path}")
    else:
        vertex_ids = farthest_point_sample_vertices(
            template_vertices,
            num_samples=args.num_points,
            seed=SEED,
        )
        vertex_ids = validate_vertex_ids(
            vertex_ids,
            total_vertices=SMPLX_NUM_VERTS,
            expected_count=args.num_points,
        )
        output_path = save_vertex_ids(
            output_path,
            vertex_ids,
            total_vertices=SMPLX_NUM_VERTS,
        )

    vis_dir.mkdir(parents=True, exist_ok=True)
    canvas = render_sampling_overview(template_vertices, vertex_ids, args.num_points)
    cv2.imwrite(str(vis_path), canvas, [cv2.IMWRITE_JPEG_QUALITY, 95])

    print(f"Output asset: {output_path}")
    print(f"Output visualization: {vis_path}")
    print(f"First 10 vertex ids: {vertex_ids[:10].tolist()}")


if __name__ == "__main__":
    main()


r'''
cd /home/yuyue/Projects/pano-stitch
conda activate stitchenv

python src/script/test_function/debug076_make_smplx_surface_sample_ids.py

# 如需重新生成：
python src/script/test_function/debug076_make_smplx_surface_sample_ids.py \
  --num-points 1024 \
  --overwrite
'''
