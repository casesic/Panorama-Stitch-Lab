"""通用 mesh surface vertex sampling 工具：[V,3] 顶点 → FPS 采样 → 固定 vertex id 子集，及其校验与 npy 读写。

仅依赖 numpy / pathlib，不依赖项目路径、人体模型、SAM/MHR/SMPL-X 或 torch。
"""

from pathlib import Path

import numpy as np


def farthest_point_sample_vertices(
    vertices: np.ndarray,
    num_samples: int,
    seed: int = 0,
) -> np.ndarray:
    """Euclidean FPS（增量式：维护每个顶点到已选集的最小平方距离，不建 [V,V] 矩阵，O(V*num_samples)），返回无重复的 [num_samples] int32；形状非法/含非有限值/num_samples 越界抛 ValueError。"""
    if vertices.ndim != 2:
        raise ValueError(f"vertices 必须是二维数组，实际 ndim={vertices.ndim}")
    if vertices.shape[1] != 3:
        raise ValueError(f"vertices 第二维必须为 3，实际 shape={vertices.shape}")
    num_vertices = vertices.shape[0]
    if num_vertices <= 0:
        raise ValueError("vertices 不能为空")
    if not np.isfinite(vertices).all():
        raise ValueError("vertices 含非有限值（nan/inf）")
    if not (1 <= num_samples <= num_vertices):
        raise ValueError(
            f"num_samples 必须落在 [1, {num_vertices}]，实际为 {num_samples}"
        )

    pts = vertices.astype(np.float64, copy=False)
    rng = np.random.default_rng(seed)

    selected = np.empty(num_samples, dtype=np.int64)
    # 每个顶点到已选集合的最小平方距离；已选顶点置为 -1 以避免被重复选中
    min_sq_dist = np.full(num_vertices, np.inf, dtype=np.float64)

    first = int(rng.integers(0, num_vertices))
    selected[0] = first
    min_sq_dist[first] = -1.0

    for i in range(1, num_samples):
        last = pts[selected[i - 1]]
        diff = pts - last
        sq_dist = np.einsum("ij,ij->i", diff, diff)
        np.minimum(min_sq_dist, sq_dist, out=min_sq_dist)
        min_sq_dist[selected[i - 1]] = -1.0
        nxt = int(np.argmax(min_sq_dist))
        selected[i] = nxt
        min_sq_dist[nxt] = -1.0

    return selected.astype(np.int32)


def validate_vertex_ids(
    vertex_ids: np.ndarray,
    total_vertices: int,
    expected_count: int | None = None,
) -> np.ndarray:
    """校验并规范化 vertex ids 为一维 int32（范围 [0,total_vertices)、非空、无重复、可选 expected_count）；浮点须为整数值，重复视为调用方错误、不静默去重，违反抛 ValueError。"""
    if total_vertices <= 0:
        raise ValueError(f"total_vertices 必须为正，实际为 {total_vertices}")

    flat = np.ravel(np.asarray(vertex_ids))
    if flat.size == 0:
        raise ValueError("vertex_ids 不能为空")

    if np.issubdtype(flat.dtype, np.floating):
        if not np.isfinite(flat).all():
            raise ValueError("vertex_ids 含非有限值（nan/inf）")
        if not np.all(flat == np.rint(flat)):
            raise ValueError("vertex_ids 含非整数值（如 1.5），无法安全转换为整数 id")
        ids = np.rint(flat).astype(np.int64)
    elif np.issubdtype(flat.dtype, np.integer):
        ids = flat.astype(np.int64)
    else:
        raise ValueError(f"vertex_ids dtype 不支持：{flat.dtype}")

    if ids.min() < 0 or ids.max() >= total_vertices:
        raise ValueError(
            f"vertex_ids 超出范围 [0, {total_vertices})："
            f"min={int(ids.min())}, max={int(ids.max())}"
        )

    if np.unique(ids).size != ids.size:
        raise ValueError("vertex_ids 含重复 id")

    if expected_count is not None and ids.size != expected_count:
        raise ValueError(
            f"vertex_ids 数量应为 {expected_count}，实际为 {ids.size}"
        )

    return ids.astype(np.int32)


def get_valid_sampled_vertex_ids(
    sample_ids: np.ndarray,
    valid_mask: np.ndarray,
) -> np.ndarray:
    """返回 sample_ids 中 valid_mask=True 的子集（int32，数量不固定，保持原顺序）；valid_mask 非一维或 sample_ids 越界抛 ValueError。"""
    mask = np.asarray(valid_mask)
    if mask.ndim != 1:
        raise ValueError(f"valid_mask 必须是一维数组，实际 ndim={mask.ndim}")

    ids = np.ravel(np.asarray(sample_ids)).astype(np.int64)
    if ids.size > 0 and (ids.min() < 0 or ids.max() >= mask.size):
        raise ValueError(
            f"sample_ids 超出 valid_mask 索引范围 [0, {mask.size})："
            f"min={int(ids.min())}, max={int(ids.max())}"
        )

    selected = ids[mask.astype(bool)[ids]]
    return selected.astype(np.int32)


def save_vertex_ids(
    output_path: str | Path,
    vertex_ids: np.ndarray,
    total_vertices: int,
) -> Path:
    """校验后保存 vertex ids 为 .npy（自动补 .npy 后缀、自动建父目录），返回实际写入的 Path。"""
    ids = validate_vertex_ids(vertex_ids, total_vertices)

    path = Path(output_path)
    if path.suffix != ".npy":
        path = Path(str(path) + ".npy")

    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, ids)
    return path


def load_vertex_ids(
    input_path: str | Path,
    total_vertices: int,
    expected_count: int | None = None,
) -> np.ndarray:
    """从 .npy 读取并经 validate_vertex_ids 校验，返回一维 int32；文件不存在抛 FileNotFoundError。"""
    path = Path(input_path)
    if not path.exists():
        raise FileNotFoundError(f"vertex ids 文件不存在：{path}")

    arr = np.load(path)
    return validate_vertex_ids(arr, total_vertices, expected_count)


# Example:
# template_vertices = np.random.randn(10475, 3).astype(np.float32)
# ids = farthest_point_sample_vertices(template_vertices, num_samples=1024)
# save_vertex_ids(
#     "assets/smplx_surface_ids_fps_1024.npy",
#     ids,
#     total_vertices=10475,
# )
