"""公共几何、ERP 有效区域筛选、SAM3D raw 138 点提取与极简 JSON 写入。

纯 CPU numpy / cv2 / json，不含 torch 前向、SMPL-X 模型、CPW、可视化。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

# tools/ 位于项目根，需确保在 sys.path 上才能 import tools.fisheye_extractor
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from tools.fisheye_extractor.config import DS_PARAMS, DS_PARAMS_CAM2, DS_PARAMS_CAM4
from tools.fisheye_extractor.ds_model import project as ds_project

# ---------------------------------------------------------------------------
# constants
# ---------------------------------------------------------------------------

ROTATION_MATRIX = np.array([1, 0, 0, 0, 0, 1, 0, -1, 0], dtype=np.float64).reshape(3, 3)
R_EXT_CAM4 = np.array([-1, 0, 0, 0, -1, 0, 0, 0, 1], dtype=np.float64).reshape(3, 3)

SMPLX_NUM_VERTS = 10475
NUM_DENSEKP = 138

# ===================================================================
# A. ERP 图像与 ERP valid-region
# ===================================================================


def find_image_path(image_dir: Path, cam: str, frame_idx: int) -> Path | None:
    """在 image_dir/cam/ 下依次查找 {frame:05d}.jpg 或 .png，返回存在的路径或 None。"""
    for suffix in (".jpg", ".png"):
        p = image_dir / cam / f"{frame_idx:05d}{suffix}"
        if p.exists():
            return p
    return None


def load_erp_image_checked(
    erp_image_dir: Path,
    cam: str,
    frame_idx: int,
    expected_width: int = 5760,
    expected_height: int = 2880,
) -> np.ndarray:
    """读取 ERP 背景图，校验存在性、可读性与尺寸。"""
    img_path = find_image_path(erp_image_dir, cam, frame_idx)
    if img_path is None:
        raise FileNotFoundError(
            f"ERP image not found: {erp_image_dir / cam / f'{frame_idx:05d}'}[.jpg|.png]"
        )
    img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Failed to read ERP image: {img_path}")
    h, w = img.shape[:2]
    if w != expected_width or h != expected_height:
        raise ValueError(
            f"ERP image size mismatch: expected {expected_width}x{expected_height}, "
            f"got {w}x{h} ({img_path})"
        )
    return img


def build_border_connected_valid_mask(
    image_bgr: np.ndarray,
    black_thresh: int = 5,
    erode_px: int = 10,
) -> np.ndarray:
    """生成 ERP 有效区域 mask，仅将与图像边界连通的黑色区域视为无效。

    内部深色区域（衣物、头发、阴影）保留为有效，再向内 erosion 收缩边界。

    Args:
        image_bgr: BGR uint8 图像 [H, W, 3]。
        black_thresh: max(R,G,B) <= 此值视为黑色。
        erode_px: erosion 半径 (kernel = 2*erode_px+1 方形)。

    Returns:
        bool [H, W] — True = 有效区域。
    """
    h, w = image_bgr.shape[:2]
    black = np.max(image_bgr, axis=2) <= black_thresh  # [H, W] bool

    # flood-fill 从 padding 边界出发，标记所有与边界连通的黑色像素
    padded = np.zeros((h + 2, w + 2), dtype=np.uint8)
    padded[1:-1, 1:-1] = np.where(black, 0, 255).astype(np.uint8)
    cv2.floodFill(padded, None, (0, 0), 128)

    border_black = padded[1:-1, 1:-1] == 128  # 与边界连通的黑色 → invalid
    valid = ~border_black

    ksize = 2 * erode_px + 1
    kernel = np.ones((ksize, ksize), dtype=np.uint8)
    eroded = cv2.erode(valid.astype(np.uint8) * 255, kernel)
    return eroded > 0


def points_in_valid_mask(points_2d: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    """判断各点是否落在 valid_mask 内。

    Args:
        points_2d: float [N, 2]。
        valid_mask: bool [H, W]。

    Returns:
        bool [N] — True = 点在 mask 内。

    Raises:
        ValueError: valid_mask 为 None。
    """
    if valid_mask is None:
        raise ValueError("valid_mask must not be None")
    if points_2d.ndim != 2 or points_2d.shape[1] != 2:
        raise ValueError(f"points_2d must be [N, 2], got {points_2d.shape}")

    n = points_2d.shape[0]
    h, w = valid_mask.shape
    result = np.zeros(n, dtype=bool)

    finite = np.isfinite(points_2d).all(axis=1)
    xs = np.zeros(n, dtype=np.int64)
    ys = np.zeros(n, dtype=np.int64)
    xs[finite] = np.round(points_2d[finite, 0]).astype(np.int64)
    ys[finite] = np.round(points_2d[finite, 1]).astype(np.int64)
    in_bounds = finite & (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)
    idx = np.where(in_bounds)[0]
    result[idx] = valid_mask[ys[idx], xs[idx]].astype(bool)
    return result


# ===================================================================
# B. ERP 筛选
# ===================================================================


def points_finite_and_in_bounds(
    points_2d: np.ndarray,
    image_width: int,
    image_height: int,
) -> np.ndarray:
    """判断各点坐标是否 finite 且在图像范围内。

    Args:
        points_2d: float [N, 2]。
        image_width, image_height: 图像尺寸。

    Returns:
        bool [N]。
    """
    if points_2d.ndim != 2 or points_2d.shape[1] != 2:
        raise ValueError(f"points_2d must be [N, 2], got {points_2d.shape}")
    finite = np.isfinite(points_2d).all(axis=1)
    return (
        finite
        & (points_2d[:, 0] >= 0)
        & (points_2d[:, 0] < image_width)
        & (points_2d[:, 1] >= 0)
        & (points_2d[:, 1] < image_height)
    )


def select_erp_valid_pairs(
    erp_cam2: np.ndarray,
    erp_cam4: np.ndarray,
    erp_mask_cam2: np.ndarray,
    erp_mask_cam4: np.ndarray,
) -> np.ndarray:
    """ERP 有效点对筛选——四类数据唯一允许的 keep mask。

    keep = finite_c2 & finite_c4 & in_bounds_c2 & in_bounds_c4
         & in_valid_mask_c2 & in_valid_mask_c4

    不用 fisheye 坐标/边界/mask，不用 confidence/values/DenseKP valid，不用 CPW min-points。

    Args:
        erp_cam2: float [N, 2] — cam2 ERP 全图坐标。
        erp_cam4: float [N, 2] — cam4 ERP 全图坐标。
        erp_mask_cam2: bool [H_c2, W_c2] — cam2 ERP valid mask。
        erp_mask_cam4: bool [H_c4, W_c4] — cam4 ERP valid mask。

    Returns:
        bool [N]。
    """
    if erp_cam2.shape != erp_cam4.shape:
        raise ValueError(
            f"erp_cam2 ({erp_cam2.shape}) and erp_cam4 ({erp_cam4.shape}) must have same shape"
        )
    if erp_cam2.ndim != 2 or erp_cam2.shape[1] != 2:
        raise ValueError(f"erp_cam2/erp_cam4 must be [N, 2], got {erp_cam2.shape}")

    h2, w2 = erp_mask_cam2.shape
    h4, w4 = erp_mask_cam4.shape

    in_c2 = points_finite_and_in_bounds(erp_cam2, w2, h2)
    in_c4 = points_finite_and_in_bounds(erp_cam4, w4, h4)
    mask_c2 = points_in_valid_mask(erp_cam2, erp_mask_cam2)
    mask_c4 = points_in_valid_mask(erp_cam4, erp_mask_cam4)

    return in_c2 & in_c4 & mask_c2 & mask_c4


# ===================================================================
# C. ERP → fisheye 投影
# ===================================================================


def project_erp_np(points_3d: np.ndarray, W: int, H: int) -> np.ndarray:
    """将 world 3D 点投影为 ERP 像素坐标 (numpy 版)。

    world: X-front, Y-left, Z-up。
    """
    x = points_3d[..., 0]
    y = points_3d[..., 1]
    z = points_3d[..., 2]
    r = np.sqrt(x * x + y * y + z * z)
    r = np.clip(r, 1e-6, None)
    lon = np.arctan2(x, z)
    lat = np.arcsin(np.clip(-y / r, -1.0 + 1e-6, 1.0 - 1e-6))
    u = (lon / (2.0 * np.pi) + 0.5) * W
    v = (0.5 - lat / np.pi) * H
    return np.stack([u, v], axis=-1).astype(np.float32)


def camera_to_world(points_cam: np.ndarray) -> np.ndarray:
    """camera frame 3D → world 3D (ROTATION_MATRIX.T 变换)。"""
    return np.asarray(points_cam, dtype=np.float64) @ ROTATION_MATRIX.T


def densekp_to_erp_2d(densekp_3d: np.ndarray, W: int, H: int) -> np.ndarray:
    """camera-frame DenseKP 3D 点 → ERP 2D 坐标。"""
    world = camera_to_world(densekp_3d)
    return project_erp_np(world, W, H)


def erp_2d_to_fisheye_2d(
    erp_pts: np.ndarray,
    erp_w: int,
    erp_h: int,
    ds_params: dict,
    cam_id: str,
) -> np.ndarray:
    """ERP 2D 坐标 → fisheye 2D 坐标。

    仅做坐标转换，不做图像边界判断、fisheye mask、clamp、筛选。

    Args:
        erp_pts: float [N, 2]。
        erp_w, erp_h: ERP 图像宽高。
        ds_params: Double Sphere 参数 dict。
        cam_id: "cam2" 或 "cam4"。

    Returns:
        float32 [N, 2] — fisheye 像素坐标。
    """
    u = erp_pts[:, 0].astype(np.float64)
    v = erp_pts[:, 1].astype(np.float64)

    lon = (u / erp_w) * 2.0 * np.pi - np.pi
    lat = (v / erp_h) * np.pi - np.pi / 2.0
    cos_lat = np.cos(lat)
    sin_lat = np.sin(lat)
    sin_lon = np.sin(lon)
    cos_lon = np.cos(lon)

    ray_x = cos_lat * sin_lon
    ray_y = sin_lat
    ray_z = cos_lat * cos_lon
    rays = np.stack([ray_x, ray_y, ray_z], axis=-1)  # world rays

    if cam_id == "cam4":
        rays = rays @ R_EXT_CAM4.T

    rays_cam = rays @ ROTATION_MATRIX.T
    pts, _ = ds_project(rays_cam, ds_params)
    return np.asarray(pts, dtype=np.float32).reshape(-1, 2)


def select_camera_params(camera_group: str) -> tuple[dict, dict]:
    """按 camera_group 返回 (ds_params_cam2, ds_params_cam4)。"""
    if camera_group == "new":
        return DS_PARAMS_CAM2, DS_PARAMS_CAM4
    return DS_PARAMS, DS_PARAMS


def get_camera_group_for_dataset(dataset: str) -> str:
    """返回数据集对应的 camera_group ('old' / 'new')。

    20260311_riding → "old"
    20260320_riding_dataset1/2 → "new"
    """
    if dataset == "20260311_riding":
        return "old"
    if dataset in ("20260320_riding_dataset1", "20260320_riding_dataset2"):
        return "new"
    raise ValueError(
        f"Unknown dataset: {dataset!r}. "
        f"Known: 20260311_riding, 20260320_riding_dataset1, 20260320_riding_dataset2"
    )


# ===================================================================
# D. DenseKP regressor
# ===================================================================



def _tensor_to_numpy(t) -> np.ndarray:
    """将 torch tensor 通过 ctypes 转为 numpy（避免旧 torch + numpy 2.x 的 .numpy() 失败）。"""
    import ctypes

    import torch as _torch

    if not t.is_cpu:
        t = t.cpu()
    t = t.contiguous()
    nbytes = t.numel() * t.element_size()
    ptr = t.data_ptr()
    buf = (ctypes.c_char * nbytes).from_address(ptr)
    dtype_map = {
        _torch.float32: np.float32,
        _torch.float64: np.float64,
        _torch.int64: np.int64,
        _torch.int32: np.int32,
    }
    np_dtype = dtype_map.get(t.dtype, np.float32)
    return np.frombuffer(bytes(buf), dtype=np_dtype).reshape(tuple(t.shape)).copy()


def load_densekp_regressor() -> np.ndarray:
    """加载 DenseKP regressor R [138, 10475]。

    优先读缓存的 npy；不存在时从 downsample_mat.pkl + smplx2smpl.pkl 构造并写回。
    """
    npy_path = _PROJECT_ROOT / "tools/CameraHMR/data/train-eval-utils/densekp_regressor_R.npy"
    if npy_path.exists():
        R = np.load(npy_path).astype(np.float32)
        if R.shape == (NUM_DENSEKP, SMPLX_NUM_VERTS):
            return R
        raise ValueError(
            f"Unexpected regressor shape {R.shape}, expected ({NUM_DENSEKP}, {SMPLX_NUM_VERTS})"
        )

    import pickle

    downsample_path = (
        _PROJECT_ROOT / "tools/CameraHMR/data/train-eval-utils/downsample_mat.pkl"
    )
    smplx2smpl_path = (
        _PROJECT_ROOT / "tools/CameraHMR/data/train-eval-utils/smplx2smpl.pkl"
    )

    with open(downsample_path, "rb") as f:
        ds = pickle.load(f)
    ds_dense = ds.to_dense() if hasattr(ds, "is_sparse") and ds.is_sparse else ds
    ds_np = _tensor_to_numpy(ds_dense)

    with open(smplx2smpl_path, "rb") as f:
        s2s = pickle.load(f)
    m = s2s["matrix"]
    if hasattr(m, "is_sparse") and m.is_sparse:
        m = m.to_dense()
    m_np = (
        _tensor_to_numpy(m) if hasattr(m, "data_ptr") else np.asarray(m, dtype=np.float64)
    )

    R = ds_np @ m_np  # [138, 6890] @ [6890, 10475] → [138, 10475]
    np.save(npy_path, R.astype(np.float32))
    return R.astype(np.float32)


# ===================================================================
# E. SAM3D raw: MHR → SMPL-X surface 3D → 138 点
# ===================================================================


def load_mhr_to_smplx_mapping(
    mapping_path: Path,
) -> tuple[np.ndarray, np.ndarray]:
    """加载 MHR → SMPL-X barycentric mapping。

    Returns:
        triangle_ids: int64 [10475] — 每个 SMPL-X 顶点对应的 MHR 三角面索引。
        baryc_coords: float64 [10475, 3] — 重心坐标。
    """
    m = np.load(str(mapping_path))
    triangle_ids = m["triangle_ids"].astype(np.int64)
    baryc_coords = m["baryc_coords"].astype(np.float64)

    if triangle_ids.shape != (SMPLX_NUM_VERTS,):
        raise ValueError(
            f"triangle_ids shape {triangle_ids.shape} != ({SMPLX_NUM_VERTS},)"
        )
    if baryc_coords.shape != (SMPLX_NUM_VERTS, 3):
        raise ValueError(
            f"baryc_coords shape {baryc_coords.shape} != ({SMPLX_NUM_VERTS}, 3)"
        )
    if not np.isfinite(triangle_ids).all():
        raise ValueError("triangle_ids contains non-finite values")
    if not np.isfinite(baryc_coords).all():
        raise ValueError("baryc_coords contains non-finite values")
    if (triangle_ids < 0).any():
        raise ValueError("triangle_ids contains negative values")

    row_sums = baryc_coords.sum(axis=1)
    if not np.allclose(row_sums, 1.0, rtol=1e-4):
        raise ValueError(
            f"baryc_coords rows do not sum to 1: min={row_sums.min():.6f} max={row_sums.max():.6f}"
        )

    return triangle_ids, baryc_coords


def mhr_recon_to_smplx_surface_3d(
    pred_vertices: np.ndarray,
    pred_cam_t: np.ndarray,
    faces: np.ndarray,
    triangle_ids: np.ndarray,
    baryc_coords: np.ndarray,
) -> np.ndarray:
    """MHR 重建 → SMPL-X 10475 surface 3D (camera frame)。

    数学与 debug074 map_and_project 前半段一致：
      verts_cam = pred_vertices + pred_cam_t
      → faces[triangle_ids] 取 MHR 三角面顶点
      → barycentric 插值 → smplx_surface_cam

    Args:
        pred_vertices:  MHR 顶点 [V, 3] camera-frame。
        pred_cam_t:     [3] camera translation。
        faces:          MHR 三角面 [F, 3] int。
        triangle_ids:   [10475] int — 每个 SMPL-X 顶点对应的 MHR 面索引。
        baryc_coords:   [10475, 3] — 重心坐标。

    Returns:
        float64 [10475, 3] camera-frame SMPL-X surface 3D。
    """
    if triangle_ids.shape != (SMPLX_NUM_VERTS,):
        raise ValueError(f"triangle_ids shape {triangle_ids.shape} != ({SMPLX_NUM_VERTS},)")
    if baryc_coords.shape != (SMPLX_NUM_VERTS, 3):
        raise ValueError(f"baryc_coords shape {baryc_coords.shape} != ({SMPLX_NUM_VERTS}, 3)")

    verts = pred_vertices.astype(np.float64)
    cam_t = pred_cam_t.astype(np.float64)
    verts_cam = verts + cam_t[None, :]  # [V, 3]

    # 校验 faces index 合法
    if (triangle_ids.max() >= faces.shape[0]) or (triangle_ids.min() < 0):
        raise ValueError(
            f"triangle_ids out of faces range: triangle_ids in [{triangle_ids.min()}, {triangle_ids.max()}], "
            f"faces has {faces.shape[0]} faces"
        )

    mhr_face_vids = faces[triangle_ids]  # [10475, 3]
    mhr_tri_verts = verts_cam[mhr_face_vids]  # [10475, 3, 3]
    smplx_surface = (baryc_coords[..., None] * mhr_tri_verts).sum(axis=1)  # [10475, 3]

    if not np.isfinite(smplx_surface).all():
        raise ValueError("smplx_surface_3d contains non-finite values")
    if smplx_surface.shape != (SMPLX_NUM_VERTS, 3):
        raise ValueError(
            f"smplx_surface_3d shape {smplx_surface.shape} != ({SMPLX_NUM_VERTS}, 3)"
        )

    return smplx_surface


def smplx_surface_to_densekp3d(
    smplx_surface_cam: np.ndarray,
    densekp_regressor: np.ndarray,
) -> np.ndarray:
    """SMPL-X surface 3D → DenseKP 138 点 3D (camera frame)。

    densekp_3d = R @ smplx_surface_cam

    Args:
        smplx_surface_cam: float [10475, 3]。
        densekp_regressor: float [138, 10475]。

    Returns:
        float32 [138, 3] — camera-frame DenseKP 3D 点。
    """
    if smplx_surface_cam.shape != (SMPLX_NUM_VERTS, 3):
        raise ValueError(
            f"smplx_surface_cam shape {smplx_surface_cam.shape} != ({SMPLX_NUM_VERTS}, 3)"
        )
    if densekp_regressor.shape != (NUM_DENSEKP, SMPLX_NUM_VERTS):
        raise ValueError(
            f"densekp_regressor shape {densekp_regressor.shape} != ({NUM_DENSEKP}, {SMPLX_NUM_VERTS})"
        )

    dkp3d = np.asarray(densekp_regressor @ smplx_surface_cam, dtype=np.float32)
    if dkp3d.shape != (NUM_DENSEKP, 3):
        raise ValueError(f"densekp3d shape {dkp3d.shape} != ({NUM_DENSEKP}, 3)")
    return dkp3d


def project_pinhole_to_upright(
    points_3d_cam: np.ndarray,
    focal_length: float,
    image_height: int,
    image_width: int,
) -> tuple[np.ndarray, np.ndarray]:
    """针孔投影：camera-frame 3D → upright crop 2D。

    u = focal * X / Z + W / 2
    v = focal * Y / Z + H / 2

    Args:
        points_3d_cam: float [N, 3] camera-frame。
        focal_length: SAM3D 预测的焦距。
        image_height, image_width: upright crop 尺寸。

    Returns:
        points_2d: float32 [N, 2]。
        geometric_valid: bool [N] — Z > 1e-6, finite, 在 upright 范围内。
    """
    if points_3d_cam.ndim != 2 or points_3d_cam.shape[1] != 3:
        raise ValueError(f"points_3d_cam must be [N, 3], got {points_3d_cam.shape}")

    pts = points_3d_cam.astype(np.float64)
    X = pts[:, 0]
    Y = pts[:, 1]
    Z = pts[:, 2]

    with np.errstate(divide="ignore", invalid="ignore"):
        u = focal_length * X / Z + image_width / 2.0
        v = focal_length * Y / Z + image_height / 2.0
    points_2d = np.stack([u, v], axis=1).astype(np.float32)

    geometric_valid = (
        (Z > 1e-6)
        & np.isfinite(points_2d).all(axis=1)
        & (points_2d[:, 0] >= 0)
        & (points_2d[:, 0] < image_width)
        & (points_2d[:, 1] >= 0)
        & (points_2d[:, 1] < image_height)
    )

    return points_2d, geometric_valid


# ===================================================================
# F. 极简 JSON
# ===================================================================

_MINIMAL_JSON_KEYS = frozenset({"frame_idx", "image_width", "image_height", "pairs"})


def build_minimal_corr_json(
    frame_idx: int,
    image_width: int,
    image_height: int,
    pairs: np.ndarray,
) -> dict:
    """构建极简点对 JSON dict。

    Args:
        frame_idx: 帧号。
        image_width, image_height: 图像尺寸。
        pairs: float [N, 4] — [cam2_x, cam2_y, cam4_x, cam4_y]，N 可为 0。

    Returns:
        dict 仅含: frame_idx, image_width, image_height, pairs。
    """
    if pairs.ndim != 2 or pairs.shape[1] != 4:
        raise ValueError(f"pairs must be [N, 4], got {pairs.shape}")
    if not np.isfinite(pairs).all():
        raise ValueError("pairs contains NaN or Inf")

    pairs_list = [
        [float(p[0]), float(p[1]), float(p[2]), float(p[3])]
        for p in pairs.astype(np.float64)
    ]

    return {
        "frame_idx": int(frame_idx),
        "image_width": int(image_width),
        "image_height": int(image_height),
        "pairs": pairs_list,
    }


def write_minimal_corr_json(
    path: Path,
    frame_idx: int,
    image_width: int,
    image_height: int,
    pairs: np.ndarray,
) -> None:
    """写入极简点对 JSON，自动创建父目录，UTF-8，indent=2。"""
    data = build_minimal_corr_json(frame_idx, image_width, image_height, pairs)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(str(path), "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
