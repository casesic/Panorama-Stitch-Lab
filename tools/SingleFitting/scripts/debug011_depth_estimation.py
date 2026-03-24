#!/usr/bin/env python3
# Estimate depth maps from one or more images and optionally overlay masks.
# Usage:
#   python tools/SingleFitting/scripts/debug011_depth_estimation.py ^
#     --images data/cam2_frame0000.jpg data/cam4_frame0000.jpg ^
#     --masks tools/SingleFitting/outputs/segmentation/mask_cam2_frame_0000.png ^
#             tools/SingleFitting/outputs/segmentation/mask_cam4_frame_0000.png ^
#     --labels cam2 cam4 ^
#     --output-dir tools/SingleFitting/outputs/depth
"""
debug011_depth_estimation.py
-----------------------------
用 Depth Anything V3 (DA3) 估计鱼眼图的深度图。

DA3 的核心优势：
  - 支持多视角输入（cam2 + cam4 同时输入），输出空间一致的深度
  - 不需要已知相机位姿
  - 直接预测深度（不是视差），几何精度更高

用法：
  v3在远程服务器上
  # 叠加分割 mask 可视化
  python debug011_depth_estimation.py \
    --images data/cam2_frame0000.jpg data/cam4_frame0000.jpg \
    --masks outputs/segmentation/mask_cam2_frame_0000.png \
            outputs/segmentation/mask_cam4_frame_0000.png \
    --labels cam2 cam4 \
    --output-dir outputs/depth
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Optional
import sys

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


# ---------------------------------------------------------------------------
# 深度图可视化工具
# ---------------------------------------------------------------------------

def colorize_depth(
    depth: np.ndarray,          # (H, W) float32
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    cmap: int = cv2.COLORMAP_INFERNO,
) -> np.ndarray:
    """把深度图转成彩色图像（BGR）。"""
    d = depth.copy().astype(np.float32)

    # 去掉无效值
    valid = np.isfinite(d) & (d > 0)
    if valid.sum() == 0:
        return np.zeros((*depth.shape, 3), dtype=np.uint8)

    if vmin is None:
        vmin = float(d[valid].min())
    if vmax is None:
        vmax = float(d[valid].max())

    d = np.clip(d, vmin, vmax)
    d = (d - vmin) / (vmax - vmin + 1e-8)
    d = (d * 255).astype(np.uint8)
    return cv2.applyColorMap(d, cmap)


def overlay_mask_on_depth(
    depth_color: np.ndarray,    # (H, W, 3) BGR
    mask: np.ndarray,           # (H, W) bool
    contour_color: tuple = (0, 255, 255),   # 青色轮廓
    thickness: int = 2,
) -> np.ndarray:
    """在深度图上叠加分割 mask 的轮廓。"""
    H, W = depth_color.shape[:2]
    if mask.shape != (H, W):
        mask = cv2.resize(
            mask.astype(np.uint8), (W, H),
            interpolation=cv2.INTER_NEAREST
        ).astype(bool)

    result = depth_color.copy()
    mask_u8 = mask.astype(np.uint8) * 255
    contours, _ = cv2.findContours(
        mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    cv2.drawContours(result, contours, -1, contour_color, thickness)
    return result


def compute_depth_stats_in_mask(
    depth: np.ndarray,
    mask: np.ndarray,
) -> dict:
    """计算 mask 区域内的深度统计信息。"""
    H, W = depth.shape[:2]
    if mask.shape != (H, W):
        mask = cv2.resize(
            mask.astype(np.uint8), (W, H),
            interpolation=cv2.INTER_NEAREST
        ).astype(bool)

    d_in  = depth[mask]
    d_out = depth[~mask & (depth > 0)]

    valid_in  = d_in[np.isfinite(d_in) & (d_in > 0)]
    valid_out = d_out[np.isfinite(d_out) & (d_out > 0)]

    stats = {
        'human_mean':       float(valid_in.mean())  if len(valid_in)  > 0 else None,
        'human_median':     float(np.median(valid_in))  if len(valid_in)  > 0 else None,
        'human_min':        float(valid_in.min())   if len(valid_in)  > 0 else None,
        'human_max':        float(valid_in.max())   if len(valid_in)  > 0 else None,
        'background_mean':  float(valid_out.mean()) if len(valid_out) > 0 else None,
        'background_median':float(np.median(valid_out)) if len(valid_out) > 0 else None,
    }
    return stats


def create_side_by_side(
    img_orig: np.ndarray,
    depth_color: np.ndarray,
    label: str = '',
    height: int = 540,
) -> np.ndarray:
    """把原图和深度图拼在一起，方便对比。"""
    def resize_h(img, h):
        r  = h / img.shape[0]
        w  = int(img.shape[1] * r)
        return cv2.resize(img, (w, h))

    orig_r  = resize_h(img_orig,    height)
    depth_r = resize_h(depth_color, height)

    # 加标签
    if label:
        cv2.putText(orig_r,  f'{label} - Original',
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 2)
        cv2.putText(depth_r, f'{label} - Depth (DA2)',
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 2)

    return np.concatenate([orig_r, depth_r], axis=1)


# ---------------------------------------------------------------------------
# 主函数
# ---------------------------------------------------------------------------

def run_depth_estimation(
    image_paths: List[Path],
    output_dir: Path,
    labels: Optional[List[str]] = None,
    mask_paths: Optional[List[Path]] = None,
    model_name: str = 'depth-anything/Depth-Anything-V2-Base-hf',
    export_format: str = 'npz',
) -> dict:
    """
    用 Depth Anything V2 估计深度（通过 transformers 调用，无需编译依赖）。

    image_paths: 输入图像路径列表
    labels:      每张图的标签
    mask_paths:  分割 mask 路径（可选）
    model_name:  模型选择：
                 - 'depth-anything/Depth-Anything-V2-Small-hf'  （最快）
                 - 'depth-anything/Depth-Anything-V2-Base-hf'   （平衡）
                 - 'depth-anything/Depth-Anything-V2-Large-hf'  （最准）
    """
    import torch
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation
    from PIL import Image

    if labels is None:
        labels = [p.stem for p in image_paths]

    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- 加载模型 ----
    print(f"Loading model: {model_name}...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    processor = AutoImageProcessor.from_pretrained(model_name)
    model     = AutoModelForDepthEstimation.from_pretrained(model_name)
    model     = model.to(device)
    model.eval()
    print("Model loaded.")

    # ---- 读取图像 & 推理 ----
    orig_imgs  = []
    depth_maps = []

    for p in image_paths:
        img_bgr = cv2.imread(str(p))
        if img_bgr is None:
            raise FileNotFoundError(f"读图失败: {p}")
        orig_imgs.append(img_bgr)

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)

        inputs  = processor(images=pil_img, return_tensors='pt')
        inputs  = {k: v.to(device) for k, v in inputs.items()}

        print(f"  Inferring {p.name}...")
        with torch.no_grad():
            outputs = model(**inputs)
            depth   = outputs.predicted_depth   # (1, H', W')

        # resize 到原图尺寸
        h, w = img_bgr.shape[:2]
        depth_np = torch.nn.functional.interpolate(
            depth.unsqueeze(1),
            size=(h, w),
            mode='bilinear',
            align_corners=False,
        ).squeeze().cpu().numpy()   # (H, W)

        depth_maps.append(depth_np)

    print(f"Depth maps: {len(depth_maps)} frame(s)")
    # 模拟 conf_maps（V2 没有置信度输出，用全1代替）
    conf_maps = [np.ones_like(d) for d in depth_maps]

    # ---- 读取 mask ----
    masks = []
    if mask_paths is not None:
        for mp in mask_paths:
            if mp is not None and Path(mp).exists():
                m = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE)
                masks.append(m > 127 if m is not None else None)
            else:
                masks.append(None)
    else:
        masks = [None] * len(image_paths)

    # ---- 可视化 + 统计 ----
    all_stats = []
    comparison_frames = []

    for i, (label, orig_img, mask) in enumerate(zip(labels, orig_imgs, masks)):
        depth = depth_maps[i]   # (H, W)
        conf  = conf_maps[i]    # (H, W) - all ones for V2

        img_h, img_w = orig_img.shape[:2]
        # depth already at original size (resized in inference)
        depth_resized = depth
        conf_resized  = conf

        # 深度图彩色化
        depth_color = colorize_depth(depth_resized)

        # 置信度图
        conf_color  = colorize_depth(conf_resized, vmin=0, vmax=1,
                                     cmap=cv2.COLORMAP_VIRIDIS)

        # 叠加 mask 轮廓
        if mask is not None:
            depth_color_masked = overlay_mask_on_depth(depth_color, mask)
        else:
            depth_color_masked = depth_color

        # 统计
        stats = {'label': label, 'image': str(image_paths[i])}
        stats['depth_global_mean']   = float(depth_resized[depth_resized > 0].mean()) \
            if (depth_resized > 0).any() else None
        stats['depth_global_median'] = float(np.median(
            depth_resized[depth_resized > 0])) \
            if (depth_resized > 0).any() else None

        if mask is not None:
            mask_stats = compute_depth_stats_in_mask(depth_resized, mask)
            stats.update(mask_stats)
            print(f"\n[{label}] 深度统计：")
            print(f"  人体区域  - 均值: {mask_stats['human_mean']:.3f}m, "
                  f"中位数: {mask_stats['human_median']:.3f}m")
            print(f"  背景区域  - 均值: {mask_stats['background_mean']:.3f}m, "
                  f"中位数: {mask_stats['background_median']:.3f}m")

        all_stats.append(stats)

        # 保存深度图
        out_depth = output_dir / f'depth_{label}.jpg'
        out_conf  = output_dir / f'conf_{label}.jpg'
        out_depth_masked = output_dir / f'depth_masked_{label}.jpg'
        out_depth_npy    = output_dir / f'depth_{label}.npy'

        cv2.imwrite(str(out_depth),        depth_color)
        cv2.imwrite(str(out_conf),         conf_color)
        cv2.imwrite(str(out_depth_masked), depth_color_masked)
        np.save(str(out_depth_npy), depth_resized)

        print(f"[{label}] → {out_depth}")
        print(f"[{label}] → {out_depth_masked}")

        # 拼接对比图
        compare = create_side_by_side(orig_img, depth_color_masked, label)
        comparison_frames.append(compare)

    # ---- 多视角对比图（把所有相机的结果拼在一起）----
    if len(comparison_frames) > 1:
        # 统一高度
        target_h = 540
        resized_frames = []
        for f in comparison_frames:
            r = target_h / f.shape[0]
            w = int(f.shape[1] * r)
            resized_frames.append(cv2.resize(f, (w, target_h)))

        combined = np.concatenate(resized_frames, axis=0)
        out_combined = output_dir / 'depth_comparison_all.jpg'
        cv2.imwrite(str(out_combined), combined,
                    [cv2.IMWRITE_JPEG_QUALITY, 92])
        print(f"\nCombined comparison → {out_combined}")

        # 接缝处深度一致性分析
        if len(all_stats) == 2 and all(
            s.get('human_mean') is not None for s in all_stats
        ):
            d1 = all_stats[0]['human_mean']
            d2 = all_stats[1]['human_mean']
            diff = abs(d1 - d2)
            print(f"\n接缝处深度一致性分析：")
            print(f"  {labels[0]} 人体深度均值: {d1:.3f}m")
            print(f"  {labels[1]} 人体深度均值: {d2:.3f}m")
            print(f"  差值: {diff:.3f}m ({diff/max(d1,d2)*100:.1f}%)")
            if diff < 0.3:
                print("  ✅ 两个视角深度一致性良好")
            else:
                print("  ⚠️  两个视角深度差异较大，接缝处可能有不连续")

    # 保存统计
    meta_path = output_dir / 'depth_meta.json'
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(all_stats, f, indent=2, ensure_ascii=False)
    print(f"\nStats → {meta_path}")

    return {'stats': all_stats, 'output_dir': str(output_dir)}


# ---------------------------------------------------------------------------
# 参数解析
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--images',   type=Path, nargs='+', required=True,
                   help='输入图像路径（1张或多张，多张时做多视角联合估计）')
    p.add_argument('--labels',   type=str,  nargs='+', default=None,
                   help='每张图的标签，默认用文件名')
    p.add_argument('--masks',    type=Path, nargs='+', default=None,
                   help='对应的分割 mask 路径（可选）')
    p.add_argument('--output-dir', type=Path,
                   default=Path('outputs/depth'))
    p.add_argument('--model',    type=str,
                   default='depth-anything/Depth-Anything-V2-Base-hf',
                   choices=[
                       'depth-anything/Depth-Anything-V2-Small-hf',
                       'depth-anything/Depth-Anything-V2-Base-hf',
                       'depth-anything/Depth-Anything-V2-Large-hf',
                   ],
                   help='模型大小（Small最快，Large最准）')
    p.add_argument('--export-format', type=str, default='npz',
                   choices=['npz', 'npy'],
                   help='深度图保存格式')
    return p.parse_args()


def main():
    args = parse_args()

    run_depth_estimation(
        image_paths=args.images,
        output_dir=args.output_dir,
        labels=args.labels,
        mask_paths=args.masks,
        model_name=args.model,
        export_format=args.export_format,
    )
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
