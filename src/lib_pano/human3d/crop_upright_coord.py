"""Crop-upright coordinate restore to equirect_rotate full-image coordinates.

Rotation rules are dataset-specific and live in
``configs/rotate_upright_rules.json`` (the same rules debug001 used to *produce*
the upright crops). ``resolve_dataset_rotate_rules`` picks the entry whose
``input_root`` matches a given crop dir, and ``resolve_rotate_rule`` maps a frame
index to its rotate string so the upright rotation can be inverted.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def get_20260311_rotate_rule(frame_idx: int) -> str:
    """返回 20260311_riding 数据集中指定帧的 upright 旋转规则。"""
    if 0 <= frame_idx <= 3494:
        return "ccw90"
    if 3495 <= frame_idx <= 5381:
        return "cw90"
    if 5951 <= frame_idx <= 5954:
        return "ccw90"
    return "none"


def upright_points_to_equirect_rotate(
    points_upright: np.ndarray,
    bbox: list[int] | tuple[int, int, int, int],
    frame_idx: int,
    rotate_rule: str | None = None,
) -> np.ndarray:
    """将 crop_upright 坐标的点还原到 equirect_rotate 全图坐标（rotate_rule 为 None 时回退到 20260311 规则）。"""
    x1, y1, x2, y2 = bbox
    crop_w = x2 - x1
    crop_h = y2 - y1

    pts = points_upright.astype(np.float32)
    out = np.empty_like(pts)

    if rotate_rule is None:
        rotate_rule = get_20260311_rotate_rule(frame_idx)

    if rotate_rule == "ccw90":
        out[:, 0] = crop_w - 1 - pts[:, 1]
        out[:, 1] = pts[:, 0]
    elif rotate_rule == "cw90":
        out[:, 0] = pts[:, 1]
        out[:, 1] = crop_h - 1 - pts[:, 0]
    elif rotate_rule == "180":
        out[:, 0] = crop_w - 1 - pts[:, 0]
        out[:, 1] = crop_h - 1 - pts[:, 1]
    elif rotate_rule == "none":
        out[:, 0] = pts[:, 0]
        out[:, 1] = pts[:, 1]
    else:
        raise ValueError(f"Unknown rotate rule: {rotate_rule}")

    out[:, 0] += x1
    out[:, 1] += y1
    return out


def resolve_dataset_rotate_rules(
    config_path: str | Path,
    crop_dir: str | Path,
) -> list[dict]:
    """按 config 里的 name 匹配 crop_dir 路径中的数据集目录段，取对应旋转规则；匹配不到则报错。"""
    entries = json.loads(Path(config_path).read_text(encoding="utf-8"))
    parts = set(str(crop_dir).replace("\\", "/").split("/"))
    for entry in entries:
        name = entry.get("name")
        if name is not None and name in parts:
            return entry.get("rules", [])
    available = [entry.get("name") for entry in entries]
    raise ValueError(
        f"No rotate-rule entry in {config_path} matches crop dir {crop_dir!r}. "
        f"Available dataset names: {available}"
    )


def resolve_rotate_rule(frame_idx: int, rules: list[dict]) -> str:
    """按帧区间返回旋转规则字符串；无匹配区间时返回 'none'（对应 debug001 的 copy 行为）。"""
    for rule in rules:
        if rule["start"] <= frame_idx <= rule["end"]:
            return rule.get("rotate", "none")
    return "none"
