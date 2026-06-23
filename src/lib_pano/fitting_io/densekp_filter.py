"""DenseKP 138 点 ID 筛选：从 JSON 加载 keep_ids/drop_ids，返回 bool mask。"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

NUM_DENSEKP = 138


def load_densekp_filter(filter_path: str | Path) -> np.ndarray:
    """从 JSON 加载 DenseKP ID 筛选规则，返回 [138] bool mask：keep_ids 不为空则只保留指定ID，否则剔除 drop_ids 中的ID。"""
    with open(filter_path, encoding="utf-8") as f:
        cfg = json.load(f)

    keep_ids = cfg.get("keep_ids")
    drop_ids = cfg.get("drop_ids")

    mask = np.ones(NUM_DENSEKP, dtype=bool)

    if keep_ids is not None and len(keep_ids) > 0:
        mask[:] = False
        mask[keep_ids] = True
    elif drop_ids is not None and len(drop_ids) > 0:
        mask[drop_ids] = False

    return mask
