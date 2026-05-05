from __future__ import annotations

from typing import Any

import numpy as np


def get_corr_vertex_ids(corr_data: dict[str, Any]) -> np.ndarray:
    return np.asarray(corr_data["matched_vertex_ids"], dtype=np.int64)


def build_keep_mask_from_exclude_ids(vertex_ids: np.ndarray, exclude_ids: set[int] | np.ndarray) -> np.ndarray:
    vertex_ids = np.asarray(vertex_ids, dtype=np.int64)
    if len(exclude_ids) == 0:
        return np.ones(len(vertex_ids), dtype=bool)
    remove_ids = np.asarray(sorted(exclude_ids), dtype=np.int64)
    return ~np.isin(vertex_ids, remove_ids)


def filter_corr_by_mask(corr_data: dict[str, Any], keep_mask: np.ndarray) -> dict[str, Any]:
    keep_mask = np.asarray(keep_mask, dtype=bool)
    filtered = dict(corr_data)
    filtered["matched_vertex_ids"] = get_corr_vertex_ids(corr_data)[keep_mask]
    filtered["pairs"] = np.asarray(corr_data["pairs"], dtype=np.float32)[keep_mask]
    return filtered


def filter_corr_by_vertex_ids(corr_data: dict[str, Any], exclude_vertex_ids: set[int] | np.ndarray) -> dict[str, Any]:
    vertex_ids = get_corr_vertex_ids(corr_data)
    keep_mask = build_keep_mask_from_exclude_ids(vertex_ids, exclude_vertex_ids)
    return filter_corr_by_mask(corr_data, keep_mask)
