from __future__ import annotations

import ast
import pickle
from pathlib import Path

import numpy as np


HAND_PART_KEYWORDS = (
    "lefthand", "righthand", "left_hand", "right_hand",
    "leftwrist", "rightwrist", "left_wrist", "right_wrist",
    "leftthumb", "rightthumb", "left_thumb", "right_thumb",
    "leftindex", "rightindex", "left_index", "right_index",
    "leftmiddle", "rightmiddle", "left_middle", "right_middle",
    "leftring", "rightring", "left_ring", "right_ring",
    "leftpinky", "rightpinky", "left_pinky", "right_pinky",
)


def parse_vertex_ids_text(text: str) -> np.ndarray:
    items = [int(item) for item in text.replace(",", " ").split()]
    return np.unique(np.asarray(items, dtype=np.int64).reshape(-1))


def load_exclude_vertex_ids(path: str | Path | None) -> np.ndarray:
    if path is None:
        return np.empty((0,), dtype=np.int64)

    path = Path(path)
    if path.suffix.lower() == ".npy":
        ids = np.load(path)
    else:
        ids = parse_vertex_ids_text(path.read_text(encoding="utf-8"))
    return np.unique(np.asarray(ids, dtype=np.int64).reshape(-1))


def save_exclude_vertex_ids(path: str | Path, vertex_ids) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ids = np.unique(np.asarray(list(vertex_ids), dtype=np.int64).reshape(-1))
    with path.open("w", encoding="utf-8") as file:
        for vertex_id in ids:
            file.write(f"{int(vertex_id)}\n")


def merge_exclude_vertex_ids(*vertex_id_sets) -> np.ndarray:
    arrays = [np.asarray(ids, dtype=np.int64).reshape(-1) for ids in vertex_id_sets if len(ids) > 0]
    if not arrays:
        return np.empty((0,), dtype=np.int64)
    return np.unique(np.concatenate(arrays)).astype(np.int64)


def _to_numpy(value) -> np.ndarray:
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _part_key_is_hand(key: object) -> bool:
    key_norm = str(key).replace(" ", "").lower()
    return any(keyword in key_norm for keyword in HAND_PART_KEYWORDS)


def _load_segmentation_file(path: Path):
    if path.suffix.lower() == ".pkl":
        with path.open("rb") as file:
            return pickle.load(file)
    return ast.literal_eval(path.read_text(encoding="utf-8"))


def _hand_ids_from_segmentation(segmentation) -> np.ndarray:
    if not isinstance(segmentation, dict):
        return np.empty((0,), dtype=np.int64)

    ids: list[int] = []
    for key, value in segmentation.items():
        if _part_key_is_hand(key):
            ids.extend(np.asarray(value, dtype=np.int64).reshape(-1).tolist())
    return np.unique(np.asarray(ids, dtype=np.int64))


def get_lbs_hand_vertex_ids(smplx_model) -> np.ndarray:
    from smplx.joint_names import JOINT_NAMES

    weights = _to_numpy(smplx_model.lbs_weights)
    hand_joint_ids = [
        joint_idx
        for joint_idx, joint_name in enumerate(JOINT_NAMES[: weights.shape[1]])
        if any(word in joint_name.lower() for word in ("wrist", "thumb", "index", "middle", "ring", "pinky"))
    ]
    if not hand_joint_ids:
        return np.empty((0,), dtype=np.int64)

    hand_weight = weights[:, hand_joint_ids].sum(axis=1)
    return np.where(hand_weight > 0.2)[0].astype(np.int64)


def get_smplx_hand_vertex_ids(smplx_model, project_root: str | Path) -> np.ndarray:
    ids: list[np.ndarray] = []
    names = (
        "smplx_vert_segmentation.json",
        "smplx_vertex_segmentation.json",
        "SMPLX_vert_segmentation.json",
        "smplx_parts_segm.pkl",
        "smplx_part_segm.pkl",
    )
    for name in names:
        for path in Path(project_root).rglob(name):
            seg_ids = _hand_ids_from_segmentation(_load_segmentation_file(path))
            if len(seg_ids) > 0:
                ids.append(seg_ids)

    lbs_ids = get_lbs_hand_vertex_ids(smplx_model)
    if len(lbs_ids) > 0:
        ids.append(lbs_ids)

    if ids:
        return np.unique(np.concatenate(ids)).astype(np.int64)

    raise RuntimeError("Failed to get SMPL-X hand vertex ids")
