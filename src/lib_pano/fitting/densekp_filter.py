from __future__ import annotations

import json
from pathlib import Path

import numpy as np

NUM_DENSEKP = 138


def load_densekp_filter(filter_path: str | Path) -> np.ndarray:
    """Load a DenseKP ID filter from JSON, returning a bool mask of shape [138].

    JSON format::

        {"drop_ids": [0, 1, 2, ...], "keep_ids": null}
        {"keep_ids": [3, 4, 5, ...], "drop_ids": null}

    Rules:
    - If *keep_ids* is not empty, only those IDs are kept.
    - Otherwise all IDs *except* those in *drop_ids* are kept.
    """
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
