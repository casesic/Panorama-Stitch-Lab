from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from .debug_shared import load_pickle


def brief_value(name, value):
    if isinstance(value, np.ndarray):
        print(f"- {name}: np.ndarray shape={value.shape}, dtype={value.dtype}")
    elif isinstance(value, torch.Tensor):
        print(f"- {name}: torch.Tensor shape={tuple(value.shape)}, dtype={value.dtype}, device={value.device}")
    else:
        print(f"- {name}: {type(value).__name__}")


def inspect_pickle_file(pkl_path: Path):
    if not pkl_path.exists():
        raise FileNotFoundError(f"File not found: {pkl_path}")

    data = load_pickle(pkl_path)
    print(f"File: {pkl_path}")
    print(f"Top-level type: {type(data).__name__}")

    if isinstance(data, dict):
        print(f"Dict keys ({len(data)}): {list(data.keys())}")
        for k, v in data.items():
            brief_value(str(k), v)
        return

    attrs = [k for k in dir(data) if not k.startswith("_")]
    show_attrs = []
    for k in attrs:
        try:
            v = getattr(data, k)
        except Exception:
            continue
        if callable(v):
            continue
        if isinstance(v, np.ndarray) or isinstance(v, torch.Tensor):
            show_attrs.append((k, v))

    if show_attrs:
        print("Main numeric fields:")
        for k, v in show_attrs:
            brief_value(k, v)

        for attr_name in ["transl", "orient", "delta_T"]:
            if hasattr(data, attr_name):
                value = getattr(data, attr_name)
                if isinstance(value, torch.Tensor):
                    print(f"{attr_name}: {value.detach().cpu().numpy().tolist()}")
                elif isinstance(value, np.ndarray):
                    print(f"{attr_name}: {value.tolist()}")
                else:
                    print(f"{attr_name}: {value}")
    else:
        print(repr(data))
