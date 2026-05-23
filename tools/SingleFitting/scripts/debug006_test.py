import pickle
import numpy as np
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


def _short_repr(value, max_len: int = 120) -> str:
    text = repr(value)
    if len(text) > max_len:
        return text[: max_len - 3] + "..."
    return text


def _print_pkl_node(name, value, indent: int = 0, max_depth: int = 3):
    prefix = " " * indent
    value_type = type(value).__name__

    if torch is not None and isinstance(value, torch.Tensor):
        print(
            f"{prefix}{name}: torch.Tensor "
            f"shape={tuple(value.shape)} dtype={value.dtype} device={value.device}"
        )
        if value.numel() > 0 and indent < max_depth * 2:
            flat = value.detach().cpu().reshape(-1)
            sample = flat[: min(8, flat.numel())].tolist()
            print(f"{prefix}  sample={sample}")
        return

    if isinstance(value, np.ndarray):
        print(f"{prefix}{name}: np.ndarray shape={value.shape} dtype={value.dtype}")
        if value.size > 0 and indent < max_depth * 2:
            flat = value.reshape(-1)
            sample = flat[: min(8, flat.size)].tolist()
            print(f"{prefix}  sample={sample}")
        return

    if isinstance(value, dict):
        print(f"{prefix}{name}: dict(len={len(value)})")
        if indent >= max_depth * 2:
            return
        for key, item in value.items():
            _print_pkl_node(str(key), item, indent + 2, max_depth)
        return

    if isinstance(value, (list, tuple)):
        print(f"{prefix}{name}: {value_type}(len={len(value)})")
        if indent >= max_depth * 2:
            return
        for idx, item in enumerate(value[:10]):
            _print_pkl_node(f"[{idx}]", item, indent + 2, max_depth)
        if len(value) > 10:
            print(f"{prefix}  ... ({len(value) - 10} more items)")
        return

    if hasattr(value, "__dict__"):
        attrs = vars(value)
        print(f"{prefix}{name}: {value_type} attrs={list(attrs.keys())}")
        if indent >= max_depth * 2:
            return
        for key, item in attrs.items():
            _print_pkl_node(key, item, indent + 2, max_depth)
        return

    print(f"{prefix}{name}: {value_type} value={_short_repr(value)}")


def print_pkl_contents(pkl_path, *, max_depth: int = 3):
    pkl_path = Path(pkl_path)
    with pkl_path.open("rb") as f:
        data = pickle.load(f)

    print(f"PKL path: {pkl_path}")
    _print_pkl_node("root", data, max_depth=max_depth)
    return data


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python debug006_test.py <pkl_path> [max_depth]")
        raise SystemExit(1)

    depth = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    print_pkl_contents(sys.argv[1], max_depth=depth)
