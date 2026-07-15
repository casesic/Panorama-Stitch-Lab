"""Minimal helpers for directly executing scripts from the source tree."""

from __future__ import annotations

import sys
from pathlib import Path


def ensure_src_on_path(current_file: str | Path) -> Path:
    """Add the repository ``src`` directory to ``sys.path`` and return the root."""
    path = Path(current_file).resolve()
    src_root = next(parent for parent in path.parents if parent.name == "src")
    value = str(src_root)
    if value not in sys.path:
        sys.path.insert(0, value)
    return src_root.parent
