"""Bootstrap direct execution of tests without an editable installation."""

from __future__ import annotations

import sys
from pathlib import Path


def ensure_test_paths(current_file: str | Path) -> Path:
    repository_root = Path(current_file).resolve().parents[3]
    src_root = repository_root / "src"
    for path in (repository_root, src_root):
        value = str(path)
        if value not in sys.path:
            sys.path.insert(0, value)
    return repository_root
