from __future__ import annotations

import sys
from pathlib import Path


def get_project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def normalize_path(path: str | Path) -> Path:
    return Path(path).expanduser()


def ensure_src_path(project_root: str | Path | None = None) -> Path:
    root = normalize_path(project_root) if project_root is not None else get_project_root()
    src_root = root / "src"
    if str(src_root) not in sys.path:
        sys.path.insert(0, str(src_root))
    return root


def ensure_single_fitting_path(project_root: str | Path | None = None) -> Path:
    root = ensure_src_path(project_root)
    sf_root = root / "tools" / "SingleFitting"
    if str(sf_root) not in sys.path:
        sys.path.insert(0, str(sf_root))
    return root
