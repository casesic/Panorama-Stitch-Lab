"""Single import boundary for the externally delivered ``densekp_delivery`` package.

The DenseKP delivery package (detector/ROI/DenseKP runtime) is an external
delivered artifact under ``Circular_Video/01_recv/<date>/``.  Production code
must import it only through this module so that exactly one place knows where
the delivery lives and no new ``sys.path`` edits appear elsewhere.

If a future delivery version lands at a different date directory, point the
``DENSEKP_DELIVERY_ROOT`` environment variable at it instead of editing source.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DELIVERY_ROOT = PROJECT_ROOT / "Circular_Video/01_recv/20260708"


def delivery_root() -> Path:
    override = os.environ.get("DENSEKP_DELIVERY_ROOT")
    root = Path(override).expanduser().resolve() if override else DEFAULT_DELIVERY_ROOT
    if not (root / "densekp_delivery").is_dir():
        raise FileNotFoundError(
            f"densekp_delivery not found under {root}; set DENSEKP_DELIVERY_ROOT"
        )
    return root


def ensure_importable() -> None:
    path = str(delivery_root())
    if path not in sys.path:
        sys.path.insert(0, path)


def densekp_delivery():
    ensure_importable()
    import densekp_delivery

    return densekp_delivery
