import sys
from pathlib import Path

_sf_root = Path(__file__).resolve().parents[3] / "tools" / "SingleFitting"
if str(_sf_root) not in sys.path:
    sys.path.insert(0, str(_sf_root))

from lib_dynamic_hvip.fitting.smplx_seq_forward_module import (
    SMPLX_ALL,
    SMPLX_BODY_22,
    SMPLX_JOINT_MAP,
    SMPLX_LHAND_15,
    SMPLX_RHAND_15,
    SMPLX_Seq_Forward_Module,
)
