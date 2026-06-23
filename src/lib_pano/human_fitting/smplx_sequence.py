"""对 lib_dynamic_hvip (tools/SingleFitting) 的 SMPL/SMPLX 前向模块与序列参数类的统一 re-export 层。"""

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
from lib_dynamic_hvip.fitting.smpl_seq_forward_module import (
    SMPL_Seq_Forward_Module,
)
from lib_dynamic_hvip.smpl.smpl_lib import (
    forward_smplx_tensor_batch,
    get_smplx_batch,
)

