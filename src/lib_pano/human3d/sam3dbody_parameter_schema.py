"""Safe, explicit schema capture for SAM 3D Body estimator outputs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np


EXACT_PARAMETER_FIELDS = (
    "global_rot", "body_pose_params", "shape_params", "pred_cam_t",
    "pred_vertices", "focal_length", "bbox",
)
PARAMETER_SYNONYMS = (
    "global_orient", "body_pose", "pred_pose", "pred_shape", "betas",
    "shape", "pose", "camera_translation", "cam_t",
)


@dataclass(frozen=True)
class RawFieldSchema:
    key: str
    python_type: str
    shape: tuple[int, ...] | None
    dtype: str | None
    device: str | None
    scalar_value: bool | int | float | str | None
    safely_serializable: bool

    def as_dict(self) -> dict[str, Any]:
        result = dict(self.__dict__)
        result["shape"] = list(self.shape) if self.shape is not None else None
        return result


def _as_numeric_array(value: Any) -> np.ndarray | None:
    if isinstance(value, np.ndarray):
        array = value
    elif hasattr(value, "detach") and hasattr(value, "cpu"):
        array = value.detach().cpu().numpy()
    elif isinstance(value, (bool, int, float, np.number)):
        array = np.asarray(value)
    else:
        return None
    return array if array.dtype.kind in "biufc" else None


def capture_raw_schema(values: Mapping[str, Any]) -> dict[str, Any]:
    fields: list[dict[str, Any]] = []
    for key in values.keys():
        value = values[key]
        array = _as_numeric_array(value)
        shape = tuple(int(v) for v in array.shape) if array is not None else None
        dtype = str(array.dtype) if array is not None else None
        device = str(getattr(value, "device", "cpu")) if array is not None else None
        scalar = None
        if array is not None and array.ndim == 0:
            scalar = array.item()
        elif isinstance(value, str):
            scalar = value
        fields.append(RawFieldSchema(
            key=str(key), python_type=type(value).__name__, shape=shape,
            dtype=dtype, device=device, scalar_value=scalar,
            safely_serializable=array is not None or isinstance(value, str),
        ).as_dict())
    keys = [str(k) for k in values.keys()]
    return {
        "fields": fields,
        "all_keys": keys,
        "exact_fields": {name: name in values for name in EXACT_PARAMETER_FIELDS},
        "synonym_candidates": [name for name in PARAMETER_SYNONYMS if name in values],
    }


def extract_optional_numeric(values: Mapping[str, Any], name: str) -> np.ndarray | None:
    if name not in values:
        return None
    array = _as_numeric_array(values[name])
    if array is None:
        raise TypeError(f"SAM 3D Body field {name!r} is not safely numeric")
    return array.copy()

