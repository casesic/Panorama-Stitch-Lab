from __future__ import annotations

import numpy as np


def load_smplx_model(*, gender: str = "neutral"):
    from flow.visibility import get_smplx_batch

    return get_smplx_batch(gender=gender)


def get_smplx_faces_from_model(smplx_model) -> np.ndarray:
    faces = smplx_model.faces
    if hasattr(faces, "detach"):
        faces = faces.detach().cpu().numpy()
    elif hasattr(faces, "numpy"):
        faces = faces.numpy()
    return np.asarray(faces, dtype=int)


def load_smplx_faces(smplx_model=None, *, gender: str = "neutral") -> np.ndarray:
    model = smplx_model if smplx_model is not None else load_smplx_model(gender=gender)
    return get_smplx_faces_from_model(model)
