"""Thin adapter around Meta's official MHR mesh -> SMPL-X PyTorch fitter.

No fitting math is implemented here.  Existing SAM recon vertices are passed
through the official topology mapping and ``Conversion.convert_mhr2smpl``.
"""

from __future__ import annotations

import contextlib
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterator

import numpy as np
import torch

from .sam3dbody_io import load_sam3dbody_recon


CONVERSION_SCHEMA_VERSION = 1


@contextlib.contextmanager
def _conversion_import_context(conversion_dir: Path) -> Iterator[None]:
    old_cwd = Path.cwd(); old_path = list(sys.path)
    os.chdir(conversion_dir)
    sys.path.insert(0, str(conversion_dir))
    try:
        yield
    finally:
        os.chdir(old_cwd); sys.path[:] = old_path


class _MeshOnlyMHRAdapter:
    """Only the face topology used by official MHR->SMPL conversion."""
    def __init__(self, faces: np.ndarray):
        self.character = SimpleNamespace(mesh=SimpleNamespace(faces=np.asarray(faces, dtype=np.int64)))

    def to(self, device: str | torch.device) -> "_MeshOnlyMHRAdapter":
        return self


@dataclass(frozen=True)
class OfficialConversionRecord:
    output_path: Path
    conversion_ms: float
    vertex_error_max: float
    vertex_error_rms: float
    roundtrip_max: float
    roundtrip_rms: float


def _percentile(values: np.ndarray, q: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def convert_recon_to_smplx_official(
    recon_path: str | Path,
    output_path: str | Path,
    *,
    conversion_dir: str | Path,
    model_path: str | Path,
    mapping_path: str | Path,
    device: str = "cuda:0",
    gender: str = "neutral",
) -> OfficialConversionRecord:
    recon_path=Path(recon_path).resolve();output_path=Path(output_path).resolve();conversion_dir=Path(conversion_dir).resolve()
    model_path=Path(model_path).resolve();mapping_path=Path(mapping_path).resolve()
    recon=load_sam3dbody_recon(recon_path)
    if recon.faces is None: raise ValueError("SAM recon must contain MHR faces")
    with _conversion_import_context(conversion_dir):
        import smplx
        from conversion import Conversion
        smpl_model=smplx.SMPLX(model_path=str(model_path),gender=gender,ext="npz",
            num_betas=10,use_pca=False,flat_hand_mean=True,batch_size=1).to(device)
        converter=Conversion(_MeshOnlyMHRAdapter(recon.faces),smpl_model,method="pytorch",batch_size=1)
        # Official SAM adapter formula. Conversion expects MHR centimeters.
        mhr_vertices=100.0*(recon.pred_vertices.astype(np.float32)+recon.pred_cam_t.astype(np.float32)[None,:])
        started=time.perf_counter()
        result=converter.convert_mhr2smpl(mhr_vertices=mhr_vertices[None],single_identity=False,
            is_tracking=False,return_smpl_meshes=False,return_smpl_parameters=True,
            return_smpl_vertices=True,return_fitting_errors=True,batch_size=1)
        if str(device).startswith("cuda"): torch.cuda.synchronize(torch.device(device))
        conversion_ms=(time.perf_counter()-started)*1000.0
        params={k:v.detach() for k,v in result.result_parameters.items()}
        required={"global_orient":3,"transl":3,"body_pose":63,"betas":10,
                  "left_hand_pose":45,"right_hand_pose":45,"expression":10}
        for name,width in required.items():
            if name not in params or tuple(params[name].shape)!=(1,width):
                raise ValueError(f"official {name} shape {getattr(params.get(name),'shape',None)} != (1,{width})")
        with torch.no_grad():
            forward=smpl_model(**params).vertices
        converted=np.asarray(result.result_vertices,dtype=np.float32)
        roundtrip=(forward.detach().cpu().numpy()-converted).astype(np.float64)
        with np.load(mapping_path,allow_pickle=False) as mapping:
            tri=mapping["triangle_ids"].astype(np.int64);bary=mapping["baryc_coords"].astype(np.float64)
        target=np.sum((mhr_vertices[recon.faces[tri]]*0.01)*bary[...,None],axis=1).astype(np.float32)[None]
        point_error=np.linalg.norm(converted.astype(np.float64)-target.astype(np.float64),axis=-1)[0]
        height=float(target[0,:,1].max()-target[0,:,1].min())
        diagonal=float(np.linalg.norm(target[0].max(0)-target[0].min(0)))
        output_path.parent.mkdir(parents=True,exist_ok=True)
        arrays={name:value[0].detach().cpu().numpy() for name,value in params.items()}
        np.savez_compressed(output_path,**arrays,converted_vertices=converted[0],mapped_mhr_target_vertices=target[0],
            vertex_error_per_point=point_error.astype(np.float32),vertex_error_max=np.float64(point_error.max()),
            vertex_error_rms=np.float64(np.sqrt(np.mean(point_error**2))),vertex_error_p50=np.float64(_percentile(point_error,50)),
            vertex_error_p95=np.float64(_percentile(point_error,95)),normalized_rms_height=np.float64(np.sqrt(np.mean(point_error**2))/max(height,1e-12)),
            normalized_rms_bbox_diagonal=np.float64(np.sqrt(np.mean(point_error**2))/max(diagonal,1e-12)),
            roundtrip_max=np.float64(np.max(np.abs(roundtrip))),roundtrip_rms=np.float64(np.sqrt(np.mean(roundtrip**2))),
            source_recon=str(recon_path),source_frame=np.int32(recon.frame_idx),source_camera=recon.cam,
            backend="official_pytorch",model_type="smplx",gender=gender,mapping_path=str(mapping_path),
            model_path=str(model_path),conversion_ms=np.float64(conversion_ms),conversion_schema_version=np.int32(CONVERSION_SCHEMA_VERSION),
            official_result_errors=np.asarray(result.result_errors))
    return OfficialConversionRecord(output_path,conversion_ms,float(point_error.max()),
        float(np.sqrt(np.mean(point_error**2))),float(np.max(np.abs(roundtrip))),float(np.sqrt(np.mean(roundtrip**2))))
