#!/usr/bin/env python3
from __future__ import annotations

if __package__ in (None, ""):
    from _bootstrap import ensure_test_paths
    ensure_test_paths(__file__)

import tempfile
from pathlib import Path

import numpy as np

from lib_pano.human3d.sam3dbody_initialization import (
    BodyPoseCompatibility, audit_crop_projection, basis_diagnostics,
    classify_body_pose, orthonormal_basis_from_rays,
    validate_smplx_beta_candidate, validate_smplx_pose_candidate,
)
from lib_pano.human3d.sam3dbody_io import Sam3DBodyRecon, load_sam3dbody_recon, save_sam3dbody_recon
from lib_pano.human3d.sam3dbody_parameter_schema import capture_raw_schema


def _recon(**extra):
    vertices=np.array([[0.0,0.0,1.0],[0.1,0.2,2.0]],np.float32);cam=np.array([0,0,2],np.float32)
    from lib_pano.human3d.sam3dbody_project import project_vertices_opencv_style
    uv,vc,vz=project_vertices_opencv_style(vertices,cam,100.0,(100,200,3))
    values=dict(image_path="x",cam="cam2",frame_idx=513,image_shape=(100,200,3),bbox=np.arange(4,dtype=np.float32),
        focal_length=100.0,pred_vertices=vertices,pred_cam_t=cam,vertices_cam=vc,vertices_2d=uv,
        valid_z=vz,inside=np.ones(2,bool));values.update(extra);return Sam3DBodyRecon(**values)


def test_backward_compatible_and_optional_roundtrip():
    with tempfile.TemporaryDirectory() as directory:
        old=Path(directory)/"old.npz";r=_recon()
        np.savez(old,image_path=r.image_path,cam=r.cam,frame_idx=r.frame_idx,image_shape=r.image_shape,bbox=r.bbox,
            focal_length=r.focal_length,pred_vertices=r.pred_vertices,pred_cam_t=r.pred_cam_t,
            vertices_cam=r.vertices_cam,vertices_2d=r.vertices_2d,valid_z=r.valid_z,inside=r.inside)
        loaded=load_sam3dbody_recon(old);assert loaded.global_rot is None and loaded.raw_parameter_schema_version==0
        new=Path(directory)/"new.npz";save_sam3dbody_recon(new,_recon(global_rot=np.zeros(3,np.float32),
            body_pose_params=np.zeros(133,np.float32),shape_params=np.zeros(45,np.float32),
            raw_parameter_schema_version=1,raw_parameter_source="test"))
        loaded=load_sam3dbody_recon(new);assert loaded.body_pose_params.shape==(133,) and loaded.shape_params.shape==(45,)


def test_schema_and_no_objects():
    schema=capture_raw_schema({"a":np.zeros((2,3),np.float32),"bad":object(),"scalar":2})
    assert schema["all_keys"]==["a","bad","scalar"]
    assert not next(x for x in schema["fields"] if x["key"]=="bad")["safely_serializable"]


def test_mhr_pose_and_shape_are_not_sliced():
    assert classify_body_pose(np.zeros(133)) is BodyPoseCompatibility.MODEL_SPECIFIC_LATENT_NOT_DIRECTLY_COMPATIBLE
    for fn,value in ((validate_smplx_pose_candidate,np.zeros(133)),(validate_smplx_beta_candidate,np.zeros(45))):
        try:fn(value)
        except ValueError:pass
        else:raise AssertionError("incompatible MHR parameter unexpectedly accepted")


def test_projection_and_basis():
    closure=audit_crop_projection(_recon());assert closure.passed
    basis=orthonormal_basis_from_rays(np.array([0,0,1.]),np.array([1e-3,0,1.]),np.array([0,1e-3,1.]))
    diag=basis_diagnostics(basis);assert diag["passed"] and abs(diag["determinant"]-1)<1e-10


def main():
    test_backward_compatible_and_optional_roundtrip();test_schema_and_no_objects()
    test_mhr_pose_and_shape_are_not_sliced();test_projection_and_basis();print("[PASS] SAM3DBody initialization unit tests")


if __name__=="__main__":main()
