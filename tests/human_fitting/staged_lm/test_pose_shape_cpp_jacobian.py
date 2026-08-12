import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[3]
REF=ROOT/"data_process/Vis/cpp_reference_dump/pose_shape/frame_000513"

def test_pose_shape_combined_layout_and_keys():
    manifest=json.loads((REF/"pose_shape_reference_manifest.json").read_text())
    assert manifest["combined_derivative_dim"]==76
    assert manifest["pose_derivative_dim"]==66
    assert manifest["beta_dim"]==10
    assert manifest["combined_layout"]=={"pose":[0,66],"beta_latent":[66,76]}
    for camera in ("cam2","cam4"):
        assert manifest["arrays"][f"{camera}_J_active"]["shape"][-1]==69
        assert manifest["arrays"][f"{camera}_J_beta"]["shape"][-1]==10
        assert manifest["arrays"][f"{camera}_d_vertices_combined"]["shape"][-1]==76

def test_pose_shape_fixture_has_no_derivative_outputs():
    allowed={"d_joints_d_beta.f32"}
    leaked=[p.name for p in (REF/"cpp_fixture").iterdir()
            if (p.name.startswith("d_") or "jacobian" in p.name.lower()) and p.name not in allowed]
    assert leaked==[]

def test_pose_shape_jacobian_invariants_and_parity():
    with np.load(REF/"pose_shape_reference.npz") as a:
        assert np.array_equal(a["cam2_d_relative_joints_beta"],a["cam4_d_relative_joints_beta"])
        for camera in ("cam2","cam4"):
            rest=a[f"{camera}_d_rest_transforms_combined"]
            assert np.count_nonzero(rest[...,:66])==0
            assert np.array_equal(rest[:,:3,3,66:],-a["d_dynamic_joints_d_beta_latent"])
    parity=json.loads((REF/"cpp_jacobian_output/pose_shape_jacobian_cpp_parity.json").read_text())
    assert parity["FIRST_DIVERGENCE"]=="NONE"
    assert parity["POSE_SHAPE_JACOBIAN_CPP_CPU_REFERENCE_READY"]=="PASS"

def test_pose_shape_manifest_axes_and_production_parity():
    manifest=json.loads((REF/"pose_shape_reference_manifest.json").read_text())
    assert all(v["axes"]=="canonical_row_major" for v in manifest["arrays"].values())
    assert manifest["reference_leakage"]=="NONE"
    assert all(manifest["production_parity"].values())
