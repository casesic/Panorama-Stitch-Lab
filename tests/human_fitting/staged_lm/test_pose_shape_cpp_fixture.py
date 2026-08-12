"""Pose/Shape C++ value fixture and parity contracts."""
import json
from pathlib import Path
from script.validation.staged_lm.export_pose_shape_cpp_fixture import FLOATS
ROOT=Path("data_process/Vis/cpp_reference_dump/pose_shape/frame_000513")
def test_fixture_no_leakage():
 forbidden=("physical_beta","dynamic_joints","dynamic_v_shaped","rotations","vertices","uv","residual","objective")
 assert not any(any(x in name for x in forbidden) for name in FLOATS)
def test_layout_and_basis():
 if not ROOT.is_dir():return
 m=json.loads((ROOT/"pose_shape_reference_manifest.json").read_text());assert m["parameter_dim"]==85 and m["frame_parameter_dim"]==75 and m["beta_dim"]==10;assert m["beta_latent_slice"]==[75,85];assert m["arrays"]["d_joints_d_beta"]["shape"]==[55,3,10];assert m["reference_leakage"]=="NONE"
def test_checker_ready_and_shared_joints():
 p=ROOT/"cpp_output/pose_shape_cpp_parity.json"
 if not p.is_file():return
 r=json.loads(p.read_text());assert r["FIRST_DIVERGENCE"]=="NONE";assert r["dynamic_joints_shared"];assert r["POSE_SHAPE_VALUE_CPP_CPU_REFERENCE_READY"]
def test_objective_schema():
 if not ROOT.is_dir():return
 m=json.loads((ROOT/"pose_shape_reference_manifest.json").read_text());assert all(f"objective_{x}" in m["arrays"] for x in ("primary","pose_anchor","shape_anchor","total"))
