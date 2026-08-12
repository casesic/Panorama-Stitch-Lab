import json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[3];REF=ROOT/"data_process/Vis/cpp_reference_dump/pose_shape_schur/frame_000513"
def test_solver_fixture_inputs_and_no_delta_leakage():
 files={p.name for p in (REF/"schur_solver_fixture").iterdir()};assert files=={"A.f64","B.f64","C.f64","g_frame.f64","g_beta.f64","fixture_meta.txt"};assert not any("delta" in x for x in files)
def test_manifest_layout_and_lambdas():
 m=json.loads((REF/"schur_reference_manifest.json").read_text());assert m["dimensions"]=={"T":1,"P":75,"B":10};assert len(m["lambda_cases"])==3;assert all(x["lambda"]>0 for x in m["lambda_cases"]);assert m["no_delta_reference_leakage"]
def test_python_structured_dense_and_blocks():
 with np.load(REF/"schur_reference.npz") as a:
  assert a["A"].shape==(75,75) and a["B"].shape==(75,10) and a["C"].shape==(10,10)
  for i in range(3):assert np.allclose(a[f"case{i}_full_delta"],a[f"case{i}_dense_delta"],rtol=1e-10,atol=1e-10)
def test_checker_schema_and_residuals():
 d=json.loads((REF/"cpp_kernel_output/pose_shape_schur_cpp_parity.json").read_text());assert d["SCHUR_KERNEL_FIRST_DIVERGENCE"]=="NONE";assert d["SCHUR_INTEGRATION_FIRST_DIVERGENCE"]=="NONE";assert d["POSE_SHAPE_SCHUR_CPP_CPU_REFERENCE_READY"]=="PASS";assert all(v["pass"] for k,v in d["kernel_metrics"].items() if "residual" in k)
