import json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[3];REF=ROOT/"data_process/Vis/cpp_reference_dump/pose_shape/frame_000513"
def test_ne_required_keys_and_dtypes():
 with np.load(REF/"pose_shape_reference.npz") as a:
  for c in ("cam2","cam4"):
   for k in ("point_weights","row_scales","weighted_residual","weighted_J_active","weighted_J_beta","H_aa","H_ab","H_bb","g_a","g_b"):assert f"{c}_{k}" in a
  assert a["weighted_J_beta" if "weighted_J_beta" in a else "cam2_weighted_J_beta"].dtype==np.float32
  assert a["A_final"].dtype==np.float64 and a["B_final"].shape==(75,10) and a["C_final"].shape==(10,10)
def test_fixture_has_no_ne_leakage():
 forbidden=("point_weights","row_scales","weighted_","H_aa","H_ab","H_bb","A_data","B_data","C_data","gradient")
 assert not [p.name for p in (REF/"cpp_fixture").iterdir() if any(x in p.name for x in forbidden)]
def test_scatter_dense_and_global_beta():
 with np.load(REF/"pose_shape_reference.npz") as a:
  assert not np.any(a["cam2_B_scatter"][6:12]);assert not np.any(a["cam4_B_scatter"][:6])
  assert np.array_equal(a["H_dense_structured"][:75,75:],a["B_final"]);assert np.array_equal(a["H_dense_structured"][75:,:75],a["B_final"].T)
  assert np.allclose(a["H_dense_structured"],a["H_dense_reference"],rtol=1e-12,atol=1e-12)
def test_checker_policy_and_first_divergence():
 d=json.loads((REF/"cpp_normal_equations_output/pose_shape_ne_cpp_parity.json").read_text());assert d["FIRST_DIVERGENCE"]=="NONE";assert d["POSE_SHAPE_NORMAL_EQUATIONS_CPP_CPU_REFERENCE_READY"]=="PASS";assert "float32" in d["dtype_boundary"]
