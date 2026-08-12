import csv,json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[3];REF=ROOT/"data_process/Vis/cpp_reference_dump/pose_shape_lm/frame_000513";FIX=ROOT/"data_process/Vis/cpp_reference_dump/pose_shape/frame_000513/cpp_fixture"
def test_trajectory_schema_and_true_handoff():
 t=json.loads((REF/"pose_shape_lm_reference_trajectory.json").read_text());assert t["schema_version"]=="pose_shape_lm_cpp_reference_v1";assert t["linearization_call_count"]==t["accepted_steps"]+1
 with np.load(REF/"pose_shape_lm_reference.npz") as a:assert a["initial_parameters"].shape==(85,);assert np.array_equal(a["initial_parameters"],np.fromfile(FIX/"parameters.f32","<f4"));assert a["final_beta_latent"].shape==(10,)
def test_fixture_has_no_trajectory_leakage():
 forbidden=("delta","candidate","objective","rho","accepted","final_parameters","trajectory","H_","g_")
 assert not [p.name for p in FIX.iterdir() if any(x in p.name for x in forbidden)]
def test_branch_counters_and_objective_reuse():
 t=json.loads((REF/"pose_shape_lm_reference_trajectory.json").read_text());rows=list(csv.DictReader((REF/"cpp_output/trajectory.tsv").open(),delimiter='\t'));assert [r['solve_backend'] for r in rows]==['schur'];assert [r['accepted'] for r in rows]==['1'];assert t['objective_call_count']==1+t['total_trials'];assert t['linearization_call_count']==1+t['accepted_steps']
def test_lm_checker_and_status():
 d=json.loads((REF/"cpp_output/pose_shape_lm_cpp_parity.json").read_text());assert d["LM_FIRST_DIVERGENCE"]=="NONE";assert d["POSE_SHAPE_LM_CPP_CPU_REFERENCE_READY"]=="PASS";assert all(d["branch_parity"].values()) and all(d["counter_parity"].values());assert not d["near_branch_boundary"]
