"""Pose C++ LM reference contracts."""

from __future__ import annotations

import json
from pathlib import Path

from script.validation.staged_lm.export_pose_cpp_fixture import FLOAT_INPUTS


ROOT = Path("data_process/Vis/cpp_reference_dump/pose_lm/frame_000513")


def test_lm_fixture_has_no_trajectory_leakage():
    forbidden = {"delta", "candidate", "rho", "accepted", "final_parameters",
                 "H_frame", "g_frame", "H_active", "g_active"}
    assert all(not any(token in name for token in forbidden) for name in FLOAT_INPUTS)


def test_reference_trajectory_schema_and_objective_reuse():
    if not ROOT.is_dir(): return
    trajectory = json.loads((ROOT / "pose_lm_reference_trajectory.json").read_text())
    assert trajectory["objective_call_count"] == 1 + trajectory["candidate_trial_count"]
    assert trajectory["normal_equation_call_count"] == 1 + trajectory["accepted_steps"]
    assert trajectory["solve_call_count"] == len(trajectory["trials"])
    assert [trial["accepted"] for trial in trajectory["trials"]] == [True, True, True, False, True, True]
    assert trajectory["status"] == "MAX_ACCEPTED_STEPS"


def test_cpp_trajectory_branch_and_counter_parity():
    path = ROOT / "cpp_output/pose_lm_cpp_parity.json"
    if not path.is_file(): return
    result = json.loads(path.read_text())
    assert result["LM_FIRST_DIVERGENCE"] == "NONE"
    assert all(result["branch_parity"].values())
    assert all(result["counter_parity"].values())
    assert result["MIN_RHO_BRANCH_MARGIN"] > result["max_rho_difference"]
    assert result["POSE_LM_CPP_CPU_REFERENCE_READY"]


def test_lm_config_and_dtype_manifest():
    path = ROOT / "pose_lm_reference_manifest.json"
    if not path.is_file(): return
    manifest = json.loads(path.read_text())
    assert manifest["max_accepted_steps"] == 5
    assert manifest["solver_config"]["lm_max_trials"] == 8
    assert manifest["candidate_dtype"] == "torch.float32"
    assert manifest["normal_equation_dtype"] == "torch.float64"
    assert manifest["reference_leakage"] == "NONE"
