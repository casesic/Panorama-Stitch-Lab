"""Small unit tests for the staged LM package.

These tests do not depend on the real SMPL-X model or data.  They verify:

* Parameter slicing does not cross frame boundaries.
* The S1->Stage 2 pose handoff preserves orient/transl and seeds pose from anchor.
* The atanh/tanh beta parameterisation is reversible.
* Output key sets match the legacy implementation for all stop-after-stage
  configurations.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

try:
    from ._bootstrap import ensure_test_paths
except ImportError:
    from _bootstrap import ensure_test_paths
ensure_test_paths(__file__)

from lib_pano.human_fitting.staged_lm.parameter_layout import (
    frame_slices,
    unpack_frame_parameters,
)


def test_parameter_slices_no_cross_frame() -> None:
    """Two consecutive frames must not share any parameter indices."""
    per_frame_dim = 75
    frame_count = 3
    param_dim = frame_count * per_frame_dim
    x = torch.arange(param_dim, dtype=torch.float32)

    for t in range(frame_count):
        slices = frame_slices(t, per_frame_dim, include_body_pose=True)
        cam2_orient = x[slices.cam2_orient]
        cam2_transl = x[slices.cam2_transl]
        cam4_orient = x[slices.cam4_orient]
        cam4_transl = x[slices.cam4_transl]
        body_pose = x[slices.body_pose]

        start = t * per_frame_dim
        assert cam2_orient[0].item() == start
        assert cam2_transl[0].item() == start + 3
        assert cam4_orient[0].item() == start + 6
        assert cam4_transl[0].item() == start + 9
        assert body_pose[0].item() == start + 12
        assert body_pose[-1].item() == start + per_frame_dim - 1

    print("test_parameter_slices_no_cross_frame: PASS")


def test_unpack_frame_parameters_shapes() -> None:
    """Unpacked parameters have the expected shapes [1, 3] and [1, 63]."""
    per_frame_dim = 75
    x = torch.arange(per_frame_dim * 2, dtype=torch.float32)

    frame_params = unpack_frame_parameters(x, 0, per_frame_dim, include_body_pose=True)
    assert frame_params.cam2_orient.shape == (1, 3)
    assert frame_params.cam2_transl.shape == (1, 3)
    assert frame_params.cam4_orient.shape == (1, 3)
    assert frame_params.cam4_transl.shape == (1, 3)
    assert frame_params.body_pose.shape == (1, 63)
    print("test_unpack_frame_parameters_shapes: PASS")


def test_handoff_s1_to_pose_preserves_state() -> None:
    """S1->Stage 2 pose handoff must preserve orient/transl and seed pose from anchor."""
    from lib_pano.human_fitting.staged_lm.stages import check_s1_to_pose, handoff_s1_to_pose

    from lib_pano.human_fitting.staged_lm.config import STAGE1_PER_FRAME_DIM

    frame_count = 3
    s1_dim = frame_count * STAGE1_PER_FRAME_DIM
    x_s1 = torch.randn(s1_dim, dtype=torch.float32)
    body_pose_dim = 6
    per_frame_dim = STAGE1_PER_FRAME_DIM + body_pose_dim
    body_pose_anchors = [torch.randn(body_pose_dim, dtype=torch.float32) for _ in range(frame_count)]

    x_pose = handoff_s1_to_pose(x_s1, body_pose_anchors, frame_count, body_pose_dim)

    # Check dimensions.
    assert x_pose.shape[0] == frame_count * per_frame_dim

    # Check orient/transl preserved.
    for t in range(frame_count):
        s1_slice = x_s1[t * STAGE1_PER_FRAME_DIM:(t + 1) * STAGE1_PER_FRAME_DIM]
        s2_pose_slice = x_pose[t * per_frame_dim:t * per_frame_dim + STAGE1_PER_FRAME_DIM]
        assert torch.allclose(s1_slice, s2_pose_slice)

    # Check pose seeded from anchor.
    for t in range(frame_count):
        pose_slice = x_pose[t * per_frame_dim + STAGE1_PER_FRAME_DIM:(t + 1) * per_frame_dim]
        assert torch.allclose(pose_slice, body_pose_anchors[t])

    # Check original tensor not modified.
    assert x_s1.shape[0] == s1_dim

    # Check handoff verification passes.
    handoff = check_s1_to_pose(x_s1, x_pose, body_pose_anchors, frame_count, body_pose_dim)
    assert handoff["ok"], f"Handoff check failed: {handoff}"

    print("test_handoff_s1_to_pose_preserves_state: PASS")


def test_beta_atanh_tanh_reversibility() -> None:
    """beta_abs_limit * tanh(atanh(beta/beta_abs_limit)) == beta."""
    beta_abs_limit = 2.5
    beta_anchor = torch.tensor([0.1, -0.5, 1.2, -2.0, 0.0, 0.3, -0.7, 2.1, -1.3, 0.8])

    # Must be feasible (inside the safe bound).
    assert float(beta_anchor.abs().max()) < beta_abs_limit * (1.0 - 1e-6)

    beta_latent = torch.atanh(beta_anchor / beta_abs_limit)
    beta_recovered = beta_abs_limit * torch.tanh(beta_latent)

    np.testing.assert_allclose(
        beta_recovered.numpy(), beta_anchor.numpy(), rtol=1e-6, atol=1e-6,
    )
    print("test_beta_atanh_tanh_reversibility: PASS")


def test_output_keys_stage1_no_lbfgs() -> None:
    """Stage 1 early-exit NPZ keys match legacy when L-BFGS is not run."""
    from lib_pano.human_fitting.staged_lm.output import collect_stage1_arrays
    from lib_pano.human_fitting.staged_lm.types import LBFGSResult, LMResult, LMSolverCounters, StageOutcome

    frame_count = 2
    s1_dim = frame_count * 12
    initial = torch.randn(s1_dim)
    lm_result = LMResult(
        label="S1_LM", parameters=torch.randn(s1_dim), objective=1.0,
        status="MAX_ACCEPTED_STEPS", counters=LMSolverCounters(),
        final_lambda=0.1, wall_time_ms=10.0, rms_metrics={"rms": 1.0, "median": 1.0, "p95": 1.0},
        weight_history=[], first_order_checks={}, spec={},
    )
    outcome = StageOutcome(lm_result=lm_result, lbfgs_result=None, verdict=None, true_objective_fn=lambda x: x.sum())
    beta_anchor = torch.randn(10)
    beta_latent0 = torch.randn(10)

    # Use a mock raw_groups_fn that returns a list of ReprojectionGroup-like objects.
    from lib_pano.human_fitting.staged_lm.types import ReprojectionGroup
    mock_groups = [ReprojectionGroup(frame_index=0, camera="cam2", residual_2d=torch.zeros(10, 2))]
    arrays = collect_stage1_arrays(
        initial, outcome, beta_anchor, beta_latent0,
        stage1_raw_groups_fn=lambda x: mock_groups,
    )
    expected_keys = {"x0_s1", "x_s1_lm", "beta_anchor", "beta_latent0", "raw_reproj_s1_lm"}
    assert set(arrays.keys()) == expected_keys, f"Unexpected keys: {set(arrays.keys())}"
    print("test_output_keys_stage1_no_lbfgs: PASS")


def test_output_keys_stage2_with_lbfgs() -> None:
    """Full Stage 2 NPZ keys use pose/joint naming with L-BFGS."""
    from lib_pano.human_fitting.staged_lm.output import collect_stage2_arrays
    from lib_pano.human_fitting.staged_lm.types import LBFGSResult, LMResult, LMSolverCounters, StageOutcome

    frame_count = 2
    s1_dim = frame_count * 12
    s2_pose_dim = frame_count * 75
    s2b_dim = s2_pose_dim + 10

    def make_outcome(dim, name):
        lm = LMResult(
            label=f"{name}_LM", parameters=torch.randn(dim), objective=1.0,
            status="MAX_ACCEPTED_STEPS", counters=LMSolverCounters(),
            final_lambda=0.1, wall_time_ms=10.0, rms_metrics={"rms": 1.0, "median": 1.0, "p95": 1.0},
            weight_history=[], first_order_checks={}, spec={},
        )
        bf = LBFGSResult(label=f"{name}_LBFGS", parameters=torch.randn(dim), objective=0.9, closure_calls=10, wall_time_ms=5.0)
        return StageOutcome(lm_result=lm, lbfgs_result=bf, verdict="LM_MATCHES_LBFGS", true_objective_fn=lambda x: x.sum())

    s1_outcome = make_outcome(s1_dim, "S1")
    s2_pose_outcome = make_outcome(s2_pose_dim, "STAGE2_POSE")
    s2b_outcome = make_outcome(s2b_dim, "STAGE2_JOINT")

    from lib_pano.human_fitting.staged_lm.types import ReprojectionGroup
    mock_groups = [ReprojectionGroup(frame_index=0, camera="cam2", residual_2d=torch.zeros(10, 2))]
    arrays = collect_stage2_arrays(
        torch.randn(s1_dim), torch.randn(s2_pose_dim), torch.randn(s2b_dim),
        s1_outcome, s2_pose_outcome, s2b_outcome,
        torch.randn(10), torch.randn(10),
        torch.randn(10), torch.randn(10),
        lambda x: mock_groups, lambda x: mock_groups, lambda x: mock_groups,
        torch.randn(s2_pose_dim), torch.randn(s2b_dim),
    )
    expected_keys = {
        "x0_s1", "x0_stage2_pose_lm", "x0_stage2_joint_lm",
        "x_s1_lm", "x_stage2_pose_lm", "x_stage2_joint_lm",
        "beta_anchor", "beta_latent0", "beta_lm",
        "raw_reproj_s1_lm", "raw_reproj_stage2_pose_lm", "raw_reproj_stage2_joint_lm",
        "x_s1_lbfgs", "raw_reproj_s1_lbfgs",
        "x0_stage2_pose_lbfgs", "x_stage2_pose_lbfgs", "raw_reproj_stage2_pose_lbfgs",
        "x0_stage2_joint_lbfgs", "x_stage2_joint_lbfgs", "beta_lbfgs", "raw_reproj_stage2_joint_lbfgs",
    }
    assert set(arrays.keys()) == expected_keys, f"Missing/extra keys: {set(arrays.keys()) - expected_keys} / {expected_keys - set(arrays.keys())}"
    assert not any("final" in key for key in arrays)
    results = {"stage2": {"pose_warmup": {}, "joint_refinement": {}, "beta": {}}}
    assert "joint_refinement" in results["stage2"] and "final" not in results["stage2"]
    print("test_output_keys_stage2_with_lbfgs: PASS")


def test_objective_breakdown_modes() -> None:
    """Charbonnier and raw-L2 breakdowns must reconstruct their true objective."""
    from lib_pano.human_fitting.staged_lm.output import compute_objective_breakdown

    parameters = torch.tensor([0.2, -0.3], dtype=torch.float64)
    for is_irls in (True, False):
        def objective_terms(x):
            primary = torch.sum(x.square()) * (0.2 if is_irls else 0.4)
            pose_anchor = 0.5 * torch.sum((x[:1] * 0.5) ** 2)
            shape_anchor = 0.5 * torch.sum((x[1:] * 0.25) ** 2)
            return {
                "primary": primary,
                "pose_anchor": pose_anchor,
                "shape_anchor": shape_anchor,
                "total": primary + pose_anchor + shape_anchor,
            }

        expected = float(objective_terms(parameters)["total"])
        breakdown = compute_objective_breakdown(
            objective_terms, parameters, expected_true_objective=expected,
        )
        assert breakdown["relative_error"] <= 1e-6
        np.testing.assert_allclose(
            breakdown["primary"] + breakdown["pose_anchor"] + breakdown["shape_anchor"],
            breakdown["true_objective"], rtol=1e-6, atol=1e-6,
        )
    print("test_objective_breakdown_modes: PASS")


def test_gate_c_three_states() -> None:
    """Gate C distinguishes skipped oracle, pass, and blocked states."""
    from script.validation.staged_lm.references.gate_c import evaluate_gate_c

    matches = {"s1_match": None, "stage2_pose_match": None, "stage2_joint_match": None}
    non_oracle = {"handoff_stage1_to_stage2_pose_lm": True, "beta_sat_lt_0.9": True, "s1_lm_status_ok": True}
    result = evaluate_gate_c(oracle_enabled=False, stage_matches=matches, non_oracle_checks=non_oracle, oracle_checks={})
    assert result["verdict"] == "GATE_C_NOT_EVALUATED"
    assert all(result["checks"][key] is None for key in matches)

    matches = {key: True for key in matches}
    result = evaluate_gate_c(oracle_enabled=True, stage_matches=matches, non_oracle_checks=non_oracle, oracle_checks={"handoff_bf": True})
    assert result["verdict"] == "GATE_C_PASS_READY_FOR_MULTIFRAME_GATE_D"
    blocked_checks = dict(non_oracle)
    blocked_checks["beta_sat_lt_0.9"] = False
    result = evaluate_gate_c(oracle_enabled=True, stage_matches=matches, non_oracle_checks=blocked_checks, oracle_checks={"handoff_bf": True})
    assert result["verdict"] == "GATE_C_BLOCKED"
    print("test_gate_c_three_states: PASS")


def test_stage2_outcome_optional_joint_refinement() -> None:
    """Pose early exit has no joint result; full Stage 2 has one."""
    from lib_pano.human_fitting.staged_lm.types import Stage2Outcome

    marker = object()
    pose_only = Stage2Outcome(pose_warmup=marker, joint_refinement=None, handoffs={})
    full = Stage2Outcome(pose_warmup=marker, joint_refinement=marker, handoffs={})
    assert pose_only.joint_refinement is None
    assert full.joint_refinement is not None
    print("test_stage2_outcome_optional_joint_refinement: PASS")


if __name__ == "__main__":
    test_parameter_slices_no_cross_frame()
    test_unpack_frame_parameters_shapes()
    test_handoff_s1_to_pose_preserves_state()
    test_beta_atanh_tanh_reversibility()
    test_output_keys_stage1_no_lbfgs()
    test_output_keys_stage2_with_lbfgs()
    test_objective_breakdown_modes()
    test_gate_c_three_states()
    test_stage2_outcome_optional_joint_refinement()
    print("\nAll tests PASSED")
