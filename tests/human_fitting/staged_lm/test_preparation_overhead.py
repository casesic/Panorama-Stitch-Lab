"""Regression tests for static preparation metadata and forward-call removal."""

from __future__ import annotations

import inspect
from types import SimpleNamespace
from unittest.mock import patch

import torch

try:
    from ._bootstrap import ensure_test_paths
except ImportError:
    from _bootstrap import ensure_test_paths
ensure_test_paths(__file__)

from tests.human_fitting.staged_lm.helpers import (  # noqa: E402
    synthetic_joint_cache,
    synthetic_joint_parameters,
)
from lib_pano.human_fitting.staged_lm.irls import (  # noqa: E402
    build_primary_aggregation_spec,
    build_primary_aggregation_spec_from_observations,
)
from lib_pano.human_fitting.staged_lm.output import stage_outcome_to_json  # noqa: E402
from script.validation.staged_lm.objective_breakdown import (  # noqa: E402
    compute_objective_breakdown,
)
from lib_pano.human_fitting.staged_lm.joint import (  # noqa: E402
    joint_fast_raw_groups,
    stage2_joint_fast_objective_terms,
    stage2_joint_fast_true_objective,
)
from lib_pano.human_fitting.staged_lm.stage_handoff import check_pose_to_joint_state  # noqa: E402
from lib_pano.human_fitting.staged_lm.stage_runtime import run_stage  # noqa: E402
from lib_pano.human_fitting.staged_lm.types import (  # noqa: E402
    LMResult,
    LMSolverCounters,
    ReprojectionGroup,
    StageDefinition,
    StageInputSummary,
    StageOutcome,
    StagePreparationTiming,
)


def _groups(frame_indices, cam2_counts, cam4_counts):
    result = []
    for frame, cam2_count, cam4_count in zip(
        frame_indices, cam2_counts, cam4_counts,
    ):
        result.append(ReprojectionGroup(
            frame, "cam2", torch.zeros(cam2_count, 2),
        ))
        result.append(ReprojectionGroup(
            frame, "cam4", torch.zeros(cam4_count, 2),
        ))
    return result


def test_static_aggregation_spec_matches_group_spec() -> None:
    cases = (
        ([513], [3], [2]),
        ([513, 519], [3, 0], [1, 4]),
        ([513, 701], [0, 5], [7, 0]),
    )
    for frames, cam2_counts, cam4_counts in cases:
        groups = _groups(frames, cam2_counts, cam4_counts)
        metadata = build_primary_aggregation_spec_from_observations(
            frames,
            [torch.arange(count) for count in cam2_counts],
            [torch.arange(count) for count in cam4_counts],
        )
        legacy = build_primary_aggregation_spec(groups)
        assert metadata == legacy
        assert list(metadata["terms"]) == [
            f"{frame}|{camera}"
            for frame in frames for camera in ("cam2", "cam4")
        ]


def _fake_result(parameters: torch.Tensor) -> LMResult:
    return LMResult(
        label="SYNTHETIC_LM", parameters=parameters, objective=0.0,
        status="CONVERGED_SCALED_GRADIENT", counters=LMSolverCounters(),
        final_lambda=0.0, wall_time_ms=0.0,
        rms_metrics={"rms": 0.0, "median": 0.0, "p95": 0.0},
        weight_history=[], first_order_checks={}, spec={},
    )


def _raising_raw_groups(_parameters):
    raise RuntimeError("UNEXPECTED_RAW_GROUP_CALL")


def test_run_stage_does_not_call_raw_groups() -> None:
    parameters = torch.zeros(2)
    summary = StageInputSummary(1, 2, 2, 5, 10, 0, 10)
    definition = StageDefinition(
        name="SYNTHETIC", parameter_dimension=2,
        initial_lm_parameters=parameters,
        raw_groups_fn=_raising_raw_groups, anchor_residual_fn=None,
        raw_metrics_fn=lambda _x: {"rms": 0.0, "median": 0.0, "p95": 0.0},
        lm_max_accepted_steps=1,
        linearization_builder=lambda _x, _spec: {},
        lm_true_objective_fn=lambda x: x.square().sum(),
        input_summary=summary,
    )
    context = SimpleNamespace(
        frame_count=1,
        args=SimpleNamespace(
            validate_device_contract=False, profile_timing=False,
        ),
        solver_config=SimpleNamespace(lm_max_trials=1),
    )
    with patch(
        "lib_pano.human_fitting.staged_lm.stage_runtime.validate_stage_initial_parameters"
    ), patch(
        "lib_pano.human_fitting.staged_lm.stage_runtime.run_lm_optimization",
        return_value=_fake_result(parameters),
    ):
        outcome = run_stage(definition, context)
    assert outcome.definition is definition


def test_stage_json_does_not_call_raw_groups() -> None:
    parameters = torch.zeros(2)
    definition = StageDefinition(
        "SYNTHETIC", 2, parameters, _raising_raw_groups, None,
        lambda _x: {"rms": 0.0, "median": 0.0, "p95": 0.0}, 1,
        input_summary=StageInputSummary(1, 2, 2, 5, 10, 0, 10),
    )
    outcome = StageOutcome(
        _fake_result(parameters), lambda x: x.sum(), definition,
    )
    result = stage_outcome_to_json(outcome)
    assert result["input_summary"] == {
        "T": 1, "param_dim": 2, "n_groups": 2,
        "total_valid_points": 5,
    }


def test_lightweight_joint_handoff() -> None:
    pose = torch.randn(75)
    beta_anchor = torch.linspace(-0.2, 0.2, 2)
    beta_abs_limit = 3.0
    beta_latent = torch.atanh(beta_anchor / beta_abs_limit)
    joint = torch.cat((pose.clone(), beta_latent))
    result = check_pose_to_joint_state(
        pose, joint, beta_anchor, beta_abs_limit, 0.1,
    )
    assert result["ok"]
    assert result["state_max_diff"] == 0.0
    assert result["raw_reprojection_max_diff"] is None
    assert result["relative_difference_joint_init_to_pose"] is None
    assert result["expensive_validation_performed"] is False
    assert result["validation_scope"] == "state_only"
    signature = inspect.signature(check_pose_to_joint_state)
    assert "raw_groups_fn" not in signature.parameters
    assert "true_objective_fn" not in signature.parameters


def test_joint_objective_terms_match_true_objective() -> None:
    cache = synthetic_joint_cache()
    parameters = synthetic_joint_parameters()
    beta_anchor = torch.tensor([0.02, -0.03], dtype=parameters.dtype)
    body_anchors = [torch.zeros(63, dtype=parameters.dtype)]
    groups = joint_fast_raw_groups(cache, parameters, 3.0)
    spec = build_primary_aggregation_spec(groups)
    for mode in ("charbonnier_irls", "raw_l2"):
        arguments = (
            cache, parameters, 3.0, beta_anchor, body_anchors,
            0.1, 0.2, mode, 1e-3, 0.01, spec,
        )
        terms = stage2_joint_fast_objective_terms(*arguments)
        objective = stage2_joint_fast_true_objective(*arguments)
        torch.testing.assert_close(terms["total"], objective, rtol=0, atol=0)
        torch.testing.assert_close(
            terms["total"],
            terms["primary"] + terms["pose_anchor"] + terms["shape_anchor"],
            rtol=0, atol=0,
        )


def test_breakdown_calls_terms_once() -> None:
    calls = 0

    def terms(parameters):
        nonlocal calls
        calls += 1
        primary = parameters.square().sum()
        zero = primary * 0.0
        return {
            "primary": primary, "pose_anchor": zero,
            "shape_anchor": zero, "total": primary,
        }

    parameters = torch.tensor([2.0])
    result = compute_objective_breakdown(
        terms, parameters, expected_true_objective=4.0,
    )
    assert calls == 1
    assert result["relative_error"] == 0.0


def test_preparation_timing_is_exact_sum() -> None:
    timing = StagePreparationTiming(1.25, 2.5, 3.75)
    assert timing.total_ms == 7.5


if __name__ == "__main__":
    test_static_aggregation_spec_matches_group_spec()
    test_run_stage_does_not_call_raw_groups()
    test_stage_json_does_not_call_raw_groups()
    test_lightweight_joint_handoff()
    test_joint_objective_terms_match_true_objective()
    test_breakdown_calls_terms_once()
    test_preparation_timing_is_exact_sum()
    print("test_preparation_overhead: PASS")
