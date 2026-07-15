"""Gate C and LM/L-BFGS comparison helpers for validation runs."""

from __future__ import annotations

from typing import Any, Callable

import torch

from lib_pano.human_fitting.staged_lm.types import LBFGSResult, LMResult


def stage_verdict(
    lm_result: LMResult,
    lbfgs_result: LBFGSResult | None,
    raw_metrics_fn: Callable[[torch.Tensor], dict[str, float]],
) -> str | None:
    if lbfgs_result is None:
        return None
    lbfgs_rms = raw_metrics_fn(lbfgs_result.parameters)
    ok = (
        lm_result.objective <= lbfgs_result.objective * 1.02
        and lm_result.rms_metrics["rms"] <= lbfgs_rms["rms"] * 1.05
        and lm_result.rms_metrics["median"] <= lbfgs_rms["median"] * 1.05
        and lm_result.status not in ("LM_NO_ACCEPTED_TRIAL", "NUMERICAL_FAILURE")
    )
    return "LM_MATCHES_LBFGS" if ok else "LM_LBFGS_GAP"


def evaluate_gate_c(
    *, oracle_enabled: bool, stage_matches: dict[str, bool | None],
    non_oracle_checks: dict[str, bool], oracle_checks: dict[str, bool],
) -> dict[str, Any]:
    checks: dict[str, bool | None] = {
        **stage_matches, **non_oracle_checks, **oracle_checks,
    }
    if not oracle_enabled:
        verdict = "GATE_C_NOT_EVALUATED"
    else:
        verdict = (
            "GATE_C_PASS_READY_FOR_MULTIFRAME_GATE_D"
            if all(value is True for value in checks.values())
            else "GATE_C_BLOCKED"
        )
    return {"verdict": verdict, "oracle_enabled": oracle_enabled, "checks": checks}
