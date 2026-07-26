# Staged LM early-stop and objective lifecycle audit

This is a read-only audit of the production solver. Production LM, stage
definitions, pipeline, output schema and timing schema v5 are unchanged.

## Control flow

The current accepted-step convergence order is:

1. frozen-IRLS surrogate scaled gradient,
2. raw Euclidean parameter-step norm,
3. relative true-objective reduction.

There is no convergence check at the initial linearization. All three normal
checks occur only after an accepted candidate has been re-linearized. If an
outer iteration accepts no trial, the separate branch reports
`CONVERGED_SCALED_GRADIENT` only when `accepted_steps > 4` and the scaled
gradient is below the hard-coded `1e-4`; otherwise it reports
`LM_NO_ACCEPTED_TRIAL`. This branch does not use
`SolverConfig.scaled_grad_tol=1e-5`.

The step criterion is `torch.norm(delta) < 1e-5`. It combines axis-angle,
translation, body-pose and beta-latent units, and its magnitude grows with the
number of frames/pose dimensions. The audit therefore also reports
`||delta|| / max(1, ||x||)` and layout-aware block norms, without changing the
production predicate.

## Frame 513 baseline

The traced runner is exact against production for parameters, status,
counters, trial/weight histories, objective, metrics and final damping.

| Stage | status | accepted/rejected | NE | solves | objective calls |
|---|---|---:|---:|---:|---:|
| Stage1 | `MAX_ACCEPTED_STEPS` | 20/6 | 21 | 26 | 67 |
| Pose | `MAX_ACCEPTED_STEPS` | 5/0 | 6 | 5 | 16 |
| Joint | `MAX_ACCEPTED_STEPS` | 1/0 | 2 | 1 | 4 |

At the production budget endpoint:

| Stage | scaled gradient / tol | raw step / tol | relative objective / tol |
|---|---:|---:|---:|
| Stage1 | 43.42 | 20,644.65 | 20,378.28 |
| Pose | 16.52 | 55,025.77 | 377,579.86 |
| Joint | 8.31 | 3,156.58 | 75,569.44 |

None of the three predicates is reached. The closest is the Joint surrogate
scaled-gradient check, but it remains 8.31 times above its configured
tolerance.

The Stage1 last step is orientation-dominated (`0.20641` orientation versus
`0.00377` translation). The Pose last step is body-pose dominated (`0.54959`
pose versus `0.02705` rigid). The Joint step has contributions from pose
(`0.02049`) and shared beta latent (`0.02365`), with physical-beta change
norm `0.05448`. Thus raw step norm is not a scale-neutral cross-stage
criterion.

## Final-solution reinitialization

The supplied production finals are budget-limited, not converged:

| Stage | initial scaled gradient | actual result from reinit | parameter drift |
|---|---:|---|---:|
| Stage1 | `4.3417e-4` | 20 accepted, 6 rejected, budget reached | 2.82097 |
| Pose | `1.6518e-4` | 5 accepted, 0 rejected, budget reached | 0.38080 |
| Joint | `8.3084e-5` | 1 accepted, 0 rejected, budget reached | 0.02659 |

All three initial gradients exceed configured `1e-5`, so frame 513 does not
demonstrate “an already converged initial point ignored by production.”
It does demonstrate that the saved production points still have accepted
descent directions.

Frame 514 Stage1 is informative: reinitializing its production final reaches
`CONVERGED_SCALED_GRADIENT` after four more accepted steps. Early-stop can
respond to a higher-quality initialization, but only after taking at least one
step because there is no initial-point check.

## Extended budgets

For frame 513, the production prefixes are exact:

| Stage | extended result | objective: production → extended |
|---|---|---|
| Stage1 | 40 accepted, still budget-limited | 37.863216 → 37.681122 |
| Pose | 15 accepted, still budget-limited | 12.398867 → 12.036642 |
| Joint | 5 accepted, still budget-limited | 12.305170 → 11.999574 |

For frame 514, Stage1 triggers scaled-gradient convergence at accepted step
27 and Pose at step 15; Joint remains budget-limited at five steps. The
two-frame and four-frame sets remain budget-limited at the requested extended
budgets. Frames 513–520 are skipped because frame 519 SAM3D dense data is
missing.

The production last-step relative reductions are far above `1e-7`, so
“early-stop is broken” is not the correct explanation for the usual full
budgets. The accurate conclusion is that current budgets end before the
strict convergence tolerances are reached.

## Objective-call lifecycle

Frame 513 calls decompose exactly as:

| Stage | outer-before | candidate | accepted recheck | final | total |
|---|---:|---:|---:|---:|---:|
| Stage1 | 20 | 26 | 20 | 1 | 67 |
| Pose | 5 | 5 | 5 | 1 | 16 |
| Joint | 1 | 1 | 1 | 1 | 4 |

For every accepted candidate, the candidate parameters equal the relative
recheck parameters exactly and their objectives are bitwise equal. The next
outer reuses exactly the same parameter point and returns the same objective.
The final evaluation is also exact at the current point. Therefore:

```text
EXACT_OBJECTIVE_REUSE_PROVEN
```

Keeping acceptance, rho, relative-objective checks and final result identical,
the theoretical calls are one initial-current objective plus one objective
per successful solve trial. Frame 513 can move from:

- Stage1: 67 to 27 calls (40 removable);
- Pose: 16 to 6 calls (10 removable);
- Joint: 4 to 2 calls (2 removable).

The final-budget re-linearization has no next solver iteration. It still
serves the final scaled-gradient status check and final IRLS weight history,
so removing it is not behavior-preserving under the current output/status
contract.

## IRLS interpretation

`scaled_gradient` is computed from the frozen-current-weight surrogate
`Jᵀr/JᵀJ`. Acceptance and relative reduction use the true Charbonnier
objective. The audit names these quantities `surrogate_scaled_gradient` and
`true_objective_relative_change`; it does not claim that the former is an
independently evaluated full Charbonnier analytic gradient.

## Recommendation

Primary classification:

```text
CURRENT_BUDGET_ENDS_BEFORE_CONVERGENCE
```

The next unique implementation task should be
`EXACT_OBJECTIVE_VALUE_REUSE`. It has an exact duplicate proof, preserves
trial acceptance, rho, status, final parameters and output values, and avoids
more calls than any other behavior-preserving option. Initial-point gradient
checking and config-consistent no-accepted status should follow because they
intentionally change status/control flow. Tolerance or budget changes require
a separate quality-versus-cost policy decision.

## Exact objective value reuse

The optimization is now implemented in production. The old lifecycle was:

```text
outer current evaluation
→ candidate evaluation
→ accepted-point recheck
→ next-outer current evaluation
→ final evaluation
```

The new lifecycle is:

```text
initial current evaluation
→ one evaluation per candidate whose solve succeeded
→ accepted candidate value becomes the current value
→ next outer and final result read that current value
```

This is one Python float local to a single `run_lm_optimization()` call. It is
not parameter memoization, does not hash or compare tensors, and is never
shared across stages or solver invocations. Rejected candidates do not update
the current value. The accepted-point re-linearization, convergence predicate
order, final-budget normal equations and weight history are unchanged.

For frame 513, true-objective executions changed from 67 to 27 in Stage1,
16 to 6 in Pose and 4 to 2 in Joint. The final parameters, objectives,
metrics, trial histories, weight histories, damping and Schur path remain
exact. Objective counters and elapsed time now describe actual executions.

CPU p50 solver-pipeline results:

| Frames | Legacy | Reuse | Reduction | FPS change |
|---|---:|---:|---:|---:|
| 513 | 472.60 ms | 379.75 ms | 19.65% | 2.12 → 2.63 |
| 513,514 | 823.78 ms | 635.70 ms | 22.83% | 2.43 → 3.15 |
| 513–516 | 1546.88 ms | 1189.21 ms | 23.12% | 2.59 → 3.36 |

CUDA was unavailable in the audit environment. T8 was skipped because no
complete continuous eight-frame interval was available in the supplied data.
