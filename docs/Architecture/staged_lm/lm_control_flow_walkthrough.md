# LM control-flow walkthrough

The canonical reading path is `linearization.py`, `lm/equations.py`,
`lm/trial.py`, `lm/convergence.py`, then `lm/runner.py`.

## Solver inputs

`run_lm_optimization` receives `make_linearization`, `raw_metrics_fn`,
`initial_parameters`, `param_dim`, the accepted-step and trial budgets,
`SolverConfig`, the timing flag, and `true_objective_fn`. `param_dim` remains
part of the stable interface.

## Linearization and normal equations

`Linearization` carries a surrogate, an optional analytic Jacobian, and
optional direct `NormalEquations`. `lm.equations._evaluate_linearization`
selects one of three unchanged paths:

1. stage-provided direct H/g;
2. stage-provided analytic J;
3. `jacfwd` fallback.

For an explicit residual and Jacobian, \(H=J^T J\) and \(g=J^T r\), with the
existing FP64 conversion boundary. Joint may instead supply the structured
Arrowhead representation.

## Initial damping and solve

After initial linearization, the runner computes
\(\lambda_0=10^{-3}\max_i H_{ii}\). The damping bounds remain \(10^{-10}\)
and \(10^{12}\).

Each trial solves

\[
(H+\lambda I)\delta=-g.
\]

Dense systems try Cholesky and retain the existing dense fallback. Structured
systems use Arrowhead Schur and retain structured-to-dense fallback.

## Candidate and reduction

The exact update remains:

```python
candidate = (x + delta.to(dtype=torch.float32)).detach()
```

The solved delta and H/g calculations remain FP64. `lm.trial` evaluates:

\[
\mathrm{predicted}=-g^T\delta-\tfrac12\delta^TH\delta,
\quad
\mathrm{actual}=F(x)-F(x+\delta),
\]

\[
\rho =
\begin{cases}
\mathrm{actual}/\max(\mathrm{predicted},10^{-12}),&
\mathrm{predicted}>0\\
0,&\text{otherwise.}
\end{cases}
\]

A candidate is accepted only when predicted reduction, actual reduction, and
rho are all strictly positive.

## Damping update

- accepted and `rho > 0.75`: divide damping by three, bounded below;
- accepted and `rho < 0.25`: multiply by three, bounded above;
- otherwise accepted: unchanged;
- rejected or solve failure: multiply by ten, bounded above.

The strict 0.75 and 0.25 boundaries are intentional.

## Exact objective reuse

The runner evaluates the true objective once at the initial current point and
once for every successfully formed candidate. On acceptance, the candidate
value becomes `current_true_objective`. Rejected candidates do not change it.
The next outer iteration, relative-objective check, and final result reuse that
Python float. This is local state propagation, not general memoization.

## Accepted-point relinearization

Every accepted point rebuilds frozen IRLS weights and H/g, recomputes the
scaled gradient, and appends weight history. This includes the final
budget-limited accepted step. The final relinearization is intentionally not
removed.

## Convergence

Normal runs check convergence only after an accepted-point relinearization:

1. scaled gradient;
2. raw Euclidean step norm;
3. relative true-objective change.

There is no initial-point convergence check. The no-accepted legacy branch is
also deliberately unchanged: it reports `CONVERGED_SCALED_GRADIENT` only when
`accepted_steps > 4` and `scaled_gradient < 1e-4`; otherwise it reports
`LM_NO_ACCEPTED_TRIAL`. This rule is not `SolverConfig`-consistent.

## Result

The final parameter clone and raw metrics retain their timing boundary. The
result contains status, counters, trial and weight histories, objective,
metrics, timing, aggregation spec, and backend provenance. Console formatting
and all status strings remain unchanged.

## State machine

```text
INITIALIZE
  ↓
OUTER
  ↓
TRIAL_SOLVE
  ├─ solve fail → increase λ → retry
  ├─ reject     → increase λ → retry
  └─ accept
        ↓
    RELINEARIZE
        ↓
    CONVERGENCE?
      ├─ yes → FINAL
      └─ no  → OUTER
```

