# Staged LM architecture

## Production data flow

```text
StageContext
  → Stage 1 rigid alignment
  → Stage 2 pose warmup
  → Stage 2 joint pose + beta refinement
  → JSON / NPZ output
```

`pipeline.py` orchestrates Stage 2, while `stage_definitions/` constructs the
three fixed production definitions. `lm/runner.py` is stage-independent: it consumes one
`Linearization` per accepted iterate and controls damping, trial acceptance,
convergence and the dense/Arrowhead solve dispatch.

| Stage | Parameter dimension | Residuals | Jacobian / normal equations | Solver |
|---|---:|---|---|---|
| Stage 1 rigid | `T × 12` (two camera `orient/transl` blocks) | dual-view ERP reprojection | analytic orientation/translation Jacobian; direct dense `H/g` | dense Cholesky, then dense solve fallback |
| Pose warmup | `T × (12 + body_pose_dim)` (currently 75 per frame) | ERP reprojection + pose anchor | analytic batched pose Jacobian; direct dense `H/g` | dense Cholesky, then dense solve fallback |
| Joint refinement | `T × (12 + body_pose_dim) + beta_dim` (currently `75T + 10`) | ERP reprojection + pose and shape anchors | analytic batched pose/shared-beta Jacobians; Arrowhead `A/B/C/g` | Arrowhead Schur, then dense fallback |

Optimization/model geometry stays in FP32. Normal-equation accumulation and
linear solves stay in FP64. This refactor does not alter either order.

## Module responsibilities

- `pipeline.py`: Stage 2 pose/joint orchestration.
- `stage_runtime.py`: generic execution of one constructed stage.
- `stage_definitions/rigid.py`: Stage 1 rigid definition construction.
- `stage_definitions/pose.py`: pose-warmup definition construction.
- `stage_definitions/joint.py`: joint pose/beta definition construction.
- `stage_handoff.py`: parameter transfer and state-only handoff checks.
- `stage_timing.py`: preparation/cache timing serialization.
- `lm/equations.py`: Jacobian/H/g evaluation and dense/structured solve dispatch.
- `lm/trial.py`: trial reduction, acceptance, and history records.
- `lm/convergence.py`: damping and convergence policy.
- `lm/runner.py`: the local-variable LM control-flow state machine.
- `linearization.py`: the typed, immutable stage-to-solver contract.
- `geometry/so3.py`: shared Rodrigues, right-Jacobian and rotated-point math.
- `geometry/erp.py`: shared ERP projection and analytic point Jacobian.
- `analytic_stage1.py`: Stage 1 cache, forward, residual, Jacobian and `H/g`.
- `pose/cache.py`: immutable Pose model/observation rows and batch layout.
- `pose/forward.py`: serial Pose value chain.
- `pose/derivatives.py`: serial/reference analytic chain and dense `H/g`.
- `pose/objective.py`: true objective and raw metrics.
- `pose/batched.py`: production FULL_BATCH Pose forward/Jacobian/dense `H/g`.
- `joint/cache.py` and `joint/shape.py`: static cache and dynamic shared beta state.
- `joint/forward.py`: Joint serial value chain.
- `joint/beta_derivatives.py` and `joint/derivatives.py`: beta-only and combined references.
- `joint/objective.py`: objective/anchors/metrics.
- `joint/arrowhead.py`: serial structured `A/B/C/g`.
- `joint/batched.py`: production FULL_BATCH Joint forward/Jacobian/Arrowhead `H/g`.
- `arrowhead_schur.py`: damped Arrowhead Schur solve.

`Linearization` is the only formal contract between a stage and the solver.
It contains the surrogate callable, optional analytic Jacobian, optional
direct normal-equation callable, frozen-weight metadata and output
provenance. It deliberately has no dictionary interface.

## Reader path

1. `pipeline.py`
2. `stage_runtime.py`
3. `stage_definitions/rigid.py`
4. `stage_definitions/pose.py`
5. `stage_definitions/joint.py`
6. `stage_handoff.py`
7. `stage_timing.py`
8. `linearization.py`
9. `lm/equations.py`
10. `lm/trial.py`
11. `lm/convergence.py`
12. `lm/runner.py`
10. `geometry/so3.py` and `geometry/erp.py`
11. `pose/cache.py`, `pose/forward.py`, `pose/derivatives.py`,
    `pose/objective.py`, and `pose/batched.py`
12. `joint/cache.py` through `joint/batched.py`, then `arrowhead_schur.py`

This order separates pipeline orchestration, generic LM execution, stage
construction and handoff/timing concerns before reaching solver control and
the mathematical kernels. New code should use these canonical modules;
The historical `stages.py`, `stage2_pose_*`, `stage2_joint_*`, and
`lm_solver.py` façade modules have been removed. Canonical imports are required.
