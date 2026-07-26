# LM symbol migration

The historical `lm_solver.py` façade has been removed. Repository code uses
the canonical modules below.

| Old symbol | Canonical module | Migration status | Main callers |
|---|---|---|---|
| `_LAMBDA_MIN`, `_LAMBDA_MAX` | `lm.convergence` | REMOVED | early-stop audit |
| `_elapsed_ms` | `lm.runner` | REMOVED | early-stop audit |
| `compute_jacobian` | `lm.equations` | REMOVED | solver, validation |
| `solve_damped_normal_equations` | `lm.equations` | REMOVED | Stage1/reference tests |
| `compute_scaled_gradient_from_normal_equations` | `lm.equations` | REMOVED | runner, audit |
| `_validate_normal_equations` | `lm.equations` | REMOVED | device tests |
| `normal_equations_max_diagonal` | `lm.equations` | REMOVED | runner, audit |
| `normal_equations_hessian_norm` | `lm.equations` | REMOVED | audit |
| `normal_equations_quadratic_form` | `lm.equations` | REMOVED | trial, audit |
| `compute_scaled_gradient` | `lm.equations` | REMOVED | runner, audit |
| `_solve_structured_normal_equations` | `lm.equations` | REMOVED | historical API map |
| `solve_normal_equations` | `lm.equations` | REMOVED | runner, shadow LM |
| `_evaluate_linearization` | `lm.equations` | REMOVED | runner, audit |
| `check_convergence` | `lm.convergence` | REMOVED | runner, audit |
| `run_lm_optimization` | `lm.runner` | REMOVED | `stage_runtime`, benchmark and validation |

The migration includes private names because the benchmark-only convergence
audit intentionally uses the production primitives as an exact reference.
Monkeypatches target the module where the runner resolves a name, normally
`lm.runner`.
