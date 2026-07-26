# Staged LM mathematics to code

## Levenberg--Marquardt control

The mapping from \(J,r\) to \(H=J^TJ,\ g=J^Tr\), the damped dense or
Arrowhead solve, and matrix diagnostics lives in `lm/equations.py`.
`lm/trial.py` maps \((H,g,\delta,F)\) to predicted reduction, actual
reduction, rho, acceptance, and the stable trial-history dictionaries.
`lm/convergence.py` owns damping and convergence scalar policy, while
`lm/runner.py::run_lm_optimization` owns their execution order. See
`lm_control_flow_walkthrough.md` for the complete state machine.

## SO(3)

For `phi [..., 3]`, with `theta = ||phi||`:

```text
R(phi) = I + sin(theta)/theta [phi]x
           + (1-cos(theta))/theta² [phi]x²
```

- Module: `geometry/so3.py`
- Function: `axis_angle_to_matrix`
- Input: `[..., 3]`
- Output: `[..., 3, 3]`

The additive axis-angle right Jacobian is `Jr(phi)`:

- Module: `geometry/so3.py`
- Function: `so3_right_jacobian`
- Input/output: `[..., 3] → [..., 3, 3]`

For local points `p [N, 3]`:

```text
d(Rp)/dphi = -R [p]x Jr(phi)
```

- Module: `geometry/so3.py`
- Function: `rotated_point_jacobian`
- Inputs: `phi [3]`, `p [N, 3]`
- Output: `[N, 3, 3]`

## ERP projection

For camera point `(x,y,z)`:

```text
longitude = atan2(x, z)
latitude  = asin(-y / ||(x,y,z)||)
u = (longitude / (2π) + 1/2) width
v = (1/2 - latitude / π) height
```

- Module: `geometry/erp.py`
- Functions: `erp_projection`, `erp_projection_with_jacobian`
- Input: points `[N, 3]`
- Outputs: UV `[N, 2]`, optional `d(uv)/d(xyz) [N, 2, 3]`

Unified staged-LM points first use the fixed production rotation:

```text
p_camera = p_unified R_fixedᵀ
J_unified = J_camera R_fixed
```

- Functions: `project_unified_erp`,
  `project_unified_erp_with_jacobian`
- Inputs/outputs: `[N, 3] → [N, 2]` and `[N, 2, 3]`

## Stage 1 rigid alignment

For each observed point:

```text
J_i = J_projection [d(Rp)/dphi, I]
H = JᵀJ
g = Jᵀr
```

- Module: `analytic_stage1.py`
- `stage1_raw_groups`: parameters `[T×12]` → grouped residuals `[Ni, 2]`
- `stage1_raw_jacobian`: parameters `[T×12]` → dense `J [M, T×12]`
- `stage1_normal_equations`: parameters `[T×12]` → dense
  `H [T×12, T×12]`, `g [T×12]`
- `build_stage1_analytic_linearization`: binds frozen IRLS scales and exposes
  the typed solver contract.
- Stage construction: `stage_definitions/rigid.py::build_stage1_definition`

## Pose warmup

The production dependency chain is:

```text
theta
 → joint rotations and dR/dtheta
 → pose blend offsets
 → forward kinematics
 → linear blend skinning
 → unified ERP projection
 → residual and analytic J
 → dense H/g
```

- Serial value module: `pose/forward.py`
- Serial derivative/reference H/g module: `pose/derivatives.py`
- Objective/metrics module: `pose/objective.py`
- Production FULL_BATCH module: `pose/batched.py`
- `stage2_pose_batched_rotations_and_jacobians`:
  packed pose parameters → rotations and rotation Jacobians
- `stage2_pose_batched_feature_offsets_from_rotations`:
  rotations → pose blend features/offsets
- `stage2_pose_batched_fk_by_depth`: local joints → global transforms
- `stage2_pose_batched_lbs_from_fk`: transforms → selected vertices
- `stage2_pose_batched_projection_from_vertices`:
  vertices → UV/projection Jacobian
- `stage2_pose_batched_group_analytic`:
  grouped inputs → residual and local parameter Jacobian
- `stage2_pose_batched_normal_equations`:
  parameters `[T×per_frame_dim]` → dense `H/g`
- Stage construction: `stage_definitions/pose.py::build_pose_stage_definition`

## Joint pose and beta refinement

Per-frame variables couple only through shared beta, producing the Arrowhead
system:

```text
[ A1   0   B1 ]
[  0  A2   B2 ]
[ B1ᵀ B2ᵀ  C ]
```

with frame gradients and one shared-beta gradient.

- Shape/value modules: `joint/shape.py`, `joint/forward.py`
- Reference derivative modules: `joint/beta_derivatives.py`, `joint/derivatives.py`
- Serial structured H/g: `joint/arrowhead.py`
- Production FULL_BATCH module: `joint/batched.py`
- `stage2_joint_batched_dynamic_shape`: beta → dynamic shape state
- `stage2_joint_batched_combined_fk_by_depth`: pose/beta derivatives → FK
- `stage2_joint_batched_group_analytic`:
  vertices → residual, pose Jacobian and beta Jacobian
- `stage2_joint_batched_accumulate_arrowhead`:
  grouped Jacobians → `Ai`, `Bi`, `C` and gradients
- `stage2_joint_batched_normal_equations`:
  joint parameters → `NormalEquations` with structured system
- Stage construction: `stage_definitions/joint.py::build_joint_stage_definition`

The damped Schur solve is implemented by
`ArrowheadNormalEquationSystem.solve_damped` in `arrowhead_schur.py`; it
returns the concatenated frame and beta step with the same parameter layout.
