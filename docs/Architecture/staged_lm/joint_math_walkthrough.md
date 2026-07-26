# Stage 2 Joint mathematical walkthrough

Notation: `T` frames, `G` non-empty frame/camera groups, `N` selected points,
`J=55` SMPL-X joints, `B=10` beta dimensions, and `Qpose=66` root/body
rotation columns.

## 1. Parameter layout

The solver vector is

`x = [x_frame_1, ..., x_frame_T, z_beta]`.

Each `[75]` frame block is
`[orient2(3), transl2(3), orient4(3), transl4(3), body_pose(63)]`.
The shared tail is beta latent `[B]`, with
`beta = beta_abs_limit * tanh(z_beta)`.

## 2. Dynamic shape

`joint.shape.physical_beta_from_latent()` implements
`d beta/dz = beta_abs_limit * (1 - tanh(z)^2)`.

`v_shaped = base_v_shaped + shapedirs * beta` and
`joints = base_joints + dJ/dbeta * beta`.
Only selected shapedir rows `[N,3,B]` are materialized for vertices; joints
remain `[J,3]`. `evaluate_joint_shape_state()` exposes joint derivatives
`[J,3,B]` for physical and latent beta.

## 3. Pose blend shapes

Joint reuses `pose.forward.assemble_full_pose_rotations()` and
`compute_selected_pose_offsets()`. Unlike Pose warmup, selected shaped
vertices and joints depend on the current shared beta.

## 4. Dynamic FK and rest correction

For joint `j`,

`L_j = [R_j, J_j - J_parent; 0, 1]`.

Pose derivatives enter the rotation block; beta derivatives enter joint
translation. The global parent traversal is unchanged. Rest correction
multiplies by `[I,-J]`, so beta contributes through both global FK translation
and the correction.

## 5. Selected LBS

`T_i = sum_j w_ij G_j`, then `v_i = T_i [v_posed_i,1]`.
Pose derivatives contain rotation, pose-offset and transform contributions;
beta derivatives contain shaped-vertex, dynamic-joint and transform
contributions. Values are `[N,3]`; derivative blocks are `[N,3,Qpose]` and
`[N,3,B]`.

## 6. ERP residual

Projection remains in `geometry/erp.py`:

`d uv/d frame = d uv/d vertex · d vertex/d frame`

`d uv/d beta = d uv/d vertex · d vertex/d beta`.

For each group, `joint.derivatives.JointGroupJacobianResult` stores flattened
residual `[2N]`, local frame Jacobian `[2N,75]`, and shared beta Jacobian
`[2N,B]`. Its compatibility field `local_pose_jacobian` includes all 12 rigid
and 63 body-pose columns.

## 7. Reference derivative paths

`joint.beta_derivatives` keeps the pose-fixed beta-only reference chain.
`joint.derivatives` propagates pose and beta columns together through dynamic
FK and LBS. They intentionally remain independent validation paths.

## 8. Arrowhead H/g

Since each group touches one frame block and the shared beta block:

`A_t += J_tᵀ W J_t`

`B_t += J_tᵀ W J_beta`

`C += J_betaᵀ W J_beta`

`g_t += J_tᵀ W r`, `g_beta += J_betaᵀ W r`.

Shapes are `A[T,75,75]`, `Bcross[T,75,B]`, `C[B,B]`,
`g_frame[T,75]`, and `g_beta[B]`. Pose anchors enter the body-pose portion
of `A/g_frame`; the physical-beta shape anchor enters `C/g_beta`.

## 9. Serial and FULL_BATCH

- `beta_derivatives.py`: beta-only reference.
- `derivatives.py`: combined serial analytic reference.
- `arrowhead.py`: serial structured H/g and Linearization.
- `batched.py`: production FULL_BATCH equivalent with unchanged padding,
  FK depth layout, FP64 reductions, and Arrowhead block order.

## 10. Schur solve

`arrowhead_schur.py` owns damping and the Schur solve and is unchanged by this
refactor.

## 11. Reading order

1. `joint/cache.py`
2. `joint/shape.py`
3. `joint/forward.py`
4. `joint/beta_derivatives.py`
5. `joint/derivatives.py`
6. `joint/objective.py`
7. `joint/arrowhead.py`
8. `joint/batched.py`
9. `stage_definitions/joint.py`
10. `arrowhead_schur.py`
