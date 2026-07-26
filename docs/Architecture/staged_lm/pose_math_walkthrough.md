# Stage 2 Pose mathematical walkthrough

Notation: `T` is the segment frame count, `N` is the selected observation
count in one frame/camera group, `J=55` is the SMPL-X joint count,
`Q_frame=75`, and the active rotation dimension is `Q_rot=66`.

## 1. Pose parameters

`parameter_layout.unpack_frame_parameters()` views each frame as:

- cam2 orientation `[3]` and translation `[3]`;
- cam4 orientation `[3]` and translation `[3]`;
- body pose `[63]` (21 axis-angle joints).

Thus the flattened Stage2 Pose vector is `[T * 75]`. The slicing remains in
`pose.forward.fast_raw_groups()` and the batched gather remains in
`pose.batched.gather_pose_batch_parameters()`.

## 2. Rotation

For each joint, `R_j(theta_j)` is evaluated in
`pose.forward.assemble_full_pose_rotations()`, producing `[J,3,3]`.
`pose.derivatives.rotation_matrix_jacobian_additive()` evaluates
`dR_j/dtheta_j` with shape `[3,3,3]`, while
`assemble_full_pose_rotation_jacobians()` assembles `[J,3,3,66]`.

## 3. Pose feature

The non-root rotations have identity removed and are flattened:

`pose_feature = flatten(R[1:] - I)`.

For 55 joints this is `[54*9] = [486]`. The value/Jacobian pair is implemented
by `compute_pose_feature_with_jacobian()` and has derivative `[486,66]`.

## 4. Pose blend shape

For selected vertices,

`Delta V_pose = posedirs · pose_feature`.

`pose.forward.compute_selected_pose_offsets()` preserves the existing
`einsum("p,pnc->nc", ...)`, producing `[N,3]`. Its serial derivative counterpart
produces `[N,3,66]`.

## 5. Forward kinematics

Local transforms combine each local rotation with the shaped joint location.
Parent transforms are traversed in model parent order to produce global
`[J,4,4]` transforms, followed by the existing rest-pose correction.
The value path calls `smplx.lbs.batch_rigid_transform`; the explicit chain is
visible in `pose.derivatives.compute_fk_with_jacobian()`, returning value
transforms `[J,4,4]` and derivatives `[J,4,4,66]`.

## 6. Linear blend skinning

For selected vertex `i`,

`T_i = sum_j w_ij G_j`, and `v_i = T_i [v_i_posed,1]^T`.

Only selected vertices are materialized. The value implementation is
`pose.forward.compute_selected_lbs()`; the derivative implementation is
`pose.derivatives.compute_selected_lbs_with_jacobian()`. Their outputs are
`[N,3]` and `[N,3,66]`.

## 7. Frame rigid transform

The installed SMPL-X path applies the selected camera orientation in the root
rotation and then:

`P_i = scale * v_i + translation`.

`pose.forward.fast_selected_vertices()` preserves this exact operation order.
The serial derivative path adds the translation columns without changing the
rotation convention.

## 8. ERP projection

Projection is owned by `geometry/erp.py`. The Pose chain is:

`d uv/dtheta = d uv/dP · dP/dv · dv/dtheta`.

`pose.derivatives.stage2_pose_group_raw_residual_and_jacobian()` chains the
projection Jacobian with selected-vertex derivatives. Projected values are
`[N,2]`; projection/residual derivatives are `[N,2,Q_frame]`.

## 9. Residual

Predicted ERP UV is subtracted from observations in frame order and, within
each frame, cam2 then cam4 group order. Horizontal seam handling and weighting
remain in the shared residual/IRLS implementation. Raw groups are `[N,2]`;
flattening remains row-major in `flatten_reprojection_groups()`.

## 10. Complete Jacobian chain

```text
body pose [63]
  → local rotations [J,3,3]
  → pose feature [486] + FK transforms [J,4,4]
  → selected LBS vertices [N,3]
  → frame rigid vertices [N,3]
  → ERP UV [N,2]
  → residual [N,2]
  → J [2N,Q_frame]
```

The main serial entry is
`pose.derivatives.stage2_pose_group_raw_residual_and_jacobian()`. It explicitly
keeps the value path and derivative path paired at each chain-rule boundary.

## 11. Normal equations

The frozen-IRLS system is:

`H = Jᵀ W J`, `g = Jᵀ W r`.

The serial/reference construction is
`pose.derivatives.stage2_pose_normal_equations()`. The production equivalent is
`pose.batched.stage2_pose_batched_normal_equations()`, which preserves static
padding, full-group batching, combined FK, selected LBS, production-order
reduction, and the existing FP64 accumulation boundaries.

## 12. Reading order

1. `pose/cache.py`
2. `pose/forward.py`
3. `pose/derivatives.py`
4. `pose/objective.py`
5. `pose/batched.py`
6. `stage_definitions/pose.py`

The historical `stage2_pose_*.py` façade modules have been removed.
