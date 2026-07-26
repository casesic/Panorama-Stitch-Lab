# Pose symbol migration

The historical façade modules have been removed. Code imports the canonical
module that owns each responsibility.

| Old module | Old symbol(s) | Canonical module | Migration status | Main callers |
|---|---|---|---|---|
| `stage2_pose_fast` | `Stage2PoseModelCache`, `Stage2PoseFrameCameraCache`, `Stage2PoseFastCache`, `build_stage2_pose_fast_cache` | `pose.cache` | REMOVED | stage definition, validation, tests |
| `stage2_pose_fast` | `assemble_full_pose_rotations`, `compute_selected_pose_offsets`, `compute_selected_lbs`, `fast_selected_vertices`, `fast_raw_groups`, `fast_raw_residual` | `pose.forward` | REMOVED | serial Jacobian, Joint kernels, validation |
| `stage2_pose_fast` | `stage2_pose_fast_true_objective`, `stage2_pose_fast_raw_metrics` | `pose.objective` | REMOVED | stage definition, validation |
| `stage2_pose_fast` | `_aa_to_rotmat` | `lib_dynamic_hvip.smpl.smpl_lib` | REMOVED | compatibility/tests |
| `stage2_pose_jacobian` | `FKJacobianResult`, `SurrogateJacobianResult`, `Stage2PoseLinearizationCommon` | `pose.derivatives` | REMOVED | validation/reference |
| `stage2_pose_jacobian` | `ROTATION_PARAMETER_DIM`, `POSE_PER_FRAME_DIM`, `validate_stage2_pose_analytic_layout` | `pose.derivatives` | REMOVED | validation/tests |
| `stage2_pose_jacobian` | `rotation_matrix_jacobian_additive`, `assemble_full_pose_rotation_jacobians`, `compute_pose_feature_with_jacobian`, `compute_selected_pose_offsets_with_jacobian`, `compute_fk_with_jacobian`, `compute_selected_lbs_with_jacobian` | `pose.derivatives` | REMOVED | serial/reference path |
| `stage2_pose_jacobian` | `fast_selected_vertices_with_jacobian`, `stage2_pose_group_raw_residual_and_jacobian`, `stage2_pose_raw_residual`, `stage2_pose_explicit_raw_jacobian` | `pose.derivatives` | REMOVED | validation/reference |
| `stage2_pose_jacobian` | `stage2_pose_anchor_residual_and_jacobian`, `stage2_pose_surrogate_residual_and_jacobian`, `stage2_pose_normal_equations`, `build_stage2_pose_linearization_common`, `build_stage2_pose_analytic_linearization` | `pose.derivatives` | REMOVED | serial/reference linearization |
| `stage2_pose_batched` | `Stage2PoseBatchLayout`, `Stage2PoseBatchParameters`, `Stage2PoseBatchCache`, `build_stage2_pose_batch_cache` | `pose.batched` | REMOVED | production stage definition |
| `stage2_pose_batched` | `gather_pose_batch_parameters`, `stage2_pose_batched_rotations_and_jacobians`, `stage2_pose_batched_feature_offsets_from_rotations`, `stage2_pose_batched_fk_by_depth`, `stage2_pose_batched_lbs_from_fk`, `stage2_pose_batched_projection_from_vertices` | `pose.batched` | REMOVED | production FULL_BATCH path/benchmarks |
| `stage2_pose_batched` | `stage2_pose_batched_group_analytic`, `pad_stage2_pose_row_scales`, `stage2_pose_batched_normal_equations`, `stage2_pose_batched_accumulate_normal_equations`, `build_stage2_pose_batched_linearization` | `pose.batched` | REMOVED | production FULL_BATCH H/g |

Repository production, benchmark, validation, and tests use canonical imports.
