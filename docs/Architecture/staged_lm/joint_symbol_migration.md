# Joint symbol migration

The historical façade modules have been removed. Code imports the module that
owns the mathematical responsibility.

| Old module | Symbols | Canonical module | Migration status | Main callers |
|---|---|---|---|---|
| `stage2_joint_fast` | `Stage2JointModelCache`, `Stage2JointFrameCameraCache`, `Stage2JointFastCache`, `validate_stage2_joint_layout`, `build_stage2_joint_fast_cache` | `joint.cache` | REMOVED | stage construction/tests |
| `stage2_joint_fast` | `Stage2JointShapeState`, `physical_beta_from_latent`, `evaluate_joint_shape_state` | `joint.shape` | REMOVED | all Joint paths |
| `stage2_joint_fast` | `evaluate_joint_group_state`, `joint_fast_selected_vertices`, `joint_fast_raw_groups`, `joint_fast_raw_residual` | `joint.forward` | REMOVED | objective/reference |
| `stage2_joint_fast` | anchor/objective/terms/metrics functions | `joint.objective` | REMOVED | stage definition/output |
| `stage2_joint_beta_jacobian` | `FKBetaJacobianResult`, `JointBetaSurrogateResult` and all beta-only Jacobian functions | `joint.beta_derivatives` | REMOVED | validation/reference |
| `stage2_joint_jacobian` | combined FK/result types and pose+beta derivative functions | `joint.derivatives` | REMOVED | serial validation |
| `stage2_joint_jacobian` | `Stage2JointLinearizationCommon`, Arrowhead accumulation/NE and analytic builder | `joint.arrowhead` | REMOVED | reference linearization |
| `stage2_joint_batched` | batch layout/cache/result types and all FULL_BATCH functions | `joint.batched` | REMOVED | production/benchmark |

Repository production, benchmark, validation, and tests use canonical imports.
