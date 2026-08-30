# CUDA PoseShape evaluator backend

## 范围

`CudaPoseShapeEvaluator` 保持 85 维参数布局：每帧 75 维 camera/pose，加一组由
cam2/cam4 共享的 10 维 latent beta。CPU `run_lm`、lambda 更新、accept/reject、
termination 和 Schur solver 均未修改。

```text
latent beta
  -> bounded beta / shaped joints / dynamic selected vertices
  -> Pose Q66 + translation columns
  -> beta10 analytic derivative
     -> shaped vertex branch
     -> joint/FK/rest-transform branch
     -> LBS
  -> DenseKP Sparse-B
  -> ERP residual, J_active69, J_beta10
  -> GPU IRLS and structured accumulation
  -> D2H: A[75,75], B[75,10], C[10,10], g_pose, g_beta
  -> CPU Schur solve and LM controller
```

production normal-equation boundary不回传完整 Jacobian。Standalone parity 路径可
显式下载 beta Jacobian，用于与 CPU 分层比较。

## Beta derivative

beta 不是只作用于顶点 shapedirs。解析导数同时包含：

1. `d(v_shaped)/d(beta_latent)`；
2. `d(joints)/d(beta_latent)`；
3. joints 对 local translation、global FK 和 rest transform 的导数；
4. LBS transform term 与 vertex term；
5. DenseKP Sparse-B 与 ERP projection chain。

当前实现以 correctness-first 为目标，没有进行 kernel fusion、reduction tuning 或
allocator 优化。CPU backend 和无 CUDA stub 保留。

单帧双相机 PoseShape 共享 beta 与 multi-frame shared-beta CUDA dispatch 均已迁移。
多帧实现见
[`staged_lm_cuda_multiframe_pose_shape_backend.md`](staged_lm_cuda_multiframe_pose_shape_backend.md)。
