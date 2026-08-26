# staged-LM CUDA Pose Jacobian backend

## 边界

本阶段只把 Pose 相机线性化迁移到 CUDA。参数布局、LM controller、IRLS、
FP64 normal equations、线性求解、lambda 更新和终止逻辑均保持 CPU
production 实现不变。本文记录 Pose Jacobian 迁移阶段；后续 PoseShape CUDA
backend 见 `staged_lm_cuda_pose_shape_backend.md`。

```text
run_lm (CPU)
  -> CudaPoseEvaluator::linearize
     -> CUDA Pose residual forward
     -> CUDA SO(3) right-Jacobian, pose-offset derivative
     -> CUDA FK derivative, LBS derivative
     -> CUDA DenseKP Sparse-B derivative
     -> CUDA ERP derivative
     -> residual [2N], J_active [2N,69]
  -> build_pose_normal_equations (CPU, unchanged)
     -> IRLS + FP64 J^T J / J^T r + 69-to-75 scatter + pose prior
```

`J_active` 的 69 列为当前相机的 3 维旋转、3 维平移和 63 维 body
pose。旋转链本身是题目中的 Q66。相机无关的 75 维 frame scatter 仍由
既有 CPU normal-equation 代码完成。

## Backend switch

`--evaluator-backend cpu` 仍是默认 production 路径。
`--evaluator-backend cuda` 目前只切换 Pose evaluator；PoseShape 自动保持
CPU，因为 beta Jacobian 尚未迁移。不存在 CUDA normal equations 或 solver。

本阶段采用直接、可审计的 CUDA kernel，并保留逐次分配和同步；未进行
kernel fusion、allocator/cache 或性能调优。
