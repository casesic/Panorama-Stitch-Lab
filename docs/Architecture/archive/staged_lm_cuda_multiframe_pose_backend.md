# Multi-frame CUDA Pose evaluator

## 目标与边界

多帧 Pose 阶段仍是 `T` 个互不耦合的 75 维 LM 问题。每帧分别保存
`x_t`、`lambda_t`、accepted/rejected trial 和收敛状态；没有把它们错误地拼成
一个大 Hessian。

当前调用关系为：

```text
CPU run_lm_batch（T 个独立状态机，锁步调度）
  -> CudaMultiFramePoseEvaluator::linearize_and_build(active frames)
       -> pose forward / Q66 Jacobian / DenseKP Sparse-B / ERP
       -> IRLS / per-frame H_pose、g_pose
  <- 每帧 75x75 H_pose、75 g_pose
  -> CPU Cholesky/dense solve、lambda/rho、accept/reject、termination
```

T=1 明确走原 `run_pose_lm()`，用于保证单帧 CUDA 路径完全不变。CPU
multi-frame 路径也保持原来的逐帧入口。PoseShape、Schur、参数布局、loss 和
tolerance 均未修改。

## CUDA 调度

批量 evaluator 为每帧长期持有一个 `CudaPoseEvaluator` 及其 cam2/cam4
workspace。一次 linearization 批次中提交所有 active frame，并把逐 kernel
同步延迟到批次末尾；同一帧内部及帧间仍使用 CUDA 默认流的确定性顺序。

`--sequential-cuda-pose` 是只用于 A/B 回归的诊断开关。默认 CUDA 多帧入口
使用 `pose_cuda_scheduling=batch`。

## 当前性能解释

后续 memory/reduction 优化已经加入连续参数上传和一次性 H/g D2H，详见
`staged_lm_cuda_multiframe_pose_memory_reduction.md`。当前尚未实现 multi-frame
fused kernel 或异步 value-output pipeline，因此 batch 接口本身仍不等于显著
FPS 收益。
