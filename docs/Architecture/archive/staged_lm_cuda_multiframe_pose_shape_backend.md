# Multi-frame CUDA PoseShape evaluator

## 参数和控制边界

多帧参数布局保持不变：

```text
[frame_0 pose75, ..., frame_(T-1) pose75, shared beta10]
```

`CudaMultiFramePoseShapeEvaluator` 持有每帧一个长期存在的
`CudaPoseShapeEvaluator`。每帧 evaluator 读取自己的 pose75 和同一份 beta10，在 GPU
上完成 forward、PoseShape Jacobian、DenseKP Sparse-B、ERP/IRLS 和 normal-equation
累积。当前是 correctness-first 的逻辑 batch，没有跨帧 kernel fusion。

```text
frame t: pose75_t + shared beta10
  -> CUDA forward/Jacobian/IRLS
  -> A_t[75,75], B_t[75,10], C_t[10,10]
  -> g_pose_t[75], g_beta_t[10]

all frames
  -> shared C = sum_t C_t
  -> shared g_beta = sum_t g_beta_t
  -> CPU multi-frame Schur
  -> CPU LM lambda/rho、accept/reject、termination
```

shape prior 是全局 shared-beta 项，只计一次。单帧 evaluator 为方便独立使用会包含一份
shape prior；多帧聚合时显式扣除重复的 `T-1` 份，避免改变原 CPU 数学。

## 接口行为

- `forward()`：按帧切片参数并调用相应 CUDA evaluator；
- `evaluate_residual()`：累加各帧 objective，并修正重复 shape prior；
- `linearize()`：产生各帧 CUDA linearization；
- `build_normal_equations()`：聚合 frame blocks 和 shared-beta blocks，继续调用现有 CPU
  Schur solver；
- `T=1`：直接走原有 single-frame PoseShape CUDA 路径，保证 trajectory 和参数文件精确一致；
- 默认 backend 仍为 `cpu`，`--evaluator-backend cuda` 显式启用 CUDA。

未修改 parameter layout、loss/IRLS、LM controller、Schur、停止条件或 tolerance。

## 当前性能含义

本实现先验证正确性。每帧仍有独立 CUDA workspace 和 evaluator 调度，因此不是最终的
高吞吐多帧 GPU 实现。后续性能工作可在不改变接口的前提下进行跨帧 kernel batching、
统一 workspace 和减少启动同步。
